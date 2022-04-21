import torch
from collections import OrderedDict
from torch.optim import SGD
from model.DCTNet import Model
from model.Spatial import Spatial
from model.Flow import Flow
from model.Depth import Depth
import os
import numpy as np
import cv2
import IOU
import datetime
import visdom

p = OrderedDict()
p['lr_bone'] = 1e-4  # Learning rate
p['lr_branch'] = 1e-3
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [9, 20]
showEvery = 50


CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)


def structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce + iou


class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None, datasampler=None):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold
        self.datasampler = datasampler
        self.build_model()

        if config.mode == 'test':
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
        print(name)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        print('mode: {}'.format(self.config.mode))
        print('------------------------------------------')
        if self.config.mode == 'pretrain_rgb':
            self.net_bone = Spatial(3, mode=self.config.mode)
            self.name = "Spatial"
            if self.config.cuda:
                self.net_bone = self.net_bone.cuda()

        elif self.config.mode == 'pretrain_depth':
            self.net_bone = Depth(3, mode=self.config.mode)
            self.name = "Depth"
            if self.config.cuda:
                self.net_bone = self.net_bone.cuda()

        elif self.config.mode == 'pretrain_flow':
            self.net_bone = Flow(3, mode=self.config.mode)
            self.name = "Flow"
            if self.config.cuda:
                self.net_bone = self.net_bone.cuda()
                
        elif self.config.mode == 'train':
            self.net_bone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Model(3, mode=self.config.mode, spatial_ckpt=self.config.spatial_ckpt,
                                                                                flow_ckpt=self.config.flow_ckpt, depth_ckpt=self.config.depth_ckpt))
            self.net_bone = torch.nn.parallel.DistributedDataParallel(self.net_bone.cuda(self.config.local_rank), device_ids=[self.config.local_rank], find_unused_parameters=True)
            self.name = "DCTNet"


        elif self.config.mode == 'test':
            self.net_bone = Model(3, mode=self.config.mode, spatial_ckpt=self.config.spatial_ckpt,
                                  flow_ckpt=self.config.flow_ckpt, depth_ckpt=self.config.depth_ckpt)
            self.name = "DCTNet"
            if self.config.cuda:
                self.net_bone = self.net_bone.cuda()
            assert (self.config.model_path != ''), ('Test mode, please import pretrained model path!')
            assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
            print('load model……all checkpoints')
            ckpt = torch.load(self.config.model_path)
            model_dict = self.net_bone.state_dict()
            pretrained_dict = {k[7:]: v for k, v in ckpt.items() if k[7:] in model_dict}
            model_dict.update(pretrained_dict)
            self.net_bone.load_state_dict(model_dict)

        base, head = [], []
        for name, param in self.net_bone.named_parameters():
            if 'rgb_bkbone' in name or 'flow_bkbone' in name or 'depth_bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        self.optimizer_bone = SGD([{'params': base}, {'params': head}], lr=p['lr_bone'],
                                  momentum=p['momentum'], weight_decay=p['wd'], nesterov=True)
        print('------------------------------------------')
        self.print_network(self.net_bone, self.name)
        print('------------------------------------------')

    def test(self):
        if not os.path.exists(self.save_fold):
            os.makedirs(self.save_fold)
        for i, data_batch in enumerate(self.test_loader):
            print(i)
            image, flow, depth, name, split, size = data_batch['image'], data_batch['flow'], data_batch['depth'], \
                                                    data_batch['name'], data_batch['split'], data_batch['size']
            dataset = data_batch['dataset']

            if self.config.cuda:
                image, flow, depth = image.cuda(), flow.cuda(), depth.cuda()
            with torch.no_grad():

                decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5 = self.net_bone(
                    image, flow, depth)

                for i in range(self.config.test_batch_size):
                    presavefold = os.path.join(self.save_fold, dataset[i], split[i])

                    if not os.path.exists(presavefold):
                        os.makedirs(presavefold)
                    pre1 = torch.nn.Sigmoid()(decoder_out1[i])
                    pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
                    pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
                    pre1 = cv2.resize(pre1, (size[0][1], size[0][0]))
                    cv2.imwrite(presavefold + '/' + name[i], pre1)

    def pretrainrgb(self):

        iter_num = len(self.train_loader)
        vis = visdom.Visdom(env='Spatial')
        for epoch in range(self.config.epoch):
            loss_all = 0
            self.optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
            self.optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):

                image, label = data_batch['image'], data_batch['label']

                if image.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    image, label = image.cuda(), label.cuda()

                decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5 = self.net_bone(image)

                loss1 = structure_loss(decoder_out1, label)
                loss2 = structure_loss(decoder_out2, label)
                loss3 = structure_loss(decoder_out3, label)
                loss4 = structure_loss(decoder_out4, label)
                loss5 = structure_loss(decoder_out5, label)

                loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

                self.optimizer_bone.zero_grad()
                loss.backward()
                self.optimizer_bone.step()
                loss_all += loss.data

                if i % showEvery == 0:
                    print(
                        '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || sum : %10.4f' % (
                            datetime.datetime.now(), epoch, self.config.epoch, i, iter_num,
                            loss1.data, loss_all / (i + 1)))

                    print('Learning rate: ' + str(self.optimizer_bone.param_groups[0]['lr']))

                if i % 50 == 0:
                    vis.images(torch.sigmoid(decoder_out1), win='predict.jpg', opts=dict(title='predict.jpg'))
                    vis.images(label, win='sal-target.jpg', opts=dict(title='sal-target.jpg'))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                p['lr_bone'] = p['lr_bone'] * 0.2
                p['lr_branch'] = p['lr_branch'] * 0.2

        torch.save(self.net_bone.state_dict(), '%s/final_bone.pth' % self.config.save_fold)

    def pretrainflow(self):

        iter_num = len(self.train_loader)
        vis = visdom.Visdom(env='Flow')
        for epoch in range(self.config.epoch):
            loss_all = 0
            self.optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
            self.optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):

                label, flow = data_batch['label'], data_batch['flow']

                if flow.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    label, flow = label.cuda(), flow.cuda()

                decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5 = self.net_bone(flow)

                loss1 = structure_loss(decoder_out1, label)
                loss2 = structure_loss(decoder_out2, label)
                loss3 = structure_loss(decoder_out3, label)
                loss4 = structure_loss(decoder_out4, label)
                loss5 = structure_loss(decoder_out5, label)

                loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

                self.optimizer_bone.zero_grad()
                loss.backward()
                self.optimizer_bone.step()
                loss_all += loss.data

                if i % showEvery == 0:
                    print(
                        '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || sum : %10.4f' % (
                            datetime.datetime.now(), epoch, self.config.epoch, i, iter_num,
                            loss1.data, loss_all / (i + 1)))

                    print('Learning rate: ' + str(self.optimizer_bone.param_groups[0]['lr']))

                if i % 50 == 0:
                    vis.images(torch.sigmoid(decoder_out1), win='predict.jpg', opts=dict(title='predict.jpg'))
                    vis.images(label, win='sal-target.jpg', opts=dict(title='sal-target.jpg'))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                p['lr_bone'] = p['lr_bone'] * 0.2
                p['lr_branch'] = p['lr_branch'] * 0.2

        torch.save(self.net_bone.state_dict(), '%s/final_bone.pth' % self.config.save_fold)

    def pretraindepth(self):

        iter_num = len(self.train_loader)
        vis = visdom.Visdom(env='Depth')
        for epoch in range(self.config.epoch):
            loss_all = 0
            self.optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
            self.optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):

                label, depth = data_batch['label'], data_batch['depth']

                if depth.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    label, depth = label.cuda(), depth.cuda()

                decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5 = self.net_bone(depth)

                loss1 = structure_loss(decoder_out1, label)
                loss2 = structure_loss(decoder_out2, label)
                loss3 = structure_loss(decoder_out3, label)
                loss4 = structure_loss(decoder_out4, label)
                loss5 = structure_loss(decoder_out5, label)

                loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

                self.optimizer_bone.zero_grad()
                loss.backward()
                self.optimizer_bone.step()
                loss_all += loss.data

                if i % showEvery == 0:
                    print(
                        '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || sum : %10.4f' % (
                            datetime.datetime.now(), epoch, self.config.epoch, i, iter_num,
                            loss1.data, loss_all / (i + 1)))

                    print('Learning rate: ' + str(self.optimizer_bone.param_groups[0]['lr']))

                if i % 50 == 0:
                    vis.images(torch.sigmoid(decoder_out1), win='predict.jpg', opts=dict(title='predict.jpg'))
                    vis.images(label, win='sal-target.jpg', opts=dict(title='sal-target.jpg'))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                p['lr_bone'] = p['lr_bone'] * 0.2
                p['lr_branch'] = p['lr_branch'] * 0.2

        torch.save(self.net_bone.state_dict(), '%s/final_bone.pth' % self.config.save_fold)

    def train_distributed(self):
        self.optimizer_bone.zero_grad()
        iter_num = len(self.train_loader)
        vis = visdom.Visdom(env='train')
        for epoch in range(self.config.epoch):
            loss_all = 0
            self.optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
            self.optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
            self.net_bone.zero_grad()
            self.datasampler.set_epoch(epoch)
            self.net_bone.train()

            for i, data_batch in enumerate(self.train_loader):

                image, label, flow, depth = data_batch['image'], data_batch['label'], data_batch['flow'], data_batch[
                    'depth']

                if image.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    image, label, flow, depth = image.cuda(self.config.local_rank), label.cuda(self.config.local_rank), flow.cuda(
                        self.config.local_rank), depth.cuda(self.config.local_rank)

                decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5 = self.net_bone(
                    image, flow, depth)

                loss1 = structure_loss(decoder_out1, label)
                loss2 = structure_loss(decoder_out2, label)
                loss3 = structure_loss(decoder_out3, label)
                loss4 = structure_loss(decoder_out4, label)
                loss5 = structure_loss(decoder_out5, label)

                loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

                self.optimizer_bone.zero_grad()
                loss.backward()
                self.optimizer_bone.step()
                loss_all += loss.data

                if i % showEvery == 0:
                    print(
                        '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || sum : %10.4f' % (
                            datetime.datetime.now(), epoch, self.config.epoch, i, iter_num,
                            loss1.data, loss_all / (i + 1)))

                    print('Learning rate: ' + str(self.optimizer_bone.param_groups[0]['lr']))

                if i % 50 == 0 and self.config.local_rank == 0:
                    vis.images(torch.sigmoid(decoder_out1), win='predict.jpg', opts=dict(title='predict.jpg'))
                    vis.images(label, win='sal-target.jpg', opts=dict(title='sal-target.jpg'))

            if (epoch + 1) % self.config.epoch_save == 0 and self.config.local_rank == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                p['lr_bone'] = p['lr_bone'] * 0.2
                p['lr_branch'] = p['lr_branch'] * 0.2

        if self.config.local_rank == 0:
            torch.save(self.net_bone.state_dict(), '%s/final_bone.pth' % self.config.save_fold)
