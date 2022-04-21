import argparse
import os
from dataset import Dataset
import torch
from solver import Solver
from torchvision import transforms
import transform
from torch.utils import data
import torch.distributed as dist

def main(config):
    composed_transforms_tr = transforms.Compose([
        transform.RandomHorizontalFlip(),
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])

    composed_transforms_te = transforms.Compose([
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])

    if config.mode == 'pretrain_rgb':
        dataset = Dataset(datasets=['DAVIS-TRAIN', 'DAVSOD', 'DUTS-TR'], transform=composed_transforms_tr, mode='train')
        train_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_thread,
                                       drop_last=True, shuffle=True)

        config.save_fold = config.save_fold + '/' + 'spatial'
        if not os.path.exists("%s" % config.save_fold):
            os.mkdir("%s" % config.save_fold)
        train = Solver(train_loader, None, config)
        train.pretrainrgb()

    elif config.mode == 'pretrain_flow':
        dataset = Dataset(datasets=['DAVIS-TRAIN', 'DAVSOD'], transform=composed_transforms_tr, mode='train')
        train_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_thread,
                                       drop_last=True, shuffle=True)

        config.save_fold = config.save_fold + '/' + 'flow'
        if not os.path.exists("%s" % config.save_fold):
            os.mkdir("%s" % config.save_fold)
        train = Solver(train_loader, None, config)
        train.pretrainflow()

    elif config.mode == 'pretrain_depth':
        dataset = Dataset(datasets=['DAVIS-TRAIN', 'DAVSOD'], transform=composed_transforms_tr, mode='train')
        train_loader = data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_thread,
                                       drop_last=True, shuffle=True)

        config.save_fold = config.save_fold + '/' + 'depth'
        if not os.path.exists("%s" % config.save_fold):
            os.mkdir("%s" % config.save_fold)
        train = Solver(train_loader, None, config)
        train.pretraindepth()

    elif config.mode == 'train':
        print("local_rank", config.local_rank)
        world_size = int(os.environ['WORLD_SIZE'])
        print("world size", world_size)
        dist.init_process_group(backend='nccl')
        dataset_train = Dataset(datasets=['DAVIS-TRAIN', 'DAVSOD'], transform=composed_transforms_tr, mode='train')
        datasampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=dist.get_world_size(),
                                                                      rank=config.local_rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, sampler=datasampler,
                                                 num_workers=8)
        print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_train), len(train_loader)))

        config.save_fold = config.save_fold + '/' + 'DCTNet'
        if not os.path.exists("%s" % config.save_fold):
            os.mkdir("%s" % config.save_fold)
        train = Solver(train_loader, None, config, datasampler=datasampler)
        train.train_distributed()

    elif config.mode == 'test':

        dataset = Dataset(datasets=config.test_dataset, transform=composed_transforms_te, mode='test')
        test_loader = data.DataLoader(dataset, batch_size=config.test_batch_size, num_workers=config.num_thread,
                                      drop_last=True, shuffle=False)
        test = Solver(train_loader=None, test_loader=test_loader, config=config, save_fold=config.testsavefold)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    print(torch.cuda.is_available())
    parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

    # train
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--save_fold', type=str, default='./checkpoints')  # 训练过程中输出的保存路径
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--spatial_ckpt', type=str, default=None)
    parser.add_argument('--flow_ckpt', type=str, default=None)
    parser.add_argument('--depth_ckpt', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    # test
    parser.add_argument('--test_dataset', type=list, default=['DAVIS', 'FBMS', 'SegTrack-V2', 'DAVSOD', 'VOS'])
    parser.add_argument('--testsavefold', type=str, default='./prediction')

    # Misc
    parser.add_argument('--mode', type=str, default='test',
                        choices=['pretrain_rgb', 'pretrain_flow', 'pretrain_depth', 'train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_fold):
        os.mkdir(config.save_fold)
    main(config)
