import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Spatial import Spatial
from model.Flow import Flow
from model.Depth import Depth


class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)

    def forward(self, x, H, W):
        x = F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)


class decoder_stage(nn.Module):
    def __init__(self, infilter, midfilter, outfilter):
        super(decoder_stage, self).__init__()
        self.layer = nn.Sequential(
            *[nn.Conv2d(infilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, outfilter, 3, padding=1, bias=False), nn.BatchNorm2d(outfilter),
              nn.ReLU(inplace=True)])

    def forward(self, x):
        return self.layer(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class RFM(nn.Module):
    def __init__(self):
        super(RFM, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

    def forward(self, rgb, depth, flow):
        rgb_aligned = F.relu(self.bn1(self.conv1(rgb)), inplace=True)
        depth_aligned = F.relu(self.bn2(self.conv2(depth)), inplace=True)
        flow_aligned = F.relu(self.bn3(self.conv3(flow)), inplace=True)
        fuse_depth = rgb_aligned * depth_aligned
        fuse_flow = rgb_aligned * flow_aligned
        depth_out = F.relu(self.bn4(self.conv4(fuse_depth + depth_aligned)), inplace=True)
        flow_out = F.relu(self.bn5(self.conv5(fuse_flow + flow_aligned)), inplace=True)
        rgb_out = F.relu(self.bn6(self.conv6(fuse_flow + fuse_depth + rgb_aligned)), inplace=True)
        return rgb_out, depth_out, flow_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv2d
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.catconv = BasicConv2d(in_planes=self.in_channels * 2, out_planes=self.in_channels, kernel_size=3,
                                   padding=1, stride=1)

        self.main_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.auxiliary_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, main_fea, auxiliary_fea):
        mainNonLocal_fea = self.main_bnRelu(main_fea)
        auxiliaryNonLocal_fea = self.auxiliary_bnRelu(auxiliary_fea)

        batch_size = mainNonLocal_fea.size(0)

        g_x = self.R_g(mainNonLocal_fea).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        l_x = self.F_g(auxiliaryNonLocal_fea).view(batch_size, self.inter_channels, -1)
        l_x = l_x.permute(0, 2, 1)

        catNonLocal_fea = self.catconv(torch.cat([mainNonLocal_fea, auxiliaryNonLocal_fea], dim=1))

        theta_x = self.F_theta(catNonLocal_fea).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.F_phi(catNonLocal_fea).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # add self_f and mutual f
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *mainNonLocal_fea.size()[2:])
        W_y = self.R_W(y)
        z = W_y + main_fea

        m = torch.matmul(f_div_C, l_x)
        m = m.permute(0, 2, 1).contiguous()
        m = m.view(batch_size, self.inter_channels, *auxiliaryNonLocal_fea.size()[2:])
        W_m = self.F_W(m)
        p = W_m + auxiliary_fea

        return z, p


class MAM(nn.Module):
    def __init__(self, inchannels):
        super(MAM, self).__init__()
        self.Nonlocal_RGB_Flow = NonLocalBlock(inchannels)
        self.Nonlocal_RGB_Depth = NonLocalBlock(inchannels)
        self.catconv = BasicConv2d(2 * inchannels, inchannels, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, flow, depth):
        rgb_f, flow = self.Nonlocal_RGB_Flow(rgb, flow)
        rgb_d, depth = self.Nonlocal_RGB_Depth(rgb, depth)
        rgb_final = self.catconv(torch.cat([rgb_f, rgb_d], dim=1))
        return rgb_final, flow, depth


class Model(nn.Module):
    def __init__(self, inchannels, mode, spatial_ckpt=None, flow_ckpt=None, depth_ckpt=None):
        super(Model, self).__init__()
        self.spatial_net = Spatial(inchannels, mode)
        self.flow_net = Flow(inchannels, mode)
        self.depth_net = Depth(inchannels, mode)

        self.ca1_rgb = ChannelAttention(64)
        self.ca2_rgb = ChannelAttention(64)
        self.ca3_rgb = ChannelAttention(64)
        self.ca4_rgb = ChannelAttention(64)
        self.ca5_rgb = ChannelAttention(64)

        self.ca1_flow = ChannelAttention(64)
        self.ca2_flow = ChannelAttention(64)
        self.ca3_flow = ChannelAttention(64)
        self.ca4_flow = ChannelAttention(64)
        self.ca5_flow = ChannelAttention(64)

        self.ca1_depth = ChannelAttention(64)
        self.ca2_depth = ChannelAttention(64)
        self.ca3_depth = ChannelAttention(64)
        self.ca4_depth = ChannelAttention(64)
        self.ca5_depth = ChannelAttention(64)

        self.catconv1_auxiliary = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv2_auxiliary = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv3_auxiliary = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv4_auxiliary = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv5_auxiliary = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.catconv1 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv2 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv3 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv4 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.catconv5 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if spatial_ckpt is not None:
            self.spatial_net.load_state_dict(torch.load(spatial_ckpt, map_location='cpu'))
            print("Successfully load spatial:{}".format(spatial_ckpt))
        if flow_ckpt is not None:
            self.flow_net.load_state_dict(torch.load(flow_ckpt, map_location='cpu'))
            print("Successfully load flow:{}".format(flow_ckpt))
        if depth_ckpt is not None:
            self.depth_net.load_state_dict(torch.load(depth_ckpt, map_location='cpu'))
            print("Successfully load depth:{}".format(depth_ckpt))

        self.rfm1 = RFM()
        self.rfm2 = RFM()
        self.rfm3 = RFM()
        self.rfm4 = RFM()
        self.rfm5 = RFM()

        self.mam3 = MAM(64)
        self.mam4 = MAM(64)
        self.mam5 = MAM(64)

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, image, flow, depth):
        out4, out1, img_conv1_feat, out2, out3, out5_aspp, course_img = self.spatial_net.rgb_bkbone(image)

        flow_out4, flow_out1, flow_conv1_feat, flow_out2, flow_out3, flow_out5_aspp, course_flo = self.flow_net.flow_bkbone(
            flow)

        depth_out4, depth_out1, depth_conv1_feat, depth_out2, depth_out3, depth_out5_aspp, course_dep = self.depth_net.depth_bkbone(
            depth)

        out1, out2, out3, out4, out5 = self.spatial_net.squeeze1(out1), \
                                       self.spatial_net.squeeze2(out2), \
                                       self.spatial_net.squeeze3(out3), \
                                       self.spatial_net.squeeze4(out4), \
                                       self.spatial_net.squeeze5(out5_aspp)

        flow_out1, flow_out2, flow_out3, flow_out4, flow_out5 = self.flow_net.squeeze1_flow(flow_out1), \
                                                                self.flow_net.squeeze2_flow(flow_out2), \
                                                                self.flow_net.squeeze3_flow(flow_out3), \
                                                                self.flow_net.squeeze4_flow(flow_out4), \
                                                                self.flow_net.squeeze5_flow(flow_out5_aspp)

        depth_out1, depth_out2, depth_out3, depth_out4, depth_out5 = self.depth_net.squeeze1_depth(depth_out1), \
                                                                     self.depth_net.squeeze2_depth(depth_out2), \
                                                                     self.depth_net.squeeze3_depth(depth_out3), \
                                                                     self.depth_net.squeeze4_depth(depth_out4), \
                                                                     self.depth_net.squeeze5_depth(depth_out5_aspp)

        out3, flow_out3, depth_out3 = self.mam3(out3, flow_out3, depth_out3)
        out4, flow_out4, depth_out4 = self.mam4(out4, flow_out4, depth_out4)
        out5, flow_out5, depth_out5 = self.mam5(out5, flow_out5, depth_out5)

        out1, depth_out1, flow_out1 = self.rfm1(out1, depth_out1, flow_out1)
        out2, depth_out2, flow_out2 = self.rfm2(out2, depth_out2, flow_out2)
        out3, depth_out3, flow_out3 = self.rfm3(out3, depth_out3, flow_out3)
        out4, depth_out4, flow_out4 = self.rfm4(out4, depth_out4, flow_out4)
        out5, depth_out5, flow_out5 = self.rfm5(out5, depth_out5, flow_out5)

        out1 = out1.mul(self.ca1_rgb(out1)) + out1
        out2 = out2.mul(self.ca2_rgb(out2)) + out2
        out3 = out3.mul(self.ca3_rgb(out3)) + out3
        out4 = out4.mul(self.ca4_rgb(out4)) + out4
        out5 = out5.mul(self.ca5_rgb(out5)) + out5

        flow_out1 = flow_out1.mul(self.ca1_flow(flow_out1)) + flow_out1
        flow_out2 = flow_out2.mul(self.ca2_flow(flow_out2)) + flow_out2
        flow_out3 = flow_out3.mul(self.ca3_flow(flow_out3)) + flow_out3
        flow_out4 = flow_out4.mul(self.ca4_flow(flow_out4)) + flow_out4
        flow_out5 = flow_out5.mul(self.ca5_flow(flow_out5)) + flow_out5

        depth_out1 = depth_out1.mul(self.ca1_depth(depth_out1)) + depth_out1
        depth_out2 = depth_out2.mul(self.ca2_depth(depth_out2)) + depth_out2
        depth_out3 = depth_out3.mul(self.ca3_depth(depth_out3)) + depth_out3
        depth_out4 = depth_out4.mul(self.ca4_depth(depth_out4)) + depth_out4
        depth_out5 = depth_out5.mul(self.ca5_depth(depth_out5)) + depth_out5

        fusion1_auxiliary = self.catconv1_auxiliary(torch.cat([flow_out1, depth_out1], dim=1))
        fusion2_auxiliary = self.catconv2_auxiliary(torch.cat([flow_out2, depth_out2], dim=1))
        fusion3_auxiliary = self.catconv3_auxiliary(torch.cat([flow_out3, depth_out3], dim=1))
        fusion4_auxiliary = self.catconv4_auxiliary(torch.cat([flow_out4, depth_out4], dim=1))
        fusion5_auxiliary = self.catconv5_auxiliary(torch.cat([flow_out5, depth_out5], dim=1))

        fusion1 = self.catconv1(torch.cat([out1, fusion1_auxiliary], dim=1))
        fusion2 = self.catconv2(torch.cat([out2, fusion2_auxiliary], dim=1))
        fusion3 = self.catconv3(torch.cat([out3, fusion3_auxiliary], dim=1))
        fusion4 = self.catconv4(torch.cat([out4, fusion4_auxiliary], dim=1))
        fusion5 = self.catconv5(torch.cat([out5, fusion5_auxiliary], dim=1))

        feature5 = self.spatial_net.decoder5(fusion5)
        feature4 = self.spatial_net.decoder4(torch.cat([feature5, fusion4], 1))
        B, C, H, W = fusion3.size()
        feature3 = self.spatial_net.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), fusion3), 1))
        B, C, H, W = fusion2.size()
        feature2 = self.spatial_net.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), fusion2), 1))
        B, C, H, W = fusion1.size()
        feature1 = self.spatial_net.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), fusion1), 1))

        decoder_out5 = self.spatial_net.out5(feature5, H * 4, W * 4)
        decoder_out4 = self.spatial_net.out4(feature4, H * 4, W * 4)
        decoder_out3 = self.spatial_net.out3(feature3, H * 4, W * 4)
        decoder_out2 = self.spatial_net.out2(feature2, H * 4, W * 4)
        decoder_out1 = self.spatial_net.out1(feature1, H * 4, W * 4)

        return decoder_out1, decoder_out2, decoder_out3, decoder_out4, decoder_out5
