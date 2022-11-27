import torch
from torch import nn
import torch.nn.functional as F
import model.resnet as models
import numpy as np
import math
from basicmgai import Sub_MGAI, ConcatNet, Sub_MGAI2
from Code.lib.res2net_v1b_base import Res2Net_model

class SA(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)


class Fusion(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x1, x2):
        out1 = x1 + x2
        out2 = x1 * x2
        out  = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1./ math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class SubFuse(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d):
        super(SubFuse, self).__init__()
        self.dim = dim

        self.conv_r = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv_l  = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv3 = nn.Sequential(BasicConv2d(self.dim*2, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.bn3 = BatchNorm(self.dim)

    def forward(self, left, down):
        down_mask = self.conv_r(down)
        left_mask = self.conv_l(left)

        z1 = F.relu(left_mask * down, inplace=True)
        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


class MGAINet(nn.Module):
    def __init__(self, dropout=0.1,  BatchNorm=nn.BatchNorm2d, pretrained=True, args=True):
        super(MGAINet, self).__init__()

        self.args = args
        models.BatchNorm = BatchNorm

        self.layer_rgb = Res2Net_model(50)

        self.edge_cat = ConcatNet(BatchNorm)

        self.mutualnet4 = Sub_MGAI(BatchNorm, dim=256, num_clusters=32, dropout=dropout)
        self.mutualnet3 = Sub_MGAI(BatchNorm, dim=256, num_clusters=32, dropout=dropout)
        self.mutualnet2 = Sub_MGAI(BatchNorm, dim=256, num_clusters=32, dropout=dropout)
        self.mutualnet1 = Sub_MGAI(BatchNorm, dim=256, num_clusters=32, dropout=dropout)
        self.edge = Sub_MGAI2(BatchNorm, dim=256, num_clusters=32, dropout=dropout)

        self.subfus4 = SubFuse(256)
        self.subfus3 = SubFuse(256)
        self.subfus2 = SubFuse(256)
        self.subfus1 = SubFuse(256)

        self.fusion = Fusion(256)

        self.linear_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.sa4 = SA(2048)
        self.sa3 = SA(1024)
        self.sa2 = SA(512)
        self.sa1 = SA(256)

    def forward(self, x, y):
        x_size = x.size()
        raw_size = x.size()[2:]

        h = int((x_size[2])  )
        w = int((x_size[3])  )

        x_0, x_1, x_2, x_3, x_4 = self.layer_rgb(x)
        y_0, y_1, y_2, y_3, y_4 = self.layer_rgb(y)

        edger = self.edge_cat(x_1, x_2, x_3, x_4)  # edge pixel-level feature
        edget = self.edge_cat(y_1, y_2, y_3, y_4)  # edge pixel-level feature



        x_1 = F.interpolate(x_1, size=(30, 30), mode='bilinear', align_corners=True)
        y_1 = F.interpolate(y_1, size=(30, 30), mode='bilinear', align_corners=True)

        x_2 = F.interpolate(x_2, size=(30, 30), mode='bilinear', align_corners=True)
        y_2 = F.interpolate(y_2, size=(30, 30), mode='bilinear', align_corners=True)

        x_3 = F.interpolate(x_3, size=(30, 30), mode='bilinear', align_corners=True)
        y_3 = F.interpolate(y_3, size=(30, 30), mode='bilinear', align_corners=True)

        x_4 = F.interpolate(x_4, size=(30, 30), mode='bilinear', align_corners=True)
        y_4 = F.interpolate(y_4, size=(30, 30), mode='bilinear', align_corners=True)

        x_4 = self.sa4(x_4) # 2048 -> 256
        x_3 = self.sa3(x_3)
        x_2 = self.sa2(x_2)
        x_1 = self.sa1(x_1)

        y_4 = self.sa4(y_4)  # 2048 -> 256
        y_3 = self.sa3(y_3)
        y_2 = self.sa2(y_2)
        y_1 = self.sa1(y_1)

        edger, edget = self.edge(edger, edget)

        x4, y4 = self.mutualnet4(x_4, y_4, edger, edget)
        x3, y3 = self.mutualnet3(x_3, y_3, edger, edget)
        x2, y2 = self.mutualnet2(x_2, y_2, edger, edget)
        x1, y1 = self.mutualnet1(x_1, y_1, edger, edget)

        x43 = self.subfus4(x4, x3)
        x432 = self.subfus3(x43, x2)
        x4321 = self.subfus2(x432, x1)

        y43 = self.subfus4(y4, y3)
        y432 = self.subfus3(y43, y2)
        y4321 = self.subfus2(y432, y1)


        out3 = self.fusion(x4321, y4321)
        # print(out3.shape)
        out1 = F.interpolate(self.linear_out(x4321), size=raw_size, mode='bilinear', )
        out2 = F.interpolate(self.linear_out(y4321), size=raw_size, mode='bilinear', )
        out3 = F.interpolate(self.linear_out(out3), size=raw_size, mode='bilinear', )

        out1 = F.interpolate(out1, size=(h, w),  mode='bilinear', align_corners=True)
        out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=True)
        out3 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=True)

        return out1, out2, out3


