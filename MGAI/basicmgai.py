import torch
from torch import nn
import torch.nn.functional as F

import cv2
import os
import h5py, math
import numpy as np


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

class ConcatNet(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d):
        super(ConcatNet, self).__init__()

        self.w = 30
        self.h = 30

        c1, c2, c3, c4 = 256, 512, 1024, 2048

        self.conv1 = nn.Sequential(BasicConv2d(c1, c1, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c1, c1, BatchNorm, kernel_size=1, padding=0))
        self.conv2 = nn.Sequential(BasicConv2d(c2, c2, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c2, c2, BatchNorm, kernel_size=1, padding=0))
        self.conv3 = nn.Sequential(BasicConv2d(c3, c2, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c2, c2, BatchNorm, kernel_size=1, padding=0))
        self.conv4 = nn.Sequential(BasicConv2d(c4, c2, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c2, c2, BatchNorm, kernel_size=1, padding=0))

        # (256+512+1024+2048=3840)
        c = c1 + c2 + c2 + c2
        self.conv5 = nn.Sequential(BasicConv2d(c, c1, BatchNorm, kernel_size=3, padding=1),
                                   BasicConv2d(c1, c1, BatchNorm, kernel_size=1, padding=0))

    def forward(self, x1, x2, x3, x4):
        x1 = F.interpolate(x1, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)

        x2 = F.interpolate(x2, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)

        x3 = F.interpolate(x3, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x3 = self.conv3(x3)

        x4 = F.interpolate(x4, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x4 = self.conv4(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1) # c=256 x 4 = 1024
        x = self.conv5(x)

        return x

 
class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous() # b x k x c
        support = torch.matmul(x_t, self.weight) # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous() # b x c x k
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert(loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous() # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x)) # b x c x k
        x = self.relu(x)
        return x

class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H*W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1) #across descriptor dim

        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma) # B x C x N(N=HxW)
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2) # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1) # l2 normalize

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign


class MutualModule0(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule0, self).__init__()
        self.gcn = CascadeGCNet(dim, loop=2)
        self.conv = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))

    def forward(self, edge_graph, region_graph1, region_graph2, assign):
        m = self.corr_matrix(edge_graph, region_graph1, region_graph2)
        edge_graph = edge_graph + m

        edge_graph = self.gcn(edge_graph)
        edge_x = edge_graph.bmm(assign)
        edge_x = self.conv(edge_x.unsqueeze(3)).squeeze(3)
        return edge_x

    def corr_matrix(self, edge, region1, region2):
        assign = edge.permute(0, 2, 1).contiguous().bmm(region1)
        assign = F.softmax(assign, dim=-1) #normalize
        m = assign.bmm(region2.permute(0, 2, 1).contiguous())
        m = m.permute(0, 2, 1).contiguous()
        return m


class Sub_MGAI(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=256, num_clusters=8, dropout=0.1):
        super(Sub_MGAI, self).__init__()

        self.dim = dim

        self.rgb_proj0   = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.t_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.rgb_conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.rgb_conv1[0].reset_params()

        self.rgb_conv2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.rgb_conv2[0].reset_params()

        self.t_conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.t_conv1[0].reset_params()

        self.t_conv2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.t_conv2[0].reset_params()

        self.t2r = MutualModule0(self.dim, BatchNorm, dropout)
        self.r2t = MutualModule0(self.dim, BatchNorm, dropout)

        self.pred = nn.Conv2d(self.dim, 1, kernel_size=1)


    def forward(self, rgb, t, edger, edget):

        edge_r = self.pred(edger)
        rgb_x = torch.sigmoid(edge_r).mul(rgb)  # elementwise-mutiply

        edge_t = self.pred(edget)
        t_x = torch.sigmoid(edge_t).mul(t)  # elementwise-mutiply

        rgb_graph, rgb_assign = self.rgb_proj0(rgb_x)
        t_graph, t_assign = self.t_proj0(t_x)
        #rgb
        rgb_graph1 = self.rgb_conv1(rgb_graph.unsqueeze(3)).squeeze(3)
        rgb_graph2 = self.rgb_conv2(rgb_graph.unsqueeze(3)).squeeze(3)
        # t
        t_graph1 = self.t_conv1(t_graph.unsqueeze(3)).squeeze(3)
        t_graph2 = self.t_conv2(t_graph.unsqueeze(3)).squeeze(3)
        # t2r
        n_rgb_x = self.t2r(rgb_graph, t_graph1, t_graph2, rgb_assign)
        rgb_x = rgb_x + n_rgb_x.view(rgb_x.size()).contiguous()
        # r2t
        n_t_x = self.r2t(t_graph, rgb_graph1, rgb_graph2, t_assign)
        t_x = t_x + n_t_x.view(t_x.size()).contiguous()

        return rgb_x,  t_x


class Sub_MGAI2(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=256, num_clusters=8, dropout=0.1):
        super(Sub_MGAI2, self).__init__()

        self.dim = dim

        self.edge_graph   = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.pred = nn.Conv2d(self.dim, 1, kernel_size=1)
        self.conv = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))

        self.gcn = CascadeGCNet(dim, loop=3)
       
        self.conv0 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
    def forward(self,  edger, edget):
        # edge fature to graph
        edger_graph, edger_assign = self.edge_graph(edger)
        edget_graph, edget_assign = self.edge_graph(edget)

        # rgbe
        coeex_graph = self.gcn(edger_graph)
        n_coeex = coeex_graph.bmm(edger_assign)
        n_coeex = self.conv0(n_coeex.view(edger.size()))
        edger_g = edger + n_coeex
        
        # te
        coeey_graph = self.gcn(edget_graph)
        n_coeey = coeey_graph.bmm(edget_assign)
        n_coeey = self.conv1(n_coeey.view(edget.size()))
        edget_g = edget + n_coeey

        return edger_g,  edget_g
