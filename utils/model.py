import numpy as np
import datetime
import tqdm

import torch
import torch.nn.functional as F
import torch_geometric
# from torch_geometric.nn import knn_graph
from torch_geometric.transforms import KNNGraph
from torch_geometric.nn import GCNConv

import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale
from torch_geometric.data import DataLoader
    
# class GCN_bn(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCN_bn, self).__init__()
#         self.conv1 = GCNConv(in_channels, out_channels)
#         self.bn1 = torch.nn.BatchNorm1d(out_channels)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x_original, edge_index):
#         x = self.conv1(x_original, edge_index)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = torch.cat((x_original, x), dim=-1)
#         return x
    

import torch
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv
from torch.nn import Linear, BatchNorm1d, ReLU

class GCN_bn(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_bn, self).__init__()
        self.dense = Linear(in_channels, out_channels)
        self.conv1 = SimpleConv(aggr='max')
        self.bn1 = BatchNorm1d(out_channels)
        self.relu = ReLU()

    def forward(self, x_original, edge_index):
        x = self.dense(x_original)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.cat((x_original, x), dim=-1)
        return x

class Dense_bn(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_bn, self).__init__()
        self.o1 = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.o2 = torch.nn.BatchNorm1d(out_features)
        self.o3 = torch.nn.ReLU()

    def forward(self, x):
        y = self.o1(x)
        y = self.o2(y)
        y = self.o3(y)
        return y
    


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.knn_16 = KNNGraph(16)
        self.knn_64 = KNNGraph(64)

        self.bn1_1 = GCN_bn(3, 32)
        self.bn1_2 = GCN_bn(35, 128)

        self.bn2_1 = GCN_bn(3, 32)
        self.bn2_2 = GCN_bn(35, 128)

        self.bn3_1 = Dense_bn(3, 32)
        self.bn3_2 = Dense_bn(32, 128)

        self.fa = Dense_bn(454, 256)
        self.output = torch.nn.Linear(256, 40)
        
    def forward(self, data):
        # Extract position and batch information from DataBatch
        pos = data.pos
        batch = data.batch
        B = batch.max().item() + 1
        N = pos.size(0) // B

        edge_index_16 = self.knn_16(data)
        edge_index_64 = self.knn_64(data)
        x1 = self.bn1_1(pos, edge_index_16.edge_index)
        x1 = self.bn1_2(x1, edge_index_16.edge_index)

        x2 = self.bn2_1(pos, edge_index_64.edge_index)
        x2 = self.bn2_2(x2, edge_index_64.edge_index)

        x3 = self.bn3_1(pos)
        x3 = self.bn3_2(x3)

        x_all = torch.cat([x1, x2, x3], dim=-1)
        x_all = self.fa(x_all)
        x_all = x_all.view(B, N, -1)
        x_all = torch.max(x_all, 1)[0]
        # x_all = F.dropout(x_all, 0.2, training=self.training)
        out = self.output(x_all)
        return out