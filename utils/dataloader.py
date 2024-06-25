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
def load_data():
    transform = torch_geometric.transforms.Compose([SamplePoints(1024), NormalizeScale()])
    train_dataset = ModelNet(root='data/ModelNet40', name='40', train=True, transform=transform)
    test_dataset = ModelNet(root='data/ModelNet40', name='40', train=False, transform=transform)

    # Create data loaders
    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader,test_loader