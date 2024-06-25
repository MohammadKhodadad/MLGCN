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


train_loader,test_loader = load_data()
model = Model().to(device)
# Assuming optimizer is defined
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

# Train the model
train(model, train_loader, test_loader, optimizer)

# Evaluate the model on the test dataset
test_loss, test_acc = test(model, test_loader)