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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_fn(output, gt):
    gt = gt.view(-1)
    loss = F.cross_entropy(output, gt)
    max_ = torch.argmax(output, dim=-1).view(-1)
    sum_ = (max_ == gt).sum()
    return loss, sum_, gt.size(0)




def train_step(data, model, optimizer):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    output = model(data)
    # print(output.shape)
    loss, corrects, num_samples = loss_fn(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item(), corrects.item(), num_samples

def eval_step(data, model):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        loss, corrects, num_samples = loss_fn(output, data.y)
    return loss.item(), corrects.item(), num_samples

def train(model, train_loader, test_loader, optimizer, num_epochs=500
):
    max_val_acc = 0
    for epoch in range(num_epochs):
        train_loss = 0
        train_corrects = 0
        train_records = 0

        # Training loop
        for data in tqdm.tqdm(train_loader):
            loss, corrects, num_samples = train_step(data, model, optimizer)
            train_loss += loss
            train_corrects += corrects
            train_records += num_samples

        test_loss = 0
        test_corrects = 0
        test_records = 0

        # Validation loop
        for data in test_loader:
            loss, corrects, num_samples = eval_step(data, model)
            test_loss += loss
            test_corrects += corrects
            test_records += num_samples

        val_acc = test_corrects / test_records
        if val_acc > max_val_acc:
            max_val_acc = val_acc

        print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Acc: {train_corrects/train_records:.4f}, '
              f'val_Loss: {test_loss:.4f}, val_Acc: {val_acc:.4f}, max_val_Acc: {max_val_acc:.4f}')

def test(model, test_loader):
    test_loss = 0
    test_corrects = 0
    test_records = 0
    model.eval()
    for data, label in test_loader:
        loss, corrects, num_samples = eval_step(data, label, model)
        test_loss += loss
        test_corrects += corrects
        test_records += num_samples
    test_acc = test_corrects / test_records
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    return test_loss, test_acc


