import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv, RGCNConv
from torch_geometric.utils import k_hop_subgraph
import kgbench as kg
from kgbench import Data

import torch
torch.cuda.empty_cache()





#data = kg.load('aifb', torch=True) 
data = torch.load("IMDb_typePeople_data.pt")
print(f'Number of entities: {data.num_entities}') #data.i2e
print(f'Number of classes: {data.num_classes}')
print(f'Types of relations: {data.num_relations}') #data.i2r
data.triples
data.num_classes = 58
idxt, clst = torch.Tensor(data.training[:, 0]), torch.Tensor(data.training[:, 1])
idxw, clsw = torch.Tensor(data.withheld[:, 0]), torch.Tensor(data.withheld[:, 1])
data.train_idx, data.train_y = idxt.long(), clst.long()
data.test_idx, data.test_y = idxw.long(), clsw.long()
data.triples = torch.tensor(data.triples)
data.edge_index = torch.stack((data.triples[:, 0], data.triples[:, 2]),dim=0)
data.edge_type = torch.tensor(data.triples[:, 1])


node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx, 2, data.edge_index, relabel_nodes=True)

data.num_nodes = node_idx.size(0)
data.edge_index = edge_index
print(data.edge_type)
data.edge_type = data.edge_type[edge_mask]
print(data.edge_type)
data.train_idx = mapping[:data.train_idx.size(0)]
data.test_idx = mapping[data.train_idx.size(0):]

dataset = data

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(data.num_nodes, 16, dataset.num_relations,
                              num_bases=30)
        self.conv2 = RGCNConv(16, dataset.num_classes, dataset.num_relations,
                              num_bases=30)

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu') if args.dataset == 'AM' else device
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, data.edge_type)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    torch.save(pred, 'pred_imdb_torch')
    return train_acc, test_acc


for epoch in range(1, 51):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')
torch.save(Net,'model_imdb_torch')
