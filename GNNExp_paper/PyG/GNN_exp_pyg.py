


from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.nn import GCNConv, Set2Set, GNNExplainer
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import matplotlib.pyplot as plt




#IMPORT THE DATASET

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]
print('dataset under study')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')




#DEFINE THE MODEL
class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)

    def forward(self, x,edge_index, data = None):
        #edge_index = adj.nonzero().t()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

    
input_dim = dataset.num_features
output_dim = dataset.num_classes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_dim, output_dim).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out= model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out= model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_incorrect = pred[data.test_mask] != data.y[data.test_mask] # Check against ground-truth labels.
      
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      torch.save(out, 'cora_chk/prediction_cora')
      return test_acc


for epoch in tqdm(range(1, 401)):
    loss = train()

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}') 


model.eval()
pred= model(data.x, data.edge_index)
torch.save(model, 'cora_chk/model_cora')



for node_idx in [0]:
#explainer boy

    x, edge_index = data.x, data.edge_index
    print(x.shape, edge_index.shape)
    explainer = GNNExplainer(model)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    print(edge_mask)
    #Full Results
    y = model(data.x, data.edge_index, data)
    res_full, predlabel_full = nn.Softmax(dim=0)(y[node_idx, :]), torch.argmax(nn.Softmax(dim=0)(y[node_idx, :]))


    #Explained results
    indices = edge_mask.to_sparse().indices()
    edge_mask.to_sparse().values()
    i = data.edge_index.t()[indices].squeeze().t()
    v = edge_mask.to_sparse().values()
    e = torch.sparse_coo_tensor(i ,v )

    indices_nodes = e.coalesce().indices().detach().numpy()
    new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) 




    ypred = model(data.x,i, data)
    print('masked adjacency:', e)
    res, predlabel = nn.Softmax(dim=0)(ypred[node_idx, :]), torch.argmax(nn.Softmax(dim=0)(ypred[node_idx, :]))

    print('true label:', data.y[node_idx], '\n full model pred label:' ,predlabel_full, '\n full model pred prob:', res_full[predlabel_full], res_full, '\n size of full graph:', len(data.edge_index[0]),
        '\n explained pred label:',predlabel, '\n explained pred prob:', res[predlabel], res, '\n size of explained graph:', len(i[0]))
    







def find_n_hop_neighbors(edge_index, n, node=None):
    """ 
    edge_index 
    n = num hops
    node = node_idx
    """
    # create dictionary of node neighborhoods
    neighborhoods = {}
    for i in range(edge_index.max().item() + 1):
        neighborhoods[i] = set()

    # find 1-hop neighbors and corresponding edges
    edges = []
    for j in range(edge_index.shape[1]):
        src, dst = edge_index[0, j].item(), edge_index[1, j].item()
        neighborhoods[src].add(dst)
        neighborhoods[dst].add(src)
        edges.append((src, dst))

    # find n-hop neighbors for the specified node or all nodes

    for k in range(2, n+1):
        new_neighbors = set()
        for neighbor in neighborhoods[node]:
            new_neighbors.update(neighborhoods[neighbor])
        neighborhoods[node].update(new_neighbors)
    sub_edges = []
    for edge in edges:
        src, dst = edge
        if src in neighborhoods[node] and dst in neighborhoods[node] or src == node or dst == node:
            sub_edges.append(edge)
            
    sub_edges_tensor = torch.tensor([sub_edges[i] for i in range(len(sub_edges))]).t()        

    #return {node: sub_edges}, {node: neighborhoods[node]}, sub_edges_tensor
    return sub_edges, neighborhoods[node], sub_edges_tensor  

def visualize(node_idx, data, masked_ver,result_weights=False):
    """ 
    Visualize important nodes for node idx prediction
    """
    #dict_index = dict_index_classes(data,masked_ver)
    sel_masked_ver = masked_ver
    indices_nodes = sel_masked_ver.coalesce().indices().detach().numpy()
    new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) #original edge indexes

    sub_edges, neighborhoods, sub_edges_tensor = find_n_hop_neighbors(data.edge_index, 2, node=0)
    G = nx.Graph()
    if result_weights:
        values = sel_masked_ver.coalesce().values().detach().numpy()
        for s,p,o in zip(indices_nodes[0],values , indices_nodes[1]):
            G.add_edge(int(s), int(o), weight=np.round(p, 2))


    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())


    pos = nx.circular_layout(G)



    labeldict = {}
    for node in G.nodes:
        labeldict[int(node)] = int(node)  


    print(data.y[list(neighborhoods)])
    
    if result_weights:
        
        nx.draw(G, pos,labels = labeldict,  edgelist=edges, edge_color=weights,  node_color = data.y[list(neighborhoods)],cmap="Set2",edge_cmap=plt.cm.Reds,font_size=8)
        nx.draw_networkx_edge_labels( G, pos,edge_labels=nx.get_edge_attributes(G,'weight'),font_size=8,font_color='red')

        plt.title("Node {}'s neighborhood important nodes".format(node_idx))

    if result_weights:
        plt.savefig(f'aifb_chk/graphs/Explanation_{node_idx}_weights.png')
        plt.show()

    
visualize(0, data, e, result_weights=True)  
