import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import kgbench as kg
import fire, sys

from kgbench import load, tic, toc, d


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


#
from torch_geometric.utils import to_networkx
import networkx as nx





def edge_index_oneadj(data):
    edge_index = torch.stack((data.triples[:, 0], data.triples[:, 2]),dim=0)
    return edge_index






def find_n_hop_neighbors(edge_index, n, node=None):
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
    if node is not None:
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
    else:
        for k in range(2, n+1):
            new_neighbors = {}
            for i in range(edge_index.max().item() + 1):
                if len(neighborhoods[i]) > 0:
                    neighbors = set(neighborhoods[i])
                    for neighbor in neighborhoods[i]:
                        neighbors.update(neighborhoods[neighbor])
                    new_neighbors[i] = neighbors
            neighborhoods.update(new_neighbors)
        sub_edge_index = torch.tensor([[edge[0] for edge in edges], [edge[1] for edge in edges]])
        sub_edge_index_mask = []
        for j in range(sub_edge_index.shape[1]):
            src, dst = sub_edge_index[0, j].item(), sub_edge_index[1, j].item()
            if src in neighborhoods and dst in neighborhoods:
                sub_edge_index_mask.append(True)
            else:
                sub_edge_index_mask.append(False)
        sub_edge_index = sub_edge_index[:, sub_edge_index_mask]
        return sub_edge_index



def match_to_triples(tensor1, tensor2):
    matching = []
    for i,i2 in zip(tensor1[:,0],tensor1[:,1]):
        for j,j1,j2, index in zip(tensor2[:,0],tensor2[:,1],  tensor2[:,2], range(len(tensor2[:,0]))):
            if i == j and i2 == j2:
                matching.append(tensor2[index])
    result = torch.stack(matching)
    return result


# def get_adjacency(neighbors, sub_edge_index):
#     adj = torch.zeros(len(neighbors), len(neighbors))

#     for edge in sub_edge_index.t():
#         adj[edge[0]][edge[1]] = 1
#     return adj

# adj = get_adjacency(neighbors, sub_edges_tensor)
# print(adj)



def main():
    data = kg.load('aifb', torch=True) 
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r

    edge_index = edge_index_oneadj(data)

    # test with n=2 and node=0
    sub_edge_index, neighbors, sub_edges_tensor = find_n_hop_neighbors(edge_index, n=2, node=2)

    print('sub_edge_index', sub_edge_index)
    print('neighbors', neighbors)
    print('sub_edges_tensor', sub_edges_tensor.t())
    print('triples', data.triples)


    tensor1 = sub_edges_tensor.t()
    tensor2 = data.triples

    result = match_to_triples(tensor1, tensor2)
    print(result)


if __name__ == '__main__':
    main()





#WHAT ABOUT WE HAVE TO WORK WITH HORGRAPH / VERGRAPH --

