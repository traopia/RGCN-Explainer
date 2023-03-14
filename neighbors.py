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
import math 

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

def sum_sparse(indices, values, size, row=True):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries

    :return:
    """

    ST = torch.cuda.sparse.FloatTensor if indices.is_cuda else torch.sparse.FloatTensor

    assert len(indices.size()) == 2

    k, r = indices.size()

    if not row:
        # transpose the matrix
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=d(indices))

    smatrix = ST(indices.t(), values, size=size)
    sums = torch.mm(smatrix, ones) # row/column sums

    sums = sums[indices[:, 0]]

    assert sums.size() == (k, 1)

    return sums.view(k)

def adj(triples, num_nodes, num_rels, cuda=False, vertical=True):
    """
     Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
     relations are stacked vertically).

     :param edges: List representing the triples
     :param i2r: list of relations
     :param i2n: list of nodes
     :return: sparse tensor
    """
    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical else (n, r * n)

    from_indices = []
    upto_indices = []

    for s, p, o in triples:

        offset = p.item() * n
        print(offset)

        if vertical:
            s = offset + s.item()
        else:
            o = offset + o.item()

        from_indices.append(s)
        upto_indices.append(o)
        

    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=d(cuda))


    assert indices.size(1) == len(triples)
    assert indices[0, :].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[1, :].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices.t(), size

def hor_ver_graph(triples, n, r):
    hor_ind, hor_size = adj(triples, n, 2*r+1, vertical=False)
    ver_ind, ver_size = adj(triples, n, 2*r+1, vertical=True)
    #number of relations is 2*r+1 because we added the inverse and self loop

    _, rn = hor_size #horizontally stacked adjacency matrix size
    r = rn // n #number of relations enriched divided by number of nodes

    vals = torch.ones(ver_ind.size(0), dtype=torch.float) #number of enriched triples
    vals = vals / sum_sparse(ver_ind, vals, ver_size) #normalize the values by the number of edges

    hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size) #size: n,r, emb


    ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)

    return hor_graph, ver_graph


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

def construct_edge_mask( num_nodes, init_strategy="normal", const_val=1.0):
    """
    Construct edge mask
    input;
        num_nodes: number of nodes in the neighborhood
        init_strategy: initialization strategy for the mask
        const_val: constant value for the mask
    output:
        mask: edge mask    
    """
    mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))  #initialize the mask
    if init_strategy == "normal":
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (num_nodes + num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
    elif init_strategy == "const":
        nn.init.constant_(mask, const_val)
    return mask


def _masked_adj(mask,adj, diag_mask):
    """ Masked adjacency matrix 
    input: edge_mask, sub_adj, diag_mask
    output: masked_adj
    """
    sym_mask = mask
    sym_mask = torch.sigmoid(mask)
    
    sym_mask = (sym_mask + sym_mask.t()) / 2
    adj = torch.tensor(adj)
    masked_adj = adj * sym_mask

    return masked_adj * diag_mask

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
    #print('triples', data.triples)


    tensor1 = sub_edges_tensor.t()
    tensor2 = data.triples

    result = match_to_triples(tensor1, tensor2)
    print(result)
    n = len(result)
    r = len(np.unique(result[:,1]))
    #hor_graph, ver_graph = hor_ver_graph(result, n, r)
    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    print(hor_graph.coalesce().values())
    mask = construct_edge_mask(n)
    print(mask)
    masked_adj = _masked_adj(mask,sub_edges_tensor , torch.eye(n))
    print(masked_adj)



if __name__ == '__main__':
    main()





#WHAT ABOUT WE HAVE TO WORK WITH HORGRAPH / VERGRAPH --

