from responses import FalseBool
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import kgbench as kg
import fire, sys
import math

from kgbench import load, tic, toc, d


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


#
from torch_geometric.utils import to_networkx
import networkx as nx


#rgcn 
from rgcn_model import adj, enrich, sum_sparse, RGCN
from rgcn_explainer_utils import dict_index_classes, dict_triples_semantics, visualize, find_n_hop_neighbors, match_to_classes, match_to_triples, edge_index_oneadj, get_relations, d_classes



#Get adjacency matrix: in this context this is hor / ver graph
def hor_ver_graph(triples, n, r):
    #triples = enrich(triples_small, n, r)

    hor_ind, hor_size = adj(triples, n, 2*r+1, vertical=False)
    ver_ind, ver_size = adj(triples, n, 2*r+1, vertical=True)
    #number of relations is 2*r+1 because we added the inverse and self loop

    _, rn = hor_size #horizontally stacked adjacency matrix size
    print(hor_size)
    r = rn // n #number of relations enriched divided by number of nodes

    vals = torch.ones(ver_ind.size(0), dtype=torch.float) #number of enriched triples
    vals = vals / sum_sparse(ver_ind, vals, ver_size) #normalize the values by the number of edges

    hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size) #size: n,r, emb


    ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)

    return hor_graph, ver_graph


#Edge mask

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
    mask = nn.Parameter(torch.FloatTensor(num_nodes))  #initialize the mask
    if init_strategy == "normal":
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (num_nodes + num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
    elif init_strategy == "const":
        nn.init.constant_(mask, const_val)
    return torch.tensor(mask)

#Masked 'adjacency' - masked hor vergraph

def _masked_adj(mask,adj, diag_mask, graph):
    """ Masked adjacency matrix 
    input: edge_mask, sub_adj, diag_mask
    output: masked_adj
    """
    sym_mask = mask
    sym_mask = torch.sigmoid(mask)
    
    sym_mask = (sym_mask + sym_mask.t()) / 2
    adj = torch.tensor(adj)
    masked_adj = adj * sym_mask

    #return masked_adj #* diag_mask
    return torch.sparse.FloatTensor(indices=graph.coalesce().indices(), values= masked_adj, size=graph.coalesce().size())








def loss_fc(edge_mask,  pred, pred_label,label, node_idx, epoch, print=False):
    """
    Args:
        pred: y_e :  prediction made by current model
        pred_label: y_hat : the label predicted by the original model.
    """
    
    #PRED LOSS
    pred_label_node = pred_label[node_idx] #pred label is the prediction made by the original model
    gt_label_node = label[node_idx]

    logit = pred[gt_label_node] #pred is the prediction made by the current model

    pred_loss = -torch.log(logit) #this is basically taking the cross entropy loss

    # MASK SIZE EDGE LOSS
    
    mask = edge_mask
    mask = torch.sigmoid(mask)

    size_loss = 0.005 * torch.sum(mask)



    # EDGE MASK ENTROPY LOSS
    mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    mask_ent_loss = 1.0 * torch.mean(mask_ent)
    

    loss = pred_loss + size_loss  + mask_ent_loss # + feat_size_loss + lap_loss

    return loss



class Explain(nn.Module):
    def __init__(self, model, data, node_idx, n_hops):
        super(Explain, self).__init__()
        #Those are the parameters of the original data and model
        self.model = model
        self.data = data
        self.n = data.num_entities
        self.r = data.num_relations
    
        #self.triples = enrich(data.triples, self.n, self.r)
        self.triples = data.triples
        self.node_idx = node_idx
        self.n_hops = n_hops
        #self.adj = get_adjacency(data)
        self.edge_index = edge_index_oneadj(self.triples)
       
        self.label = self.data.withheld[:, 1]

        self.epoch = 1
        self.hor_graph, self.ver_graph = hor_ver_graph(self.triples, self.n, self.r)
        self.pred_label = torch.load('/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/aifb_chk/prediction_aifb')
        self.sub_edges, self.neighbors, self.sub_edges_tensor = find_n_hop_neighbors(self.edge_index, n=self.n_hops, node=self.node_idx)
        self.sub_triples = match_to_triples(self.sub_edges_tensor.t(),self.triples)
        self.sub_hor_graph, self.sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)
        self.num_nodes = self.sub_hor_graph.coalesce().values().shape[0]   
        self.edge_mask = construct_edge_mask(self.num_nodes)
    
    def _masked_adj(self, sub_graph):
        """ Masked adjacency matrix 
        input: edge_mask, sub_adj, diag_mask
        output: masked_adj
        """
        adj = sub_graph.coalesce().values()
        indices = sub_graph.coalesce().indices()
        size = sub_graph.coalesce().size()
        sym_mask = self.edge_mask
        sym_mask = torch.sigmoid(self.edge_mask)
        
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = torch.tensor(adj)
        masked_adj = adj * sym_mask

        return torch.sparse.FloatTensor(indices=indices, values= masked_adj, size=size )
    
    def softmax(self, pred):
        """Compute softmax values for each sets of scores in x."""
        x_index = pred[self.node_idx].detach().numpy()
        e_x = np.exp(x_index - np.max(x_index))
        return e_x / e_x.sum(axis=0),  np.argmax(e_x / e_x.sum(axis=0))
    
    def new_index(self):
        idxw, clsw = self.data.withheld[:, 0], self.data.withheld[:, 1]
        idxw, clsw = idxw.long(), clsw.long()
        idxw_list = list(idxw)
        self.new_node_idx = idxw_list.index(self.node_idx)
        return self.new_node_idx 
        
    def forward(self):
        """
        Returns:
            ypred: prediction of the query node made by the current model (on the subgraph)

        """
        masked_hor = self._masked_adj(self.sub_hor_graph)
        masked_ver = self._masked_adj(self.sub_ver_graph)

        ypred = self.model.forward2(masked_hor, masked_ver)
        
        node_pred = ypred[self.node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)
  
        return res, masked_hor, masked_ver
    
    def criterion(self, epoch):
        """
        Computes the loss of the current model
        """
        #prediction of explanation model
        pred, masked_hor, masked_ver = self.forward()


        self.new_node_idx = self.new_index()
        loss_val = loss_fc(self.edge_mask,  pred, self.pred_label,self.label, self.new_node_idx, epoch, print=False)

        return loss_val 
    

    def return_stuff(self):
        return self.neighbors,self.n_hops, self.node_idx


def main(node_idx, n_hops, threshold, train):
    data = kg.load('aifb', torch=True) 
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    d_classes(data)
    if train:
        model = torch.load('/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/aifb_chk/model_aifb')

        explainer = Explain(model = model, data = data, node_idx = node_idx, n_hops = n_hops)
        optimizer = torch.optim.Adam(explainer.parameters(), lr=0.01)
        print('start training')
        explainer.train()
        for epoch in range(100):
            explainer.zero_grad()
            optimizer.zero_grad()
            ypred, masked_hor, masked_ver = explainer.forward()
            loss = explainer.criterion(epoch)
            neighbors,n_hops, node_idx = explainer.return_stuff()
            #pred_label, original_label, neighbors, sub_label, sub_feat, num_hops = explainer.return_stuff()
            



            loss.backward()
            optimizer.step()
        


            if epoch % 10 == 0:

                print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),

                "; pred: ",
                ypred )
        torch.save(masked_ver, f'aifb_chk/masked_ver{node_idx}')
    # sem_triples = dict_triples_semantics(data, masked_ver)
    # print('semantic triples: ', sem_triples)
    # visualize_result(node_idx, masked_ver, neighbors,data,n_hops)
    # visualize_data(node_idx, data, n_hops)
    else:
        masked_ver = torch.load(f'aifb_chk/masked_ver{node_idx}')
    visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=False, low_threshold=False)
    visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=True, low_threshold=FalseBool())
    dict_index = dict_index_classes(data,masked_ver)
    print('dict index:' ,dict_index)
    #print('masked ver values: ', masked_ver.coalesce().values())
    


if __name__ == "__main__":
    main(5797,0,0.5, train=False)



    

    
