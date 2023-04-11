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
from rgcn_explainer_utils import dict_index_classes, dict_triples_semantics, selected, visualize, find_n_hop_neighbors, match_to_classes, match_to_triples, edge_index_oneadj, get_relations, d_classes



#Get adjacency matrix: in this context this is hor / ver graph
def hor_ver_graph(triples, n, r):
    """ 
    input: triples, number of nodes, number of relations
    output: hor_graph, ver_graph : horizontally and vertically stacked adjacency matrix
    """
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
    torch.manual_seed(42)

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

    size_loss = 0.8 * torch.sum(mask)



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
        #ypred = self.model.forward3(masked_ver)
        # print('ypred',ypred[0])
        # print('masked_ver', masked_ver)
        #print(len(self.neighbors))
        #print(self.edge_mask.shape)
        
        node_pred = ypred[self.node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)
        #print(res)
        torch.save(masked_hor,'masked_hor_forward')
        torch.save(masked_ver,'masked_ver_forward')
  
        return res, masked_hor, masked_ver
    
    def criterion(self, epoch):
        """
        Computes the loss of the current model
        """
        #prediction of explanation model
        pred, masked_hor, masked_ver = self.forward()


        self.new_node_idx = self.new_index()
        loss_val = loss_fc(self.edge_mask,  pred, self.pred_label,self.label, self.new_node_idx, epoch, print=False)
        #loss_val = loss_fc(self.masked_adj,  pred, self.pred_label,self.label, self.new_node_idx, epoch, print=False)

        return loss_val 
    

    def return_stuff(self):
        return self.neighbors,self.n_hops, self.node_idx


#def main(node_idx, n_hops, threshold, train):
def main(n_hops, threshold, train):
    data = kg.load('aifb', torch=True) 
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    d_classes(data)
    d = {key.item(): data.withheld[:, 0][data.withheld[:, 1] == key].tolist() for key in torch.unique(data.withheld[:, 1])}
    high = []
    low = []
    for node_idx in d[3]:
        if train:
            model = torch.load('/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/aifb_chk/model_aifb')

            explainer = Explain(model = model, data = data, node_idx = node_idx, n_hops = n_hops)
            optimizer = torch.optim.Adam(explainer.parameters(), lr=0.1)
            print('start training')
            explainer.train()
            for epoch in range(50):
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
                torch.save(masked_hor, f'aifb_chk/masked_hor{node_idx}')
            print('masked_ver', masked_ver)

        else:
            masked_ver = torch.load(f'aifb_chk/masked_ver{node_idx}')
            masked_hor = torch.load(f'aifb_chk/masked_hor{node_idx}')
        #h = visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=False, low_threshold=False)
        h = selected(masked_ver, threshold,data, low_threshold=False)
        print('most important relations: ', h)
        high.append(h)
        #print('high:', high)
        if len(h) < len(masked_ver.coalesce().values()):
            l = selected(masked_ver, threshold,data, low_threshold=True)
            print('least important relations: ', l)
            low.append(l)
        with open('aifb_chk/Important_relation.txt', 'a') as f:
            f.write('\n node: ')
            f.write(str(node_idx))
            f.write('\n Most important relations:')
            f.write(str(h))
            f.write('\n Least important relations:')
            f.write(str(l))
            f.close()
        #print('low:', low)



        # l =  visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=False, low_threshold=True)

        # l = selected(masked_ver, threshold,data, low_threshold=True)
        # print('least important relations: ', l)
        # low.append(l)



        # visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=True, low_threshold=False)
        # dict_index = dict_index_classes(data,masked_ver)


    #print('dict index:' ,dict_index)
    #print('masked ver values: ', masked_ver.coalesce().values())
    print('high', high)

    print('low', low)    
    with open('aifb_chk/Counter_imp_relations.txt', 'a') as f:
        f.write('\n high:')
        f.write(str(high))
        f.write('\n low:')
        f.write(str(low))
        f.close()


def main2(node_idx, n_hops, threshold, train):

    data = kg.load('aifb', torch=True) 
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    d_classes(data)
    d = {key.item(): data.withheld[:, 0][data.withheld[:, 1] == key].tolist() for key in torch.unique(data.withheld[:, 1])}
    high = []
    low = []

    if train:

        model = torch.load('/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/aifb_chk/model_aifb')

        explainer = Explain(model = model, data = data, node_idx = node_idx, n_hops = n_hops)
        optimizer = torch.optim.Adam(explainer.parameters(), lr=0.01)
        print('start training')
        explainer.train()
        for epoch in range(50):
            explainer.zero_grad()
            optimizer.zero_grad()
            ypred, masked_hor, masked_ver = explainer.forward()
            print('masked_ver', masked_ver.to_sparse())
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
            torch.save(masked_ver, f'masked_ver{node_idx}_{epoch}')
            torch.save(masked_hor, f'masked_hor{node_idx}_{epoch}')
        #print('masked_ver', masked_ver)

    else:
        masked_ver = torch.load(f'aifb_chk/masked_ver{node_idx}')
        masked_hor = torch.load(f'aifb_chk/masked_hor{node_idx}')
    print('masked_ver', masked_ver)
    #h = visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=False, low_threshold=False)
    h = selected(masked_ver, threshold,data, low_threshold=False)
    print('most important relations: ', h)
    high.append(h)
    print('high:', high)


    # l =  visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=False, low_threshold=True)
    if len(h) < len(masked_ver.coalesce().values()):
        l = selected(masked_ver, threshold,data, low_threshold=True)
        print('least important relations: ', l)
        low.append(l)
    # l = selected(masked_ver, threshold,data, low_threshold=True)
    # print('least important relations: ', l)
    # low.append(l)



    #visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=True, low_threshold=False)
    # dict_index = dict_index_classes(data,masked_ver)


#print('dict index:' ,dict_index)
#print('masked ver values: ', masked_ver.coalesce().values())
    print('high', high)

    print('low', low)        
    model = torch.load('/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/aifb_chk/model_aifb')
    ypred = model.forward2(masked_hor, masked_ver)
    node_pred = ypred[node_idx, :]
    #print(nn.Softmax(dim=1)(ypred))
    #print(nn.Softmax(dim=0)(node_pred))
    res = nn.Softmax(dim=0)(node_pred)
    print('ypred', res)

    #ypred = model.forward2(torch.load('masked_hor_forward'), torch.load('masked_ver_forward'))
    # node_pred = ypred[node_idx, :]
    # res = nn.Softmax(dim=0)(node_pred)
    # print('ypred', res)


    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    y_full = model.forward2(hor_graph, ver_graph)
    print(nn.Softmax(dim=1)(y_full))
    node_pred_full = y_full[node_idx, :]
    #print(nn.Softmax(dim=0)(node_pred_full))
    res_full = nn.Softmax(dim=0)(node_pred_full)
    print('ypred full', res_full)
    h,v = torch.load('masked_hor_forward'), torch.load('masked_ver_forward')
    y_full3 = model.forward3(ver_graph)
    node_pred_full3 = y_full3[0][node_idx, :]
    res_full3 = nn.Softmax(dim=0)(node_pred_full3)
    print('ypred full3', res_full3)





    print('original label',[k for k, v in d.items() if node_idx in v])



if __name__ == "__main__":
    main2(5757,2,0.5, train=True)
    #main(1,0.5, train=True)



    

    
