from pickle import FALSE
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
from src.rgcn_model import adj, enrich, sum_sparse, RGCN
from src.rgcn_explainer_utils import *
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
    #print(hor_size)
    r = rn // n #number of relations enriched divided by number of nodes

    vals = torch.ones(ver_ind.size(0), dtype=torch.float) #number of enriched triples
    #vals = vals / sum_sparse(ver_ind, vals, ver_size) #normalize the values by the number of edges

    hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size) #size: n,r, emb


    ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)

    return hor_graph, ver_graph

class Explainer:
    def __init__(self,
                 model,
                 data,
                 name,
                 node_idx,
                 n_hops,
                 prune=True):
        self.model = model
        self.model.eval()
        self.data = data
        self.name = name
        self.n = data.num_entities
        self.r = data.num_relations
        self.triples = data.triples
        self.edge_index = edge_index_oneadj(self.triples)
       
        self.label = {int(k):int(v) for k,v in zip(data.withheld[:, 0], data.withheld[:, 1])} #self.data.withheld[:, 1]
        self.pred_label = torch.load(f'{self.name}_chk/prediction_{self.name}_prune_{prune}')
        self.node_idx = node_idx
        self.n_hops = n_hops
        #self.hor_graph, self.ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        #self.n_hops = 0 if prune else 2 # number layers to propagate (in the paper it is 2)
        self.sub_edges, self.neighbors, self.sub_edges_tensor = find_n_hop_neighbors(self.edge_index, n=self.n_hops, node=self.node_idx)
        self.sub_triples = match_to_triples(self.sub_edges_tensor.t(),self.triples)

    def new_index(self):
        idxw, clsw = self.data.withheld[:, 0], self.data.withheld[:, 1]
        idxw, clsw = idxw.long(), clsw.long()
        idxw_list = list(idxw)
        self.new_node_idx = idxw_list.index(self.node_idx)
        return self.new_node_idx 
    def explain(self, node_idx):
        print("node label:", self.label[node_idx])
        node_idx_new = self.new_index()
        neighbors = list(self.neighbors)
        print(len(neighbors))
        #sub_label = torch.Tensor([self.label[i] for i in neighbors if i in self.label.keys()] )#self.label[neighbors]
        #print("o: ", sub_label)
        sub_hor_graph, sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)
        #node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood()
        print("neigh graph idx: ", self, node_idx, node_idx_new)

        #sub_hor_graph = np.expand_dims(sub_hor_graph, axis=0)
        #sub_ver_graph = np.expand_dims(sub_ver_graph, axis=0)
        hor_graph, ver_graph = torch.tensor(sub_hor_graph, dtype=torch.float), torch.tensor(sub_ver_graph, dtype=torch.float)
    
        #label = torch.tensor(sub_label, dtype=torch.long)
        label = torch.tensor([self.label[node_idx]], dtype=torch.long)
        pred_label = torch.argmax(self.pred_label[node_idx_new])
        print("Node predicted label: ", pred_label)

        # Explainer model whose parameters are to be learned
        # print(ver_graph.size(), hor_graph.size())
        # print(ver_graph.coalesce().indices().size(), hor_graph.coalesce().indices().size())
        # print(ver_graph)
        #breakpoint()
        explainer = ExplainModule(
            hor_graph, ver_graph,
            self.model,
            label
        )

        self.model.eval()
        explainer.train()  # set the explainer model to training mode
        for epoch in range(30):
            explainer.zero_grad()  # zero the gradient buffers of all parameters
            explainer.optimizer.zero_grad()
            ypred, masked_hor, masked_ver = explainer(self.node_idx)  # forward pass of the explainer
            loss = explainer.loss(ypred, pred_label, self.node_idx, epoch)  # loss function
            print('loss:', loss)

            loss.backward()

            explainer.optimizer.step()
   
            print(epoch)
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; pred: ",
                ypred,
            )

        print('Finished Training')

        masked_hor_values = (masked_hor.coalesce().values() * hor_graph.coalesce().values())
        masked_ver_values = (masked_ver.coalesce().values() * ver_graph.coalesce().values())
        #masked_hor_values, masked_ver_values = masked_hor.coalesce().values(), masked_ver.coalesce().values()
        masked_hor = torch.sparse.FloatTensor(masked_hor.coalesce().indices(), masked_hor_values, hor_graph.size())
        masked_ver = torch.sparse.FloatTensor(masked_ver.coalesce().indices(), masked_ver_values, ver_graph.size())

        return masked_hor, masked_ver

class ExplainModule(nn.Module):
    def __init__(
            self,
            hor_graph,
            ver_graph,
            model,
            label):
        super(ExplainModule, self).__init__()
        self.hor_graph, self.ver_graph = hor_graph, ver_graph #hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        self.model = model
        self.label = label

        num_nodes = self.hor_graph.coalesce().indices().size()[1] #self.hor_graph.size()[0]#self.hor_graph.coalesce().values().shape[0]
        self.mask = self.construct_edge_mask(num_nodes)
        print(self.mask.size())
        
        params = [self.mask]
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        self.optimizer = torch.optim.Adam(params, lr=0.01)

        self.coeffs = {
            "pred": 1,
            "size": 0.9,  # 0.005,
            "feat_size": 1.0,
            "ent": 2.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0, }



    def construct_edge_mask(self, num_nodes, init_strategy="const", const_val=1.0):
        """
        Construct edge mask
        """
        torch.manual_seed(42)
        mask = nn.Parameter(torch.FloatTensor(num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)        
        return mask

    def _masked_adj_ver(self):
        "" "Mask the adjacency matrix with the learned mask" ""
        sym_mask = self.mask

        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2

        adj = self.ver_graph.coalesce().values()
        adj = torch.tensor(adj)

        #adj = adj.unsqueeze(0)
        # print('size ')
        # print(adj.size())
        # print(sym_mask.size())
        # #adj = adj.to_dense()
        # print(adj, adj.size())
        # #sym_mask = sym_mask.to_sparse()
        # print(adj, adj.shape)
        # print(sym_mask, sym_mask.shape)

        masked_adj = adj * sym_mask

        #return masked_adj * self.diag_mask
        result = torch.sparse.FloatTensor(indices=self.ver_graph.coalesce().indices(), values= masked_adj, size=self.ver_graph.coalesce().size())
        return result   

    def _masked_adj_hor(self):
        "" "Mask the adjacency matrix with the learned mask" ""
        sym_mask = self.mask

        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.hor_graph.coalesce().values()
        adj = torch.tensor(adj)
        masked_adj = adj * sym_mask
        print('masked_adj', masked_adj)

        #return masked_adj * self.diag_mask

        result = torch.sparse.FloatTensor(indices=self.hor_graph.coalesce().indices(), values= masked_adj, size=self.hor_graph.coalesce().size())
        return result

    

    def forward(self, node_idx):
        print('node_idx', node_idx)
        self.masked_ver = self._masked_adj_ver()  # masked adj is the adj matrix with the mask applied
        self.masked_hor = self._masked_adj_hor()  # masked adj is the adj matrix with the mask applied
        print('masked_hor', self.masked_hor)

        #ypred = self.model(self.masked_hor, self.masked_ver)
        ypred = self.model.forward2(self.masked_hor, self.masked_ver)
        print('ypred', ypred)
        node_pred = ypred[node_idx,:]
        res = nn.Softmax(dim=0)(node_pred[0:])

        print('res:', res)
        return res,   self.masked_hor, self.masked_ver



    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """

        # prediction loss
        #pred_label_node = pred_label[node_idx]

        #pred_label_node = pred_label
        print('label', self.label)
        
        gt_label_node = self.label#[node_idx]
        logit = pred[gt_label_node]
        print('logit', logit)
        pred_loss =  -torch.log(logit) * self.coeffs["pred"]

        # size loss
        mask = self.mask
        # print('mask', mask)
        #print('gradient of the mask:', mask.grad)  # None at the beginning

        mask = torch.sigmoid(self.mask)  # sigmoid of the mask

        size_loss = self.coeffs["size"] * torch.sum(mask)



        # entropy edge mask 
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)


        # laplacian loss
        # D = torch.diag(torch.sum(self.masked_adj[0], 0))
        # m_adj = self.masked_adj
        # L = D - m_adj

        # pred_label_t = torch.tensor(pred_label, dtype=torch.float)

        # lap_loss = (self.coeffs["lap"] * (pred_label_t @ L @ pred_label_t) / self.masked_adj())

        loss = pred_loss + size_loss +  + mask_ent_loss #+ lap_loss  # feat_mask_ent_loss 

        return loss








def main(name,prune=True):
    n_hops = 0 if prune else 2
    if name in ['aifb', 'mutag', 'bgs', 'am']:
        data = kg.load(name, torch=True, final=False)
    else:    
    #data = kg.load(name, torch=True)  
        data = torch.load(f'data/IMDB/finals/{name}.pt')
    if prune:
        data = prunee(data, 2)
        data.triples = torch.tensor(data.triples)
        data.withheld = torch.tensor(data.withheld)
        data.training = torch.tensor(data.training)

          
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    d_classes(data)
    #breakpoint()
    d = {key.item(): data.withheld[:, 0][data.withheld[:, 1] == key].tolist() for key in torch.unique(data.withheld[:, 1])}
    #print(len(d.keys()))
    node_idx = 5678 #d[0][0]
    print('node_idx', node_idx)
    model = torch.load(f'{name}_chk/model_{name}_prune_{prune}')
    explainer = Explainer(model, data,name,  node_idx, n_hops, prune)
    masked_hor, masked_ver = explainer.explain(node_idx)
    if not os.path.exists(f'{name}_chk/masked_adj'):
            os.makedirs(f'{name}_chk/masked_adj') 
    torch.save(masked_ver, f'{name}_chk/masked_adj/masked_ver{node_idx}_new')
    torch.save(masked_hor, f'{name}_chk/masked_adj/masked_hor{node_idx}_new') 
    print('masked_ver', masked_ver)
    h = visualize(node_idx, n_hops, data, masked_ver,threshold=0.5, name = name, result_weights=False, low_threshold=False)
    h = selected(masked_ver, threshold=0.5,data=data, low_threshold=False)
    res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
    print('ypred explain', res)

    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    y_full = model.forward2(hor_graph, ver_graph)
    node_pred_full = y_full[node_idx, :]
    res_full = nn.Softmax(dim=0)(node_pred_full)
    print('ypred full', res_full)

if __name__ == "__main__":
    main('aifb',prune= True)    