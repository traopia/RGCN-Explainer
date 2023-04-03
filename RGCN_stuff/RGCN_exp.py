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
#from GNN_exp import ExplainModule


#rgcn 
from rgcn_model import adj, enrich, sum_sparse, RGCN
from rgcn_explainer_utils import dict_index_classes, dict_triples_semantics, visualize, find_n_hop_neighbors, match_to_classes, match_to_triples, edge_index_oneadj, get_relations, d_classes
from RGCN_exp_utils import hor_ver_graph, find_n_hop_neighbors, construct_edge_mask


class Explainer:
    def __init__(self,
        model,
        data,
        node_idx, 
        n_hops):

        self.model =torch.load('/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/aifb_chk/model_aifb')
        self.data = data
        self.label = self.data.withheld[:, 1]
        self.pred = torch.load('aifb_chk/prediction_aifb')
        self.node_idx = node_idx
        self.n_hops = n_hops
        self.n = data.num_entities
        self.r = data.num_relations
        self.triples = data.triples
        self.edge_index = edge_index_oneadj(self.triples)
        self.model.eval()
        self.hor_graph, self.ver_graph = hor_ver_graph(self.triples, self.n, self.r)
        self.new_node_idx = self.new_index()
        #self.neighborhoods = neighborhoods(adj=self.adj, n_hops=self.n_hops)


 
    def new_index(self):
        idxw, clsw = self.data.withheld[:, 0], self.data.withheld[:, 1]
        idxw, clsw = idxw.long(), clsw.long()
        idxw_list = list(idxw)
        self.new_node_idx = idxw_list.index(self.node_idx)
        return self.new_node_idx 
    


    def explain(self, node_idx):
            print("node label:", self.label[self.new_node_idx])
            neighbors, sub_edges_tensor = find_n_hop_neighbors(self.edge_index, self.n_hops, self.new_node_idx)
            neighbors = list(neighbors)
            #pred_label = torch.argmax(self.pred[neighbors], dim=1)
            pred_label = torch.argmax(self.pred[self.new_node_idx])
            print("Node predicted label: ", pred_label)

            sub_triples = match_to_triples(sub_edges_tensor.t(),self.triples)
            sub_hor_graph, sub_ver_graph = hor_ver_graph(sub_triples, self.n, self.r)

            explainer = ExplainModule(
                                                model = self.model,
                                                label = self.label,
                                                hor_graph = sub_hor_graph, 
                                                ver_graph = sub_ver_graph,
                                                num_nodes = len(neighbors))

            self.model.eval()
            explainer.train()
            for epoch in range(100):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, hor_att, ver_att = explainer() # forward pass of the explainer
                loss = explainer.loss(ypred, pred_label, self.node_idx, epoch) # loss function
                loss.backward() 

                explainer.optimizer.step()
                mask_density = explainer.mask_density()
                print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )
                #single_subgraph_label = sub_label.squeeze() 
                print('epoch:', epoch)
                # if epoch % 25== 0:
                # #         # explainer.log_mask(epoch)
                # #         # explainer.log_masked_adj(
                # #         #     node_idx_new, epoch, label=single_subgraph_label
                # #         # )
                #     explainer.log_adj_grad(
                #         node_idx_new, pred_label, epoch, label=single_subgraph_label
                #     )
            print('Finished Training')
            # if model == "exp":
            #         masked_adj = (
            #             explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
            #         )
            # else:
            hor_att = nn.functional.sigmoid(hor_att)
            ver_att = nn.functional.sigmoid(ver_att)
            masked_hor = hor_att.cpu().detach().numpy() * sub_hor_graph.squeeze()
            masked_ver = ver_att.cpu().detach().numpy() * sub_ver_graph.squeeze()

            torch.save(masked_hor, 'cora_chk/masked_hor1')
            torch.save(masked_ver, 'cora_chk/masked_ver1')
            return masked_hor, masked_ver, neighbors
    

class ExplainModule(nn.Module):
    def __init__(
        self,
        model,
        label,
        hor_graph, 
        ver_graph,
        num_nodes):
        super(ExplainModule, self).__init__()
        self.hor_graph = hor_graph
        self.ver_graph = ver_graph
        self.model = model
        self.label = label
        init_strategy = "normal"
        self.mask = construct_edge_mask(
            num_nodes)
        params = [self.mask]
        self.optimizer = torch.optim.Adam(params, lr=0.01)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,}
        
    def _masked_adj(self, sub_graph):
        """ Masked adjacency matrix 
        input: edge_mask, sub_adj, diag_mask
        output: masked_adj
        """
        adj = sub_graph.coalesce().values()
        indices = sub_graph.coalesce().indices()
        size = sub_graph.coalesce().size()
        sym_mask = self.mask
        sym_mask = torch.sigmoid(self.mask)
        
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = torch.tensor(adj)
        print(sym_mask.shape)
        print(adj.shape)
        masked_adj = adj * sym_mask

        return torch.sparse.FloatTensor(indices=indices, values= masked_adj, size=size )




    
    def forward(self):
        """
        Returns:
            ypred: prediction of the query node made by the current model (on the subgraph)

        """
        masked_hor = self._masked_adj(self.hor_graph)
        masked_ver = self._masked_adj(self.ver_graph)

        ypred, hor_att, ver_att = self.model.forward3(masked_hor, masked_ver)
        
        node_pred = ypred[self.node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)

        return res,hor_att, ver_att
    

    def adj_feat_grad(self, node_idx, pred_label_node):
        """
        Compute the gradient of the prediction w.r.t. the adjacency matrix
        and the node features.
        """
        self.model.zero_grad()
        self.ver_graph.requires_grad = True
        self.hor_graph.requires_grad = True

        if self.ver_graph.grad and self.hor_graph.grad is not None:
            print('self.adj.grad', self.ver_graph.grad, self.hor_graph.grad)
            self.ver_graph.grad.zero_() # zero out the gradient
            self.hor_graph.grad.zero_() # zero out the gradient
        else:    
            ver_graph, hor_graph = self.ver_graph, self.hor_graph
        print('self.adj.grad', self.ver_graph.grad, self.hor_graph.grad)
        ypred, _ = self.model.forward3(hor_graph, ver_graph)

        logit = nn.Softmax(dim=0)(ypred[node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return  self.ver_graph.grad, self.hor_graph.grad
    
    def log_adj_grad(self, node_idx, pred_label):

        predicted_label = pred_label[node_idx]
        #adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])
        ver_graph_grad, hor_graph_grad = self.adj_feat_grad(node_idx, predicted_label)
        print(ver_graph_grad, hor_graph_grad)
        if ver_graph_grad and hor_graph_grad is not None:
            ver_graph_grad, hor_graph_grad = torch.abs(ver_graph_grad), torch.abs(hor_graph_grad)


    def loss(self, pred, pred_label, node_idx, epoch):
        pred_label_node = pred_label[node_idx]
        gt_label_node = self.label[node_idx]
        logit = pred[gt_label_node]
        pred_loss = -torch.log(logit)
        # size
        mask = self.mask

        mask = torch.sigmoid(self.mask)

        size_loss = self.coeffs["size"] * torch.sum(mask)

        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        loss = pred_loss + size_loss  + mask_ent_loss
        return loss





def main(node_idx, n_hops, threshold):
    data = kg.load('aifb', torch=True) 
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    d_classes(data)

    model = torch.load('/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/aifb_chk/model_aifb')

    explainer = Explainer(model = model,
                                        data = data,
                                        node_idx = node_idx, 
                                        n_hops = n_hops,)
    masked_hor, masked_ver, neighbors = explainer.explain(node_idx)
    visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=False, low_threshold=False)
    visualize(node_idx, n_hops, data, masked_ver,threshold, result_weights=True, low_threshold=FalseBool())
    dict_index = dict_index_classes(data,masked_ver)
    print('dict index:' ,dict_index)



if __name__ == "__main__":
    main(5757,0,0)

    





