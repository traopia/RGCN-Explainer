from pickle import FALSE, TRUE
from colorama import init
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
        self.pred_label = torch.load(f'chk/{self.name}_chk/prediction_{self.name}_prune_{prune}')
        self.node_idx = node_idx
        self.n_hops = n_hops
        #self.hor_graph, self.ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        #self.n_hops = 0 if prune else 2 # number layers to propagate (in the paper it is 2)
        self.sub_edges, self.neighbors, self.sub_edges_tensor = find_n_hop_neighbors(self.edge_index, n=self.n_hops, node=self.node_idx)
        self.sub_triples = match_to_triples(self.sub_edges_tensor.t(),self.triples, sparse=False)
        self.overall_rel_frequency = dict(Counter(self.data.triples[:,1].tolist()))
        
    def new_index(self):
        idxw, clsw = self.data.withheld[:, 0], self.data.withheld[:, 1]
        idxw, clsw = idxw.long(), clsw.long()
        idxw_list = list(idxw)
        self.new_node_idx = idxw_list.index(self.node_idx)
        return self.new_node_idx 
    def explain(self, node_idx):
        print(f"node {node_idx} label: {self.label[node_idx]}")
        node_idx_new = self.new_index()
        neighbors = list(self.neighbors)
        #print(len(neighbors))
        #sub_label = torch.Tensor([self.label[i] for i in neighbors if i in self.label.keys()] )#self.label[neighbors]
        #print("o: ", sub_label)
        sub_hor_graph, sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)
        #node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood()
        #print("neigh graph idx: ", self, node_idx, node_idx_new)

        #sub_hor_graph = np.expand_dims(sub_hor_graph, axis=0)
        #sub_ver_graph = np.expand_dims(sub_ver_graph, axis=0)
        hor_graph, ver_graph = torch.tensor(sub_hor_graph, dtype=torch.float), torch.tensor(sub_ver_graph, dtype=torch.float)
    
        #label = torch.tensor(sub_label, dtype=torch.long)
        label = torch.tensor([self.label[node_idx]], dtype=torch.long)
        pred_label = torch.argmax(self.pred_label[node_idx_new])
        #print("Node predicted label: ", pred_label)

        # Explainer model whose parameters are to be learned
        # print(ver_graph.size(), hor_graph.size())
        # print(ver_graph.coalesce().indices().size(), hor_graph.coalesce().indices().size())
        # print(ver_graph)

        explainer = ExplainModule(
            hor_graph, ver_graph,
            self.model,
            label,
            self.data
        )


        run = wandb.init(
        # Set the project where this run will be logged
        project="RGCN-Explainer",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.01,
            "epochs": 5,
        })
        wandb.login()

        self.model.eval()
        explainer.train()  # set the explainer model to training mode

        for epoch in range(30):
            explainer.zero_grad()  # zero the gradient buffers of all parameters
            explainer.optimizer.zero_grad()
            ypred, masked_hor, masked_ver = explainer(self.node_idx)  # forward pass of the explainer
            #loss, pred_loss, size_loss, size_num_loss, mask_ent_loss, reg_loss, squared_loss = explainer.loss(ypred, pred_label, self.node_idx, epoch)  # loss function
            loss, pred_loss, size_loss, mask_ent_loss, size_std_loss, size_num_loss = explainer.loss(ypred, pred_label, self.node_idx, epoch) 
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
            # wandb.log({"len mask > 0.5": len([i for i in masked_ver.coalesce().values() if i > 0.5]), "loss": loss,
            #            "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss,
            #            "size_num_loss": size_num_loss,"reg_loss": reg_loss, "squae_loss": squared_loss})
            
            wandb.log({"len mask > 0.5": len([i for i in masked_ver.coalesce().values() if i > 0.5]), "loss": loss,
            "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss, "size_std_loss": size_std_loss, "size_num_loss": size_num_loss})
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
            label, 
            data):
        super(ExplainModule, self).__init__()
        self.hor_graph, self.ver_graph = hor_graph, ver_graph #hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        self.model = model
        self.label = label
        self.data = data


        num_nodes = self.hor_graph.coalesce().indices().size()[1] #self.hor_graph.size()[0]#self.hor_graph.coalesce().values().shape[0]
        self.mask = self.construct_edge_mask(num_nodes, self.hor_graph,self.data)

        params = [self.mask]
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        #self.optimizer = torch.optim.Adam(params, lr=0.5, weight_decay=0.001)
       

        self.coeffs = {
            "pred": 1,
            "size": 0.0005,  # 0.005,
            "size_std": -1,
            "feat_size": 1.0,
            "ent": 1,
            "feat_ent": 0.1,
            "grad": 1,
            "lap": 1.0, 
            "size_num": 0.1,
            "lr": 0.1,
            "lambda_reg": 0.1}
        
        self.coeffs_grid = {
            "pred": 1,
            "size": 0.0005,
            "size_std": [-100,-10,-1],
            "feat_size": 1.0,
            "ent": 1,
            "feat_ent": 0.1,
            "grad": 1,
            "lap": 1.0, 
            "size_num": [0.005, 0.05, 0.5],
            "lr": [0.05, 0.1, 0.5],
            "lambda_reg": 0.1}
        self.optimizer = torch.optim.Adam(params, lr=self.coeffs["lr"], weight_decay=0.1)


    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
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
    
    def construct_edge_mask(self, num_nodes,sparse_tensor,data, init_strategy="normal", const_val=1.0, relation_id = 2):
        """
        Construct edge mask
        """
        if num_nodes > 1000:
            init(strategy="const", const_val=0.1)
        data = self.data
        num_entities = data.num_entities
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
        elif init_strategy == "zero_out":
            '''initialize the mask with the zero out strategy: we zero out edges belonging to specific relations'''
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
            output_indices, output_values, value_indices=select_relation(sparse_tensor,relation_id)
            _,_,value_indices1=select_relation(sparse_tensor,33)
            #print(value_indices, value_indices1)
            value_indices = torch.cat((value_indices, value_indices1), 0)
            mask.data[[value_indices]] = 0
        

        elif init_strategy == "overall_frequency":
            '''Initialize the mask with the overall frequency of the relations'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            overall_rel_frequency = dict(Counter(data.triples[:,1].tolist()))#.most_common()

            overall_rel_frequency_  = {key: round(value/len(data.triples[:,1].tolist()),5) for key, value in overall_rel_frequency.items()}
            for i in p:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = overall_rel_frequency_[i]
        
        elif init_strategy == "relative_frequency":
            ''' Initialize the mask with the relative frequency of the relations-relative for the node to be explained'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            rel_frequency = dict(Counter(p))
            rel_frequency_  = {key: round(value/len(p),5) for key, value in rel_frequency.items()}
            for i in p:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = rel_frequency_[i]

        elif init_strategy == "inverse_relative_frequency":
            ''' Initialize the mask with the relative frequency of the relations-relative for the node to be explained'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            rel_frequency = dict(Counter(p))
            rel_frequency_  = {key: 1 - round(value/len(p),5) for key, value in rel_frequency.items()}
            for i in p:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = rel_frequency_[i]


        elif init_strategy == "domain_frequency":
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            dict_domain, dict_range = domain_range_freq(data, len(d_classes(data)))
            for i in p:

                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = dict_domain[i]

        elif init_strategy == "range_frequency":
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            dict_domain, dict_range = domain_range_freq(data, len(d_classes(data)))
            for i in p:
                    _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                    mask.data[[value_indices]] = dict_range[i]
        elif init_strategy == "rdf":
            rdf = [i for i in range(data.num_relations) if 'rdf' in data.i2r[i]]
            for i in rdf:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = 0
        elif init_strategy == "owl":
            owl = [i for i in range(data.num_relations) if 'owl' in data.i2r[i]]
            for i in owl:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = 0
                
        return mask

    def _masked_adj_ver(self):
        "" "Mask the adjacency matrix with the learned mask" ""
        sym_mask = self.mask

        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2

        adj = self.ver_graph.coalesce().values()
        adj = torch.Tensor(adj)#adj.clone().detach()


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
        adj = torch.Tensor(adj)#adj.clone().detach()
        masked_adj = adj * sym_mask
        #print('masked_adj', masked_adj)

        #return masked_adj * self.diag_mask

        result = torch.sparse.FloatTensor(indices=self.hor_graph.coalesce().indices(), values= masked_adj, size=self.hor_graph.coalesce().size())
        return result

    

    def forward(self, node_idx):
        #print('node_idx', node_idx)
        self.masked_ver = self._masked_adj_ver()  # masked adj is the adj matrix with the mask applied
        self.masked_hor = self._masked_adj_hor()  # masked adj is the adj matrix with the mask applied

        ypred = self.model.forward2(self.masked_hor, self.masked_ver)
        node_pred = ypred[node_idx,:]
        res = nn.Softmax(dim=0)(node_pred[0:])
        
        return res,   self.masked_hor, self.masked_ver

    def size_loss_f(self,mask, coeffs):
        center_weight = coeffs.get("center_weight", 10)
        edge_weight = coeffs.get("edge_weight", 1.0)
        mask_center = mask.mean()
        weight = torch.exp(-(torch.arange(mask.numel()) - mask.numel() / 2.0).pow(2) / (2.0 * (mask.numel() / 4.0) ** 2))
        weight /= weight.mean()
        weight = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)
        weight = torch.where(torch.isinf(weight), torch.ones_like(weight), weight)
        weight = torch.where(torch.abs(weight) < 0.01, torch.zeros_like(weight), weight)
        weight = (center_weight - edge_weight) * (weight - weight.min()) / (weight.max() - weight.min()) + edge_weight
        size_loss = (mask - mask_center).pow(2) * weight
        return coeffs.get("size", 1.0) * size_loss.sum()

    def get_frequency_relations(self,v):
        _ ,p = torch.div(v.coalesce().indices(), v.size()[0], rounding_mode='floor')
        return dict(Counter(p))

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        selected_mask, num_high, p,relation_counter = subset_sparse(self.masked_hor, self.data)
        #print('relation_counter', relation_counter)
        
        mask = torch.sigmoid(self.mask)  # sigmoid of the mask
        #mask = self.mask
        mask_without_small = mask[mask > 0.5]
        #print('mask_without_small', mask_without_small)
        num_high = len(mask_without_small)
        print('num_high', num_high,'len(mask)', len(mask))
        #print('mask', mask)
        node_degree_regulizer = (len(mask) + num_high) *0.1

        #print(node_degree_regulizer)


        # prediction loss

        
        gt_label_node = self.label#[node_idx]
        logit = pred[gt_label_node]
        pred_loss =  -torch.log(logit) * self.coeffs["pred"]

        #size loss
        # lambda_reg = self.coeffs["lambda_reg"]
        # if num_high>300:
        #     lambda_reg = 0.5



        # lambda_l1 = 0.5
        # lambda_squared = 0.5

        # # Compute the L1 regularization loss
        # l1_loss = lambda_l1 * torch.sum(torch.abs(mask))

        # # Compute the squared regularization loss
        # squared_loss = lambda_squared * torch.sum(mask**2) + l1_loss
        # if len(mask_without_small) < len(mask):

        #     size_loss =  100*self.coeffs["size"] * torch.std(mask_without_small)
        #     reg_loss = 0.1 * lambda_reg * torch.norm(mask_without_small, p=1)
        # else:   
        #     reg_loss =  lambda_reg * torch.norm(mask_without_small, p=1)
        #     size_loss = self.coeffs["size"] * torch.std(mask) 


        # size_loss = self.size_loss_f(mask, self.coeffs)
        #size_loss = 10 * torch.sum(mask)

        #size_num_loss = self.coeffs["size_num"] * num_high #(num_high - self.num_nodes / 2) ** 2
        #size_num_loss = self.coeffs["size_num"] * (len(mask) - num_high / 2) ** 2
        #ize_num_loss = self.coeffs["size_num"] * num_high+0.00001/len(mask)
        size_num_loss = (len(mask) + num_high) *0.00001

        size_loss = self.coeffs["size"] * torch.sum(torch.abs(mask_without_small))
        #size_loss_std = (0.1*len(mask))*self.coeffs["size_std"] * torch.std(mask_without_small)
        size_loss_std = (len(mask))*self.coeffs["size_std"] * torch.std(mask_without_small)



        # entropy edge mask 
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        
        # if len(mask_without_small) <=3:
        #     loss = pred_loss
        # else:
        #     loss = pred_loss + size_num_loss + mask_ent_loss + reg_loss + squared_loss + size_loss
        loss = torch.exp(pred_loss + size_loss + mask_ent_loss + size_loss_std) #+ size_num_loss

        #print('pred_loss', pred_loss)
        #print('size_loss', size_loss)
        #print('size_num_loss', size_num_loss)
        #print('mask_ent_loss', mask_ent_loss)

        #print('size_loss_std', size_loss_std)
        # print('reg_loss', reg_loss)
        # print('squared_loss', squared_loss)


        #return loss, pred_loss, size_loss, size_num_loss, mask_ent_loss, reg_loss, squared_loss
        return loss, pred_loss, size_loss, mask_ent_loss,size_loss_std, size_num_loss


    

    def per_relation_loss(self, pred):
        _ ,p = torch.div(self.masked_hor.coalesce().indices(), self.masked_hor.size[0], rounding_mode='floor')
        print(p)






import wandb

def main(name,node_idx, prune=True, explain_all = False, train=False, threshold =0.5):



    n_hops = 0 if prune else 2
    #n_hops = 2
    if name in ['aifb', 'mutag', 'bgs', 'am']:
        data = kg.load(name, torch=True, final=False)
    else:    
    #data = kg.load(name, torch=True)  
        data = torch.load(f'data/IMDB/finals/{name}.pt')
    if prune:
        data = prunee(data, 2)
        data.triples = torch.Tensor(data.triples).to(int)#data.triples.clone().detach()
        data.withheld = torch.Tensor(data.withheld).to(int)#data.withheld.clone().detach()
        data.training = torch.Tensor(data.training).to(int)#data.training.clone().detach()

          
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    d_classes(data)
    d = {key.item(): data.withheld[:, 0][data.withheld[:, 1] == key].tolist() for key in torch.unique(data.withheld[:, 1])}


    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    high = []
    high_floats = []
    low = []
    relations = [data.i2rel[i][0] for i in range(len(data.i2rel))]
    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    relations = ['label', 'node_idx'] + relations
    df = pd.DataFrame(columns=relations)
    df_floats = pd.DataFrame(columns=relations)
    if explain_all == True:
        for target_label in range(len(d.keys())):
            for node_idx in d[target_label]:
                
                explainer = Explainer(model, data,name,  node_idx, n_hops, prune)
                masked_hor, masked_ver = explainer.explain(node_idx)
                if not os.path.exists(f'chk/{name}_chk/masked_adj'):
                        os.makedirs(f'chk/{name}_chk/masked_adj') 
                torch.save(masked_ver, f'chk/{name}_chk/masked_adj/masked_ver{node_idx}_new')
                torch.save(masked_hor, f'chk/{name}_chk/masked_adj/masked_hor{node_idx}_new') 
                #print('masked_ver', masked_ver)
                h = visualize(node_idx, n_hops, data, masked_ver,threshold=0.5, name = name, result_weights=False, low_threshold=False)
                h = selected(masked_ver, threshold=0.5,data=data, low_threshold=False)
                res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
                #print('ypred explain', res)

                hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
                y_full = model.forward2(hor_graph, ver_graph)
                node_pred_full = y_full[node_idx, :]
                res_full = nn.Softmax(dim=0)(node_pred_full)
                #print('ypred full', res_full)

                high.append(h)
                h = dict(h)
                info = {'label': target_label, 'node_idx': str(node_idx)}
                h.update(info)
                df.loc[str(node_idx)] = h

                # h_floats = selected(masked_ver, threshold=0.5,data=data, low_threshold=False,float=True)
                # high_floats.append(h_floats)
                # h_floats = dict(h_floats)
                # h_floats.update(info)
                # df_floats.loc[str(node_idx)] = h_floats
                if not os.path.exists(f'Relation_Importance_{name}'):
                    os.makedirs(f'Relation_Importance_{name}') 
                    df.to_csv(f'Relation_Importance_{name}/Relations_Important_{name}_{node_idx}.csv', index=False)

                print('node_idx', node_idx, 
                    '\n node original label',[k for k, v in d.items() if node_idx in v],
                    '\n node predicted label explain', torch.argmax(res).item(),
                    '\n node prediction probability explain', res,
                        '\n node predicted label full', torch.argmax(res_full).item(),
                        'most important relations ', h,
                        '\n final masks and lenght', masked_ver, len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>threshold ]),
                        '\n ---------------------------------------------------------------')
        if not os.path.exists(f'Relation_Importance_{name}'):
            os.makedirs(f'Relation_Importance_{name}') 
        df.to_csv(f'Relation_Importance_{name}/Relations_Important_all_{name}.csv', index=False) 
        #df_floats.to_csv(f'Relation_Importance_{name}/Relations_Important_all_{name}_{node_idx}_floats.csv', index=False) 
                
    else:
        if name != 'aifb':
            node_idx = d[0][0]
        if train:
            explainer = Explainer(model, data,name,  node_idx, n_hops, prune)
            masked_hor, masked_ver = explainer.explain(node_idx)
            if not os.path.exists(f'chk/{name}_chk/masked_adj'):
                    os.makedirs(f'chk/{name}_chk/masked_adj') 
            torch.save(masked_ver, f'chk/{name}_chk/masked_adj/masked_ver{node_idx}_new')
            torch.save(masked_hor, f'chk/{name}_chk/masked_adj/masked_hor{node_idx}_new') 
        else:
            masked_ver = torch.load(f'chk/{name}_chk/masked_adj/masked_ver{node_idx}_new')
            masked_hor = torch.load(f'chk/{name}_chk/masked_adj/masked_hor{node_idx}_new')
        h = visualize(node_idx, n_hops, data, masked_ver,threshold=threshold , name = name, result_weights=False, low_threshold=False)
        h = selected(masked_ver, threshold=0.5,data=data, low_threshold=False)
        res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
        #print('ypred explain', res)

        hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        y_full = model.forward2(hor_graph, ver_graph)
        node_pred_full = y_full[node_idx, :]
        res_full = nn.Softmax(dim=0)(node_pred_full)
        #print('ypred full', res_full)
        high.append(h)
        h = dict(h)



        target_label = str([k for k, v in d.items() if node_idx in v])
        info = {'label': str(target_label), 'node_idx': str(node_idx)}
        h.update(info)
        
        df.loc[str(node_idx)] = h

        # h_floats = selected(masked_ver, threshold=0.5,data=data, low_threshold=False,float=True)
        # high_floats.append(h_floats)
        # h_floats = dict(h_floats)
        # h_floats.update(info)
        # df_floats.loc[str(node_idx)] = h_floats


        print('node_idx', node_idx, 
            '\n node original label',target_label,
            '\n node predicted label explain', torch.argmax(res).item(),
            '\n node prediction probability explain', res,
            '\n node predicted label full', torch.argmax(res_full).item(),
            '\n node prediction probability full', res_full,
            '\n final masks and lenght', masked_ver, len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>threshold ]))
        if not os.path.exists(f'Relation_Importance_{name}'):
            os.makedirs(f'Relation_Importance_{name}') 
        df.to_csv(f'Relation_Importance_{name}/Relations_Important_{name}_{node_idx}.csv', index=False)
        #df_floats.to_csv(f'Relation_Importance_{name}/Relations_Important_{name}_{node_idx}_floats.csv', index=False)


if __name__ == "__main__":
    main('aifb',node_idx= 5757, prune= True, explain_all =False, train=True, threshold = 0.5)    
    #main('aifb',node_idx= 5791, prune= True, explain_all =False, train=True)    