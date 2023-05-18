from http.client import INSUFFICIENT_STORAGE
from logging import config
from pickle import FALSE, TRUE
from re import A
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

#params
import wandb




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
                 prune,
                 config):
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
        self.config = config
        
    def new_index(self):
        ''' Find the index of the node in the subgraph'''
        idxw, clsw = self.data.withheld[:, 0], self.data.withheld[:, 1]
        idxw, clsw = idxw.long(), clsw.long()
        idxw_list = list(idxw)
        self.new_node_idx = idxw_list.index(self.node_idx)
        return self.new_node_idx 
    


    def explain(self):
        ''' Explain the prediction of the node with index node_idx: main method'''
        node_idx = self.node_idx
        config = self.config
        print(f"node {node_idx} label:", self.label[node_idx])
        node_idx_new = self.new_index()
        neighbors = list(self.neighbors)
        print(len(neighbors))
        sub_hor_graph, sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)
        hor_graph, ver_graph = torch.tensor(sub_hor_graph, dtype=torch.float), torch.tensor(sub_ver_graph, dtype=torch.float)
    
        label = torch.tensor([self.label[node_idx]], dtype=torch.long)
        pred_label = torch.argmax(self.pred_label[node_idx_new])

        #wandb.init(project='RGCNExplainer', config=self.params)
        #config = wandb.config

        explainer = ExplainModule(
            hor_graph, ver_graph,
            self.model,
            label,
            self.data, 
            self.config
        )




        # sweep_config = {
        #     'method': 'grid',
        #     'parameters': {
        #         'lr': {'values': [0.05, 0.1, 0.5]},
        #         'size': {'values': [0.05, 0.005, 0.0005]},
        #         'ent': {'values': [10,1]},
        #         'size_std': {'values': [10,1]},
        #         'size_num': {'values': [0.5, 0.1, 0.05]},
        #     }
        # }
        # sweep_id = wandb.sweep(sweep_config, project='RGCNExplainer')
        # wandb.agent(sweep_id, function=main)

        # wandb.init(project='RGCNExplainer', config=params)  
        # wandb.login()
        config = wandb.config

        self.model.eval()
        explainer.train()  # set the explainer model to training mode

        for epoch in range(config.epochs):
            explainer.zero_grad()  # zero the gradient buffers of all parameters
            explainer.optimizer.zero_grad()
            ypred, masked_hor, masked_ver = explainer(self.node_idx)  # forward pass of the explainer
            loss, pred_loss, size_loss, mask_ent_loss, size_std_loss, size_num_loss, num_high = explainer.loss(ypred, config) 
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
            
            wandb.log({f"len mask > {config['threshold']}": len([i for i in masked_ver.coalesce().values() if i > config['threshold']]), "loss": loss,
            "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss, "size_std_loss": size_std_loss, "size_num_loss": size_num_loss, "num_high": num_high})
        print('Finished Training')

        masked_hor_values = (masked_hor.coalesce().values() * hor_graph.coalesce().values())
        masked_ver_values = (masked_ver.coalesce().values() * ver_graph.coalesce().values())
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
            data,
            config):
        super(ExplainModule, self).__init__()
        self.hor_graph, self.ver_graph = hor_graph, ver_graph #hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        self.model = model
        self.label = label
        self.data = data

        print('init_strategy:', config["init_strategy"])
        self.init_strategy = config["init_strategy"]

        num_nodes = self.hor_graph.coalesce().indices().size()[1] #self.hor_graph.size()[0]#self.hor_graph.coalesce().values().shape[0]
        self.mask = self.construct_edge_mask(num_nodes, self.hor_graph,self.data)

        params = [self.mask]
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        self.config = config


       
        
        self.optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=0.1)


    
    def construct_edge_mask(self, num_nodes,sparse_tensor,data, const_val=1.0, relation_id = 2):
        """
        Construct edge mask
        """
        init_strategy = self.init_strategy
        # if num_nodes > 1000:
        #     init(strategy="const", const_val=0.1)
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
            print(value_indices, value_indices1)
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
        print(f'mask initialized with {init_strategy} strategy: {mask}')   
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
        self.masked_ver = self._masked_adj_ver()  # masked adj is the adj matrix with the mask applied
        self.masked_hor = self._masked_adj_hor()  # masked adj is the adj matrix with the mask applied
        masked_ver,masked_hor = convert_binary(self.masked_ver, self.config["threshold"]), convert_binary(self.masked_hor, self.config["threshold"])
        ypred = self.model.forward2(masked_hor, masked_ver)
        #ypred = self.model.forward2(self.masked_hor, self.masked_ver)
        node_pred = ypred[node_idx,:]
        res = nn.Softmax(dim=0)(node_pred[0:])
        
        return res,   self.masked_hor, self.masked_ver



    def get_frequency_relations(self,v):
        _ ,p = torch.div(v.coalesce().indices(), v.size()[0], rounding_mode='floor')
        return dict(Counter(p))

    def loss(self, pred, config):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #selected_mask, num_high, p,relation_counter = subset_sparse(self.masked_hor, self.data, self.config["threshold"])
        #print('relation_counter', relation_counter)
        #self.mask = ((self.mask-0.5)**2)
        mask = torch.sigmoid(self.mask)  # sigmoid of the mask
   
        #mask = self.mask
        mask_without_small = mask[mask > config["threshold"]]
        print('mask_without_small', mask_without_small)
        num_high = len(mask_without_small)
        print('num_high', num_high,'len(mask)', len(mask))
        print('mask', mask)


        # prediction loss

        
        gt_label_node = self.label#[node_idx]te
        logit = pred[gt_label_node]
        pred_loss =  -torch.log(logit) 

        #size_num_loss = -(config["size_std"]*config["size_num"] * (len(mask) - num_high+3) )
        size_num_loss = config["size_std"]* (num_high+3)/len(mask)

        size_loss = config["size"] * torch.sum(torch.abs(mask))
        size_loss_std = -config["size_std"] * torch.std(mask)



        mask = mask/ torch.sum(mask)
        

        # entropy edge mask 
        #mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent = - torch.log(mask)
        mask_ent_loss = config["ent"] * torch.mean(mask_ent)


        #loss = torch.exp(pred_loss + mask_ent_loss + size_num_loss + size_loss_std + size_loss ) #
        loss = torch.exp(pred_loss + size_loss + mask_ent_loss + size_loss_std)

        print('pred_loss', pred_loss)
        print('size_loss', size_loss)
        #
        #print('size_num_loss', size_num_loss)
        print('mask_ent_loss', mask_ent_loss)

        print('size_loss_std', size_loss_std)

        return loss, pred_loss, size_loss, mask_ent_loss,size_loss_std, size_num_loss, num_high


    

    def per_relation_loss(self, pred):
        _ ,p = torch.div(self.masked_hor.coalesce().indices(), self.masked_hor.size[0], rounding_mode='floor')
        print(p)






import wandb

def main(name,node_idx, prune=True, explain_all = False, train=False):

    n_hops = 0 if prune else 2

    if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre']:
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
    edge_index = edge_index_oneadj(data.triples)
    _,neighbors,sub_edge = find_n_hop_neighbors(edge_index, n_hops, node_idx)
    num_neighbors = len(sub_edge.t())
    print('num_neighbors', num_neighbors)
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'lr': {'values': [0.1, 0.5]},
            'size': {'values': [0.005]},
            'ent': {'values': [10]},
            'size_std': {'values': [10]},
            'size_num': {'values': [ 0.1]},
            'epochs': {'values': [2]},
            'init_strategy': {'values': ['normal', 'const', 'overall_frequency', 'relative_frequency', 'inverse_relative_frequency', 'domain_frequency', 'range_frequency', 'owl']},
            'threshold': {'values': [0.5]},
        }
    }
    #sweep_id = wandb.sweep(sweep_config, project='RGCNExplainer_AIFB_5757')
    params={
                "size": 0.005,  
                "size_std": 10, #num_neighbors*0.1,#-10,
                "ent": 1,
                "size_num": 0.001,
                "lr": 0.1,
                "epochs": 30,
                "init_strategy": "overall_frequency", #[],
                "threshold": 0.5,
                "experiment": "RGCNExplainer_AIFB_",
            }
    wandb.login()
    wandb.init(project=params['experiment'], config=params)
    
    config = wandb.config
    print(config)
    experiment_name = f'size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}'
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
                _,neighbors,sub_edge = find_n_hop_neighbors(edge_index, n_hops, node_idx)
                num_neighbors = len(sub_edge.t())
                if num_neighbors*0.1 > 10:
                    config.update({'size_std': num_neighbors*0.1}, allow_val_change=True)
  
                #elif num_neighbors*0.1 < 1:
                else:
                    config.update({'size_std': 10}, allow_val_change=True)
                #config.update({'size_std': num_neighbors*0.1}, allow_val_change=True)
                print('config size std:',config['size_std'])
                
                explainer = Explainer(model, data,name,  node_idx, n_hops, prune,config)
                masked_hor, masked_ver = explainer.explain()
                #wandb.agent(sweep_id,  explainer.explain)
                # if not os.path.exists(f'/chk/{name}_chk/{experiment_name}_/masked_adj'):
                #         os.makedirs(f'chk/{name}_chk/{experiment_name}_/masked_adj') 
                # else:
                #     print('directory exists')
                directory = f'chk/{name}_chk/{experiment_name}/masked_adj'

                if not os.path.exists(directory):
                    os.makedirs(directory)
                else:
                    print(f"Directory '{directory}' already exists.")
                torch.save(masked_ver, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_ver{node_idx}')
                torch.save(masked_hor, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_hor{node_idx}') 
                #h = visualize(node_idx, n_hops, data, masked_ver,threshold=config['threshold'], name = name, result_weights=False, low_threshold=False)
                h = selected(masked_ver, threshold=config['threshold'],data=data, low_threshold=False)
                masked_ver,masked_hor = convert_binary(masked_ver, config['threshold']), convert_binary(masked_hor, config['threshold'])
                res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])

                hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
                y_full = model.forward2(hor_graph, ver_graph)
                node_pred_full = y_full[node_idx, :]
                res_full = nn.Softmax(dim=0)(node_pred_full)

                high.append(h)
                h = dict(h)
                info = {'label': target_label, 'node_idx': str(node_idx),'number_neighbors': num_neighbors, 'prediction_explain': str(res.detach().numpy()), 'prediction_full': str(res_full.detach().numpy())}
                h.update(info)
                #predictions =  {'number_neighbors': num_neighbors, 'prediction_explain': str(res.detach().numpy()), 'prediction_full': str(res_full.detach().numpy())}
                #h.update(predictions)
                df.loc[str(node_idx)] = h
                #df.loc[str(node_idx)].update(predictions)




                # h_floats = selected(masked_ver, threshold=0.5,data=data, low_threshold=False,float=True)
                # high_floats.append(h_floats)
                # h_floats = dict(h_floats)
                # h_floats.update(info)
                # df_floats.loc[str(node_idx)] = h_floats
                experiment_name = f'size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}'
                if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
                    os.makedirs(f'Relation_Importance_{name}/{experiment_name}')
                #df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_{name}_{node_idx}.csv', index=False)

                print('node_idx', node_idx, 
                    '\n node original label',[k for k, v in d.items() if node_idx in v],
                    '\n node predicted label explain', torch.argmax(res).item(),
                    '\n node prediction probability explain', res,
                        '\n node predicted label full', torch.argmax(res_full).item(),
                        'most important relations ', h,
                        '\n final masks and lenght', masked_ver, len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>config['threshold'] ]),
                        '\n ---------------------------------------------------------------')
        
        if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
            os.makedirs(f'Relation_Importance_{name}/{experiment_name}')
        df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_all_{name}.csv', index=False) 
        #df_floats.to_csv(f'Relation_Importance_{name}/Relations_Important_all_{name}_{node_idx}_floats.csv', index=False) 
                
    else:
        if name != 'aifb':
            node_idx = d[0][0]
        if train:
            #config.update({'size_std': num_neighbors*0.1}, allow_val_change=True)
            config.update({'size_std': num_neighbors*10}, allow_val_change=True)
            #config.update({'size': num_neighbors*0.01 * 0.005}, allow_val_change=True)
            print('config size std:',config['size_std'])
            print('config size:',config['size'])
            explainer = Explainer(model, data,name,  node_idx, n_hops, prune, config)
            masked_hor, masked_ver = explainer.explain()
            #wandb.agent(sweep_id, explainer.explain)
            if not os.path.exists(f'chk/{name}_chk/masked_adj'):
                    os.makedirs(f'chk/{name}_chk/masked_adj') 
            torch.save(masked_ver, f'chk/{name}_chk/masked_adj/masked_ver{node_idx}_new')
            torch.save(masked_hor, f'chk/{name}_chk/masked_adj/masked_hor{node_idx}_new') 
        else:
            masked_ver = torch.load(f'chk/{name}_chk/masked_adj/masked_ver{node_idx}_new')
            masked_hor = torch.load(f'chk/{name}_chk/masked_adj/masked_hor{node_idx}_new')
        h = visualize(node_idx, n_hops, data, masked_ver,threshold=config['threshold'] , name = name, result_weights=False, low_threshold=False, experiment_name=experiment_name)
        h = selected(masked_ver, threshold=config['threshold'],data=data, low_threshold=False)
        masked_ver,masked_hor = convert_binary(masked_ver, config['threshold']), convert_binary(masked_hor, config['threshold'])
        res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])


        hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        hor_graph, ver_graph = convert_binary(hor_graph, config['threshold']), convert_binary(ver_graph, config['threshold'])
        y_full = model.forward2(hor_graph, ver_graph)
        node_pred_full = y_full[node_idx, :]
        res_full = nn.Softmax(dim=0)(node_pred_full)
        high.append(h)
        h = dict(h)



        target_label = str([k for k, v in d.items() if node_idx in v])
        info = {'label': str(target_label), 'node_idx': str(node_idx)}
        h.update(info)
        
        df.loc[str(node_idx)] = h
        df['number neighbors'] = num_neighbors
        df['prediction explain'] = str(res.detach().numpy())
        df['prediction full'] = str(res_full.detach().numpy())
        

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
            '\n final masks and lenght', masked_ver, len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>config['threshold'] ]))
        experiment_name = f'size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}'
        if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
            os.makedirs(f'Relation_Importance_{name}/{experiment_name}') 
        df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_{name}_{node_idx}.csv', index=False)
        #df_floats.to_csv(f'Relation_Importance_{name}/Relations_Important_{name}_{node_idx}_floats.csv', index=False)


if __name__ == "__main__":

    main('mdgenre',node_idx= 263997, prune= False, explain_all = False, train=True)      
