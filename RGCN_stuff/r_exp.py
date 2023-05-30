from ast import List
from http.client import INSUFFICIENT_STORAGE
from logging import config
from pickle import FALSE, TRUE
from re import A
from colorama import init
from responses import FalseBool
from sqlalchemy import false, true
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
        self.hor_graph, self.ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)

        #hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        self.edge_index_h, self.edge_index_v = self.hor_graph.coalesce().indices(), self.ver_graph.coalesce().indices()
        self.sub_edges, self.neighbors_h, self.sub_edges_tensor_h  = find_n_hop_neighbors(self.edge_index_h, n=self.n_hops, node=self.node_idx)
        self.sub_edges, self.neighbors_v, self.sub_edges_tensor_v  = find_n_hop_neighbors(self.edge_index_v, n=self.n_hops, node=self.node_idx)
        print('shape sub',self.sub_edges_tensor_h.shape, self.sub_edges_tensor_v.shape)


        #self.sub_edges, self.neighbors, self.sub_edges_tensor = find_n_hop_neighbors(self.edge_index, n=self.n_hops, node=self.node_idx)


        self.sub_triples = match_to_triples(self.sub_edges_tensor_v, self.sub_edges_tensor_h,self.data, sparse=False)
        print(Counter(self.sub_triples[:,1].tolist()))

        


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
        neighbors_h, neighbors_v  = list(self.neighbors_h), list(self.neighbors_v)
        print('num_neighbors:' ,len(set(neighbors_v + neighbors_h)))
        sub_hor_graph, sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)
        print('sub_hor_graph ', sub_hor_graph.shape)
        hor_graph, ver_graph = torch.tensor(sub_hor_graph, dtype=torch.float), torch.tensor(sub_ver_graph, dtype=torch.float)
        print(hor_graph.shape, ver_graph.shape)
        label = torch.tensor([self.label[node_idx]], dtype=torch.long)
        pred_label = torch.argmax(self.pred_label[node_idx_new])


        explainer = ExplainModule(
            hor_graph, ver_graph,
            self.model,
            label,
            self.data, 
            self.config
        )



        config = wandb.config

        self.model.eval()
        explainer.train()  # set the explainer model to training mode
        print('start training')
        for epoch in range(config.epochs):
            #config.update({'lr': config['lr']*1/(1+epoch)}, allow_val_change=True)
            explainer.zero_grad()  # zero the gradient buffers of all parameters
            explainer.optimizer.zero_grad()
            ypred, masked_hor, masked_ver = explainer(self.node_idx)  # forward pass of the explainer
            loss, pred_loss, size_loss, mask_ent_loss, size_std_loss,  num_high, wrong_pred = explainer.loss(ypred, config,epoch) 

            m_ver, m_hor = sub(masked_ver, 0.5), sub(masked_hor,0.5)

            m = match_to_triples(m_ver, m_hor, data, node_idx)
            counter = Counter(m[:,1].tolist())

            loss.backward()

            explainer.optimizer.step()
            
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; pred: ",
                ypred,
                "; Counter",
                counter,
            )
            print('--------------------------------------------------------------')
            # wandb.log({"len mask > 0.5": len([i for i in masked_ver.coalesce().values() if i > 0.5]), "loss": loss,
            #            "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss,
            #            "size_num_loss": size_num_loss,"reg_loss": reg_loss, "squae_loss": squared_loss})
            
            wandb.log({f"len mask > {config['threshold']}": len([i for i in masked_ver.coalesce().values() if i > config['threshold']]) , "loss": loss,
            "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss, "size_std_loss": size_std_loss,  "num_high": num_high, "wrong_pred": wrong_pred})
        print('Finished Training')


        #here I can kill some relations
        h_0 ,v_0= select_on_relation_sparse(hor_graph,self.data, 34), select_on_relation_sparse(ver_graph,self.data, 34)
        h_0 ,v_0= select_on_relation_sparse(hor_graph,self.data, 38), select_on_relation_sparse(ver_graph,self.data, 38)
        h_0 ,v_0= select_on_relation_sparse(hor_graph,self.data, 39), select_on_relation_sparse(ver_graph,self.data, 39)
        #h_0 ,v_0= select_on_relation_sparse(hor_graph,self.data, 2), select_on_relation_sparse(ver_graph,self.data, 2)
        #hor_graph, ver_graph = h_0,v_0 

        masked_hor_values = (masked_hor.coalesce().values() * hor_graph.coalesce().values())
        masked_ver_values = (masked_ver.coalesce().values() * ver_graph.coalesce().values())


        masked_hor = torch.sparse.FloatTensor(hor_graph.coalesce().indices(), masked_hor_values, hor_graph.size())
        masked_ver = torch.sparse.FloatTensor(ver_graph.coalesce().indices(), masked_ver_values, ver_graph.size())


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
        _, _, self.type_indices1=select_relation(self.hor_graph,self.data.num_entities,39)
        _, _, self.type_indices2=select_relation(self.ver_graph,self.data.num_entities,39)
        self.type_indices = torch.cat((self.type_indices1, self.type_indices2), 0)
        print('self.type_indices', self.type_indices)

        print('init_strategy:', config["init_strategy"])
        self.init_strategy = config["init_strategy"]
        print('size', self.hor_graph.coalesce().indices().size(), self.ver_graph.coalesce().indices().size())
        num_nodes = self.hor_graph.coalesce().indices().size()[1] #self.hor_graph.size()[0]#self.hor_graph.coalesce().values().shape[0]



        self.mask = self.construct_edge_mask(num_nodes, self.hor_graph,self.data)

        params = [self.mask]
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        self.config = config

        # self.edge_index_h, self.edge_index_v = self.hor_graph.coalesce().indices(), self.ver_graph.coalesce().indices()
        # self.sub_edges_h, self.neighbors_h, self.sub_edges_tensor_h  = find_n_hop_neighbors(self.edge_index_h, n=0, node=self.node_idx)
        # self.sub_edges_v, self.neighbors_v, self.sub_edges_tensor_v  = find_n_hop_neighbors(self.edge_index_v, n= 0, node=self.node_idx)





       
        
        self.optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=0.1)


    
    def construct_edge_mask(self, num_nodes,sparse_tensor,data, const_val=1.0, relation_id = 30):
        """
        Construct edge mask
        """
        init_strategy = self.init_strategy
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
            #nn.init.constant_(mask, const_val) 
            nn.init.uniform_(mask, 0,1) 
            #nn.init.xavier_uniform(mask, gain=nn.init.calculate_gain('relu'))

        elif init_strategy == "zero_out":
            '''initialize the mask with the zero out strategy: we zero out edges belonging to specific relations'''
            # std = nn.init.calculate_gain("relu") * math.sqrt(
            #     2.0 / (num_nodes + num_nodes)
            # )
            # with torch.no_grad():
            #     mask.normal_(1.0, std)
            nn.init.constant_(mask,1) 
            #output_indices, output_values, value_indices=select_relation(sparse_tensor,self.data.num_entities,relation_id)
            # output_indices, output_values, value_indices1=select_relation(sparse_tensor,self.data.num_entities,34)
            # output_indices, output_values, value_indices2=select_relation(sparse_tensor,self.data.num_entities,38)



            output_indices, output_values, value_indices3=select_relation(self.hor_graph,self.data.num_entities,39)
            output_indices, output_values, value_indices1=select_relation(self.ver_graph,self.data.num_entities,39)
            
            value_indices = torch.cat((value_indices1, value_indices3), 0)




            mask.data[[self.type_indices]] = -10

        elif init_strategy == "one_out":
            nn.init.uniform_(mask, 0,1) 
            output_indices, output_values, value_indices=select_relation(sparse_tensor,self.data.num_entities,relation_id)
            mask.data[[value_indices]] = 10
            # output_indices, output_values, value_indices1=select_relation(sparse_tensor,self.data.num_entities,34)
            # output_indices, output_values, value_indices2=select_relation(sparse_tensor,self.data.num_entities,38)
            # output_indices, output_values, value_indices3=select_relation(sparse_tensor,self.data.num_entities,39)
            # value_indices = torch.cat((value_indices1, value_indices2, value_indices3), 0)
            # mask.data[[value_indices]] = 0


        

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
        #masked_adj = adj * self.mask

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
        #masked_adj = adj * self.mask

        result = torch.sparse.FloatTensor(indices=self.hor_graph.coalesce().indices(), values= masked_adj, size=self.hor_graph.coalesce().size())
        return result

    

    def forward(self, node_idx):
        self.masked_ver = self._masked_adj_ver()  # masked adj is the adj matrix with the mask applied
        self.masked_hor = self._masked_adj_hor()  # masked adj is the adj matrix with the mask applied
        masked_ver, masked_hor = self.masked_ver, self.masked_hor
        masked_ver,masked_hor = convert_binary(self.masked_ver, self.config["threshold"]), convert_binary(self.masked_hor, self.config["threshold"])


        ypred = self.model.forward2(masked_hor, masked_ver)
        #ypred = self.model.forward3(masked_hor)
        #ypred = self.model.forward2(self.masked_hor, self.masked_ver)
        node_pred = ypred[node_idx,:]
        res = nn.Softmax(dim=0)(node_pred[0:])
        
        return res,   self.masked_hor, self.masked_ver



    def get_frequency_relations(self,v):
        _ ,p = torch.div(v.coalesce().indices(), v.size()[0], rounding_mode='floor')
        return dict(Counter(p))

    # def loss(self, pred, config, epoch):
    #     """
    #     Args:
    #         pred: prediction made by current model
    #         pred_label: the label predicted by the original model.
    #     """
    #     #selected_mask, num_high, p,relation_counter = subset_sparse(self.masked_hor, self.data, self.config["threshold"])
    #     #print('relation_counter', relation_counter)
    #     #self.mask = ((self.mask-0.5)**2)


    #     mask = torch.sigmoid(self.mask)  # sigmoid of the mask
   
    #     mask_without_small = mask[mask > config["threshold"]]
    #     print('mask_without_small', torch.mean(mask_without_small))#, torch.max(mask_without_small,0), torch.min(mask_without_small,0))
    #     num_high = len(mask_without_small)
    #     print('num_high', num_high,'len(mask)', len(mask))
    #     print('mask', torch.mean(mask), torch.std(mask))


    #     # prediction loss

        
    #     gt_label_node = self.label#[node_idx]te
    #     logit = pred[gt_label_node]
    #     # pred_loss = config["pred"]* -torch.log(logit) 
    #     if torch.argmax(pred) != gt_label_node:
    #         pred_loss = 5 * -torch.log(logit) 
    #         size_loss, mask_ent_loss,size_loss_std, size_num_loss, num_high, type_loss = 0,0,0,0,0,0


    #     else:
    #         adaptive_pred_coeff = config["pred"] * (epoch+1)*0.1
    #         print('adaptive_pred_coeff', adaptive_pred_coeff)
    #         #pred_loss = config["pred"]* -torch.log(logit)
    #         pred_loss = adaptive_pred_coeff* -torch.log(logit)


    #         size_num_loss = config["size_std"]* (num_high+3)/len(mask)


    #         adaptive_size_coeff = config["size"] * 1/(epoch+1)
    #         print('adaptive_size_coeff', adaptive_size_coeff)
    #         #size_loss = config["size"] * torch.sum(torch.abs(mask))
    #         size_loss = adaptive_size_coeff * torch.sum(torch.abs(mask))


    #         adaptive_size_std_coeff = config["size_std"] * (epoch+1)*0.01
    #         print('adaptive_size_std_coeff', adaptive_size_std_coeff)
    #         size_loss_std = -config["size_std"] * torch.std(mask_without_small) 
            

    #         #type loss
    #         print('types',len(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5]))
    #         type_len = len(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5])
    #         #type_loss = -0.01 *len(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5])
    #         adaptive_type_coeff =  0.1 * 1/(epoch+1)
    #         type_loss = adaptive_type_coeff *torch.sum(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5])



    #         mask = mask/ torch.sum(mask)
            

    #         # entropy edge mask 
    #         mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    #         #mask_ent = - torch.log(mask)
    #         adaptive_ent_coeff =  config["ent"] * 1/(epoch+1)
    #         #mask_ent_loss =  config["ent"] * torch.mean(mask_ent)
    #         mask_ent_loss =  adaptive_ent_coeff * torch.mean(mask_ent)







    #         #loss = torch.exp(pred_loss + mask_ent_loss + size_num_loss + size_loss_std + size_loss ) #
    #     loss = torch.exp(pred_loss + size_loss + mask_ent_loss + size_loss_std + type_loss)
    #     #loss = pred_loss + size_loss + mask_ent_loss + size_loss_std
    #     #loss = pred_loss + mask_ent_loss
    #     print('pred_loss', pred_loss)
    #     print('size_loss', size_loss)
    #     print('type_loss', type_loss)
    #     #
    #     #print('size_num_loss', size_num_loss)
    #     print('mask_ent_loss', mask_ent_loss)

    #     print('size_loss_std', size_loss_std)

    #     return loss, pred_loss, size_loss, mask_ent_loss,size_loss_std, size_num_loss, num_high


     
    def loss(self, pred, config, epoch, adaptive=True):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #selected_mask, num_high, p,relation_counter = subset_sparse(self.masked_hor, self.data, self.config["threshold"])
        #print('relation_counter', relation_counter)
        #self.mask = ((self.mask-0.5)**2)

        mask = torch.sigmoid(self.mask)  # sigmoid of the mask
        mask_without_small = mask[mask > config["threshold"]]
        print('mask_without_small', torch.mean(mask_without_small))#, torch.max(mask_without_small,0), torch.min(mask_without_small,0))
        num_high = len(mask_without_small)
        print('num_high', num_high,'len(mask)', len(mask))
        print('mask', torch.mean(mask), torch.std(mask))

        size_std = config['size_std']
        size = config['size']
        ent = config['ent']
        pred_coeff = config['pred']
        e = (epoch + 1)*0.1
        if adaptive:
            config.update({'size_std': size_std*e}, allow_val_change=True)
            config.update({'size': size*e}, allow_val_change=True)
            config.update({'ent': ent*1/(epoch+1)}, allow_val_change=True)
            config.update({'pred': pred_coeff*e}, allow_val_change=True)
            print('adaptive', config['size_std'], config['size'], config['ent'], config['pred'])
   
        # prediction loss

        
        gt_label_node = self.label#[node_idx]te
        logit = pred[gt_label_node]

        pred_loss = config["pred"]* -torch.log(logit)




        size_loss = config['size'] * torch.sum(torch.abs(mask))



        size_loss_std = -config["size_std"] * torch.std(mask) 
            




        #mask = mask/ torch.sum(mask)
        

        # entropy edge mask 
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)

        mask_ent_loss =  config["ent"] * torch.mean(mask_ent)




        #type loss
        
        type_len = len(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5])
        print('types',type_len)
        #type_loss = config['type'] * type_len/len(mask)
        type_loss = - config['type']*torch.sum(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5])/torch.sum(mask)



        # adaptive_type_coeff =  0.1 * 1/(epoch+1)
        # type_loss = adaptive_type_coeff *torch.sum(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5])


        wrong_pred = 10 if torch.argmax(pred) != gt_label_node else 0

        

        loss = torch.exp(pred_loss + size_loss + mask_ent_loss + size_loss_std + wrong_pred + type_loss)
        #loss = pred_loss + size_loss + mask_ent_loss + size_loss_std + wrong_pred + type_loss

        print('pred_loss', pred_loss)
        print('size_loss', size_loss)

        print('type_loss', type_loss)
        #print('size_num_loss', size_num_loss)
        print('mask_ent_loss', mask_ent_loss)
        print('wrong_pred', wrong_pred)

        print('size_loss_std', size_loss_std)
        print(torch.argmax(pred), gt_label_node)
        

        return loss, pred_loss, size_loss, mask_ent_loss,size_loss_std, num_high , wrong_pred 

    def per_relation_loss(self, pred):
        _ ,p = torch.div(self.masked_hor.coalesce().indices(), self.masked_hor.size[0], rounding_mode='floor')
        print(p)









import wandb



def main1(n_hops, node_idx, model, data,name,  prune, config, num_neighbors):
    experiment_name = f'hops_{n_hops}_size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}_exp_{config["try"]}'
    print('experiment_name', experiment_name)


    explainer = Explainer(model, data,name,  node_idx, n_hops, prune, config)
    masked_hor, masked_ver = explainer.explain()



    #wandb.agent(sweep_id, explainer.explain)
    directory = f'chk/{name}_chk/{experiment_name}/masked_adj'

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print(f"Directory '{directory}' already exists.")
    torch.save(masked_ver, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_ver{node_idx}')
    torch.save(masked_hor, f'chk/{name}_chk/{experiment_name}/masked_adj/masked_hor{node_idx}') 

    #h = visualize(node_idx, n_hops, data, masked_ver,masked_hor, threshold=config['threshold'] , name = name, result_weights=False, low_threshold=False, experiment_name=experiment_name)
    h = visualize(node_idx, n_hops, data, masked_ver, threshold=config['threshold'] , name = name, result_weights=False, low_threshold=False, experiment_name=experiment_name)

    #h = selected(masked_ver,masked_hor,  threshold=config['threshold'],data=data, low_threshold=False)
    res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
    #res = nn.Softmax(dim=0)(model.forward3(masked_hor)[node_idx, :])

    m_ver,m_hor = convert_binary(masked_ver,0), convert_binary(masked_hor, 0)
    res_sub = nn.Softmax(dim=0)(model.forward2(m_hor, m_ver)[node_idx, :])

    masked_ver,masked_hor = convert_binary(masked_ver, config['threshold']), convert_binary(masked_hor, config['threshold'])

    #masked_ver,masked_hor = convert_binary(masked_ver,0), convert_binary(masked_hor, 0)
    res_binary = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
    #res_binary = nn.Softmax(dim=0)(model.forward3(masked_hor)[node_idx, :])


    masked_ver, masked_hor = sub(masked_ver, 0.5), sub(masked_hor,0.5)
    m = match_to_triples(masked_ver, masked_hor, data, node_idx)
    counter = dict(Counter(m[:,1].tolist()))
    # counter = {k: [data.i2rel[i][0] for i in v] for k,v in counter.items()}


    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    #hor_graph, ver_graph = convert_binary(hor_graph, config['threshold']), convert_binary(ver_graph, config['threshold'])
    y_full = model.forward2(hor_graph, ver_graph)
    #y_full = model.forward3(hor_graph)
    node_pred_full = y_full[node_idx, :]
    res_full = nn.Softmax(dim=0)(node_pred_full)









    h = dict(h)
    print('Important relations', h)



    target_label = str([k for k, v in d.items() if node_idx in v])
    info = {'label': str(target_label), 'node_idx': str(node_idx), 'number_neighbors': str(num_neighbors), 'prediction_explain_binary': str(res_binary.detach().numpy()), 'prediction_full': str(res_full.detach().numpy()), 'prediction_explain': str(res.detach().numpy())}
    h.update(info)
    df.loc[str(node_idx)] = h



    print('node_idx', node_idx, 
        '\n node original label',target_label,
        '\n node predicted label explain', torch.argmax(res).item(),
        '\n node prediction probability explain', res, 'explain binary', torch.argmax(res_binary).item(),
        '\n node prediction probability explain binary', res_binary,
        '\n node predicted label full', torch.argmax(res_full).item(),
        '\n node prediction probability full', res_full,
        '\n node predicted label sub', torch.argmax(res_sub).item(),
        '\n node prediction probability sub', res_sub,
        '\n final masks and lenght', masked_ver, len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>config['threshold'] ]))
    #experiment_name = f'size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}'
    if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
        os.makedirs(f'Relation_Importance_{name}/{experiment_name}') 
    df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_{name}_{node_idx}.csv', index=False)
    return h, experiment_name




if __name__ == "__main__":

    explain_all = True
    node_idx = 5678
    prune = True
    name = 'aifb'
    if explain_all == True:
        exp = 'all'
    else:
        exp = node_idx
    n_hops = 0 if prune else 2
    n_hops = 2


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
    relations = [data.i2rel[i][0] for i in range(len(data.i2rel))]
        
    relations = ['label', 'node_idx','number_neighbors', 'prediction_explain', 'prediction_full', 'prediction_explain_binary'] + relations
    df = pd.DataFrame(columns=relations)
    d = d_classes(data)
        
    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    edge_index_h, edge_index_v = hor_graph.coalesce().indices(), ver_graph.coalesce().indices()


    #model 
    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    if exp == 'all':
        for target_label in range(len(d.keys())):
            for node_idx in d[target_label]:
                sub_edges, neighbors_h, sub_edges_tensor_h  = find_n_hop_neighbors(edge_index_h, n_hops, node_idx)
                sub_edges, neighbors_v, sub_edges_tensor_v  = find_n_hop_neighbors(edge_index_v, n_hops, node_idx)
                num_neighbors = len(list(neighbors_h) + list(neighbors_v))
                print(f'num_neighbors {num_neighbors} for node {node_idx}')

                params={
                "pred": 1,
                "size":  0.0005,#0.005,#0.005, #0.005,  
                "size_std": num_neighbors*0.1,#-10,
                "ent": 1,
                "type": 1,
                "lr": 0.1,
                "epochs": 30,
                "init_strategy": "normal", #[],
                "threshold": 0.5,
                "experiment": f"RGCNExplainer_AIFB_{exp}",
                "hops": n_hops,
                "try": 'adaptive_new',
            }
                wandb.init(project= f"RGCNExplainer_AIFB_{node_idx}",config=params, reinit=True)
                config = wandb.config
                #experiment_name = f'hops_{n_hops}_size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}_exp_{config["try"]}'
                


                
                h, experiment_name = main1(n_hops, node_idx, model, data,name,  prune, config, num_neighbors)
                
                #df = df.append(h, ignore_index=True)
                df.loc[str(node_idx)] = h

                wandb.finish()
        if not os.path.exists(f'Relation_Importance_{name}/{experiment_name}'):
            os.makedirs(f'Relation_Importance_{name}/{experiment_name}') 
        df.to_csv(f'Relation_Importance_{name}/{experiment_name}/Relations_Important_{name}_full.csv', index=False)
    else:

        sub_edges, neighbors_h, sub_edges_tensor_h  = find_n_hop_neighbors(edge_index_h, n_hops, node_idx)
        sub_edges, neighbors_v, sub_edges_tensor_v  = find_n_hop_neighbors(edge_index_v, n_hops, node_idx)
        num_neighbors = len(list(neighbors_h) + list(neighbors_v))

        params={
        "pred": 1,
        "size":  0.0005,#0.005, #0.005,  
        "size_std": num_neighbors*0.1,#-10,
        "ent": 1,
        "type": 1,
        "lr": 0.1,
        "epochs": 30,
        "init_strategy": "normal", #[],
        "threshold": 0.5,
        "experiment": f"RGCNExplainer_AIFB_{exp}_playground",
        "hops": n_hops,
        "try": 'adaptive_',
    }
        wandb.init(project= params["experiment"],config=params, reinit=True)
        config = wandb.config
        


        
        h, experiment_name = main1(n_hops, node_idx, model, data,name,  prune, config, num_neighbors)
        
        #df = df.append(h, ignore_index=True)
        df.loc[str(node_idx)] = h

        wandb.finish()


