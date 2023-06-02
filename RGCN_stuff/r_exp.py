
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
    """ 
    Explainer class for RGCN - node classification, inspired by GNNExplainer (Ying et al., 2019)
    :param model: RGCN node classification model whose predictions we want to explain
    :param data: relational graph dataset (in the format of kgbenchmark)
    :param name: name of the dataset
    :param node_idx: index of the node we want to explain the prediction of
    :param n_hops: number of hops for the subgraph (which corresponds to the number of RGCN layers)
    :param prune: whether to prune the graph or not
    :param config: configuration dictionary that contains the hyperparameters for the training (loss coefficients and lr)


    :function explain: main method that explains the prediction of the node with index node_idx over a training loop

    :return masked_hor, masked_ver: masked adjacency matrices of the subgraph

    """
    def __init__(self,
                 model,
                 pred_label,
                 data,
                 name,
                 node_idx,
                 n_hops,
                 prune,
                 config,):
        self.model = model
        self.pred_label = pred_label 
        self.data = data
        self.name = name
        self.n = data.num_entities
        self.r = data.num_relations
        self.triples = data.triples
        self.node_idx = node_idx
        self.n_hops = n_hops
        self.config = config
        print(self.node_idx)
        self.label = data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([self.node_idx])),1]
        print(self.node_idx, self.label)


        #get hor and ver graph and the subgraphs
        self.hor_graph, self.ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        self.edge_index_h, self.edge_index_v = self.hor_graph.coalesce().indices(), self.ver_graph.coalesce().indices()

        self.sub_edges_h, self.neighbors_h, self.sub_edges_tensor_h  = find_n_hop_neighbors(self.edge_index_h, n=self.n_hops, node=self.node_idx)
        self.sub_edges_v, self.neighbors_v, self.sub_edges_tensor_v  = find_n_hop_neighbors(self.edge_index_v, n=self.n_hops, node=self.node_idx)
        self.sub_triples = match_to_triples(self.sub_edges_tensor_v, self.sub_edges_tensor_h,self.data, sparse=False)
        self.overall_rel_frequency = dict(Counter(self.data.triples[:,1].tolist()))
        self.num_neighbors = len(set(list(self.neighbors_h) + list(self.neighbors_v)))
        self.num_edges = len(self.sub_edges_v)+len(self.sub_edges_h)

        


    def explain(self):
        ''' Explain the prediction of the node with index node_idx: main method
            :return masked_hor, masked_ver: masked adjacency matrices of the subgraph'''

        print(f"node {self.node_idx} label:", self.label)
        print('num_neighbors:' ,self.num_neighbors)
        print('num_edges:', self.num_edges)

        sub_hor_graph, sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)

        explainer = ExplainModule(
            sub_hor_graph, sub_ver_graph,
            self.model,
            self.label,
            self.data, 
            self.config,
            self.num_edges,
            self.node_idx
        )


        self.model.eval()
        explainer.train()  
        print('start training')
        for epoch in range(self.config.epochs):
            #self.config.update({'lr': self.config['lr']*1/(1+epoch)}, allow_val_change=True)
            explainer.zero_grad()  # zero the gradient buffers of all parameters
            explainer.optimizer.zero_grad()
            ypred, masked_hor, masked_ver = explainer(self.node_idx)  # forward pass of the explainer
            loss, pred_loss, size_loss, mask_ent_loss, size_std_loss,  num_high, wrong_pred = explainer.loss(ypred, self.config,epoch) 

            m_ver, m_hor = sub(masked_ver, 0.5), sub(masked_hor,0.5)

            m = match_to_triples(m_ver, m_hor, data, node_idx)
            counter = Counter(m[:,1].tolist())
            counter = {data.i2rel[k][0]:v for k,v in counter.items() if k!=0}

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
            
            wandb.log({f"len mask > {self.config['threshold']}": len([i for i in masked_ver.coalesce().values() if i > self.config['threshold']]) , "loss": loss,
            "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss, "size_std_loss": size_std_loss,  "num_high": num_high, "wrong_pred": wrong_pred})
        print('Finished Training')


        if self.config['kill_type']:
            h_0 ,v_0= select_on_relation_sparse(sub_hor_graph,self.data, 34), select_on_relation_sparse(sub_ver_graph,self.data, 34)
            h_0 ,v_0= select_on_relation_sparse(sub_hor_graph,self.data, 38), select_on_relation_sparse(sub_ver_graph,self.data, 38)
            h_0 ,v_0= select_on_relation_sparse(sub_hor_graph,self.data, 39), select_on_relation_sparse(sub_ver_graph,self.data, 39)
            h_0 ,v_0= select_on_relation_sparse(hor_graph,self.data, 2), select_on_relation_sparse(ver_graph,self.data, 2)
            hor_graph, ver_graph = h_0,v_0 

        # if add_zero_hops:
        #     edge_index = edge_index_oneadj(self.data.triples)
        #     sub_edges, _, sub_edges_tensor = find_n_hop_neighbors(edge_index, n=0, node=self.node_idx)
        #     masked_hor = 

        masked_hor_values = (masked_hor.coalesce().values() * sub_hor_graph.coalesce().values())
        masked_ver_values = (masked_ver.coalesce().values() * sub_ver_graph.coalesce().values())


        masked_hor = torch.sparse.FloatTensor(sub_hor_graph.coalesce().indices(), masked_hor_values, sub_hor_graph.size())
        masked_ver = torch.sparse.FloatTensor(sub_ver_graph.coalesce().indices(), masked_ver_values, sub_ver_graph.size())


        return masked_hor, masked_ver

class ExplainModule(nn.Module):
    def __init__(
            self,
            hor_graph,
            ver_graph,
            model,
            label, 
            data,
            config,
            num_edges,
            node_idx):
        super(ExplainModule, self).__init__()
        self.hor_graph, self.ver_graph = hor_graph, ver_graph 
        self.model = model
        self.label = label
        self.data = data
        self.config = config
        self.node_idx = node_idx

        type = [i for i in range(data.num_relations) if 'type' in data.i2r[i]]
        tensor_list = []
        for i in type:
            _,_,value_indices_h=select_relation(self.hor_graph,self.data.num_entities,i)
            tensor_list.append(value_indices_h)
        self.type_indices = torch.cat(tensor_list, 0)


        print('init_strategy:', config["init_strategy"])
        num_nodes = num_edges
        self.mask = self.construct_edge_mask(num_nodes, self.hor_graph,self.data)
        params = [self.mask]
        self.optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])





    
    def construct_edge_mask(self, num_nodes,sparse_tensor,data, const_val=10):
        """
        Construct edge mask
        """
        init_strategy = self.config["init_strategy"]
        data = self.data
        num_entities = data.num_entities
        torch.manual_seed(42)
        mask = nn.Parameter(torch.FloatTensor(num_nodes))
        relation_id = self.config["relation_id"]


        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)

        elif init_strategy == "const":
            nn.init.constant_(mask, const_val) 
            #nn.init.uniform_(mask, 0,1) 



        elif init_strategy == "zero_out":
            '''initialize the mask with the zero out strategy: we zero out edges belonging to specific relations'''
            # std = nn.init.calculate_gain("relu") * math.sqrt(
            #     2.0 / (num_nodes + num_nodes)
            # )
            # with torch.no_grad():
            #     mask.normal_(1.0, std)
            nn.init.constant_(mask,1) 
            _, _, value_indices3=select_relation(self.hor_graph,self.data.num_entities,relation_id)
            _, _, value_indices1=select_relation(self.ver_graph,self.data.num_entities,relation_id)
            
            value_indices = torch.cat((value_indices1, value_indices3), 0)
            mask.data[[self.type_indices]] = -10

        elif init_strategy == "one_out":
            #nn.init.uniform_(mask, 0,1) 
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
            _, _, value_indices=select_relation(sparse_tensor,self.data.num_entities,relation_id)
            mask.data[[value_indices]] = 10  

        elif init_strategy == "overall_frequency":
            '''Initialize the mask with the overall frequency of the relations'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            overall_rel_frequency = dict(Counter(data.triples[:,1].tolist()))

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
            ''' Initialize proportionally to the domain frequency of the relations'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            dict_domain, dict_range = domain_range_freq(data, len(d_classes(data)))
            for i in p:

                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = dict_domain[i]

        elif init_strategy == "range_frequency":
            ''' Initialize proportionally to the range frequency of the relations'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            dict_domain, dict_range = domain_range_freq(data, len(d_classes(data)))
            for i in p:
                    _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                    mask.data[[value_indices]] = dict_range[i]

        elif init_strategy == "rdf":
            ''' Initialization that zeroes out the rdf relations'''
            rdf = [i for i in range(data.num_relations) if 'rdf' in data.i2r[i]]
            for i in rdf:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = 0

        elif init_strategy == "owl":
            ''' Initialization that zeroes out the owl relations'''
            owl = [i for i in range(data.num_relations) if 'owl' in data.i2r[i]]
            for i in owl:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = 0

        elif init_strategy == "type":
            ''' Initialization that zeroes out the type relations'''
            type = [i for i in range(data.num_relations) if 'type' in data.i2r[i]]
            for i in type:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = 0

        print(f'mask initialized with {init_strategy} strategy: {mask}')   
        return mask
    

    def _masked_adj_ver(self):
        "" "Mask the adjacency matrix with the learned mask" ""

        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2

        adj = torch.Tensor(self.ver_graph.coalesce().values())
        masked_adj = adj * sym_mask 

        result = torch.sparse.FloatTensor(indices=self.ver_graph.coalesce().indices(), values= masked_adj, size=self.ver_graph.coalesce().size())
        
        return result   

    def _masked_adj_hor(self):
        "" "Mask the adjacency matrix with the learned mask" ""
        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = torch.Tensor(self.hor_graph.coalesce().values())
        masked_adj = adj * sym_mask 


        result = torch.sparse.FloatTensor(indices=self.hor_graph.coalesce().indices(), values= masked_adj, size=self.hor_graph.coalesce().size())
        return result

    

    def forward(self, node_idx):
        self.masked_ver = self._masked_adj_ver()  # masked adj is the adj matrix with the mask applied
        self.masked_hor = self._masked_adj_hor()  # masked adj is the adj matrix with the mask applied
        masked_ver,masked_hor = convert_binary(self.masked_ver, self.config["threshold"]), convert_binary(self.masked_hor, self.config["threshold"])


        ypred = self.model.forward2(masked_hor, masked_ver)
        node_pred = ypred[node_idx,:]
        res = nn.Softmax(dim=0)(node_pred[0:])
        
        return res, self.masked_hor, self.masked_ver



    def get_frequency_relations(self,v):
        _ ,p = torch.div(v.coalesce().indices(), v.size()[0], rounding_mode='floor')
        return dict(Counter(p))




     
    def loss(self, pred, config, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        adaptive = config['adaptive']
        mask = torch.sigmoid(self.mask)  # sigmoid of the mask
        mask_without_small = mask[mask > config["threshold"]]
        print('mask', torch.mean(mask), torch.std(mask))
        print('mask_without_small', torch.mean(mask_without_small))#, torch.max(mask_without_small,0), torch.min(mask_without_small,0))
        num_high = len(mask_without_small)
        print('num_high', num_high,'len(mask)', len(mask))
       

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

        logit = pred[self.label]

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
        
        type_loss = - config['type']*torch.sum(mask[[self.type_indices]][mask[[self.type_indices]] > 0.5])/torch.sum(mask)
        type_loss = config['type'] * type_len/len(mask)

        wrong_pred = 10 if torch.argmax(pred) != self.label else 0

        

        loss = torch.exp(pred_loss + size_loss + mask_ent_loss + size_loss_std +  type_loss)
        #loss = pred_loss + size_loss + mask_ent_loss + size_loss_std + wrong_pred + type_loss

        print('pred_loss', pred_loss)
        print('size_loss', size_loss)
        print('type_loss', type_loss)
        print('mask_ent_loss', mask_ent_loss)
        print('wrong_pred', wrong_pred)
        print('size_loss_std', size_loss_std)
        print('pred',torch.argmax(pred), 'original label',self.label)
        

        return loss, pred_loss, size_loss, mask_ent_loss,size_loss_std, num_high , wrong_pred 
    
    def loss(self, pred, config, epoch):

        mask = torch.sigmoid(self.mask)  # sigmoid of the mask
        logit = pred[self.label]

        pred_loss =  -torch.log(logit)


        size_loss = 0.005 * torch.sum(mask)
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)

        mask_ent_loss =  1 * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss, pred_loss, size_loss, mask_ent_loss, 0, 0, 0


    def per_relation_loss(self, pred):
        _ ,p = torch.div(self.masked_hor.coalesce().indices(), self.masked_hor.size[0], rounding_mode='floor')
        print(p)













def main1(n_hops, node_idx, model,pred_label, data,name,  prune, config, num_neighbors):

    experiment_name = f'hops_{n_hops}_size_{config["size"]}_lr_{config["lr"]}_epochs_{config["epochs"]}_threshold_{config["threshold"]}_init_{config["init_strategy"]}_exp_{config["try"]}'
    print('experiment_name', experiment_name)
    directory = f'chk/{name}_chk/{experiment_name}'


    explainer = Explainer(model,pred_label, data,name,  node_idx, n_hops, prune, config)
    masked_hor, masked_ver = explainer.explain()



    

    if not os.path.exists(directory + f'/masked_adj'):
        os.makedirs(directory + f'/masked_adj')
    else:
        print(f"Directory '{directory}' already exists.")
    torch.save(masked_ver, f'{directory}/masked_adj/masked_ver{node_idx}')
    torch.save(masked_hor, f'{directory}/masked_adj/masked_hor{node_idx}') 

    #h = visualize(node_idx, n_hops, data, masked_ver,masked_hor, threshold=config['threshold'] , name = name, result_weights=False, low_threshold=False, experiment_name=experiment_name)
    #h = visualize(node_idx, n_hops, data, masked_ver, threshold=config['threshold'] , name = name, result_weights=False, low_threshold=False, experiment_name=experiment_name)

    #Explain prediction
    res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])

    #Niehborhood subgraph prediction
    m_ver,m_hor = convert_binary(masked_ver,0), convert_binary(masked_hor, 0)
    res_sub = nn.Softmax(dim=0)(model.forward2(m_hor, m_ver)[node_idx, :])

    #Explain prediction binary
    masked_ver,masked_hor = convert_binary(masked_ver, config['threshold']), convert_binary(masked_hor, config['threshold'])
    res_binary = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])


    #Prediction full graph - the one to be explained
    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    res_full = nn.Softmax(dim=0)(model.forward2(hor_graph, ver_graph)[node_idx, :])

    #Important relations - mask > threshold
    masked_ver, masked_hor = sub(masked_ver, config["threshold"]), sub(masked_hor,config["threshold"])
    m = match_to_triples(masked_ver, masked_hor, data, node_idx)
    counter = dict(Counter(m[:,1].tolist()))
    counter = {data.i2rel[k][0]:v for k,v in counter.items() if k!=0}
    print('Important relations', counter)

    #Save in the csv: label, node, number neighbors, predictions
    target_label = str([k for k, v in d.items() if node_idx in v])
    info = {'label': str(target_label), 'node_idx': str(node_idx), 'number_neighbors': str(num_neighbors), 'prediction_explain_binary': str(res_binary.detach().numpy()), 'prediction_full': str(res_full.detach().numpy()), 'prediction_explain': str(res.detach().numpy())}
    counter.update(info)
    df.loc[str(node_idx)] = counter



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

    # if not os.path.exists(directory + f'/Relation_Importance'):
    #     os.makedirs(directory + f'/Relation_Importance')
    # df.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_{node_idx}.csv', index=False)
    return counter, experiment_name




if __name__ == "__main__":

    explain_all = True
    node_idx = 7185

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
        data.triples = torch.Tensor(data.triples).to(int)
        data.withheld = torch.Tensor(data.withheld).to(int)
        data.training = torch.Tensor(data.training).to(int)

          
    print(f'Number of entities: {data.num_entities}') #data.i2e
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') #data.i2r
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    get_relations(data)
    relations = [data.i2rel[i][0] for i in range(len(data.i2rel))]
        
    relations = ['label', 'node_idx','number_neighbors', 'prediction_explain', 'prediction_full', 'prediction_explain_binary'] + relations
    df = pd.DataFrame(columns=relations)
    d = d_classes(data)
    # if name != 'aifb':
    #     node_idx = d[list(d.keys())[0]][0]
        
    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    edge_index_h, edge_index_v = hor_graph.coalesce().indices(), ver_graph.coalesce().indices()


    #model 
    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    pred_label = torch.load(f'chk/{name}_chk/prediction_{name}_prune_{prune}')
    if exp == 'all':
        for target_label in range(len(d.keys())):
            for node_idx in d[target_label]:
                sub_edges, neighbors_h, sub_edges_tensor_h  = find_n_hop_neighbors(edge_index_h, n_hops, node_idx)
                sub_edges, neighbors_v, sub_edges_tensor_v  = find_n_hop_neighbors(edge_index_v, n_hops, node_idx)
                num_neighbors = len(list(neighbors_h) + list(neighbors_v))
                print(f'num_neighbors {num_neighbors} for node {node_idx}')

                params={
                "pred": 1,
                "size":  0.00005,#0.005, #0.005,  
                "size_std": num_neighbors,#-10,
                "ent": -1,
                "type": 1,
                "lr": 0.001,
                "weight_decay": 0.005,
                "adaptive": False,
                "epochs": 30,
                "init_strategy": "normal", #[],
                "threshold": 0.5,
                "experiment": f"RGCNExplainer_{name}_{exp}",
                "hops": n_hops,
                "try": 'GNN_explainer_vibez',
                "kill_type": False,
                "relation_id": 39 #relation for mask initialization
            }
                wandb.init(project= f"RGCNExplainer_{name}_{node_idx}",config=params, reinit=True)
                config = wandb.config
                
                counter, experiment_name = main1(n_hops, node_idx, model,pred_label, data,name,  prune, config, num_neighbors)
                directory = f'chk/{name}_chk/{experiment_name}'
                df.loc[str(node_idx)] = counter

                wandb.finish()

        if not os.path.exists(directory + f'/Relation_Importance'):
            os.makedirs(directory + f'/Relation_Importance')
        df.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_full.csv', index=False)
    else:

        sub_edges, neighbors_h, sub_edges_tensor_h  = find_n_hop_neighbors(edge_index_h, n_hops, node_idx)
        sub_edges, neighbors_v, sub_edges_tensor_v  = find_n_hop_neighbors(edge_index_v, n_hops, node_idx)
        num_neighbors = len(list(neighbors_h) + list(neighbors_v))

    #     params={
    #     "pred": 10,#10
    #     "size": 0.00005,#0.005, #0.005,  
    #     "size_std":  num_neighbors, #*0.1,#-10,
    #     "ent": -1,#10
    #     "type": 1,#10
    #     "lr": 0.1,
    #     "weight_decay": 0.9,
    #     "adaptive": False,
    #     "epochs": 30,
    #     "init_strategy": "normal", #[],
    #     "threshold": 0.5,
    #     "experiment": f"RGCNExplainer_AIFB_{exp}_playground",
    #     "hops": n_hops,
    #     "try": 'adaptive_',
    #     "kill_type": False,
    #     "relation_id": 39 #relation for mask initialization
    # }
        params={
        "pred": 1,#10
        "size": 0.005,#0.005, #0.005,  
        "size_std":  num_neighbors, #*0.1,#-10,
        "ent": -1,#10
        "type": 1,#10
        "lr": 0.1,
        "weight_decay": 0.9,
        "adaptive": False,
        "epochs": 30,
        "init_strategy": "normal", #[],
        "threshold": 0.5,
        "experiment": f"RGCNExplainer_{name}_{exp}_playground",
        "hops": n_hops,
        "try": '',
        "kill_type": False,
        "relation_id": 39 #relation for mask initialization
        }
        wandb.init(project= params["experiment"],config=params, reinit=True)
        config = wandb.config
        
        counter, experiment_name = main1(n_hops, node_idx, model, pred_label, data,name,  prune, config, num_neighbors)
        
        df.loc[str(node_idx)] = counter

        wandb.finish()


