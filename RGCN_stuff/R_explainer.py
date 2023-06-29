from numpy import True_
from sympy import true
import torch 
from collections import Counter
from baseline import baseline_pred
import torch.nn as nn
import math
import pandas as pd

#rgcn 
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
        self.label = data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([self.node_idx])),1]
        self.label = self.label.long()

        #get hor and ver graph and the subgraphs
        self.hor_graph, self.ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
        self.edge_index_h, self.edge_index_v = self.hor_graph.coalesce().indices(), self.ver_graph.coalesce().indices()
        self.res_full = nn.Softmax(dim=0)(model.forward2(self.hor_graph, self.ver_graph)[node_idx, :])
        self.sub_edges_h, self.neighbors_h, self.sub_edges_tensor_h  = find_n_hop_neighbors(self.edge_index_h, n=self.n_hops, node=self.node_idx)
        self.sub_edges_v, self.neighbors_v, self.sub_edges_tensor_v  = find_n_hop_neighbors(self.edge_index_v, n=self.n_hops, node=self.node_idx)
        self.sub_triples = match_to_triples(self.sub_edges_tensor_v, self.sub_edges_tensor_h,self.data, sparse=False)
        self.overall_rel_frequency = dict(Counter(self.data.triples[:,1].tolist()))
        self.num_neighbors = len(set(list(self.neighbors_h) + list(self.neighbors_v)))
        self.num_edges = len(self.sub_edges_v)+len(self.sub_edges_h)
        self.baseline = baseline_pred(self.data, self.model, self.node_idx)
        #self. = [i for i in range(data.num_relations) if 'type' in data.i2r[i]]


    
    def most_freq_rel_f(self, sub_hor_graph, sub_ver_graph):
        sub_h, sub_v = sub_hor_graph, sub_ver_graph
        m = match_to_triples(sub_v, sub_h,self.data, self.node_idx)
        freq = Counter(m[:,1].tolist())
        sorted_freq = {self.data.i2r[k]: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True) if k!=0}
        most_freq_rel = list(sorted_freq.keys())[0]
        id_most_freq_rel = self.data.r2i[most_freq_rel]
        return id_most_freq_rel


        

    def explain(self):
        ''' Explain the prediction of the node with index node_idx: main method
            :return masked_hor, masked_ver: masked adjacency matrices of the subgraph'''

        print(f'Explanation for {self.node_idx} with original label {self.label}')
        print('num_neighbors:' ,self.num_neighbors,'num_edges:', self.num_edges)
        data, node_idx = self.data, self.node_idx
        sub_hor_graph, sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)
        self.most_freq_rel_node = self.most_freq_rel_f(sub_hor_graph, sub_ver_graph)
        print('most_freq_relation is:',  self.most_freq_rel_node )
        
        if self.config['kill_most_freq_rel']:
            #for i in self.most_freq_rel_node:
            sub_ver_graph, sub_hor_graph= select_on_relation_sparse(sub_ver_graph,data, 
                                                                    self.most_freq_rel_node), select_on_relation_sparse(sub_hor_graph,data, 
                                                                                                                      self.most_freq_rel_node)

        explainer = ExplainModule(
            sub_hor_graph, sub_ver_graph,
            self.model,
            self.label,
            self.data, 
            self.config,
            self.num_edges,
            self.node_idx,
            self.most_freq_rel_node
        )
        
        # explainer = ExplainModule(
        #     self.hor_graph, self.ver_graph,
        #     self.model,
        #     self.label,
        #     self.data, 
        #     self.config,
        #     74227,
        #     self.node_idx,
        #     self.most_freq_rel_node
        # ) #what if the 2 hop neighborhood is not enough??????

        self.model.eval()
        explainer.train()  
        print('start training')
        ypred_before, masked_hor_before, masked_ver_before = None, None, None
        for epoch in range(self.config.epochs):
            explainer.zero_grad()  # zero the gradient buffers of all parameters
            explainer.optimizer.zero_grad()
            
            ypred, masked_hor, masked_ver = explainer(self.node_idx)  # forward pass of the explainer

            if self.config["break_if_wrong_pred"] and epoch > 0:
                if torch.argmax(ypred) != self.label or list(ypred) == list(self.baseline) :
                    ypred, masked_hor, masked_ver = ypred_before, masked_hor_before, masked_ver_before
                    break

            if self.config["break_on_number_of_high"]:
                num_high = len([i for i in masked_ver.coalesce().values() if i > self.config['threshold']])
                if num_high < 3:
                    ypred, masked_hor, masked_ver = ypred_before, masked_hor_before, masked_ver_before
                    break

            ypred_before, masked_hor_before, masked_ver_before = ypred, masked_hor, masked_ver
            m_ver, m_hor = sub(masked_ver, 0.5), sub(masked_hor,0.5)

            

            m = match_to_triples(m_ver, m_hor, data, node_idx)
            counter = Counter(m[:,1].tolist())
            counter = {data.i2rel[k][0]:v for k,v in counter.items() if k!=0}
            if self.config.funky_loss==False:
                loss, pred_loss, size_loss, mask_ent_loss, size_std_loss,  num_high, wrong_pred = explainer.loss(ypred, self.config,epoch) 
                loss.backward()
                explainer.optimizer.step()

            if self.config.funky_loss:
                ypred_1_m, masked_hor_1_m, masked_ver_1_m = explainer.forward_1_m(self.node_idx)
                loss, fidelity_minus, fidelity_plus, sparsity = explainer.funky_loss(ypred, self.res_full, ypred_1_m, self.config)
                loss.backward(retain_graph=True)
                #loss.backward()
                explainer.optimizer.step()
            explainer.scheduler.step()
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
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
            
            if self.config.funky_loss==False:
                wandb.log({f"len mask > {self.config['threshold']}": len([i for i in masked_ver.coalesce().values() if i > self.config['threshold']]) , "loss": loss,
                "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss, "size_std_loss": size_std_loss,  "num_high": num_high, "wrong_pred": wrong_pred})
            if self.config.funky_loss:
                wandb.log({f"len mask > {self.config['threshold']}": len([i for i in masked_ver.coalesce().values() if i > self.config['threshold']]) , "loss": loss,
                "fidelity_minus": fidelity_minus, "fidelity_plus": fidelity_plus, "sparsity": sparsity})
        print('Finished Training')


        if self.config['kill_most_freq_rel']:
            #for i in self.most_freq_rel_node:

            h_0 ,v_0= select_on_relation_sparse(sub_hor_graph,self.data, self.most_freq_rel_node), select_on_relation_sparse(sub_ver_graph,self.data, self.most_freq_rel_node)
            sub_hor_graph, sub_ver_graph = h_0,v_0 



        masked_hor_values = (masked_hor.coalesce().values() * sub_hor_graph.coalesce().values())
        masked_ver_values = (masked_ver.coalesce().values() * sub_ver_graph.coalesce().values())


        masked_hor = torch.sparse.FloatTensor(sub_hor_graph.coalesce().indices(), masked_hor_values, sub_hor_graph.size())
        masked_ver = torch.sparse.FloatTensor(sub_ver_graph.coalesce().indices(), masked_ver_values, sub_ver_graph.size())


        return masked_hor, masked_ver, self.res_full
    

class ExplainModule(nn.Module):
    def __init__(
            self,
            sub_hor_graph,
            sub_ver_graph,
            model,
            label, 
            data,
            config,
            num_edges,
            node_idx, 
            most_freq_rel_node ):
        super(ExplainModule, self).__init__()
        self.hor_graph, self.ver_graph = sub_hor_graph, sub_ver_graph 
        self.model = model
        self.label = label
        self.data = data
        self.config = config
        self.node_idx = node_idx
        self.most_freq_rel_node = most_freq_rel_node

        #type = [i for i in range(data.num_relations) if 'type' in data.i2r[i]]
        tensor_list = []
        #for i in self.most_freq_rel_node:
        _,_,value_indices_h=select_relation(self.hor_graph,self.data.num_entities,self.most_freq_rel_node)
        _,_, value_indices_v=select_relation(self.ver_graph,self.data.num_entities,self.most_freq_rel_node)
        tensor_list.append(value_indices_h)
        tensor_list.append(value_indices_v)
        self._indices = torch.cat(tensor_list, 0)

        #num_nodes = num_edges
        num_nodes = torch.Tensor(self.ver_graph.coalesce().values()).shape
        self.mask = self.construct_edge_mask(num_nodes, self.ver_graph,self.data)
        #self.mask = self.construct_edge_mask(num_nodes, self.ver_graph,self.data)
        params = [self.mask]
        self.optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)





    
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
            #mask.data[[self._indices]] = 0

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
            mask.data[[self._indices]] = 0

        # elif init_strategy == "zero_out_most_freq_node":
        #     sub_edges = [self.hor_graph.coalesce().indices()[0],self.ver_graph.coalesce().indices()[1]]
        #     class_id= Counter([num for tup in sub_edges for num in tup]).most_common(1)[0][0]
        #     nn.init.constant_(mask,1) 
        #     value_indices = torch.where(sparse_tensor.coalesce().indices() == class_id)
        #     mask.data[[value_indices[1]]] = -10

        elif init_strategy == "one_out":
            #nn.init.uniform_(mask, 0,1) 
            nn.init.constant_(mask,0)
            _, _, value_indices3=select_relation(self.hor_graph,self.data.num_entities,relation_id)
            _, _, value_indices1=select_relation(self.ver_graph,self.data.num_entities,relation_id)
            
            value_indices = torch.cat((value_indices1, value_indices3), 0)
            mask.data[[value_indices]] = 1
            mask.data[[self._indices]] = 1


        elif init_strategy == "overall_frequency":
            '''Initialize the mask with the overall frequency of the relations'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            overall_rel_frequency = dict(Counter(data.triples[:,1].tolist()))

            overall_rel_frequency_  = {key: round(value/len(data.triples[:,1].tolist()),5) for key, value in overall_rel_frequency.items()}
            for i in p:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = overall_rel_frequency_[i]

        elif init_strategy == "overall_frequency_inverse":
            '''Initialize the mask with the overall frequency of the relations'''
            _ ,p = torch.div(sparse_tensor.coalesce().indices(), num_entities, rounding_mode='floor').tolist()
            overall_rel_frequency = dict(Counter(data.triples[:,1].tolist()))
            overall_rel_frequency_  = {key: 1 - round(value/len(data.triples[:,1].tolist()),5) for key, value in overall_rel_frequency.items()}
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

        elif init_strategy == "most_freq_rel":
            ''' Initialization that zeroes out the type relations'''
            #type = [i for i in range(data.num_relations) if 'type' in data.i2r[i]]

            for i in self.most_freq_rel_node:
                _,_,value_indices=select_relation(sparse_tensor,num_entities,i)
                mask.data[[value_indices]] = 0
        if self.config.print:
            print(f'mask initialized with {init_strategy} strategy: {mask}')   
        return mask
    

    def _masked_adj_ver(self):
        "" "Mask the adjacency matrix with the learned mask" ""

        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2

        adj = torch.Tensor(self.ver_graph.coalesce().values())
        print('shapes bruh', adj.shape, sym_mask.shape)
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

        # h_threshold, v_threshold,t_h, t_v = threshold_mask(self.masked_hor, self.masked_ver, self.data, self.config.num_exp)
        # res = nn.Softmax(dim=0)(self.model.forward2(h_threshold, v_threshold)[node_idx, :])
        #h_threshold, v_threshold,t = threshold_mask(masked_hor, masked_ver, self.data, num_exp = 10)
        #masked_ver,masked_hor = self.masked_ver, self.masked_hor
        #ypred = self.model.forward2(h_threshold, v_threshold)

        
        return res, self.masked_hor, self.masked_ver
    
    def forward_1_m(self, node_idx):
        self.masked_ver = self._masked_adj_ver()  # masked adj is the adj matrix with the mask applied
        self.masked_hor = self._masked_adj_hor() 
        masked_ver, masked_hor = inverse_tensor(self.masked_ver), inverse_tensor(self.masked_hor)
        res1_m = nn.Softmax(dim=0)(self.model.forward2(masked_hor, masked_ver)[node_idx])
        return res1_m , masked_hor, masked_ver



    def get_frequency_relations(self,v):
        _ ,p = torch.div(v.coalesce().indices(), v.size()[0], rounding_mode='floor')
        return dict(Counter(p))



    def funky_loss(self, pred,res_full, res1_m,  config):
        res_binary = pred
        label = self.label
        mask = torch.sigmoid(self.mask)
        explanation_lenght = torch.mean(mask[mask > config["threshold"]])
        fidelity_minus = torch.absolute(1 - (res_full[int(label)] - res_binary[int(label)]))
        fidelity_plus = (res_full[int(label)] - res1_m[int(label)])
        mask_len  = len(mask)
        sparsity = torch.tensor(1 - (explanation_lenght)/torch.mean(mask))
        most_freq_rel_len = len(mask[[self._indices]][mask[[self._indices]] > 0.5])
        most_freq_rel_loss = most_freq_rel_len/len(mask)
        size_loss = config['size'] * torch.sum(torch.abs(mask))
        loss = -(fidelity_minus + fidelity_plus + sparsity - most_freq_rel_loss - size_loss)
        #print(loss, fidelity_minus, fidelity_plus, sparsity)
        return loss, fidelity_minus, fidelity_plus, sparsity
     
    def loss(self, pred, config, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        adaptive = config['adaptive']
        mask = torch.sigmoid(self.mask)  # sigmoid of the mask
        mask_without_small = mask[mask > config["threshold"]]
#, torch.max(mask_without_small,0), torch.min(mask_without_small,0))
        num_high = len(mask_without_small)
        
       
        if self.config.print:
            print('mask', torch.mean(mask), torch.std(mask))
            print('mask_without_small', torch.mean(mask_without_small))
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

        

        # entropy edge mask 
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)

        mask_ent_loss =  config["ent"] * torch.mean(mask_ent)


        #type loss
        
        most_freq_rel_len = len(mask[[self._indices]][mask[[self._indices]] > 0.5])
        if config.print:
            print('mosgt freq rel',most_freq_rel_len)
        
        #type_loss = - config['type']*torch.sum(mask[[self._indices]][mask[[self._indices]] > 0.5])/torch.sum(mask)
        most_freq_rel_loss = config['most_freq_rel'] * most_freq_rel_len/len(mask)
        #this is to be minimized 
        wrong_pred = 1 if torch.argmax(pred) != self.label else 0

        

        loss = torch.exp(pred_loss + size_loss + mask_ent_loss + size_loss_std +  most_freq_rel_loss)
        #loss = pred_loss + size_loss + mask_ent_loss
        #loss = pred_loss + size_loss + mask_ent_loss + size_loss_std + wrong_pred + type_loss
        if config.print:
            print('pred_loss', pred_loss)
            print('size_loss', size_loss)
            print('type_loss', most_freq_rel_loss)
            print('mask_ent_loss', mask_ent_loss)
            print('wrong_pred', wrong_pred)
            print('size_loss_std', size_loss_std)
            print('pred',torch.argmax(pred), 'original label',self.label)
        

        return loss, pred_loss, size_loss, mask_ent_loss,size_loss_std, num_high , wrong_pred 
    


def scores(res_full, res_expl,res1_m,label,masked_ver, config):
    ''' res_expl: res_binary , res_threshold'''
    fidelity_minus = float(1 - (res_full[int(label)] - res_expl[int(label)]))
    fidelity_plus = float((res_full[int(label)] - res1_m[int(label)]))
    print(fidelity_minus, fidelity_plus)
    explanation_lenght = len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>config['threshold'] ])
    sparsity = float(1 - explanation_lenght/len(masked_ver.coalesce().values()))

    if sparsity == 1:
        sparsity_loss = 0
    if sparsity == 0:
        sparsity_loss = - 1
    else:
        sparsity_loss = sparsity
    score = fidelity_minus + fidelity_plus + sparsity_loss
    return fidelity_minus, fidelity_plus, sparsity, score


def main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,config):
    if sweep:
        wandb.init(config = config, reinit = True, project= f"RGCN_Explainer_{name}_{node_idx}")
    else:
        wandb.init(config = config, reinit = True, project= f"RGCN_Explainer_{name}", mode="disabled")
    config = wandb.config
    wandb.config.update({"size_std": num_neighbors})
    #wandb.config.update({"size_std": 10})

    label = int(data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([node_idx])),1])
    df = pd.DataFrame(columns=relations)
    df_threshold = pd.DataFrame(columns=relations) 

    explainer = Explainer(model,pred_label, data,name,  node_idx, n_hops, prune, config)
    masked_hor, masked_ver, res_full = explainer.explain()

    breaking = 'wrong_pred' if config["break_if_wrong_pred"] else 'num_high' if config["break_on_number_of_high"] else 'no'
    
    if sweep:

        experiment_name = f'exp/init_{config["init_strategy"]}_hops_{n_hops}_lr_{config["lr"]}_adaptive_{config["adaptive"]}_size_{config["size"]}_sizestd_adaptive_ent_{config["ent"]}_type_{config["most_freq_rel"]}_killtype_{config["kill_most_freq_rel"]}_break_{breaking}'
        wandb.run.name = experiment_name
    else:
        experiment_name = f'exp/init_{config["init_strategy"]}_hops_{n_hops}_lr_{config["lr"]}_adaptive_{config["adaptive"]}_size_{config["size"]}_sizestd_adaptive_ent_{config["ent"]}_type_{config["most_freq_rel"]}_killtype_{config["kill_most_freq_rel"]}_break_{breaking}'
        wandb.run.name = str(f'{node_idx}_{config["init_strategy"]}')

    directory = f'chk/{name}_chk/{experiment_name}'
    


    
    #if config.explain_all == True:
    if not os.path.exists(directory + f'/masked_adj'):
        os.makedirs(directory + f'/masked_adj')
    else:
        print(f"Directory '{directory}' already exists.")
    torch.save(masked_ver, f'{directory}/masked_adj/masked_ver{node_idx}')
    torch.save(masked_hor, f'{directory}/masked_adj/masked_hor{node_idx}') 

    #Explain prediction
    res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])

    #Niehborhood subgraph prediction
    m_ver,m_hor = convert_binary(masked_ver,0), convert_binary(masked_hor, 0)
    res_sub = nn.Softmax(dim=0)(model.forward2(m_hor, m_ver)[node_idx, :])

    #Explain prediction binary
    masked_ver_bin,masked_hor_bin= convert_binary(masked_ver, config['threshold']), convert_binary(masked_hor, config['threshold'])
    res_binary = nn.Softmax(dim=0)(model.forward2(masked_hor_bin, masked_ver_bin)[node_idx, :])

    #Prediction on inverse of explanation binary
    v_inv, h_inv = inverse_tensor(masked_ver), inverse_tensor(masked_hor)
    res1_m = nn.Softmax(dim=0)(model.forward2(h_inv,v_inv)[node_idx])




    #threshold to max of explanation edges
    h_threshold, v_threshold,t_h, t_v = threshold_mask(masked_hor, masked_ver, data, config.num_exp)
    res_threshold = nn.Softmax(dim=0)(model.forward2(h_threshold, v_threshold)[node_idx, :])
    # counter_threshold = important_relation(h_threshold, v_threshold,data, node_idx, config['threshold'])
    # print('Important relations thresholded to 10', counter_threshold)

    #Important relations - mask > threshold 
    counter = important_relation(masked_hor, masked_ver, data,node_idx, config['threshold'])
    print('Important relations', counter)
    v_, h_ = masked_ver, masked_hor
    v_._values().zero_()
    h_._values().zero_()
    res_baseline = nn.Softmax(dim=0)(model.forward2(h_,v_)[node_idx])
    print('res baseline', res_baseline)



 
    #Threshold until lekker
    i = 0
    res_threshold_lekker = res_binary
    while res_threshold_lekker.argmax() != res_full.argmax() and not torch.equal(res_threshold_lekker, res_baseline):
        print(res_threshold_lekker, res_baseline)
        h_threshold, v_threshold,t_h, t_v = threshold_mask(masked_hor, masked_ver, data, 1+i, equal=False)
        i+=1
        res_threshold_lekker = nn.Softmax(dim=0)(model.forward2(h_threshold, v_threshold)[node_idx, :])
        if i== masked_hor.shape[0]:
            break
    if not os.path.exists(directory + f'/masked_adj'):
        os.makedirs(directory + f'/masked_adj')
    else:
        print(f"Directory '{directory}' already exists.")
    torch.save(v_threshold, f'{directory}/masked_adj/masked_ver_thresh{node_idx}')
    torch.save(h_threshold, f'{directory}/masked_adj/masked_hor_thresh{node_idx}') 
    counter_threshold = important_relation(h_threshold, v_threshold,data, node_idx, config['threshold'])
    print('Important relations thresholded to 10', counter_threshold)

    #Random explanation
    h_random, v_random = random_explanation_baseline(h_threshold), random_explanation_baseline(v_threshold)
    counter = important_relation(h_random, v_random, data,node_idx, 0.5)
    print('Random baseline Important relations', counter)
    res_random = nn.Softmax(dim=0)(model.forward2(h_random, v_random)[node_idx, :])

    res_random_inverse = res1_m = nn.Softmax(dim=0)(model.forward2(inverse_tensor(h_random),inverse_tensor(v_random))[node_idx])
    ##Inverse of threshold until lekker
    v_inv, h_inv = inverse_tensor(v_threshold), inverse_tensor(h_threshold)
    res_threshold_lekker_inverse = nn.Softmax(dim=0)(model.forward2(h_inv,v_inv)[node_idx])
    
    fidelity_minus, fidelity_plus, sparsity, score = scores(res_full, res_binary,res1_m,label,masked_ver, config)
    fidelity_minus_threshold, fidelity_plus_threshold, sparsity_threshold, score_threshold = scores(res_full, res_threshold_lekker,res_threshold_lekker_inverse,label,v_threshold, config)
    fidelity_minus_random, fidelity_plus_random, sparsity_random, score_random = scores(res_full, res_random ,res_random_inverse,label,v_random, config)
    wandb.log({'score_threshold': score_threshold})
    wandb.log({'score': score})
    print('score', score)
    #metrics
    # fidelity_minus = torch.mean(1 - (res_full - res_binary))
    # fidelity_plus = torch.mean((res_full - res1_m))

    # fidelity_minus = float(1 - (res_full[int(label)] - res_binary[int(label)]))
    # fidelity_plus = float((res_full[int(label)] - res1_m[int(label)]))
    # print(fidelity_minus, fidelity_plus)
    # explanation_lenght = len(masked_ver.coalesce().values()[masked_ver.coalesce().values()>config['threshold'] ])
    # sparsity = float(1 - explanation_lenght/len(masked_ver.coalesce().values()))

    # if sparsity == 1:
    #     sparsity_loss = 0
    # if sparsity == 0:
    #     sparsity_loss = - 1
    # else:
    #     sparsity_loss = sparsity
    # score = fidelity_minus + fidelity_plus + sparsity_loss



    #Save in the csv: label, node, number neighbors, predictions
    target_label = str([k for k, v in dict_classes.items() if node_idx in v])
    info = {'label': str(target_label), 'node_idx': str(node_idx), 'number_neighbors': str(num_neighbors),
             'prediction_explain_binary': str(res_binary.detach().numpy()), 'prediction_full': str(res_full.detach().numpy()), 
             'prediction_explain': str(res.detach().numpy()), 'prediction_inverse_binary': str(res1_m.detach().numpy()),
             'prediction_random': str(res_random.detach().numpy()), 
             'prediction_sub': str(res_sub.detach().numpy()), 'prediction_threshold': str(res_threshold.detach().numpy()),
             'prediction_threshold_lekker': str(res_threshold_lekker.detach().numpy()),
             'res_random_inverse': str(res_random_inverse.detach().numpy()),
             'res_threshold_lekker_inverse': str(res_threshold_lekker_inverse.detach().numpy()),
            'fidelity_minus': str(fidelity_minus), 'fidelity_plus': str(fidelity_plus), 'sparsity': str(sparsity),
            'fidelity_minus_threshold': str(fidelity_minus_threshold), 'fidelity_plus_threshold': str(fidelity_plus_threshold), 'sparsity_threshold': str(sparsity_threshold),
            'fidelity_minus_random': str(fidelity_minus_random), 'fidelity_plus_random': str(fidelity_plus_random), 'sparsity_random': str(sparsity_random)
            }
    counter.update(info)
    df.loc[str(node_idx)] = counter
    counter_threshold.update(info)
    df_threshold.loc[str(node_idx)] = counter_threshold

    

    print('score', score)
    print('node_idx', node_idx, 
        '\n node original label',target_label,
        '\n VS label full', torch.argmax(res_full).item(),
        '\n VS label explain', torch.argmax(res).item(),
        '\n VS label explain binary', torch.argmax(res_binary).item(),
        '\n VS label threshold', torch.argmax(res_threshold).item(),
        '\n VS label threshold lekker', torch.argmax(res_threshold_lekker).item(),
        '\n VS label sub', torch.argmax(res_sub).item(),
        '\n VS label 1-m explain binary', torch.argmax(res1_m).item(),

        ' \n pred prob explain', res, 
        '\n pred prob explain binary', res_binary,
        '\n pred prob threshold', res_threshold,'threshold',t_h,t_v, 'with num edges', len(v_threshold.coalesce().values()[v_threshold.coalesce().values()>config['threshold'] ]),
        '\n pred prob threshold lekker', res_threshold_lekker,
        '\n pred prob full', res_full,       
        '\n pred prob sub', res_sub,
        '\n pred prob 1-m explain binary', res1_m,
        '\n pred prob random', res_random,
        '\n final masks and lenght', torch.mean(v_threshold.coalesce().values()[v_threshold.coalesce().values()>config['threshold'] ]), torch.std(v_threshold.coalesce().values()[v_threshold.coalesce().values()>config['threshold'] ]),#convert_back(masked_ver,data), 
        len(v_threshold.coalesce().values()[v_threshold.coalesce().values()>config['threshold'] ]),
        '\n overall mean', torch.mean(v_threshold.coalesce().values()), torch.std(v_threshold.coalesce().values()),
        '\n Sparsity', sparsity, '\n fidelity_minus', fidelity_minus, '\n fidelity_plus', fidelity_plus, '\n score', score)

    if not os.path.exists(directory + f'/Relation_Importance'):
        os.makedirs(directory + f'/Relation_Importance')
    if config.explain_all == False:
        df.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_{node_idx}.csv', index=False)
        df_threshold.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_{node_idx}_threshold.csv', index=False)
    if config.explain_all == True:
        if node_idx == dict_classes[list(dict_classes.keys())[0]][0]:
            df.to_csv(f'{directory}/Relation_Importance/Relations_Important_full.csv', mode='a', index=False)
            df_threshold.to_csv(f'{directory}/Relation_Importance/Relations_Important_full_threshold.csv', mode='a', index=False)
        else:
            df.to_csv(f'{directory}/Relation_Importance/Relations_Important_full.csv', mode='a', header=False, index=False)
            df_threshold.to_csv(f'{directory}/Relation_Importance/Relations_Important_full_threshold.csv', mode='a', header=False, index=False)
    return counter,counter_threshold,  experiment_name


