
import torch 
from collections import Counter
from Relation_Attribution import baseline_pred
import torch.nn as nn
import math
import pandas as pd
import matplotlib.pyplot as plt
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
                 prediction_model,
                 data,
                 name,
                 node_idx,
                 n_hops,
                 prune,
                 config,):
        self.model = model
        self.prediction_model = prediction_model
        self.label = data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([node_idx])),1].long()
        self.pred_label = self.prediction_model.argmax(dim=1)[torch.where(data.withheld[:, 0] == torch.tensor([node_idx]))]
        self.data = data
        self.name = name
        self.n = data.num_entities
        self.r = data.num_relations
        self.triples = enrich(data.triples,self.n,self.r)
        self.node_idx = node_idx
        self.n_hops = n_hops
        self.config = config


        #get hor and ver graph and the subgraphs
        self.hor_graph, self.ver_graph = data.hor_graph, data.ver_graph
        self.edge_index_h, self.edge_index_v = self.hor_graph.coalesce().indices(), self.ver_graph.coalesce().indices()
        self.res_full = self.prediction_model[torch.where(data.withheld[:, 0] == torch.tensor([node_idx]))][0]#nn.Softmax(dim=0)(model.forward2(self.hor_graph, self.ver_graph)[node_idx, :])
        self.sub_edges_h, self.neighbors_h, self.sub_edges_tensor_h  = find_n_hop_neighbors(self.edge_index_h, n=self.n_hops, node=self.node_idx)
        self.sub_edges_v, self.neighbors_v, self.sub_edges_tensor_v  = find_n_hop_neighbors(self.edge_index_v, n=self.n_hops, node=self.node_idx)
        self.sub_triples = match_to_triples(self.sub_edges_tensor_v, self.sub_edges_tensor_h,self.data, sparse=False)
        self.overall_rel_frequency = dict(Counter(self.data.triples[:,1].tolist()))
        self.num_neighbors = len(set(list(self.neighbors_h) + list(self.neighbors_v)))
        self.num_edges = len(self.sub_edges_v)+len(self.sub_edges_h)
        self.baseline = baseline_pred(self.data, self.model, self.node_idx)



    
    def most_freq_rel_f(self, sub_h, sub_v):
        m = match_to_triples(sub_v, sub_h,self.data, self.node_idx)
        freq = Counter(m[:,1].tolist())
        sorted_freq = {self.data.i2r[k]: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True) if k!=0}
        most_freq_rel = list(sorted_freq.keys())[0]
        most_freq_rel_val = list(sorted_freq.values())[0]
        id_most_freq_rel = self.data.r2i[most_freq_rel]
        return id_most_freq_rel, most_freq_rel_val


        

    def explain(self):
        ''' Explain the prediction of the node with index node_idx: main method
            :return masked_hor, masked_ver: masked adjacency matrices of the subgraph'''

        print(f'Explanation for {self.node_idx} with original label {self.label}, label predicted by the model to explain {self.pred_label}')
        print('num_neighbors:' ,self.num_neighbors,'num_edges:', self.num_edges)
        data, node_idx = self.data, self.node_idx
        sub_hor_graph, sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)
        self.most_frequent_relation, mfr_val = self.most_freq_rel_f(sub_hor_graph, sub_ver_graph)
        print('most_freq_relation is:',  data.i2r[self.most_frequent_relation],self.most_frequent_relation ,mfr_val/self.num_edges)

        if mfr_val/self.num_edges > 0.7 and self.name != 'dbo_gender':
            print(mfr_val/self.num_edges)
            self.config.update({'kill_most_freq_rel': True}, allow_val_change=True)
        if self.config['kill_most_freq_rel']:
            sub_ver_graph, sub_hor_graph= select_on_relation_sparse(sub_ver_graph,data, 
                                                                    self.most_frequent_relation), select_on_relation_sparse(sub_hor_graph,data, 
                                                                                                                      self.most_frequent_relation)

        explainer = ExplainModule(
            sub_hor_graph, sub_ver_graph,
            self.model,
            self.pred_label,
            self.data, 
            self.config,
            self.num_edges,
            self.node_idx,
            self.most_frequent_relation
        )


        self.model.eval()
        explainer.train()  
        print('start training')
        ypred_before, masked_hor_before, masked_ver_before = None, None, None
        for epoch in range(self.config.epochs):
            explainer.zero_grad()  # zero the gradient buffers of all parameters
            explainer.optimizer.zero_grad()
            
            ypred, masked_hor, masked_ver = explainer(self.node_idx)  # forward pass of the explainer

            m_ver, m_hor = sub(masked_ver, 0.5), sub(masked_hor,0.5)
            m = match_to_triples(m_ver, m_hor, data, node_idx)
            counter = Counter(m[:,1].tolist())
            counter = {data.i2rel[k][0]:v for k,v in counter.items()}# if k!=0}
            loss, pred_loss, size_loss, mask_ent_loss, size_std_loss,  num_high, wrong_pred, most_freq_rel_loss = explainer.loss(ypred, self.config,epoch) 
            loss.backward()
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
            

            wandb.log({f"len mask > {self.config['threshold']}": len([i for i in masked_ver.coalesce().values() if i > self.config['threshold']]) , "loss": loss,
                "pred_loss": pred_loss, "size_loss": size_loss, "mask_ent_loss": mask_ent_loss, "size_std_loss": size_std_loss,  "num_high": num_high, "wrong_pred": wrong_pred, "most_freq_rel_loss": most_freq_rel_loss})

        print('Finished Training')
        m = match_to_triples(sub_ver_graph, sub_hor_graph, data, node_idx)
        counter = dict(Counter(m[:,1].tolist()))
        counter_full = {data.i2rel[k][0]:v for k,v in counter.items() if k!=0}


        if self.config['kill_most_freq_rel']:
            #for i in self.most_frequent_relation:

            h_0 ,v_0= select_on_relation_sparse(sub_hor_graph,self.data, self.most_frequent_relation), select_on_relation_sparse(sub_ver_graph,self.data, self.most_frequent_relation)
            sub_hor_graph, sub_ver_graph = h_0,v_0 
            del counter_full[data.i2rel[self.most_frequent_relation][0]]

        masked_hor_values = (masked_hor.coalesce().values() * sub_hor_graph.coalesce().values())
        masked_ver_values = (masked_ver.coalesce().values() * sub_ver_graph.coalesce().values())


        masked_hor = torch.sparse.FloatTensor(sub_hor_graph.coalesce().indices(), masked_hor_values, sub_hor_graph.size())
        masked_ver = torch.sparse.FloatTensor(sub_ver_graph.coalesce().indices(), masked_ver_values, sub_ver_graph.size())


        return masked_hor, masked_ver, self.res_full, counter_full
    

class ExplainModule(nn.Module):
    def __init__(
            self,
            sub_hor_graph,
            sub_ver_graph,
            model,
            pred_label, 
            data,
            config,
            num_edges,
            node_idx, 
            most_freq_relation ):
        super(ExplainModule, self).__init__()
        self.hor_graph, self.ver_graph = sub_hor_graph, sub_ver_graph 
        self.model = model
        self.pred_label = pred_label
        self.data = data
        self.config = config
        self.node_idx = node_idx
        self.most_frequent_relation = most_freq_relation

        self.mfr_indices = torch.cat([select_relation(self.hor_graph, self.data.num_entities, self.most_frequent_relation)[2], select_relation(self.ver_graph, self.data.num_entities, self.most_frequent_relation)[2]], 0)

        num_nodes = torch.Tensor(self.ver_graph.coalesce().values()).shape[0]
        self.mask = self.construct_edge_mask(num_nodes, self.ver_graph,self.data)
        self.optimizer = torch.optim.Adam([self.mask], lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)





    
    def construct_edge_mask(self, num_nodes,sparse_tensor,data, const_val=0.75):
        """
        Construct edge mask
        """
        init_strategy = self.config["init_strategy"]
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




        elif "Domain_Knowledge" in init_strategy:
            relations_id = self.config["relation_id"]
            value_for_relations_id = self.config["value_for_relations_id"]
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
            if relations_id != None:
                if type(relations_id)== int:
                    _, _, value_indices3=select_relation(self.hor_graph,self.data.num_entities,relations_id)
                    _, _, value_indices1=select_relation(self.ver_graph,self.data.num_entities,relations_id)
                    
                    value_indices = torch.cat((value_indices1, value_indices3), 0)
                    mask.data[[value_indices]] = value_for_relations_id
                else:
                    for i in relations_id:
                        _, _, value_indices3=select_relation(self.hor_graph,self.data.num_entities,i)
                        _, _, value_indices1=select_relation(self.ver_graph,self.data.num_entities,i)
                        
                        value_indices = torch.cat((value_indices1, value_indices3), 0)
                        mask.data[[value_indices]] = value_for_relations_id
            else:
                pass




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
            if torch.isnan(mask).any():
                # Replace NaN values with 0
                mask[torch.isnan(mask)] = 0.0

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
            ''' Initialization that zeroes out the most frequent relation'''
            #type = [i for i in range(data.num_relations) if 'type' in data.i2r[i]]

            for i in self.most_frequent_relation:
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
        masked_ver, masked_hor = self.masked_ver, self.masked_hor
        # threshold = torch.mean(masked_ver.coalesce().values()) + torch.std(masked_ver.coalesce().values())
        # self.config.update({'threshold': threshold}, allow_val_change=True)
        masked_ver,masked_hor = convert_binary(self.masked_ver, self.config["threshold"]), convert_binary(self.masked_hor, self.config["threshold"])
    
        ypred = self.model.forward2(masked_hor, masked_ver)
        node_pred = ypred[node_idx,:]
        res = nn.Softmax(dim=0)(node_pred[0:])  
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


     
    def loss(self, pred, config, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """

        mask = torch.sigmoid(self.mask)  # sigmoid of the mask
        mask_without_small = mask[mask > config["threshold"]]
        num_high = len(mask_without_small)
        
       
        if self.config.print:
            print('mask', torch.mean(mask), torch.std(mask))
            print('mask_without_small', torch.mean(mask_without_small))

   
        # prediction loss

        logit = pred[self.pred_label]

        pred_loss = config["pred"]* -torch.log(logit)


        size_loss = config['size'] * torch.sum(mask)

        size_loss_std = -config["size_std"] * torch.std(mask) 

        
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)

        if torch.isnan(mask_ent).any():
            # Replace NaN values with 0
            mask_ent_loss = 0
        else:
            mask_ent_loss =  config["ent"] * torch.mean(mask_ent)


        most_freq_rel_len = len(mask[[self.mfr_indices]][mask[[self.mfr_indices]] > config.threshold])/2
        if most_freq_rel_len/len(mask) > 0.5:
            most_freq_rel_loss = 10 * most_freq_rel_len/len(mask)
        else:
            most_freq_rel_loss = config['most_freq_rel'] * most_freq_rel_len/len(mask)

        wrong_pred = 1 if torch.argmax(pred) != self.pred_label else 0

        

        loss = torch.exp(pred_loss + size_loss + mask_ent_loss + size_loss_std +  most_freq_rel_loss)
        if config.print:
            print('pred_loss', pred_loss)
            print('size_loss', size_loss)
            print('type_loss', most_freq_rel_loss)
            print('mask_ent_loss', mask_ent_loss)
            print('wrong_pred', wrong_pred)
            print('size_loss_std', size_loss_std)
            print('pred',torch.argmax(pred), 'original label',self.label)
        

        return loss, pred_loss, size_loss, mask_ent_loss,size_loss_std, num_high , wrong_pred , most_freq_rel_loss
    


def scores(res_full, res_expl,res1_m,label,masked_ver, config,num_edges, probability=False):
    ''' res_expl: res_binary , res_threshold'''

    if probability:
        fidelity_minus = float(1 - (res_full[int(label)] - res_expl[int(label)]))
        fidelity_plus = float((res_full[int(label)] - res1_m[int(label)]))
    else:
        fidelity_minus = 1 if res_full.argmax() == res_expl.argmax() else 0
        fidelity_plus = 1 if res_full.argmax() != res1_m.argmax() else 0
    explanation_lenght = len(masked_ver.coalesce().values()[masked_ver.coalesce().values()==1 ])
    sparsity = float(1 - explanation_lenght/num_edges)

    score = fidelity_minus + fidelity_plus + sparsity
    return fidelity_minus, fidelity_plus, sparsity, score



def main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_edges,sweep,config):
    if sweep:
        wandb.init(config = config, reinit = True, project= f"RGCN_Explainer_{name}_{node_idx}")
    else:
        wandb.init(config = config, reinit = True, project= f"RGCN_Explainer_{name}", mode="disabled")
    config = wandb.config

    experiment_name = f'exp/init_{config["init_strategy"]}_lr_{config["lr"]}_size_{config["size"]}_ent_{config["ent"]}_type_{config["most_freq_rel"]}_wd_{config["weight_decay"]}_MFR_{config["most_freq_rel"]}'
    if sweep:    
        wandb.run.name = experiment_name
    else:
        wandb.run.name = str(f'{node_idx}_{config["init_strategy"]}')

    directory = f'chk/{name}_chk/{experiment_name}'
    if not os.path.exists(directory + f'/masked_adj'):
        os.makedirs(directory + f'/masked_adj')
    else:
        pass

    label = int(data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([node_idx])),1])

    relations_plus = ['label', 'node_idx','number_neighbors', 
                'prediction_explain', 'prediction_full', 'prediction_explain_binary',
                'prediction_inverse_binary', 
                'prediction_random','prediction_sub', 'prediction_threshold',
                'prediction_threshold_lekker',
                'res_random_inverse','res_threshold_lekker_inverse',
                'fidelity_minus', 'fidelity_plus', 'sparsity',
                'fidelity_minus_threshold','fidelity_plus_threshold','sparsity_threshold',
                'fidelity_minus_random','fidelity_plus_random','sparsity_random','score_threshold_prob'] + relations
    relations_neighborhood = ['label', 'node_idx','number_neighbors']  + relations 
    df = pd.DataFrame(columns=relations_plus)
    df_threshold = pd.DataFrame(columns=relations_plus) 
    df_full_neighborhood = pd.DataFrame(columns=relations_neighborhood)
    explainer = Explainer(model,pred_label, data,name,  node_idx, n_hops, prune, config)
    masked_hor, masked_ver, res_full, counter_full = explainer.explain()

    #breaking = 'wrong_pred' if config["break_if_wrong_pred"] else 'num_high' if config["break_on_number_of_high"] else 'no'
    torch.save(masked_ver, f'{directory}/masked_adj/masked_ver{node_idx}')
    torch.save(masked_hor, f'{directory}/masked_adj/masked_hor{node_idx}') 
    
    #Explain prediction
    prediction_explain = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])

    #Niehborhood subgraph prediction
    h_neighborhood, v_neighborhood = convert_binary(masked_hor, 0), convert_binary(masked_ver,0)
    prediction_neighborhood = nn.Softmax(dim=0)(model.forward2(h_neighborhood, v_neighborhood)[node_idx, :])

    #Explain prediction binary
    h_binary, v_binary = convert_binary(masked_hor, config['threshold']), convert_binary(masked_ver, config['threshold'])
    prediction_explain_binary = nn.Softmax(dim=0)(model.forward2(h_binary, v_binary)[node_idx, :])
    prediction_binary_inverse = nn.Softmax(dim=0)(model.forward2(inverse_tensor(masked_hor),inverse_tensor(masked_ver))[node_idx])



    # baseline
    v_, h_ = masked_ver.clone(), masked_hor.clone()
    v_._values().zero_()
    h_._values().zero_()
    res_baseline = nn.Softmax(dim=0)(model.forward2(h_,v_)[node_idx])
    

    #threshold to max of explanation edges
    h_threshold, v_threshold,t_h, t_v = threshold_mask(masked_hor, masked_ver, data, config.num_exp)
    h_threshold, v_threshold = get_n_highest_sparse(masked_hor, config.num_exp),get_n_highest_sparse(masked_ver, config.num_exp)
    print('masked ver',v_threshold.coalesce().values().count_nonzero())
    res_threshold = nn.Softmax(dim=0)(model.forward2(h_threshold, v_threshold)[node_idx, :])
    i = 0
    row_indices =  torch.nonzero(v_threshold.coalesce().indices()[1] == node_idx, as_tuple=False)[:, 0]

    while res_threshold.argmax() != res_full.argmax():# and v_threshold.coalesce().values()[row_indices].count_nonzero() == 0:# and res_threshold[label] == res_baseline[label]:

        i += 1
        
        h_threshold, v_threshold,t_h, t_v = threshold_mask(masked_hor, masked_ver, data, config.num_exp+i)
        h_threshold, v_threshold = get_n_highest_sparse(masked_hor, config.num_exp+i),get_n_highest_sparse(masked_ver, config.num_exp+i)
        res_threshold_lekker = nn.Softmax(dim=0)(model.forward2(h_threshold, v_threshold)[node_idx, :])
        if res_threshold_lekker.argmax() == res_full.argmax() or i>500:
            break
    

    else:
        res_threshold_lekker = res_threshold
    
    print('i is:', config.num_exp + i)
    if not os.path.exists(directory + f'/masked_adj'):
        os.makedirs(directory + f'/masked_adj')
    else:
        pass
    torch.save(v_threshold, f'{directory}/masked_adj/masked_ver_thresh{node_idx}')
    torch.save(h_threshold, f'{directory}/masked_adj/masked_hor_thresh{node_idx}') 

    res_threshold_lekker_inverse = nn.Softmax(dim=0)(model.forward2(inverse_tensor(h_threshold),inverse_tensor(v_threshold))[node_idx])

    #random baseline
    h_random, v_random = random_explanation_baseline(masked_hor, config.num_exp + i), random_explanation_baseline(masked_ver, config.num_exp + i)
    res_random = nn.Softmax(dim=0)(model.forward2(h_random,v_random)[node_idx, :])
    res_random_inverse = nn.Softmax(dim=0)(model.forward2(inverse_tensor(h_random),inverse_tensor(v_random))[node_idx])
    #COUNTERS
    counter = important_relation(h_binary, v_binary, data,node_idx,0)
    counter_threshold = important_relation(h_threshold, v_threshold,data, node_idx, 0)
    counter_random = important_relation(h_random, v_random, data,node_idx, 0)
    print('Important relations', counter)
    print(f'Important relations thresholded to {config.num_exp + i}', counter_threshold)
    print('Random baseline Important relations', counter_random)
    print('All relations', counter_full)

    
    fidelity_minus, fidelity_plus, sparsity, score = scores(res_full, prediction_explain_binary,prediction_binary_inverse,label,masked_ver, config, num_edges)
    fidelity_minus_threshold, fidelity_plus_threshold, sparsity_threshold, score_threshold = scores(res_full, res_threshold_lekker,res_threshold_lekker_inverse,label,v_threshold, config,num_edges)
    _,_,_, score_threshold_prob = scores(res_full, res_threshold_lekker,res_threshold_lekker_inverse,label,v_threshold, config,num_edges, probability=True)
    fidelity_minus_random, fidelity_plus_random, sparsity_random, score_random = scores(res_full, res_random ,res_random_inverse,label,v_random, config,num_edges)
    wandb.log({'score_threshold': score_threshold})
    wandb.log({'score': score})
    wandb.log({'score_threshold_prob': score_threshold_prob})
    wandb.log({'Explanation Size':config.num_exp + i })



    #Save in the csv: label, node, number neighbors, predictions

    info = {'label': str(label), 'node_idx': str(node_idx), 'number_neighbors': str(num_edges),
             'prediction_explain_binary': str(prediction_explain_binary.detach().numpy()), 'prediction_full': str(res_full.detach().numpy()), 
             'prediction_explain': str(prediction_explain.detach().numpy()), 
             'prediction_inverse_binary': str(prediction_binary_inverse.detach().numpy()),
             'prediction_random': str(res_random.detach().numpy()), 
             'prediction_sub': str(prediction_neighborhood.detach().numpy()), 'prediction_threshold': str(res_threshold.detach().numpy()),
             'prediction_threshold_lekker': str(res_threshold_lekker.detach().numpy()),
             'res_random_inverse': str(res_random_inverse.detach().numpy()),
             'res_threshold_lekker_inverse': str(res_threshold_lekker_inverse.detach().numpy()),
            'fidelity_minus': str(fidelity_minus), 'fidelity_plus': str(fidelity_plus), 'sparsity': str(sparsity),
            'fidelity_minus_threshold': str(fidelity_minus_threshold), 'fidelity_plus_threshold': str(fidelity_plus_threshold), 'sparsity_threshold': str(sparsity_threshold),
            'fidelity_minus_random': str(fidelity_minus_random), 'fidelity_plus_random': str(fidelity_plus_random), 'sparsity_random': str(sparsity_random),
            'score_threshold_prob': str(score_threshold_prob),
            }
    counter.update(info)
    df.loc[str(node_idx)] = counter
    counter_threshold.update(info)
    df_threshold.loc[str(node_idx)] = counter_threshold


    info_neighborhood = {'label': str(label), 'node_idx': str(node_idx), 'number_neighbors': str(num_edges)}
    counter_full.update(info_neighborhood)
    df_full_neighborhood.loc[str(node_idx)] = counter_full


    print('node_idx', node_idx, 
        '\n node original label',label,
        '\n VS label full', torch.argmax(res_full).item(), 
        '\n VS label explain', torch.argmax(prediction_explain).item(),
        '\n VS label explain binary', torch.argmax(prediction_explain_binary).item(), 'with edges', len( v_binary.coalesce().values()[ v_binary.coalesce().values()==1 ]),
        '\n VS label threshold lekker', torch.argmax(res_threshold_lekker).item(),
        '\n VS label sub', torch.argmax(prediction_neighborhood).item(), 'with edges', len(v_neighborhood.coalesce().values()[v_neighborhood.coalesce().values()==1 ]),
        '\n label 1-m threshold lekker', torch.argmax(res_threshold_lekker_inverse).item(),
        '\n VS label 1-m explain binary', torch.argmax(prediction_binary_inverse).item(), 
        '\n VS label random', torch.argmax(res_random).item(),
        '\n VS label baseline', torch.argmax(res_baseline).item(),

    

        '\n lekker equal baseline', res_baseline.argmax() == res_threshold_lekker.argmax(),
        #'\n pred baseline', res_baseline,
        # ' \n pred prob explain', prediction_explain, 
        #'\n pred prob explain binary', prediction_explain_binary,
        # '\n pred prob threshold', res_threshold,
        #'\n pred prob threshold lekker', res_threshold_lekker,
        # '\n pred prob full', res_full,       
        #'\n pred prob sub', prediction_neighborhood,
        # '\n pred prob 1-m explain binary', prediction_binary_inverse,
        # '\n pred prob random', res_random,
        '\n explanation length', len(v_threshold.coalesce().values()[v_threshold.coalesce().values()==1 ]) , 
        '\n Sparsity', sparsity_threshold, '\n fidelity_minus', fidelity_minus_threshold, '\n fidelity_plus', fidelity_plus_threshold, '\n score', score_threshold,
        '\n score prob', score_threshold_prob)

    if not os.path.exists(directory + f'/Relation_Importance'):
        os.makedirs(directory + f'/Relation_Importance')
    
    nodes_exp = node_idx if config.explain_one else 'full' if config.explain_all == True else 'sample' 
    path = f'{directory}/Relation_Importance/Relations_Important_{nodes_exp}.csv'
    path_t = f'{directory}/Relation_Importance/Relations_Important_{nodes_exp}_threshold.csv'
    path_full = f'{directory}/Relation_Importance/Relations_Important_full_neighborhood.csv'
    file_exists_and_non_empty = os.path.exists(path) and os.path.getsize(path) > 0
    if file_exists_and_non_empty:
        df_check = pd.read_csv(path)
        if node_idx in df_check.iloc[:, 1].values: 

            df.to_csv(path, mode='w', index=False)
            df_threshold.to_csv(path_t, mode='w', index=False)
            df_full_neighborhood.to_csv(path_full, mode='w', index=False)
        else:

            if not file_exists_and_non_empty:
                df.to_csv(path, mode='a', index=False)
                df_threshold.to_csv(path_t, mode='a', index=False)
                df_full_neighborhood.to_csv(path_full, mode='a', index=False)
            else:
                df.to_csv(path, mode='a', header=False, index=False)
                df_threshold.to_csv(path_t, mode='a', header=False, index=False)
                df_full_neighborhood.to_csv(path_full, mode='a', header=False, index=False)

    else:
        df.to_csv(path, mode='a', index=False)
        df_threshold.to_csv(path_t, mode='a', index=False)
        df_full_neighborhood.to_csv(path_full, mode='a', index=False)

    row_indices =  torch.nonzero(v_threshold.coalesce().indices()[1] == node_idx, as_tuple=False)[:, 0]
    if v_threshold.coalesce().values()[row_indices].count_nonzero() != 0:
        print('node_idx in explanation')
    else:
        print('node_idx not in explanation')
    
    print(directory)


    return counter,counter_threshold,  experiment_name



