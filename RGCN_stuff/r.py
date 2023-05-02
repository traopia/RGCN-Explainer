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
import os
import wandb
from itertools import product
from kgbench import load, tic, toc, d


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm


#
from torch_geometric.utils import to_networkx
import networkx as nx


#rgcn 
from rgcn_model import * #adj, enrich, sum_sparse, RGCN
#from rgcn_torch import Net
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

    r = rn // n #number of relations enriched divided by number of nodes

    vals = torch.ones(ver_ind.size(0), dtype=torch.float) #number of enriched triples
    vals = vals / sum_sparse(ver_ind, vals, ver_size) #normalize the values by the number of edges

    hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size) #size: n,r, emb


    ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)

    return hor_graph, ver_graph














class Explain(nn.Module):
    def __init__(self, model, data, node_idx, n_hops,name,prune=True, pyg_torch = False):
        super(Explain, self).__init__()
        #Those are the parameters of the original data and model
        self.model = model
        self.data = data
        self.n = data.num_entities
        self.r = data.num_relations
        self.name = name
        #self.triples = enrich(data.triples, self.n, self.r)
        self.triples = data.triples
        self.node_idx = node_idx
        self.n_hops = 0 if prune else n_hops
        #self.adj = get_adjacency(data)
        self.edge_index = edge_index_oneadj(self.triples)
       
        self.label = self.data.withheld[:, 1]

        self.epoch = 1
        self.hor_graph, self.ver_graph = hor_ver_graph(self.triples, self.n, self.r)


        
        self.pred_label = torch.load(f'chk/{name}_chk/prediction_{name}_prune_{prune}')
        self.pyg_torch = pyg_torch
        if pyg_torch: 
            self.pred_label = torch.load(f'chk/{name}_chk/prediction_{name}_torch')
        self.sub_edges, self.neighbors, self.sub_edges_tensor = find_n_hop_neighbors(self.edge_index, n=self.n_hops, node=self.node_idx)
        self.sub_triples = match_to_triples(self.sub_edges_tensor.t(),self.triples)
        self.sub_hor_graph, self.sub_ver_graph = hor_ver_graph(self.sub_triples, self.n, self.r)


        self.num_nodes = self.sub_hor_graph.coalesce().values().shape[0]   
        print(self.num_nodes)
        self.mask = self.construct_edge_mask(self.num_nodes)

    
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

        masked_adj = adj * sym_mask

        result = torch.sparse_coo_tensor(indices=sub_graph.coalesce().indices(), values= masked_adj, size=sub_graph.coalesce().size())

        return result
    
    def construct_edge_mask( self, num_nodes, init_strategy="normal", const_val=1.0):
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
        return mask #torch.tensor(mask)
        #
        #return torch.tensor(mask)


        def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0, prior=None):
            """
            Construct edge mask
            input;
                num_nodes: number of nodes in the neighborhood
                init_strategy: initialization strategy for the mask
                const_val: constant value for the mask
                prior: prior information about certain relations
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
            if prior is not None:
                # Apply prior knowledge to the mask initialization
                for i in range(num_nodes):
                    if prior[i] == 1:
                        mask.data[i] = const_val
            return mask

    
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
        self.masked_hor = self._masked_adj(self.sub_hor_graph)
        self.masked_ver = self._masked_adj(self.sub_ver_graph)
        if self.pyg_torch:
            ypred = self.model.forward(self.masked_hor.coalesce().indices(), self.masked_hor.coalesce().values())
        ypred = self.model.forward2(self.masked_hor, self.masked_ver)
        
        name = self.name
        node_pred = ypred[self.node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)

  
        return res, self.masked_hor, self.masked_ver

    def variance_loss(self, mask):
        std = torch.std(mask)
        mse = torch.mean((std - 1)**2)
        return mse
    
    def size_var_loss(self, mask):
        "Computes the variance of the mask and promuoves it"
        mask_without_small = mask[mask > 0.5]
        if len(mask_without_small) < len(mask):
            size_loss = 10*  torch.var(mask_without_small)
        else:    
            size_loss =  torch.var(mask_without_small)
        #size_loss_f = torch.mean((size_loss - 1)**2) 
        return size_loss


    def target_label(self,node_idx):
        pass

    


    # def loss_fc(self, pred, pred_label, node_idx, epoch, grid_coeff, size, size_num):
    #     """
    #     Args:
    #         pred: prediction made by current model
    #         pred_label: the label predicted by the original model.
    #     """
    #     budget = 1
    #     lambda_reg = 0.1
    #     mask = self.mask
    #     mask = torch.sigmoid(self.mask)  # sigmoid of the mask- because of the normal initialization of the mask some values are > 1 or < 0
    #     print(mask)
    #     self.num_high = len(mask[mask > 0.5])#len([i for i in mask if i > 0.5])

    #     #Prediction loss
    #     gt_label_node = self.label[node_idx]
    #     logit = pred[gt_label_node]
    #     pred_loss =  -torch.log(logit) * grid_coeff["pred"]

    #     #Size loss (as defined in the paper + budget)
    #     #size_loss_1 = -size * torch.relu((torch.sum(self.mask)- budget)  )  #size * torch.sum(mask)

    #     #Size loss to maximize the mask variance
    #     #size_loss_2 = size * self.size_var_loss(mask)
    #     mask_without_small = mask[mask > 0.5]
    #     if len(mask_without_small) < len(mask):
    #         size_loss_2 = 1000* size*  torch.var(mask_without_small)
    #     else:    
    #         size_loss_2 =  size *torch.var(mask_without_small)

    #     #Size loss using budget
    #     size_loss_3 = size * self.variance_loss(mask)

    #     #Size loss to minimize the number of edges
    #     print("num_high", self.num_high)
    #     size_num_loss = size_num * self.num_high/len(mask)
    #     #size_num_loss = size_num * (len(mask) - self.num_high / 2) ** 2
    #     #size_num_loss = - size_num_loss * num_high

    #     loss_reg = lambda_reg * torch.norm(mask, p=1)





    #     # entropy edge mask 
    #     mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    #     mask_ent_loss = grid_coeff["ent"] * torch.mean(mask_ent)
    #     # scale=0.99
    #     # mask = self.mask*(2*scale-1.0)+(1.0-scale)
    #     # mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    #     # mask_ent_loss = grid_coeff["ent"] * torch.mean(mask_ent)


    #     # laplacian loss
    #     # D = torch.diag(torch.sum(self.masked_adj[0], 0))
    #     # m_adj = self.masked_adj
    #     # L = D - m_adj

    #     # pred_label_t = torch.tensor(pred_label, dtype=torch.float)

    #     # lap_loss = (self.coeffs["lap"] * (pred_label_t @ L @ pred_label_t) / self.masked_adj())

    #     #loss = pred_loss + size_loss + size_num_loss + size_loss_2 #+ mask_ent_loss #+ lap_loss  # feat_mask_ent_loss 
    #     loss = pred_loss  +  size_loss_2 + size_loss_3 + size_num_loss + loss_reg 
    #     print('pred_loss', pred_loss)
    #     print('size_loss',  size_loss_2, size_loss_3)
    #     print('size_num_loss', size_num_loss)
    #     print('loss_reg', loss_reg)


    #     return loss
    
    def loss_fc(self, pred, pred_label, node_idx, epoch, grid_coeff, size, size_num):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """

        # prediction loss
        #pred_label_node = pred_label[node_idx]

        #pred_label_node = pred_label


        #print('label', self.label)
        
        gt_label_node = self.label[node_idx]
        print('gt_label_node', gt_label_node)
        print('gt_label_node', type(gt_label_node))
        logit = pred[gt_label_node]
        print('logit', logit)
        pred_loss =  -torch.log(logit) * grid_coeff["pred"]

        # size loss
        mask = self.mask
        # print('mask', mask)
        #print('gradient of the mask:', mask.grad)  # None at the beginning

        mask = torch.sigmoid(self.mask)  # sigmoid of the mask

        #size_loss = self.coeffs["size"] * torch.sum(mask)
        mask_without_small = mask[mask > 0.5]
        print('mask_without_small', mask_without_small)
        if len(mask_without_small) < len(mask):

            size_loss = 1000* size * torch.var(mask_without_small)
            loss_reg = 0.5 * torch.norm(mask_without_small, p=1)
            size_loss = size_loss + loss_reg
        else:    
            size_loss = size * torch.var(mask_without_small) + 0.1 * torch.norm(mask_without_small, p=1)

        #size_loss = self.size_loss_f(mask, self.coeffs)

        num_high = len([i for i in mask if i > 0.5])
        print('num_high', num_high)
        #size_num_loss = self.coeffs["size_num"] * num_high #(num_high - self.num_nodes / 2) ** 2
        #size_num_loss = size_num * (len(mask) - num_high / 2) ** 2
        size_num_loss = size_num * num_high/len(mask)





        # entropy edge mask 
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = grid_coeff["ent"] * torch.mean(mask_ent)
        


        # laplacian loss
        # D = torch.diag(torch.sum(self.masked_adj[0], 0))
        # m_adj = self.masked_adj
        # L = D - m_adj

        # pred_label_t = torch.tensor(pred_label, dtype=torch.float)

        # lap_loss = (self.coeffs["lap"] * (pred_label_t @ L @ pred_label_t) / self.masked_adj())
        if len(mask_without_small) <=3:
            loss = pred_loss
        else:
            loss = pred_loss + size_loss + size_num_loss# + mask_ent_loss #+ lap_loss  # feat_mask_ent_loss 
        print('pred_loss', pred_loss)
        print('size_loss', size_loss)
        print('size_num_loss', size_num_loss)


        return loss
    
    def criterion(self, epoch, grid_coeff, size, size_num):
        """
        Computes the loss of the current model
        """
        #prediction of explanation model
        pred, masked_hor, masked_ver = self.forward()


        self.new_node_idx = self.new_index()
        loss_val = self.loss_fc( pred, self.pred_label, self.new_node_idx, epoch, grid_coeff, size, size_num)
        #loss_val = loss_fc(self.masked_adj,  pred, self.pred_label,self.label, self.new_node_idx, epoch, print=False)

        return loss_val 
    

    def return_stuff(self):
        return self.neighbors,self.n_hops, self.node_idx



import torch.nn.functional as F

class SparseBinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self, lambda_reg):
        super(SparseBinaryCrossEntropyLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, input, target):
        # Compute binary cross-entropy loss
        loss_ce = F.binary_cross_entropy(input, target, reduction='mean')

        # Compute L1 regularization loss
        loss_reg = self.lambda_reg * torch.norm(input, p=1)

        # Combine the two loss terms
        loss = loss_ce + loss_reg

        return loss

#def main(node_idx, n_hops, threshold, train):
def main(n_hops, threshold, train,name,prune, grid_search=False):
    if name in ['aifb', 'mutag', 'bgs', 'am']:
        data = kg.load(name, torch=True, final=False)
    else:     
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

    d = {key.item(): data.withheld[:, 0][data.withheld[:, 1] == key].tolist() for key in torch.unique(data.withheld[:, 1])}
    high = []
    low = []
    relations = [data.i2rel[i][0] for i in range(len(data.i2rel))]
    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    relations = ['label', 'node_idx'] + relations
    df = pd.DataFrame(columns=relations)
    for target_label in range(len(d.keys())):
        for node_idx in d[target_label]:
            if train:
                print('train modality')


                hyperparameter_grid = {
                    "pred": 1,
                    "size": [-100,-10,-1],
                    "feat_size": 1.0,
                    "ent": 1,
                    "feat_ent": 0.1,
                    "grad": 1,
                    "lap": 1.0, 
                    "size_num": [0.005, 0.05, 0.5],
                    "lr": [0.05, 0.1, 0.5]
                }

                params = {
                    "pred": 1,
                    "size": -10,
                    "feat_size": 1.0,
                    "ent": 1,
                    "feat_ent": 0.1,
                    "grad": 1,
                    "lap": 1.0, 
                    "size_num": 1,
                    "lr": 0.5
                }
                if grid_search:
                    wandb.init(project='RGCNExpl-gridsearch', config=hyperparameter_grid)
                else:
                    wandb.init(project='RGCNExpl', config=params)  


                print('node_idx:', node_idx, 'belonging to ', target_label)

                config = wandb.config
                if grid_search:
                    for size, size_num, lr in product(
                    config.size, config.size_num, config.lr):
                        explainer = Explain(model = model, data = data, node_idx = node_idx, name = name, n_hops = n_hops)
                        optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)
                        print('start training')
                        model.eval()
                        explainer.train()
                            
                else:
                    size = config.size
                    size_num = config.size_num
                    lr = config.lr
                    explainer = Explain(model = model, data = data, node_idx = node_idx, name = name, n_hops = n_hops)
                    optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)
                    print('start training')
                    model.eval()
                    explainer.train()





        # Initialize Wandb



                    for epoch in tqdm(range(10)):
                        explainer.zero_grad()
                        optimizer.zero_grad()
                        ypred, masked_hor, masked_ver = explainer.forward()
                        print('masked_hor:', masked_hor)
                        ypred = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
                        loss = explainer.criterion(epoch,hyperparameter_grid , size, size_num)
                        neighbors,n_hops, node_idx = explainer.return_stuff()

                        



                        loss.backward()
                        optimizer.step()

                            
                        


                        if epoch % 2 == 0:

                            print(
                            "epoch: ",
                            epoch,
                            "; loss: ",
                            loss.item(),

                            "; pred: ",
                            ypred )
                        wandb.log({'loss': loss, 'size': size,
                    'size_num': size_num, 'lr': lr, "len mask > 0.5": len([i for i in masked_ver.coalesce().values() if i > 0.5])})    

                    if epoch ==49:
                        print('masked_ver', masked_ver)
                    if not os.path.exists(f'chk/{name}_chk/masked_adj'):
                            os.makedirs(f'chk/{name}_chk/masked_adj') 
                    torch.save(masked_ver, f'chk/{name}_chk/masked_adj/masked_ver{node_idx}')
                    torch.save(masked_hor, f'chk/{name}_chk/masked_adj/masked_hor{node_idx}')    
   
  
                

            else:
                masked_ver = torch.load(f'chk/{name}_chk/masked_adj/masked_ver{node_idx}')
                masked_hor = torch.load(f'chk/{name}_chk/masked_adj/masked_hor{node_idx}') 
                #h = visualize(node_idx, n_hops, data, masked_ver,threshold, name= name, result_weights=False, low_threshold=False)
        h = selected(masked_ver, threshold,data, low_threshold=False)
        print(f'most important relations {node_idx}: ', h)
        high.append(h)
        print('high:', high)


        # l =  visualize(node_idx, n_hops, data, masked_ver,threshold,name = name, result_weights=False, low_threshold=True)
        if len(h) < len(masked_ver.coalesce().values()):
            l = selected(masked_ver, threshold,data, low_threshold=True)
            print(f'least important relations: {node_idx}', l)
            low.append(l)
        # relations = [data.i2rel[i][0] for i in range(len(data.i2rel))]

        # relations = ['label', 'node_idx'] + relations
        #df = pd.DataFrame(columns=relations)
        h = dict(h)
        
        info = {'label': target_label, 'node_idx': str(node_idx)}
        h.update(info)
        df.loc[str(node_idx)] = h

    df.to_csv('RGCN_stuff/Relations_Important_all.csv', index=False)    



def main2(name, node_idx, n_hops, threshold, train, prune = True, pyg_torch = False, grid_search = False):
    #name = 'aifb'
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
    node_idx = d[0][1]


    if pyg_torch:
        model = torch.load(f'chk/{name}_chk/model_{name}_torch')
    else:
        model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')

    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    y_full = model.forward2(hor_graph, ver_graph)
    node_pred_full = y_full[node_idx, :]
    res_full = nn.Softmax(dim=0)(node_pred_full)    
    if train:
        print('train modality')


        hyperparameter_grid = {
            "pred": 1,
            "size": [-100,-10,-1],
            "feat_size": 1.0,
            "ent": 1,
            "feat_ent": 0.1,
            "grad": 1,
            "lap": 1.0, 
            "size_num": [0.005, 0.05, 0.5],
            "lr": [0.05, 0.1, 0.5]
        }

        params = {
            "pred": 1,
            "size": -10,
            "feat_size": 1.0,
            "ent": 1,
            "feat_ent": 0.1,
            "grad": 1,
            "lap": 1.0, 
            "size_num": 1,
            "lr": 0.5
        }
        if grid_search:
            wandb.init(project='RGCNExpl-gridsearch', config=hyperparameter_grid)
        else:
            wandb.init(project='RGCNExpl', config=params)    

        # Set the hyperparameters for this run based on the current sweep configuration
        config = wandb.config
        if grid_search:
            for size, size_num, lr in product(
            config.size, config.size_num, config.lr):
                explainer = Explain(model = model, data = data, node_idx = node_idx, name = name, n_hops = n_hops)
                optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)
                print('start training')
                model.eval()
                explainer.train()
                    
        else:
            size = config.size
            size_num = config.size_num
            lr = config.lr
            explainer = Explain(model = model, data = data, node_idx = node_idx, name = name, n_hops = n_hops)
            optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)
            print('start training')
            model.eval()
            explainer.train()





        # Initialize Wandb



            for epoch in tqdm(range(10)):
                explainer.zero_grad()
                optimizer.zero_grad()
                ypred, masked_hor, masked_ver = explainer.forward()
                print('masked_hor:', masked_hor)
                if pyg_torch:
                    ypred = nn.Softmax(dim=0)(model.forward(masked_hor.coalesce().indices(), masked_hor.coalesce().values())[node_idx, :])
                else:
                    ypred = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
                #criterion = SparseBinaryCrossEntropyLoss(lambda_reg=0.01)
                #loss = criterion(ypred, res_full)
                loss = explainer.criterion(epoch,hyperparameter_grid , size, size_num)
                neighbors,n_hops, node_idx = explainer.return_stuff()

                



                loss.backward()
                optimizer.step()
            


                if epoch % 2 == 0:

                    print(
                    "epoch: ",
                    epoch,
                    "; loss: ",
                    loss.item(),

                    "; pred: ",
                    ypred )
                wandb.log({'loss': loss, 'size': size,
               'size_num': size_num, 'lr': lr, "len mask > 0.5": len([i for i in masked_ver.coalesce().values() if i > 0.5])})    

            if epoch ==49:
                print('masked_ver', masked_ver)
            if not os.path.exists(f'chk/{name}_chk/masked_adj'):
                    os.makedirs(f'chk/{name}_chk/masked_adj') 
            torch.save(masked_ver, f'chk/{name}_chk/masked_adj/masked_ver{node_idx}')
            torch.save(masked_hor, f'chk/{name}_chk/masked_adj/masked_hor{node_idx}')    
   
  
                

    else:
        masked_ver = torch.load(f'chk/{name}_chk/masked_adj/masked_ver{node_idx}')
        masked_hor = torch.load(f'chk/{name}_chk/masked_adj/masked_hor{node_idx}') 
    #print('masked_ver', masked_ver)
    h = visualize(node_idx, n_hops, data, masked_ver,threshold, name = name, result_weights=False, low_threshold=False)
    h = selected(masked_ver, threshold,data, low_threshold=False)
    #print('most important relations: ', h)

    sel_masked_ver1 = sub_sparse_tensor(masked_ver, 0.5, data, low_threshold=False)
    sel_masked_hor1 = sub_sparse_tensor(masked_hor, 0.5, data, low_threshold=False)

    # l =  visualize(node_idx, n_hops, data, masked_ver,threshold, name = name, result_weights=False, low_threshold=True)

    # if len(h) < len(masked_ver.coalesce().values()):
    #     l = selected(masked_ver, threshold,data, low_threshold=True)
    #     print('least important relations: ', l)



    # l = selected(masked_ver, threshold,data, low_threshold=True)
    # print('least important relations: ', l)
    # low.append(l)



    #visualize(node_idx, n_hops, data, masked_ver,threshold, name = name,result_weights=True, low_threshold=False)
    relations = [data.i2rel[i][0] for i in range(len(data.i2rel))]
    relations = ['label', 'node_idx'] + relations
    df = pd.DataFrame(columns=relations)
    h = dict(h)
    info = {'label':0, 'node_idx': str(node_idx)}
    h.update(info)
    df.loc[str(node_idx)] = h

    df.to_csv(f'RGCN_stuff/Relations_Important_{name}_{node_idx}.csv', index=False)
    # dict_index = dict_index_classes(data,masked_ver)

    #CHECK RESULTS WITH SUB AND FULL GRAPH 
    #Explain prediction with subgraph
    # masked_hor =  torch.sparse_coo_tensor(masked_hor.indices(), torch.ones(masked_hor._nnz()), masked_hor.size(), requires_grad=True)
    # masked_ver =  torch.sparse_coo_tensor(masked_ver.indices(), torch.ones(masked_ver._nnz()), masked_ver.size(), requires_grad=True)
    #masked_ver.values = torch.ones(len(masked_ver.values))
    model.eval()
    res = nn.Softmax(dim=0)(model.forward2(masked_hor, masked_ver)[node_idx, :])
    print('ypred explain', res)
   
    # res_sel = nn.Softmax(dim=0)(model.forward2(sel_masked_hor1, sel_masked_ver1)[node_idx, :])
    # print('ypred explain sel', res_sel)




    #Prediciction with full graph 
    # hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    # y_full = model.forward2(hor_graph, ver_graph)
    # node_pred_full = y_full[node_idx, :]
    # res_full = nn.Softmax(dim=0)(node_pred_full)
    print('ypred full', res_full)






    print('original label',[k for k, v in d.items() if node_idx in v])
    mislabeled, mislabeled_exp = [], []
    correct, correct_exp  = [], []
    for i in range(data.withheld.shape[0]):
        if torch.load(f'chk/{name}_chk/prediction_{name}_prune_{prune}').argmax(dim=1)[i] != data.withheld[i][1]:
            mislabeled.append(data.withheld[i][0])
        else:
            correct.append(data.withheld[i][0])
        #if ypred.argmax(dim=1)[i] != data.withheld[i][1]:
        if res.argmax != data.withheld[i][1]:
            mislabeled_exp.append(data.withheld[i][0])
        else:
            correct_exp.append(data.withheld[i][0])
    print('num mislabel', len(mislabeled))
    print('num correct', len(correct))
    if node_idx in mislabeled:
        print(f'{node_idx} was mislabeled tho')
    # print('mislabeled', mislabeled)
    # print('correct', correct)c
    # print('mislabeled_exp', mislabeled_exp)
    # print('correct_exp', correct_exp)



if __name__ == "__main__":
    #main2(name = 'aifb', node_idx = 5757, n_hops = 0,threshold = 0.5, train= True)

    main(n_hops = 2,threshold = 0.5, train=True, name='aifb', prune = True)



    

    