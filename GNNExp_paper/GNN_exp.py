#THIS IS THE IMPLEMENTATION OF THE EXPLAINABLE GRAPH NEURAL NETWORKS 
#Following the structure of the source code: https://arxiv.org/pdf/1903.03894.pdf

#Main differences:
#1. The model is a GCN
#2. The model is trained on the Cora dataset - to make it easier to compare to PyG implementation
#3. Everything needed in one file 
#4. 


#Importing the libraries
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt



#Explain Module
import torch.nn as nn
import math

#to import the data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx



torch.set_printoptions(threshold=100)
from GNN_exp_utils import *




class Explainer:
    def __init__(self,
        model,
        adj,
        feat,
        label,
        pred,
        node_idx, 
        n_hops):
        self.model = model
        self.model.eval()
        self.feat = feat
        self.label = label
        self.pred = pred
        self.node_idx = node_idx
        self.adj = adj
        self.n_hops = n_hops #number layers to propagate (in the paper it is 2)
        self.neighborhoods = neighborhoods(adj=self.adj, n_hops=self.n_hops)

        # self.num_classes = num_classes
        # self.num_features = num_feature


    def extract_neighborhood(self):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = neighborhoods(self.adj,self.n_hops)[self.node_idx, :] #take row of the node in the new adj matrix
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:self.node_idx]) #sum of all the nodes before the query node (since they are 1 or 0) - it becomes count of nodes before the query node
        neighbors = np.nonzero(neighbors_adj_row)[0] #return the indices of the nodes that are connected to the query node (and thus are non zero)
        sub_adj = self.adj[neighbors][:, neighbors]
        sub_feat = self.feat[neighbors]
        sub_label = self.label[neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors    

    def explain(self, node_idx,epochs, model = "exp"):
        print("node label:", self.label[node_idx])
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood()
        print("neigh graph idx: ", self,node_idx, node_idx_new)

        sub_adj = np.expand_dims(sub_adj, axis=0) 
        sub_feat = np.expand_dims(sub_feat, axis=0)
        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        pred_label = torch.argmax(self.pred[neighbors], axis=1)
        print("Node predicted label: ", pred_label[node_idx_new])

        #Explainer model whose parameters are to be learned
        explainer = ExplainModule( 
            adj=adj,
            x=x,
            model=self.model,
            label=label
        )
        
        self.model.eval()
        explainer.train() #set the explainer model to training mode
        for epoch in range(100):
                explainer.zero_grad() #zero the gradient buffers of all parameters
                explainer.optimizer.zero_grad() 
                ypred, adj_atts = explainer(node_idx_new) # forward pass of the explainer
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch) # loss function

                loss.backward() 

                a = adj_atts.to_sparse()

                print('values: ', a.coalesce().values())
                #print('neighbors', len(neighbors))

                explainer.optimizer.step()
                mask_density = explainer.mask_density()
                print(epoch)
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
                single_subgraph_label = sub_label.squeeze() 
                print('epoch:', epoch)
                if epoch % 25== 0:
                    #Thos are the functions that log the results : we can omit them 
                    # explainer.log_mask(epoch)
                    # explainer.log_masked_adj(
                    #     node_idx_new, epoch, label=single_subgraph_label
                    # )
                    explainer.log_adj_grad(
                        node_idx_new, pred_label, epoch, label=single_subgraph_label
                    )
        print('Finished Training')

        #masked_adj = (explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze())

        adj_atts = torch.sigmoid(adj_atts).squeeze()
        masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()         
        torch.save(masked_adj, 'cora_chk/masked_adj')
        return masked_adj, neighbors, node_idx_new, adj_atts  




class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )
        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes) 
        self.optimizer = torch.optim.Adam(params, lr=0.001)

        self.coeffs = {
            "size": 0.005, #0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,}
        

        
    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        """
        Initialize feature mask parameter
        """
        torch.manual_seed(42)
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        """
        Construct edge mask

        """
        torch.manual_seed(42)
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)
        return mask

    def _masked_adj(self):
        "" "Mask the adjacency matrix with the learned mask" ""
        sym_mask = self.mask

        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj
        masked_adj = adj * sym_mask
        return masked_adj * self.diag_mask

    def mask_density(self):
        """Compute the density of the learned mask"""
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum
    

    def forward(self, node_idx, mask_features=True, marginalize=False):
        x = self.x
        print('node_idx', node_idx)
        print(type(node_idx))

        self.masked_adj = self._masked_adj() #masked adj is the adj matrix with the mask applied
        feat_mask = (torch.sigmoid(self.feat_mask)) #

        x = x * feat_mask
        #print(x.shape)
        #print(self.masked_adj.shape)
        ypred, adj_att = self.model(x.squeeze(), self.masked_adj.squeeze())
        #print(self.mask)
        node_pred = ypred[node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)
        print('adj here:', adj_att.to_sparse())
        torch.save(adj_att, 'cora_chk/adj_att')
        #return res, adj_att
        return res, self.masked_adj.squeeze() 
    

    def adj_feat_grad(self, node_idx, pred_label_node):
        """
        Compute the gradient of the prediction w.r.t. the adjacency matrix
        and the node features.
        """
        self.model.zero_grad() # zero out the gradient
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            #print('self.adj.grad', self.adj.grad)
            self.adj.grad.zero_() # zero out the gradient
            self.x.grad.zero_() # zero out the gradient
        else:    

            x, adj = self.x, self.adj
        x, adj = self.x, self.adj    
        #print('self.adj.grad', self.adj.grad)
        ypred, _ = self.model(x.squeeze(), adj.squeeze())

        logit = nn.Softmax(dim=0)(ypred[node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad
    
    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """

        # prediction loss
        pred_label_node = pred_label[node_idx]
        gt_label_node = self.label[node_idx]
        logit = pred[gt_label_node]
        pred_loss = -torch.log(logit)

        # size loss
        mask = self.mask
        #print('mask', mask)
        print('gradient of the mask:', mask.grad) #None at the beginning


        mask = torch.sigmoid(self.mask) #sigmoid of the mask
 
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # feature size loss
        feat_mask = (self.feat_mask)
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy edge mask 
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)


        # entropy feature mask
        feat_mask_ent = - feat_mask * torch.log(feat_mask) - (1 - feat_mask) * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)
        print('feat_mask_ent_loss:', feat_mask_ent)
        #TODO: feat mask ent loss is NONE !!!!

        # laplacian loss
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj
        L = D - m_adj

        pred_label_t = torch.tensor(pred_label, dtype=torch.float)

        lap_loss = (self.coeffs["lap"]* (pred_label_t @ L @ pred_label_t) / self.adj.numel())


        loss = pred_loss + size_loss + feat_size_loss  + mask_ent_loss + lap_loss #feat_mask_ent_loss 
        # print("optimization/size_loss", size_loss, epoch)
        # print("optimization/feat_size_loss", feat_size_loss, epoch)
        # print("optimization/mask_ent_loss", mask_ent_loss, epoch)
        # print(
        #     "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
        # )

        # print("optimization/pred_loss", pred_loss, epoch)
        # print("optimization/lap_loss", lap_loss, epoch)
        # print("optimization/overall_loss", loss, epoch)
        return loss
    


    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False


        predicted_label = pred_label[node_idx]
        #adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])
        adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label) #adj_grad is the gradient of the adj matrix

        if adj_grad is not None:
            adj_grad = torch.abs(adj_grad)
            x_grad = x_grad.squeeze()
            x_grad = x_grad[node_idx][:, np.newaxis]
            adj_grad = adj_grad.squeeze()  
            adj_grad = (adj_grad + adj_grad.t()) / 2
            adj_grad = (adj_grad * self.adj).squeeze()
        
            masked_adj = self.masked_adj[0].cpu().detach().numpy()
            
            adj_grad = adj_grad.detach().numpy()
            print('adj_grad is:' , adj_grad) 
            #TODO: adj_grad is none!!!!!!!! 

            G = denoise_graph(adj_grad, node_idx, threshold = 0.5)
        else:
            print('adj grad is none ')
            pass    


def main(node_idx, n_hops, epochs):
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]
    print('dataset under study')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')

    input_dim = dataset.num_features
    output_dim = dataset.num_classes


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(input_dim, output_dim).to(device)

    #model = Net(dataset)
    model = torch.load('cora_chk/model_cora') #load the trained model



    explainer = Explainer(model = model,
        adj = torch.load('cora_chk/adj_cora'), #load the adj matrix
        feat = torch.Tensor(data.x),
        label = torch.Tensor(data.y),
        pred = torch.load('cora_chk/prediction_cora'), #load the prediction
        node_idx = node_idx,
        n_hops = n_hops,)
    masked_adj, neighbors, node_idx_new, adj_att  = explainer.explain(node_idx, n_hops, epochs)
    visualize_result(node_idx, masked_adj, neighbors,data,n_hops)


    #Mislabeled and non of original model prediction
    mislabeled = []
    correct = []
    for i in range(1000):
        if torch.load('cora_chk/prediction_cora').argmax(dim=1)[i] != data.y[i]:
            mislabeled.append(i)
        else:
            correct.append(i)
    # print('mislabeled', mislabeled)    
    # print('correct',correct)
    torch.save(correct, 'cora_chk/correct')
    
    masked_adj = torch.Tensor(masked_adj)
    
    print('masked_adj', masked_adj.to_sparse())
    ypred = model(data.x, masked_adj)
    ypred_adj_att = model(data.x, torch.load('cora_chk/adj_att'))
    yp = model(data.x, torch.load('cora_chk/adj_cora'))
    print('ypred model', nn.Softmax(dim=0)(yp[0][node_idx, :]), torch.argmax(nn.Softmax(dim=0)(yp[0][node_idx, :])))
    print('ypred explainer', nn.Softmax(dim=0)(ypred[0][node_idx, :]), torch.argmax(nn.Softmax(dim=0)(ypred[0][node_idx, :])))
    print('ypred_adj_att', nn.Softmax(dim=0)(ypred_adj_att[0][node_idx, :]), torch.argmax(nn.Softmax(dim=0)(ypred_adj_att[0][node_idx, :])))
    print('original', data.y[node_idx])







if __name__ == '__main__':
    # correct = torch.load('cora_chk/correct')
    # for i in correct:
    #     node_idx = i

    main(node_idx = 0, n_hops=2, epochs=100)








        











