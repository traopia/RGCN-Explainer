import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

#for the model 
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


import matplotlib.pyplot as plt
#Explain Module
import torch.nn as nn
import math

#to import the data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

from GNN_explainer_easy import adj_feat_grad

def neighborhoods(adj, n_hops):
    """Returns the n_hops degree adjacency matrix adj."""

    adj = torch.tensor(adj, dtype=torch.float)
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        #print(type(hop_adj))
        hop_adj = (hop_adj > 0).float()
        #print(hop_adj)
    return hop_adj.cpu().numpy().astype(int)


def denoise_graph(adj, node_idx, feat=None, label=None, threshold=None, threshold_num=None, max_component=True):
    """Cleaning a graph by thresholding its node values.

    Args:
        - adj               :  Adjacency matrix.
        - node_idx          :  Index of node to highlight (TODO ?)
        - feat              :  An array of node features.
        - label             :  A list of node labels.
        - threshold         :  The weight threshold.
        - theshold_num      :  The maximum number of nodes to threshold.
        - max_component     :  TODO
    """
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.nodes[node_idx]["self"] = 1
    if feat is not None:
        for node in G.nodes():
            G.nodes[node]["feat"] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.nodes[node]["label"] = label[node]

    if threshold_num is not None:
        # this is for symmetric graphs: edges are repeated twice in adj
        adj_threshold_num = threshold_num * 2
        #adj += np.random.rand(adj.shape[0], adj.shape[1]) * 1e-4
        neigh_size = len(adj[adj > 0])
        threshold_num = min(neigh_size, adj_threshold_num)
        threshold = np.sort(adj[adj > 0])[-threshold_num]

    if threshold is not None:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] >= threshold
        ]
    else:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] > 1e-6
        ]
    G.add_weighted_edges_from(weighted_edge_list)
    if max_component:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    else:
        # remove zero degree nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    return G


def visualize_result(node_idx, masked_adj, neighbors, data, num_hops):
    """Visualizes the n-hop neighborhood of a given node."""
    G = nx.from_pandas_adjacency(pd.DataFrame(masked_adj))

    labeldict = {}
    for i,j in zip(range(len(neighbors)),neighbors):
        labeldict[i] = j
    #print(labeldict)    
    a = nx.get_edge_attributes(G,'weight')
    weights = {key : round(a[key], 3) for key in a}

    edge_colors = [masked_adj[u][v] for u, v in G.edges()]
    print(edge_colors)
    # draw graph with edge colors
    plt.figure()  
    plt.title("Node {}'s {}-hop neighborhood important nodes".format(node_idx, num_hops))
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Reds,labels = labeldict, node_color = data.y[neighbors], cmap="Set2" )
    nx.draw_networkx_edge_labels( G, pos,edge_labels=weights,font_size=8,font_color='red')

    # add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array(edge_colors)
    cbar = plt.colorbar(sm)
    cbar.ax.set_title('Weight')

    plt.show()  




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
        self.n_hops = n_hops #number layers to propagate
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

        #pred_label = torch.argmax(self.pred_label[neighbors], dim=1)
        #pred_label = np.argmax(self.pred[neighbors]) # predicted label of the neighborhood )
        pred_label = torch.argmax(self.pred[neighbors], dim=1)
        print("Node predicted label: ", pred_label[node_idx_new])

        
        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label
        )
        
        self.model.eval()
        explainer.train()
        for epoch in range(epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts = explainer(node_idx_new) # forward pass of the explainer
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch) # loss function
                loss.backward() 

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
                # if epoch % 25== 0:
                # #         # explainer.log_mask(epoch)
                # #         # explainer.log_masked_adj(
                # #         #     node_idx_new, epoch, label=single_subgraph_label
                # #         # )
                #     explainer.log_adj_grad(
                #         node_idx_new, pred_label, epoch, label=single_subgraph_label
                #     )
        print('Finished Training')
        if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
        else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()         
        print(masked_adj)
        torch.save(masked_adj, 'cora_chk/masked_adj')
        return masked_adj, neighbors, node_idx_new  




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
        self.optimizer = torch.optim.Adam(params, lr=0.01)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,}
        

        
    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        """
        Construct feature mask
        """
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
        sym_mask = self.mask

        sym_mask = torch.sigmoid(self.mask)

        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj
        masked_adj = adj * sym_mask
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum
    

    def forward(self, node_idx, mask_features=True, marginalize=False):
        x = self.x
        print('node_idx', node_idx)
        print(type(node_idx))
        print('x', x)

        self.masked_adj = self._masked_adj()
        #feat_mask = (torch.sigmoid(self.feat_mask))

        #x = x * feat_mask
        print(x.shape)
        print(self.masked_adj.shape)
        ypred, adj_att = self.model(x.squeeze(), self.masked_adj.squeeze())

        node_pred = ypred[node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att
    

    def adj_feat_grad(self, node_idx, pred_label_node):
        """
        Compute the gradient of the prediction w.r.t. the adjacency matrix
        and the node features.
        """
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            print('self.adj.grad', self.adj.grad)
            self.adj.grad.zero_() # zero out the gradient
            self.x.grad.zero_() # zero out the gradient
        #else:    
        x, adj = self.x, self.adj
        print('self.adj.grad', self.adj.grad)
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

        pred_label_node = pred_label[node_idx]
        gt_label_node = self.label[node_idx]
        logit = pred[gt_label_node]
        pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        print('mask shape', mask.shape)
        print('pred shape', pred.shape)

        mask = torch.sigmoid(self.mask)

        size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (self.feat_mask)
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj
        L = D - m_adj
        print('L shape', L.shape)
        print()
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)

        lap_loss = (self.coeffs["lap"]
            * (pred_label_t @ L @ pred_label_t)
            / self.adj.numel()
        )


        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        print("optimization/size_loss", size_loss, epoch)
        print("optimization/feat_size_loss", feat_size_loss, epoch)
        print("optimization/mask_ent_loss", mask_ent_loss, epoch)
        print(
            "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
        )

        print("optimization/pred_loss", pred_loss, epoch)
        print("optimization/lap_loss", lap_loss, epoch)
        print("optimization/overall_loss", loss, epoch)
        return loss
    


    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False


        predicted_label = pred_label[node_idx]
        #adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])
        adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
        print(adj_grad)
        if adj_grad is not None:
            adj_grad = torch.abs(adj_grad)
            x_grad = x_grad.squeeze()
            x_grad = x_grad[node_idx][:, np.newaxis]
                # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
            print(adj_grad.shape)  
            adj_grad = adj_grad.squeeze()  
            adj_grad = (adj_grad + adj_grad.t()) / 2
            adj_grad = (adj_grad * self.adj).squeeze()
        
            masked_adj = self.masked_adj[0].cpu().detach().numpy()

            # only for graph mode since many node neighborhoods for syn tasks are relatively large for
            # visualization
            
            adj_grad = adj_grad.detach().numpy()

                # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            #G = denoise_graph(adj_grad, node_idx, threshold_num=12)
        else:
            pass    

class Net(torch.nn.Module):
    def __init__(self,dataset):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, adj):
        edge_index = adj.nonzero().t()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), adj

def main(node_idx, n_hops, epochs):
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]
    print('dataset under study')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')

    model = Net(dataset)
    model = torch.load('cora_chk/model_cora')

    explainer = Explainer(model = model,
        adj = torch.load('cora_chk/adj_cora'),
        feat = torch.Tensor(data.x),
        label = torch.Tensor(data.y),
        pred = torch.load('cora_chk/prediction_cora'),
        node_idx = node_idx,
        n_hops = n_hops,)
    masked_adj, neighbors, node_idx_new  = explainer.explain(node_idx, n_hops, epochs)
    visualize_result(node_idx, masked_adj, neighbors,data,n_hops)





if __name__ == '__main__':
    main(node_idx = 1, n_hops=2, epochs=500)






        











