#Here we import the necessary libraries
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

def extract_neighborhood(node_idx,adj,feat,label,n_hops):
    """Returns the neighborhood of a given ndoe."""
    neighbors_adj_row = neighborhoods(adj,n_hops)[node_idx, :] #take row of the node in the new adj matrix
    # index of the query node in the new adj
    node_idx_new = sum(neighbors_adj_row[:node_idx]) #sum of all the nodes before the query node (since they are 1 or 0) - it becomes count of nodes before the query node
    neighbors = np.nonzero(neighbors_adj_row)[0] #return the indices of the nodes that are connected to the query node (and thus are non zero)
    sub_adj = adj[neighbors][:, neighbors]
    sub_feat = feat[neighbors]
    sub_label = label[neighbors]
    return node_idx_new, sub_adj, sub_feat, sub_label, neighbors


def visualize_hop_neighbor_subgraph(node_idx, data, neighbors, n_hops,adj,feat,label):
    """Visualizes the n-hop neighborhood of a given node."""
    node_idx_new, sub_adj, sub_feat, sub_label, neighbors = extract_neighborhood(node_idx,adj,feat,label,n_hops)
    subdata = data.subgraph(torch.tensor(neighbors))
    subindex = subdata.edge_index
    Gsub = G = to_networkx(subdata, to_undirected=False)
    labeldict = {}
    for i,j in zip(range(len(neighbors)),neighbors):
        labeldict[i] = j
    nx.draw(Gsub, labels = labeldict, node_color = data.y[neighbors], cmap="Set2")
    return subdata, subindex

def get_adjacency(data):
    adj = torch.zeros(data.num_nodes, data.num_nodes)
    for edge in data.edge_index.t():
        adj[edge[0]][edge[1]] = 1
    return adj

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
    
def construct_edge_mask( num_nodes, init_strategy="normal", const_val=1.0):
    """
    Construct edge mask
    input;
        num_nodes: number of nodes in the neighborhood
        init_strategy: initialization strategy for the mask
        const_val: constant value for the mask
    output:
        mask: edge mask    
    """
    mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))  #initialize the mask
    if init_strategy == "normal":
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (num_nodes + num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
    elif init_strategy == "const":
        nn.init.constant_(mask, const_val)
    return mask    

def construct_feat_mask( feat_dim, init_strategy="normal"):
    """
    Construct feature mask
    input:
        feat_dim: dimension of the feature
        init_strategy: initialization strategy
    output:
        mask: feature mask    
    """
    mask = nn.Parameter(torch.FloatTensor(feat_dim))
    if init_strategy == "normal":
        std = 0.1
        with torch.no_grad():
            mask.normal_(1.0, std)
    elif init_strategy == "constant":
        with torch.no_grad():
            nn.init.constant_(mask, 0.0)
            # mask[0] = 2
    return mask

def _masked_adj(mask,adj, diag_mask):
    """ Masked adjacency matrix 
    input: edge_mask, sub_adj, diag_mask
    output: masked_adj
    """
    sym_mask = mask
    sym_mask = torch.sigmoid(mask)
    
    sym_mask = (sym_mask + sym_mask.t()) / 2
    adj = torch.tensor(adj)
    masked_adj = adj * sym_mask

    return masked_adj * diag_mask

def loss_fc(edge_mask, feat_mask, masked_adj,adj, pred, pred_label,label, node_idx, epoch, print=False):
    """
    Args:
        pred: y_e :  prediction made by current model
        pred_label: y_hat : the label predicted by the original model.
    """
    #PRED LOSS
    pred_label_node = pred_label[node_idx] #pred label is the prediction made by the original model
    gt_label_node = label[node_idx]

    logit = pred[gt_label_node] #pred is the prediction made by the current model

    pred_loss = -torch.log(logit) #this is basically taking the cross entropy loss

    # MASK SIZE EDGE LOSS
    
    mask = edge_mask
    mask = torch.sigmoid(mask)

    size_loss = 0.005 * torch.sum(mask)

    
    #MASK SIZE FEATURE LOSS
    feat_mask = (torch.sigmoid(feat_mask))
    feat_size_loss = 1.0 * torch.mean(feat_mask)

    # EDGE MASK ENTROPY LOSS
    mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    mask_ent_loss = 1.0 * torch.mean(mask_ent)
    
    # FEATURE MASK ENTROPY LOSS
    feat_mask_ent = - feat_mask * torch.log(feat_mask) - (1 - feat_mask) * torch.log(1 - feat_mask)

    feat_mask_ent_loss = 0.1  * torch.mean(feat_mask_ent)

    # LAPLACIAN LOSS
    D = torch.diag(torch.sum(masked_adj, 0))
    m_adj = masked_adj 
    L = D - m_adj

    pred_label_t = torch.tensor(pred_label, dtype=torch.float)


    lap_loss = ( 1.0
        * (pred_label_t @ L @ pred_label_t)
        / torch.Tensor(adj).numel())


    loss = pred_loss + size_loss  + mask_ent_loss + feat_size_loss + lap_loss
    if print== True:
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

def construct_diag_mask(neighbors):
    """
    Args:
        adj: adjacency matrix of the graph
    """
    num_nodes = len(neighbors)
    diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes) 

    return diag_mask

def mask_density(edge_mask):
    """
    Args:
        mask: edge mask
    """
    mask = torch.sigmoid(edge_mask)
    return torch.sum(mask) / torch.Tensor(mask).numel()


def adj_feat_grad(node_idx, pred_label_node, model , adj,x, ):
    """
    Compute the gradient of the prediction w.r.t. the adjacency matrix
    and the node features.
    """
    model.zero_grad()
    adj.requires_grad = True
    x.requires_grad = True
    print(adj)
    if adj.grad is not None:
        adj.grad.zero_() # zero out the gradient
        x.grad.zero_() # zero out the gradient

    x, adj = x, adj
    ypred, _ = model(x, adj)

    logit = nn.Softmax(dim=0)(ypred[ node_idx, :])
    logit = logit[pred_label_node]
    loss = -torch.log(logit)
    loss.backward()
    return adj.grad, x.grad

def log_adj_grad(adj, masked_adj, node_idx, pred_label,pred_label_node, x, epoch, label=None):
    """ 
    Computes the gradient of the adjacency matrix with respect to the loss
    
    """
    log_adj = False

    predicted_label = pred_label
    # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
    adj_grad, x_grad = adj_feat_grad(node_idx, pred_label_node, model , adj,x)
    adj_grad = torch.abs(adj_grad)
    x_grad = x_grad[node_idx][:, np.newaxis]
        # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
    adj_grad = (adj_grad + adj_grad.t()) / 2
    adj_grad = (adj_grad * adj).squeeze()


    masked_adj = masked_adj[0].cpu().detach().numpy()
    return masked_adj

def adj_feat_grad(node_idx, pred_label_node, model , adj,x, ):
    """
    Compute the gradient of the prediction w.r.t. the adjacency matrix
    and the node features.
    """
    model.zero_grad()
    adj.requires_grad = True
    x.requires_grad = True
    print(adj)
    if adj.grad is not None:
        adj.grad.zero_() # zero out the gradient
        x.grad.zero_() # zero out the gradient

    x, adj = x, adj
    ypred, _ = model(x, adj)

    logit = nn.Softmax(dim=0)(ypred[ node_idx, :])
    logit = logit[pred_label_node]
    loss = -torch.log(logit)
    loss.backward()
    return adj.grad, x.grad


def visualize_result(node_idx, masked_adj, neighbors, data, num_hops):
    """Visualizes the n-hop neighborhood of a given node."""
    G = nx.from_pandas_adjacency(pd.DataFrame(masked_adj.detach().numpy()))

    labeldict = {}
    for i,j in zip(range(len(neighbors)),neighbors):
        labeldict[i] = j
    #print(labeldict)    
    a = nx.get_edge_attributes(G,'weight')
    weights = {key : round(a[key], 3) for key in a}

    edge_colors = [masked_adj.detach().numpy()[u][v] for u, v in G.edges()]
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

class Explain(nn.Module):
    def __init__(self, model, data, node_idx, n_hops):
        super(Explain, self).__init__()
        #Those are the parameters of the original data and model
        self.model = model
        self.data = data
        self.node_idx = node_idx
        self.n_hops = n_hops
        self.adj = get_adjacency(data)
        self.label = torch.Tensor(data.y)
        self.feat = torch.Tensor(data.x)
        self.feat_dim = data.num_features
        self.epoch = 1
        self.x = data.x

        self.pred_label = torch.load('cora_chk/prediction_cora')

        self.node_idx_new, self.sub_adj, self.sub_feat, self.sub_label, self.neighbors = extract_neighborhood(self.node_idx, self.adj, self.feat, self.label, self.n_hops)
        self.num_nodes = len(self.neighbors)
        self.diag_mask = construct_diag_mask(self.neighbors)
        self.subdata = torch.Tensor(data.subgraph(torch.tensor(self.neighbors)).x)
        self.edge_mask = construct_edge_mask(self.num_nodes)
        self.feat_mask = construct_feat_mask(self.feat_dim, init_strategy="normal")



    
    def _masked_adj(self):
        """ Masked adjacency matrix 
        input: edge_mask, sub_adj, diag_mask
        output: masked_adj
        """
        sym_mask = self.edge_mask
        sym_mask = torch.sigmoid(sym_mask)
        
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = torch.tensor(self.sub_adj)
        masked_adj = adj * sym_mask

        return masked_adj * self.diag_mask
    
        
        
    def forward(self):
        """
        Returns:
            ypred: prediction of the query node made by the current model (on the subgraph)

        """

        self.masked_adj = self._masked_adj()
        feat_mask = (torch.sigmoid(self.feat_mask))
        x = self.sub_feat * feat_mask
        #ypred, adj_att = model(self.subdata, masked_adj)
        ypred, adj_att = self.model(x, self.masked_adj)
        node_pred = ypred[self.node_idx_new, :]
        res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att, self.sub_adj
    
    def criterion(self, epoch):
        """
        Computes the loss of the current model
        """
        #prediction of explanation model
        pred, adj_e, sub_adj = self.forward()

        #prediction of original model
        pred_label = torch.argmax(self.pred_label[self.neighbors], dim=1)


        loss_val = loss_fc(self.edge_mask, self.feat_mask, self.masked_adj,self.adj, pred, pred_label, self.label,self.node_idx, self.epoch)

        return loss_val 
    
    def mask_density(self):
        """
        Computes the density of the edge mask
        """
        mask_sum = torch.sum(self.masked_adj)
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum
    
    def return_stuff(self):
        pred_label = torch.argmax(self.pred_label[self.neighbors], dim=1)
        return pred_label[self.node_idx], self.label[self.node_idx], self.neighbors, self.sub_label, self.sub_feat, self.n_hops, self.node_idx_new
    
    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        """ 
        Computes the gradient of the adjacency matrix with respect to the loss
        
        """
        log_adj = False

        predicted_label = pred_label
        # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
        adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
        adj_grad = torch.abs(adj_grad)
        x_grad = x_grad[node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()


        masked_adj = self.masked_adj[0].cpu().detach().numpy()


    

        adj_grad = adj_grad.detach().numpy()
        print('adj_grad', adj_grad)
        #G = denoise_graph(adj_grad, node_idx, threshold_num=12)
        # io_utils.log_graph(
        #     self.writer, G, name="grad/graph", epoch=epoch, args=self.args
        # )

        return adj_grad

    def adj_feat_grad(self, node_idx, pred_label_node):
        """
        Compute the gradient of the prediction w.r.t. the adjacency matrix
        and the node features.
        """
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True


        x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)


        logit = nn.Softmax(dim=0)(ypred[ node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        print(self.adj.grad)
        print(self.x.grad)
        return self.adj, self.x.grad    
    

        



def main(node_idx, n_hops):
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]
    print('dataset under study')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')


    explainer = Explain(model = torch.load('cora_chk/model_cora'), data = data, node_idx= node_idx, n_hops= n_hops)
    optimizer = torch.optim.Adam(explainer.parameters(), lr=0.01)
    
    explainer.train()
    for epoch in range(100):
        explainer.zero_grad()
        optimizer.zero_grad()
        ypred, adj_atts, sub_adj = explainer.forward()
        loss = explainer.criterion(epoch)
        pred_label, original_label, neighbors, sub_label, sub_feat, num_hops , node_idx_new = explainer.return_stuff()
        



        loss.backward(retain_graph=True)
        optimizer.step()
        mask_density = explainer.mask_density()
        single_subgraph_label = sub_label.squeeze()
        if epoch % 25 == 0:
        #     explainer.log_mask(epoch)
        #     explainer.log_masked_adj(
        #         node_idx_new, epoch, label=single_subgraph_label
        #     )


            explainer.log_adj_grad(
                node_idx_new, pred_label, epoch, label=single_subgraph_label)



        if epoch % 10 == 0:

            print(
            "epoch: ",
            epoch,
            "; loss: ",
            loss.item(),
            "; mask density: ",
            mask_density.item(),
            "; pred: ",
            ypred,
            "; labels equal: ",
            torch.argmax(ypred) == original_label== pred_label,

        )


    adj_atts = torch.sigmoid(adj_atts).squeeze()
    masked_adj = adj_atts * sub_adj.squeeze()
    print(masked_adj)

    visualize_result(1, masked_adj, neighbors,data,num_hops)


if __name__ == '__main__':
    main(node_idx = 2, n_hops = 2)