#UTILS FOR GNN_exp
import torch 
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

#for the model 
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

#Extract neighborhood of node of interest
def neighborhoods(adj, n_hops):
    """Returns the n_hops degree adjacency matrix adj."""

    adj = torch.tensor(adj, dtype=torch.float)
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj

        hop_adj = (hop_adj > 0).float()

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