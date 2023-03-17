import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch


def dict_index_classes(data, masked_ver):
    indices_nodes = masked_ver.coalesce().indices().detach().numpy()

    d = list(data.e2i.keys())
    values_indices_nodes = [d[i] for i in indices_nodes[1]]
    dict = {}
    for i in range(len(values_indices_nodes)):
        try:
            dict[values_indices_nodes[i][0]] = str(values_indices_nodes[i][1]).split('/')[3]
            
        except IndexError :
            dict[values_indices_nodes[i][0]] = str(values_indices_nodes[i][1])
    return dict        
  
def dict_triples_semantics(data,  masked_ver, sub_triples):
    """ 
    Function to get a dictionary where the keys are the triples in indexes and values their semantic values
    Input:
        - Data
        - Sub Triples
        - Dict index
        
    """
    dict_index = dict_index_classes(data,masked_ver)
    indices_nodes = masked_ver.coalesce().indices().detach().numpy()
    new_ver = indices_nodes[0]%data.num_entities
    new_index = np.transpose(np.stack((new_ver, indices_nodes[1])))
    #sub_triples = match_to_triples(new_index, data.triples)
    dict = {}
    for i in range(len(sub_triples)):

        s = dict_index[int(sub_triples[:,0][i])]

        p = str(data.i2r[int(sub_triples[:,1][i])]).split('/')[3]
        o = dict_index[int(sub_triples[:,2][i])]
        dict[sub_triples[i]] = s + ' ' + p + ' ' + o 
    return dict 


#Visualize the result
def visualize_result(node_idx, masked_ver, neighbors, data, num_hops):
    """Visualizes the n-hop neighborhood of a given node."""
    indices_nodes = masked_ver.coalesce().indices().detach().numpy()
    new_ver = indices_nodes[0]%data.num_entities #get original indexes
    new_index = np.transpose(np.stack((new_ver, indices_nodes[1]))) #original edge indexes
    G = nx.from_edgelist(new_index)
    
    #create dict of index - node: to visualize index of the node
    labeldict = {}
    for node in G.nodes:
        labeldict[node] = node 
    print(G.nodes)
    dict_index = dict_index_classes(data,masked_ver)
    #order dict index according to G nodes in networkx
    ordered_dict = {}
    for item in list(G.nodes):
        ordered_dict[item] = dict_index[item]

    dict_index = ordered_dict
    

    #get inverse of dict to allow mapping of different 'classes' of nodes to different nodes
    inv_map = {v: k for k, v in dict_index.items()}
    print(inv_map) #use inv:map to get a legend of node colors for later 
    color_list = list(dict_index.values())
    
    #make a list out of it 
    for i in range(len(color_list)):
        if color_list[i] in inv_map:
            color_list[i] = inv_map[color_list[i]]   

            
    #edge colors reflect the masked ver values - more important relations have darker color
    #to check why we have relations than expexted :')
    edge_colors = list(masked_ver.coalesce().values().detach().numpy())[:int(G.number_of_edges())]

    # draw graph with edge colors
    plt.figure()  
    plt.title("Node {}'s {}-hop neighborhood important nodes".format(node_idx, num_hops))
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True, edge_color = edge_colors, edge_cmap=plt.cm.Reds,node_color =  color_list  ,labels = labeldict, cmap="Set2" )

    #add colorbar legend


    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array(edge_colors)
    cbar = plt.colorbar(sm)
    cbar.ax.set_title('Weight')

    plt.show()  

#Neighborhood
#Extract neighborhood

def find_n_hop_neighbors(edge_index, n, node=None):
    """ 
    edge_index 
    n = num hops
    node = node_idx
    """
    # create dictionary of node neighborhoods
    neighborhoods = {}
    for i in range(edge_index.max().item() + 1):
        neighborhoods[i] = set()

    # find 1-hop neighbors and corresponding edges
    edges = []
    for j in range(edge_index.shape[1]):
        src, dst = edge_index[0, j].item(), edge_index[1, j].item()
        neighborhoods[src].add(dst)
        neighborhoods[dst].add(src)
        edges.append((src, dst))

    # find n-hop neighbors for the specified node or all nodes

    for k in range(2, n+1):
        new_neighbors = set()
        for neighbor in neighborhoods[node]:
            new_neighbors.update(neighborhoods[neighbor])
        neighborhoods[node].update(new_neighbors)
    sub_edges = []
    for edge in edges:
        src, dst = edge
        if src in neighborhoods[node] and dst in neighborhoods[node] or src == node or dst == node:
            sub_edges.append(edge)
            
    sub_edges_tensor = torch.tensor([sub_edges[i] for i in range(len(sub_edges))]).t()        

    #return {node: sub_edges}, {node: neighborhoods[node]}, sub_edges_tensor
    return sub_edges, neighborhoods[node], sub_edges_tensor      

#match tro classes
def match_to_classes(tensor1, tensor2):
    """
    tensor1: sub graph indices
    tensor2: data.y labelsss
    """
    matching = []
    for i in (tensor1[:,0]):
        for j, index in zip(tensor2[:,0],range(len(tensor2[:,0]))):
            if i == j:
                matching.append(tensor2[index])
    return matching   


#match triples 
def match_to_triples(tensor1, tensor2):
    """
    tensor1: sub_edge tensor: edges of the neighborhood - transpose!!
    tensor2: data.triples: all edges
    """
    matching = []
    for i,i2 in zip(tensor1[:,0],tensor1[:,1]):
        for j,j1,j2, index in zip(tensor2[:,0],tensor2[:,1],  tensor2[:,2], range(len(tensor2[:,0]))):
            if i == j and i2 == j2:
                matching.append(tensor2[index])

    result = torch.stack(matching)
    return result

#edge index
def edge_index_oneadj(triples):
    edge_index = torch.stack((triples[:, 0], triples[:, 2]),dim=0)
    return edge_index


def visualize_data(node_idx, data, num_hops):
    """Visualizes the n-hop neighborhood of a given node."""
    edge_index = edge_index_oneadj(data.triples)
    sub_edges, neighborhoods, sub_edges_tensor = find_n_hop_neighbors(edge_index, num_hops, node_idx)

    G = nx.from_edgelist(sub_edges)
    
    #create dict of index - node: to visualize index of the node
    labeldict = {}
    for node in G.nodes:
        labeldict[node] = node 
    print(G.nodes)
    print(G.number_of_edges)




    #can get it through original triples 
    triples_matched = match_to_triples(np.array(sub_edges), data.triples)
    print(triples_matched)
    edge_colors = triples_matched[:,1].detach().numpy()
    print(edge_colors)
    colormap = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)), cmap = "rainbow")

    # Create a color map for each value in the list
    colors = [colormap.to_rgba(val) for val in edge_colors]
    #cmap=cm.rainbow(np.array(edge_colors))


    # draw graph with edge colors
    plt.figure()  
    plt.title("Node {}'s {}-hop neighborhood".format(node_idx, num_hops))
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True, edge_color = colors, edge_cmap=plt.cm.Reds,labels = labeldict, cmap="Set2" )



    plt.show()        