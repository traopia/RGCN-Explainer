import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from collections import Counter
from kgbench import load, tic, toc, d, Data
import os

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
    all_p = []
    for i in range(len(sub_triples)):

        s = dict_index[int(sub_triples[:,0][i])]
        p = data.i2rel[int(sub_triples[:,1][i])][0]
        o = dict_index[int(sub_triples[:,2][i])]
        all_p.append(p)
        dict[sub_triples[i]] = s + ' ' + p + ' ' + o 
    print(set(all_p))    
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
    #print(G.nodes)
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
# def match_to_triples(tensor1, tensor2):
#     """
#     tensor1: sub_edge tensor: edges of the neighborhood - transpose!!
#     tensor2: data.triples: all edges
#     """
#     matching = []
#     for i,i2 in zip(tensor1[:,0],tensor1[:,1]):
#         for j,j1,j2, index in zip(tensor2[:,0],tensor2[:,1],  tensor2[:,2], range(len(tensor2[:,0]))):
#             if i == j and i2 == j2:
#                 matching.append(tensor2[index])
                

#     result = torch.stack(matching)
#     return result

# def match_to_triples(v, data):
#     p,_ = v.coalesce().indices()//data.num_entities
#     s,o = v.coalesce().indices()%data.num_entities
#     result = torch.stack([s,p,o], dim=1)
#     return result



def match_to_triples(v, data, sparse=True):
    if sparse:
        # p,_ = torch.div(v.coalesce().indices(), data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
        # s,o = v.coalesce().indices()%data.num_entities
        # result = torch.stack([s,p,o], dim=1)
        matching = []
        for i,i2 in zip(v[:,0],v[:,1]):
            for j,j1,j2, index in zip(data[:,0],data[:,1],  data[:,2], range(len(data[:,0]))):
                if i == j and i2 == j2:
                    matching.append(data[index])
        result = torch.stack(matching)
                    
    else:
        matching = []
        for i,i2 in zip(v[:,0],v[:,1]):
            for j,j1,j2, index in zip(data[:,0],data[:,1],  data[:,2], range(len(data[:,0]))):
                if i == j and i2 == j2:
                    matching.append(data[index])
                    

        result = torch.stack(matching)
    
    return result

#edge index
def edge_index_oneadj(triples):
    edge_index = torch.stack((triples[:, 0], triples[:, 2]),dim=0)
    return edge_index


def sub_sparse_tensor(sparse_tensor, threshold, data, low_threshold=False):
    if low_threshold:
        nonzero_indices = sparse_tensor.coalesce().indices()[:, sparse_tensor.coalesce().values() < threshold]
        nonzero_indices[0] = nonzero_indices[0]%data.num_entities
        nonzero_values = sparse_tensor.coalesce().values()[sparse_tensor.coalesce().values() < threshold]
        sel_masked_ver = torch.sparse_coo_tensor(nonzero_indices, nonzero_values)
    else:
        nonzero_indices = sparse_tensor.coalesce().indices()[:, sparse_tensor.coalesce().values() > threshold]
        nonzero_indices[0] = nonzero_indices[0]%data.num_entities
        nonzero_values = sparse_tensor.coalesce().values()[sparse_tensor.coalesce().values() > threshold]
        sel_masked_ver = torch.sparse_coo_tensor(nonzero_indices, nonzero_values)    
    return sel_masked_ver



def sel_masked(sparse_tensor, threshold, data):
    nonzero_indices = sparse_tensor.coalesce().indices()[:, sparse_tensor.coalesce().values() > threshold]
    nonzero_values = sparse_tensor.coalesce().values()[sparse_tensor.coalesce().values() > threshold]
    sel_masked_ver = torch.sparse_coo_tensor(nonzero_indices, nonzero_values, size =  (data.num_entities, data.num_entities*(2*data.num_relations+data.num_relations)))
    return sel_masked_ver

def encode_classes(dict_index):
    d = []
    for k,v in dict_index.items():
        d.append(v)
    a = np.unique(d)
    dict = {}
    for i,j in zip(a, range(len(a))):
        dict[i] = j
        
    return dict


def encode_dict(dict_index):
    encoded_dict = {}
    dict = encode_classes(dict_index)
    for k,v in dict_index.items():
        for k1,v1 in dict.items():
            if v==k1:
                encoded_dict[k] = v1
    return encoded_dict


def selected(masked_ver, threshold,data, low_threshold, float=False):
    sel_masked_ver = sub_sparse_tensor(masked_ver, threshold,data, low_threshold)
    indices_nodes = sel_masked_ver.coalesce().indices().detach().numpy()
    new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) 
    triples_matched = match_to_triples(np.array(new_index), data.triples)
    #triples_matched = match_to_triples(sel_masked_ver, data)
    #print(triples_matched)
    if float:
            l = {}
            for i,j in zip(triples_matched[:,1],sel_masked_ver.coalesce().values()):

                if data.i2rel[int(i)][0] in l.keys():
                    l[data.i2rel[int(i)][0]] += np.float(j)
                else:
                    l[data.i2rel[int(i)][0]] = np.float(j)
            return l

    else:
        l = []
        for i in triples_matched[:,1]:
            l.append(data.i2rel[int(i)][0])

        return Counter(l)


def selected_float(masked_ver, threshold,data, low_threshold):
    ''' 
    masked_ver: masked adjacency matrix
    threshold: threshold for the masked adjacency matrix
    data: dataset
    low_threshold: if True, the threshold is applied to the masked adjacency matrix, otherwise to the original adjacency matrix
    returns: a dictionary with the sum of the mask for each relation'''
    sel_masked_ver = sub_sparse_tensor(masked_ver, threshold,data, low_threshold)
    sel_masked_ver = masked_ver
    indices_nodes = sel_masked_ver.coalesce().indices().detach().numpy()
    new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) 
    triples_matched = match_to_triples(np.array(new_index), data.triples)

    l = {}
    for i,j in zip(triples_matched[:,1],sel_masked_ver.coalesce().values()):

        if data.i2rel[int(i)][0] in l.keys():
            l[data.i2rel[int(i)][0]] += j
        else:
            l[data.i2rel[int(i)][0]] = j
   

    return Counter(l)
    

def visualize(node_idx, n_hop, data, masked_ver,threshold,name, result_weights=True, low_threshold=False ):
    """ 
    Visualize important nodes for node idx prediction
    """
    dict_index = dict_index_classes(data,masked_ver)
    
    #select only nodes with a certain threshold
    sel_masked_ver = sub_sparse_tensor(masked_ver, threshold,data, low_threshold)
    print('sel masked ver',sel_masked_ver)
    indices_nodes = sel_masked_ver.coalesce().indices().detach().numpy()
    new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) #original edge indexes

    
    
    G = nx.Graph()
    if result_weights:
        values = sel_masked_ver.coalesce().values().detach().numpy()
        for s,p,o in zip(indices_nodes[0],values , indices_nodes[1]):
            G.add_edge(int(s), int(o), weight=np.round(p, 2))

    else:
        #get triples to get relations 
        triples_matched = match_to_triples(np.array(new_index), data.triples)
        #triples_matched = match_to_triples(sel_masked_ver, data)
        l = []
        for i in triples_matched[:,1]:
            l.append(data.i2rel[int(i)][0])
        print(Counter(l))
        for s,p,o in triples_matched:
            G.add_edge(int(s), int(o), weight=int(p))

    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())


    pos = nx.circular_layout(G)

    ordered_dict = {}
    for item in list(G.nodes):
        if item in ordered_dict:
            ordered_dict[item].append(dict_index[item])
        # else:
        #     ordered_dict[item] =  dict_index[item]

    dict_index = ordered_dict

    labeldict = {}
    for node in G.nodes:
        labeldict[int(node)] = int(node)  

    print('dict index:', dict_index)

    dict = {}
    for k,v in dict_index.items():
        for k1,v1 in data.entities_classes.items():
            if v==k1: 

                dict[k] = v1
            else:
                if k not in dict:
                    dict[k] = 0
                

    color_list = list(dict.values())
    color_list = list(encode_dict(dict_index).values())


    
    if result_weights:
        
        nx.draw(G, pos,labels = labeldict,  edgelist=edges, edge_color=weights, node_color =  color_list, cmap="Set2",edge_cmap=plt.cm.Reds,font_size=8)
        nx.draw_networkx_edge_labels( G, pos,edge_labels=nx.get_edge_attributes(G,'weight'),font_size=8,font_color='red')
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        # sm.set_array(weights)
        # cbar = plt.colorbar(sm)
        # cbar.ax.set_title('Weight')
        plt.title("Node {}'s {}-hop neighborhood important nodes".format(node_idx, n_hop))
    else:
        rel = nx.get_edge_attributes(G,'weight')
        for k,v in rel.items():
            rel[k] = data.i2rel[v][0]
        nx.draw(G, pos,labels = labeldict,  edgelist=edges, edge_color=weights,node_color =  color_list, cmap="Set2",font_size=8)
        nx.draw_networkx_edge_labels( G, pos,edge_labels=rel,font_size=8,font_color='red')
        res = Counter(rel.values())
    if result_weights:
        if not os.path.exists(f'chk/{name}_chk/graphs'):
            os.makedirs(f'chk/{name}_chk/graphs')  
        plt.savefig(f'chk/{name}_chk/graphs/Explanation_{node_idx}_{n_hop}_weights.png')
        plt.show()
    else:
        if not os.path.exists(f'chk/{name}_chk/graphs'):
            os.makedirs(f'chk/{name}_chk/graphs')  
        plt.savefig(f'chk/{name}_chk/graphs/Explanation_{node_idx}_{n_hop}_relations.png')    
        plt.show()
        return res
    

def get_relations(data):
    all_relations = []
    for i in range(data.num_relations):
        rel = str(data.i2r[i]).split('/')[-1]
        if '#' in rel:
            rel = rel.split('#')[1]
        all_relations.append(rel)
    dict = {}
    for i in range(len(all_relations)):
        dict[i] = [all_relations[i]]
        dict[i].append(data.i2r[i])

    print(all_relations)

    data.i2rel = dict
    return data.i2rel       


def d_classes(data):
    """ 
    Get classes of nodes (select only the alphanum - not literals)
    """
    indices_nodes = data.entities
    d = list(data.e2i.keys())
    values_indices_nodes = [d[i] for i in indices_nodes]
    dict = {}
    for i in range(len(values_indices_nodes)):
        try:
            dict[values_indices_nodes[i][0]] = str(values_indices_nodes[i]).split('/')[3]
            
        except IndexError :
            dict[values_indices_nodes[i][0]] = str(values_indices_nodes[i])

    a = encode_classes(dict)   
    d = {}

    c = 0
    for k in a.keys():
        if k.isalpha():
            d[k] = c
            c+=1
    data.entities_classes = d
    return d    





def prunee(data , n=2):
    """
    Prune a given dataset. That is, reduce the number of triples to an n-hop neighborhood around the labeled nodes. This
    can save a lot of memory if the model being used is known to look only to a certain depth in the graph.

    Note that switching between non-final and final mode will result in different pruned graphs.

    :param data:
    :return:
    """

    data_triples = data.triples
    data_training = data.training
    data_withheld = data.withheld

    if data.torch:
        data_triples = data_triples.numpy()
        data_training = data_training.numpy()
        data_withheld = data_withheld.numpy()

    assert n >= 1

    entities = set()

    for e in data_training[:, 0]:
        entities.add(e)
    for e in data_withheld[:, 0]:
        entities.add(e)

    entities_add = set()
    for _ in range(n):
        for s, p, o in data_triples:
            if s in entities:
                entities_add.add(o)
            if o in entities:
                entities_add.add(s)
        entities.update(entities_add)
    #print(entities)
    # new index to old index
    n2o = list(entities)
    o2n = {o: n for n, o in enumerate(entities)}

    nw = Data(dir=None)
    nw.num_entities_new = len(n2o)
    nw.num_entities = data.num_entities
    nw.num_relations = data.num_relations
    nw.i2e = data.i2e
    nw.e2i = data.e2i
    #nw.i2e = [data.i2e[i] for i in range(len(n2o))] #[data.i2e[n2o[i]] for i in range(len(n2o))]
    #nw.e2i = {e: i for i, e in enumerate(nw.i2e)}

    # relations are unchanged, but copied for the sake of GC
    nw.i2r = list(data.i2r)
    nw.r2i = dict(data.r2i)

    # count the new number of triples
    num = 0
    for s, p, o in data_triples:
        if s in entities and o in entities:
            num += 1

    nw.triples = np.zeros((num, 3), dtype=int)

    row = 0
    for s, p, o in data_triples:
        if s in entities and o in entities:
            #s, o =  o2n[s], o2n[o]
            s,o = s,o
            nw.triples[row, :] = (s, p, o)
            row += 1

    nw.training = data_training.copy()
    for i in range(nw.training.shape[0]):
        nw.training[i, 0] = nw.training[i, 0] #o2n[nw.training[i, 0]]

    nw.withheld = data_withheld.copy()
    for i in range(nw.withheld.shape[0]):
        nw.withheld[i, 0] = nw.withheld[i, 0] #o2n[nw.withheld[i, 0]]

    nw.num_classes = data.num_classes

    nw.final = data.final
    nw.torch = data.torch
    if nw.torch:  # this should be constant-time/memory
        nw.triples = torch.from_numpy(nw.triples)
        nw.training = torch.from_numpy(nw.training)
        nw.withheld = torch.from_numpy(nw.withheld)

    return nw


