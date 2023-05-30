from re import M
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from collections import Counter
from kgbench import load, tic, toc, d, Data
import os
from rgcn_model import adj, enrich, sum_sparse, RGCN

def dict_index_classes(data, masked_ver):
    indices_nodes = masked_ver.coalesce().indices().tolist()

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
    indices_nodes = masked_ver.coalesce().indices().tolist()
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




#Neighborhood
#Extract neighborhood

def find_n_hop_neighbors(edge_index, n, node=None):
    """ 
    edge_index 
    n = num hops
    node = node_idx
    returns: 
        - sub_edges: list of edges in the n-hop neighborhood of the specified node
        - neighborhoods: dictionary of node neighborhoods
        - sub_edges: tensor of edges in the n-hop neighborhood of the specified node

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









########## the one

def match_to_triples(v,h, data, sparse=True):
    if sparse:
        pv,_ = torch.div(v.coalesce().indices(), data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
        sv,ov = v.coalesce().indices()%data.num_entities
        result_v = torch.stack([sv,pv,ov], dim=1)
        ph,_ = torch.div(h.coalesce().indices(), data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
        sh,oh = h.coalesce().indices()%data.num_entities
        result_h = torch.stack([sh,ph,oh], dim=1)
        result = torch.cat((result_v, result_h), 0)


                    
    else:

        # _,ph = torch.div(h, data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
        # sh,oh = h%data.num_entities
        # result_h = torch.stack([sh,ph,oh], dim=1)

        # pv, _ = torch.div(v, data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
        # sv,ov = v%data.num_entities
        # result_v = torch.stack([sv,pv,ov], dim=1)

        # result = torch.cat((result_v, result_h), 0)

        if len(h )!= 0:
            _,ph = torch.div(h, data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
            sh,oh = h%data.num_entities
            result_h = torch.stack([sh,ph,oh], dim=1)
        if len(v)!=0:
            pv, _ = torch.div(v, data.num_entities, rounding_mode='floor')#v.coalesce().indices()//data.num_entities
            sv,ov = v%data.num_entities
            result_v = torch.stack([sv,pv,ov], dim=1)
        if len(h) != 0 and len(v) != 0:
            result = torch.cat((result_v, result_h), 0)
            print(pv,ph)
        if len(h) == 0:
            result = result_v
            print(pv)
        if len(v) == 0:
            result = result_h
            print(ph)
        

                    
    
    return result

#edge index
def edge_index_oneadj(triples):
    edge_index = torch.stack((triples[:, 0], triples[:, 2]),dim=0)
    return edge_index


def sub_sparse_tensor(sparse_tensor, threshold, data, low_threshold=False):
    if low_threshold:
        nonzero_indices = sparse_tensor.coalesce().indices()[:, sparse_tensor.coalesce().values() < threshold]
        nonzero_indices[0] = nonzero_indices[0]#%data.num_entities
        nonzero_values = sparse_tensor.coalesce().values()[sparse_tensor.coalesce().values() < threshold]
        sel_masked_ver = torch.sparse_coo_tensor(nonzero_indices, nonzero_values)
    else:
        nonzero_indices = sparse_tensor.coalesce().indices()[:, sparse_tensor.coalesce().values() > threshold]
        nonzero_indices[0] = nonzero_indices[0]#%data.num_entities
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


def selected(masked_ver, masked_hor, threshold,data, low_threshold, float=False):
    sel_masked_ver, sel_masked_hor = sub_sparse_tensor(masked_ver, threshold,data, low_threshold), sub_sparse_tensor(masked_hor, threshold,data, low_threshold)
    indices_nodes_v, indices_nodes_h = sel_masked_ver.coalesce().indices().tolist(), sel_masked_hor.coalesce().indices().tolist()
    new_index_v, new_index_h = np.transpose(np.stack((indices_nodes_v[0], indices_nodes_v[1]))) , np.transpose(np.stack((indices_nodes_h[0], indices_nodes_h[1])))
    #triples_matched = match_to_triples(np.array(new_index), data.triples)
    triples_matched = match_to_triples(sel_masked_ver,sel_masked_hor,  data)
    #print(triples_matched)
    if float:
            l = {}
            for i,j in zip(triples_matched[:,1],sel_masked_ver.coalesce().values()):
                
                if data.i2rel[int(i)][0] in l.keys():
                    l[data.i2rel[int(i)][0]] += j
                else:
                    l[data.i2rel[int(i)][0]] = j
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
    indices_nodes = sel_masked_ver.coalesce().indices().tolist()
    new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) 
    triples_matched = match_to_triples(np.array(new_index), data.triples)

    l = {}
    for i,j in zip(triples_matched[:,1],sel_masked_ver.coalesce().values()):

        if data.i2rel[int(i)][0] in l.keys():
            l[data.i2rel[int(i)][0]] += j
        else:
            l[data.i2rel[int(i)][0]] = j
   

    return Counter(l)
    


    

def get_relations(data):
    ''' Get all relations in the dataset'''
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

    #print(all_relations)

    data.i2rel = dict
    return data.i2rel       


def d_classes(data):
    """ 
    Get classes of nodes (select only the alphanum - not literals)
    """
    data.entities = np.append(data.triples[:,0].tolist(),(data.triples[:,2].tolist()))
    indices_nodes = data.entities
    d = list(data.e2i.keys())
    values_indices_nodes = [d[int(i)] for i in indices_nodes]
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
    d = {key.item(): data.withheld[:, 0][data.withheld[:, 1] == key].tolist() for key in torch.unique(data.withheld[:, 1])}
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


def subset_sparse(sparse_full,data,  threshold=0.5):
    ''' Select the subset of the tensor based on the threshold value'''
    num_entities = data.num_entities
    x_values = sparse_full.coalesce().values()
    x_indices = sparse_full.coalesce().indices()
    x = torch.sparse_coo_tensor(indices = x_indices, values = x_values)
    subset_mask = (x._values() > threshold).to_dense()
    subset_indices = x._indices()[:, subset_mask.nonzero().squeeze()]
    subset_values = x._values()[subset_mask]
    subset_sparse = torch.sparse_coo_tensor(subset_indices, subset_values)
    num_high = subset_sparse._nnz()
    _ ,p = torch.div(subset_sparse.coalesce().indices(),num_entities, rounding_mode='floor')
    relation_counter = dict(Counter(p.tolist()))
    return subset_sparse, num_high,p, relation_counter


def select_relation(sparse_tensor,num_entities,relation_id):
    ''' Select the subset of the tensor based on the relation id'''
    p = torch.div(sparse_tensor.coalesce().indices(),num_entities, rounding_mode='floor')
    if sparse_tensor.shape[0]> sparse_tensor.shape[1]:
        p = p[0]
    else:
        p = p[1]
    output_indices = sparse_tensor.coalesce().indices()[:, p==relation_id]
    output_values = sparse_tensor.coalesce().values()[p==relation_id]
    value_indices = torch.where(p == relation_id)[0]
    #I can then use the value indices to change the values in the mask??
    return output_indices, output_values, value_indices


def get_class_entity(node,data):
    ''' Get class of a node'''
    url = data.i2e[int(node)][0]
    try:
        if '#' in url:
            url = url.split('#')[1]
        if 'http' in url:
            return str(url).split('/')[3]

    except IndexError:
        return 'blank' #str(url)
    
def get_class_relation(node,data):
    ''' Get class of a relation node'''
    rel =str(data.i2r[node]).split('/')[-1]
    if '#' in rel:
            rel = rel.split('#')[1]

    return str(rel)



def domain_range_freq(data,num_classes):
    '''Get the frequency of the domain and range of the relations'''
    dict_domain = {}
    dict_range = {}
    for m in data.triples:


        if int(m[1]) in dict_domain:
            dict_domain[int(m[1]) ].append(get_class_entity(m[0],data))
        else:
            dict_domain[int(m[1]) ] = [get_class_entity(m[0],data)]
        if int(m[1])  in dict_range:
            dict_range[int(m[1]) ].append(get_class_entity(m[2],data))
        else:
            dict_range[int(m[1]) ] = [get_class_entity(m[2],data)]
    for k,v in dict_domain.items():
        dict_domain[k] = len(set(v))/num_classes
    for k,v in dict_range.items():
        dict_range[k] = set(v)
    return dict_domain, dict_range


def convert_binary(sparse_tensor, threshold=0.5):
    ''' Converts a sparse tensor to a binary sparse tensor based on a threshold'''
    # convert values to either 0 or 1 based on a threshold of 0.5
    mask = sparse_tensor._values() > threshold
    converted_values = torch.zeros_like(sparse_tensor._values())
    converted_values[mask] = 1
    #print("Number of non zero values: ", converted_values.nonzero().size(0))

    # create a new sparse tensor with the converted values
    converted_sparse_tensor = torch.sparse_coo_tensor(sparse_tensor._indices(), converted_values, size=sparse_tensor.size())

    return converted_sparse_tensor



def find_repeating_sublists(sublists):
    ''' Find repeating sublists in a list'''
    repeating_elements = {}

    for sublist in sublists:
        key1 = (sublist[0], sublist[2])
        key2 = (sublist[2], sublist[0])

        if key1 in repeating_elements:
            repeating_elements[key1].append(sublist[1])
        elif key2 in repeating_elements:
            repeating_elements[key2].append(sublist[1])
        else:
            repeating_elements[key1] = [sublist[1]]

    result_array = []
    for key, values in repeating_elements.items():
        if len(values) > 1:
            result_array.append([key[0], values, key[1]])
        else:
            result_array.append([key[0], [values[0]], key[1]])

    return result_array

def unnest_list(nested_list):
    return [item for sublist in nested_list for item in (unnest_list(sublist) if isinstance(sublist, list) else [sublist])]

def visualize(node_idx, n_hop, data, masked_ver,threshold,name, result_weights=True, low_threshold=False,experiment_name=None ):
    """ 
    Visualize important nodes for node idx prediction
    """
    dict_index = dict_index_classes(data,masked_ver)
    
    #select only nodes with a certain threshold
    sel_masked_ver = sub_sparse_tensor(masked_ver, threshold,data, low_threshold)
    sel_masked_hor = sub_sparse_tensor(masked_ver, threshold,data, low_threshold)
    if len(sel_masked_ver)==0:
        sel_masked_ver=sub_sparse_tensor(masked_ver, 0,data, low_threshold)
    #print('sel masked ver',sel_masked_ver)
    indices_nodes = sel_masked_ver.coalesce().indices().detach().numpy()
    new_index = np.transpose(np.stack((indices_nodes[0], indices_nodes[1]))) #original edge indexes

    
    
    G = nx.Graph()
    if result_weights:
        values = sel_masked_ver.coalesce().values().tolist()
        for s,p,o in zip(indices_nodes[0],values , indices_nodes[1]):
            G.add_edge(int(s), int(o), weight=np.round(p, 2))

    else:

        triples_matched = match_to_triples(sel_masked_ver,sel_masked_hor, data)
        l = []
        for i in triples_matched[:,1]:
            l.append(data.i2rel[int(i)][0])
        triples_matched = find_repeating_sublists(triples_matched.tolist())
        for s,p,o in triples_matched:
            G.add_edge(int(s), int(o), weight=p)


    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
    weights = [[item] if not isinstance(item, list) else item for item in weights]


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


    col_weights = [weights[i][0] for i in range(len(weights))]
    if result_weights:
        
        nx.draw(G, pos,labels = labeldict,  edgelist=edges, edge_color=col_weights, node_color =  color_list, cmap="Set2",edge_cmap=plt.cm.Reds,font_size=8)
        nx.draw_networkx_edge_labels( G, pos,edge_labels=nx.get_edge_attributes(G,'weight'),font_size=8,font_color='red')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array(weights)
        cbar = plt.colorbar(sm)
        cbar.ax.set_title('Weight')
        plt.title("Node {}'s {}-hop neighborhood important nodes".format(node_idx, n_hop))
    else:
        rel = nx.get_edge_attributes(G,'weight')
        rel = {k: [data.i2rel[i][0] for i in v] for k,v in rel.items()}
        col_weights = [sum(weights[i], 3) if len(weights[i]) > 1 else weights[i][0] for i in range(len(weights))]
        nx.draw(G, pos,labels = labeldict, edge_color=col_weights,edgelist=edges,node_color =  color_list, cmap="Set2",font_size=7, arrows = True)
        nx.draw_networkx_edge_labels( G, pos,edge_labels=rel,font_size=8,font_color='red')
        
        res = Counter(unnest_list(rel.values()))
        print(res)
    if result_weights:
        if not os.path.exists(f'chk/{name}_chk/{experiment_name}⁄graphs'):
            os.makedirs(f'chk/{name}_chk/{experiment_name}⁄graphs')  
        plt.savefig(f'chk/{name}_chk/{experiment_name}⁄graphs/Explanation_{node_idx}_weights.png')

        #plt.show()

    else:
        if not os.path.exists(f'chk/{name}_chk/{experiment_name}⁄graphs'):
            os.makedirs(f'chk/{name}_chk/{experiment_name}⁄graphs')  
        plt.savefig(f'chk/{name}_chk/{experiment_name}⁄graphs/Explanation_{node_idx}_relations.png')    
        #plt.show()
        return res
    


def sub(v, threshold):
    nonzero_indices = v.coalesce().indices()[:, v.coalesce().values() > threshold]
    nonzero_indices[0] = nonzero_indices[0]#%data.num_entities
    nonzero_values = v.coalesce().values()[v.coalesce().values() > threshold]
    sel_masked_ver = torch.sparse_coo_tensor(nonzero_indices, nonzero_values)
    return sel_masked_ver

  


def select_on_relation_sparse(sparse_tensor,data, relation):
    ''' Selects the values of a sparse tensor based on the relation'''
    output_indices, output_values, value_indices=select_relation(sparse_tensor,data.num_entities,relation)
    coalesced_tensor = sparse_tensor.coalesce()
    coalesced_values = coalesced_tensor._values()
    coalesced_indices = coalesced_tensor._indices()
    coalesced_values[value_indices] = 0
    masked_sparse_tensor = torch.sparse_coo_tensor(coalesced_indices, coalesced_values, sparse_tensor.size())
    return masked_sparse_tensor



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
    #print(hor_size)
    r = rn // n #number of relations enriched divided by number of nodes

    vals = torch.ones(ver_ind.size(0), dtype=torch.float) #number of enriched triples
    #vals = vals / sum_sparse(ver_ind, vals, ver_size) #normalize the values by the number of edges

    hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size) #size: n,r, emb


    ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)

    return hor_graph, ver_graph



def select_one_relation(sparse_tensor,data, relation,value =1):
    """ Selects the values of a sparse tensor based on the relation"""
    sparse_tensor = torch.sparse_coo_tensor(sparse_tensor._indices(), torch.zeros(sparse_tensor._indices().shape[1]), sparse_tensor.size() )
    output_indices, output_values, value_indices=select_relation(sparse_tensor,data.num_entities,relation)
    coalesced_tensor = sparse_tensor.coalesce()
    coalesced_values = coalesced_tensor._values()
    coalesced_indices = coalesced_tensor._indices()
    coalesced_values[value_indices] = value
    masked_sparse_tensor = torch.sparse_coo_tensor(coalesced_indices, coalesced_values, sparse_tensor.size())
    return masked_sparse_tensor



def keep_columns_with_non_zero_values(df):
    # Get the column names with non-zero values
    df = df.fillna(0)
    non_zero_columns = df.columns[df.astype(bool).any(axis=0)]

    # Create a new DataFrame with only the columns containing non-zero values
    modified_df = df[non_zero_columns]

    return modified_df
