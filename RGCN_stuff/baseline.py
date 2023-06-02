from networkx import dfs_postorder_nodes
from sklearn import base
from sqlalchemy import false
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import kgbench as kg

from kgbench import load, tic, toc, d

from src.rgcn_explainer_utils import *
from rgcn_model import RGCN


def prediction_full(data, model, node_idx):
    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    #y_full = model.forward2(hor_graph, ver_graph)
    #node_pred_full = y_full[node_idx, :]
    #res_full = nn.Softmax(dim=0)(node_pred_full)
    m = match_to_triples(ver_graph,hor_graph,data, node_idx)

    #return res_full, m, hor_graph, ver_graph
    return m, hor_graph, ver_graph


def prediction_wrong_if(data,model, node_idx,label):
    ''' 
    Prediction of the full graph.
    Results  by canceling out the contribution of each relation.'''
    id = 'wrong_if'
    print('label', label)
    print('node_idx', node_idx)
    count = {}
    ones = {}

    count['label'], ones['label'] = label.tolist()[0], label.tolist()[0]
    count['node_idx'], ones['node_idx'] = node_idx, node_idx
    #res_full, m, h,v = prediction_full(data, model, node_idx)
    m, h,v = prediction_full(data, model, node_idx)
    for key in Counter(m[:,1].tolist()).keys():
        v_ = select_on_relation_sparse(v,data, key)
        h_ = select_on_relation_sparse(h,data, key)
        out = model.forward2(h_,v_)
        res = nn.Softmax(dim=0)(out[node_idx])
        # for i in label:
        #     if torch.argmax(res)!=i:
        #         print(f'for node {node_idx}, wrong prediction without {data.i2r[key]}')
        #         count[data.i2rel[key][0]] = res.detach().tolist()
        #         ones[data.i2rel[key][0]] = 1
        # else:
        #     pass
        #for i in label:
        if len(label) > 1:
            if torch.argmax(res)!=label[0] or torch.argmax(res)!=label[1]:
                print(f'for node {node_idx}, wrong prediction without {data.i2r[key]}')
                count[data.i2rel[key][0]] = res.detach().tolist()
                ones[data.i2rel[key][0]] = 1
        if len(label) == 1:
            if torch.argmax(res)!=label:
                print(f'for node {node_idx}, wrong prediction without {data.i2r[key]}')
                count[data.i2rel[key][0]] = res.detach().tolist()
                ones[data.i2rel[key][0]] = 1

        else:
            pass

    return count, ones, id




def baseline_pred(data, model, node_idx):
    ''' Baseline prediction of the full graph - namely all relations are 0s'''
    #res_full, m, h,v = prediction_full(data, model, node_idx)
    m, h,v = prediction_full(data, model, node_idx)
    for key in Counter(m[:,1].tolist()).keys():
        v_ = select_one_relation(v,data, key, value =0)
        h_ = select_one_relation(h,data, key, value = 0)
        out = model.forward2(h_,v_)
        res = nn.Softmax(dim=0)(out[node_idx])
    return res


def prediction_with_one_relation(data, model, node_idx,label):
    ''' Get prediction with one relation only to check which relation is the most important for the prediction'''
    id = 'one_relation'
    print('label', label)
    print('node_idx', node_idx)
    count = {}
    ones = {}
    count['label'], ones['label'] = label.tolist()[0], label.tolist()[0]
    count['node_idx'], ones['node_idx'] = node_idx, node_idx
    #res_full, m, h,v = prediction_full(data, model, node_idx)
    m, h,v = prediction_full(data, model, node_idx)
    baseline = baseline_pred(data, model, node_idx)
    for key in Counter(m[:,1].tolist()).keys():
        v_ = select_one_relation(v,data, key)
        h_ = select_one_relation(h,data, key)
        #print(v_, h_)
        out = model.forward2(h_,v_)
        res = nn.Softmax(dim=0)(out[node_idx])
        for i in label:
            if torch.argmax(res)!=i:
            #count[data.i2rel[key][0]] = None
                pass
        else:
            if list(res) != list(baseline):
                #print(f'correct only with {data.i2rel[key][0]}, {key}', res)
                print(f'correct only with {data.i2rel[key][0]}, {key}')
                count[data.i2rel[key][0]] = res.detach().tolist()
                ones[data.i2rel[key][0]] = 1

    print(f'correct with {len(count)} relations out of {len(Counter(m[:,1].tolist()).keys())} relations')
    return count, ones, id 





def main(name,node_idx, prune=True, all = True, test = False):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    model.to(device)
    if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre']:
        data = kg.load(name, torch=True, final=False)
        if test:
            data = kg.load(name, torch=True, final=True)
            id_test = 'test'
        else:
            data = kg.load(name, torch=True) 
            id_test = 'withheld'
    else:    
        data = torch.load(f'data/IMDB/finals/{name}.pt')
        id_test = 'withheld'


    data = prunee(data, 2)
    data.to(device)
    data.triples = torch.Tensor(data.triples).to(int)
    data.withheld = torch.Tensor(data.withheld).to(int)
    data.training = torch.Tensor(data.training).to(int)
    d = d_classes(data)
    relation = get_relations(data)
    relations = ['node_idx','label'] + [data.i2rel[i][0] for i in range(len(data.i2rel))]
    df = pd.DataFrame(columns=relations)
    df_ones = pd.DataFrame(columns=relations)

    if all:
        for target_label in range(len(d.keys())):
            for node_idx in d[target_label]:
                if node_idx in data.withheld[:,0]:
                    label = data.withheld[data.withheld[:,0]==node_idx,1]
                #count, ones, id  = prediction_with_one_relation(data, model, node_idx,label)
                count, ones, id = prediction_wrong_if(data, model, node_idx,label)
                df.loc[str(node_idx)] = count
                df_ones.loc[str(node_idx)] = ones
        df.to_csv(f'chk/{name}_chk/Important_{id}_{name}_results_{id_test}.csv', index=False)
        df_ones.to_csv(f'chk/{name}_chk/Important_{id}_{name}_results_ones_{id_test}.csv', index=False)

    else: 
        node_idx = d[list(d.keys())[0]][0]       
        label = data.withheld[data.withheld[:,0]==node_idx,1]
        #count, ones, id = prediction_with_one_relation(data, model, node_idx,label)
        count, ones, id = prediction_wrong_if(data, model, node_idx,label)
        print(count)
        df.loc[str(node_idx)] = count
        df.to_csv(f'chk/{name}_chk/Important_{id}_{name}_results{node_idx}_{id_test}.csv', index=False)





    

if __name__ == '__main__':
    main('IMDb_us',7185, prune=True, all = True, test = True)
    