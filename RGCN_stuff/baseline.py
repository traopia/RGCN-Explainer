from networkx import dfs_postorder_nodes
from sklearn import base
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import kgbench as kg

from kgbench import load, tic, toc, d
import numpy as np

from src.rgcn_explainer_utils import *
from rgcn_model import RGCN


def prediction_full(data, model, node_idx):
    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    y_full = model.forward2(hor_graph, ver_graph)
    node_pred_full = y_full[node_idx, :]
    res_full = nn.Softmax(dim=0)(node_pred_full)
    m = match_to_triples(ver_graph,hor_graph,data, node_idx)

    return res_full, m, hor_graph, ver_graph


def prediction_wrong_if(data,model, node_idx,label):
    ''' 
    Prediction of the full graph.
    Results  by canceling out the contribution of each relation.'''
    res_full, m, h,v = prediction_full(data, model, node_idx)
    for key in Counter(m[:,1].tolist()).keys():
        v_ = select_on_relation_sparse(v,data, key)
        h_ = select_on_relation_sparse(h,data, key)
        out = model.forward2(h_,v_)
        res = nn.Softmax(dim=0)(out[node_idx])
        if torch.argmax(res)!=label:
            print(f'for node {node_idx}, wrong prediction without {data.i2r[key]}')



def baseline_pred(data, model, node_idx):
    ''' Baseline prediction of the full graph - namely all relations are 0s'''
    res_full, m, h,v = prediction_full(data, model, node_idx)
    for key in Counter(m[:,1].tolist()).keys():
        v_ = select_one_relation(v,data, key, value =0)
        h_ = select_one_relation(h,data, key, value = 0)
        out = model.forward2(h_,v_)
        res = nn.Softmax(dim=0)(out[node_idx])
    return res


def prediction_with_one_relation(data, model, node_idx,label):
    ''' Get prediction with one relation only to check which relation is the most important for the prediction'''

    print('label', label)
    print('node_idx', node_idx)
    count = {}
    ones = {}
    count['label'], ones['label'] = label.detach().numpy()[0], label.detach().numpy()[0]
    count['node_idx'], ones['node_idx'] = node_idx, node_idx
    res_full, m, h,v = prediction_full(data, model, node_idx)
    baseline = baseline_pred(data, model, node_idx)
    for key in Counter(m[:,1].tolist()).keys():
        v_ = select_one_relation(v,data, key)
        h_ = select_one_relation(h,data, key)
        out = model.forward2(h_,v_)
        res = nn.Softmax(dim=0)(out[node_idx])

        if torch.argmax(res)!=label:
            count[data.i2rel[key][0]] = None
        else:
            if list(res) != list(baseline):
                print(f'correct only with {data.i2rel[key][0]}, {key}', res)
                count[data.i2rel[key][0]] = res.detach().numpy().tolist()
                ones[data.i2rel[key][0]] = 1

    print(f'correct with {len(count)} relations out of {len(Counter(m[:,1].tolist()).keys())} relations')
    print(count)
    return count, ones





def main(name,node_idx, prune=True, all = True):

    model = torch.load(f'/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/chk/{name}_chk/model_{name}_prune_{prune}')
    data = kg.load(name, torch=True) 
    data = prunee(data, 2)
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
                count, ones = prediction_with_one_relation(data, model, node_idx,label)
                df.loc[str(node_idx)] = count
                df_ones.loc[str(node_idx)] = ones


    else:        
        label = data.withheld[data.withheld[:,0]==node_idx,1]
        count, ones = prediction_with_one_relation(data, model, node_idx,label)
        print(count)
        df.loc[str(node_idx)] = count
        df.to_csv(f'/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/chk/{name}_chk/Important_baseline_{name}_results{node_idx}.csv', index=False)


    df.to_csv(f'/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/chk/{name}_chk/Important_baseline_{name}_results.csv', index=False)
    df_ones.to_csv(f'/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/chk/{name}_chk/Important_baseline_{name}_results_ones.csv', index=False)

    

if __name__ == '__main__':
    main('mdgenre',5757, prune=False, all = True)
    

