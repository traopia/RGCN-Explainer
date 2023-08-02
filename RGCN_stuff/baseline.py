from networkx import dfs_postorder_nodes
from sklearn import base
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from src.kgbench import load, tic, toc, d, Data
import src.kgbench as kg
from src.rgcn_explainer_utils import *
from rgcn import RGCN
import psutil
import argparse
import random


def print_cpu_utilization():
    # Get CPU utilization percentage
    cpu_percent = psutil.cpu_percent()
    # Print CPU utilization
    print(f"CPU utilization: {cpu_percent}%")

    # Get virtual memory usage percentage
    mem_percent = psutil.virtual_memory().percent

    # Print virtual memory usage percentage
    print(f"Virtual memory: {mem_percent}%")

    # Get disk usage percentage
    disk_percent = psutil.disk_usage('/').percent

    # Print disk usage percentage
    print(f"Disk usage: {disk_percent}%")

    # Get cpu statistics
    cpu_stats = psutil.cpu_stats()

    # Print number of context switches
    print(f"Number of context switches: {cpu_stats}")


  

# Call the function to print CPU utilization



def prediction_full(data, model, node_idx):
    hor_graph, ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)
    m = match_to_triples(ver_graph,hor_graph,data, node_idx)
    return m, hor_graph, ver_graph


def prediction_wrong_if(data,model, node_idx,label):
    ''' 
    Prediction of the full graph.
    Results  by canceling out the contribution of each relation.
    --> backward '''
    id = 'wrong_if'
    print('label', label)
    print('node_idx', node_idx)
    count = {}
    ones = {}

    count['label'], ones['label'] = label.tolist()[0], label.tolist()[0]
    count['node_idx'], ones['node_idx'] = node_idx, node_idx
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
    ''' Get prediction with one relation only to check which relation is the most important for the prediction
    
    --> forward'''
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
    print(count)
    print(f'correct with {len(count)-2} relations out of {len(Counter(m[:,1].tolist()).keys())} relations')
    return count, ones, id 





def main(prune=True,test = False):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('name',help='name of the dataset')
    parser.add_argument('explanation_type',help='Relation importance: \'one_relation\' or \'wrong_if\'')
    parser.add_argument('--explain_all', action='store_true', help='if True: explain all nodes')
    parser.add_argument('--random_sample', action='store_true', help='if True: explain sample of nodes')
    parser.add_argument('--num_samples_per_class', help='Number of samples per class (Required if --random_sample is used)', type=int)
    parser.add_argument('--explain_one', action='store_true', help='if True: explain one node')
    name = parser.parse_args().name
    explain_all = parser.parse_args().explain_all
    random_sample = parser.parse_args().random_sample
    explain_one = parser.parse_args().explain_one
    explanation_type = parser.parse_args().explanation_type
    num_samples_per_class = parser.parse_args().num_samples_per_class
    
    if name == 'mdgenre' or 'dbo' in name:
        prune = False
    #     device = torch.device("cpu")
    # print(device)
    model = torch.load(f'chk/{name}_chk/models/model_{name}_prune_{prune}')
    # model.to(device)
    if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre', 'amplus']:
        data = load(name, torch=True, final=False)
        if test:
            data = load(name, torch=True, final=True)
            id_test = 'test'
        else:
            data = load(name, torch=True) 
            id_test = 'withheld'
    if 'IMDb' in name:    
        data = torch.load(f'data/IMDB/finals/{name}.pt')
        id_test = 'withheld'
    if 'dbo' in name:
        data = torch.load(f'data/DBO/finals/{name}.pt')
        id_test = 'withheld'


    if prune:
        data = prunee(data, 2)
    print(type(data))
    # data.to(device)
    data.triples = torch.Tensor(data.triples).to(int)
    data.withheld = torch.Tensor(data.withheld).to(int)
    data.training = torch.Tensor(data.training).to(int)
    dict_classes = d_classes(data)
    relation = get_relations(data)
    relations = ['node_idx','label'] + [data.i2rel[i][0] for i in range(len(data.i2rel))]
    df = pd.DataFrame(columns=relations)
    df_ones = pd.DataFrame(columns=relations)
    if not os.path.exists(f'chk/{name}_chk/Relation_Selection'):
        os.makedirs(f'chk/{name}_chk/Relation_Selection')

    if explain_all:
        for target_label in range(len(dict_classes.keys())):
            for node_idx in dict_classes[target_label]:
                if node_idx in data.withheld[:,0]:
                    label = data.withheld[data.withheld[:,0]==node_idx,1]
                if explanation_type=='one_relation':
                    count, ones, id = prediction_with_one_relation(data, model, node_idx,label) 
                if explanation_type=='wrong_if':
                    count, ones, id = prediction_wrong_if(data, model, node_idx,label)
                df.loc[str(node_idx)] = count
                df_ones.loc[str(node_idx)] = ones
        print_cpu_utilization()
        df.to_csv(f'chk/{name}_chk/Relation_Selection/Important_{id}_{name}_results_{id_test}.csv', index=False)
        df_ones.to_csv(f'chk/{name}_chk/Relation_Selection/Important_{id}_{name}_results_ones_{id_test}.csv', index=False)

    if explain_one: 
        node_idx = dict_classes[list(dict_classes.keys())[0]][0]       
        label = data.withheld[data.withheld[:,0]==node_idx,1]
        if explanation_type=='one_relation':
            count, ones, id = prediction_with_one_relation(data, model, node_idx,label) 
        if explanation_type=='wrong_if':
            count, ones, id = prediction_wrong_if(data, model, node_idx,label)
        print(count)
        print_cpu_utilization()
        df.loc[str(node_idx)] = count
        df.to_csv(f'chk/{name}_chk/Relation_Selection/Important_{id}_{name}_results{node_idx}_{id_test}.csv', index=False)

    if random_sample:
        print('random sample')
        random.seed(42)
        min_length = min(len(value) for value in dict_classes.values())
        if num_samples_per_class is None:
            num_samples_per_class = 30 if min_length > 30 else min_length 
        print('num_samples_per_class', num_samples_per_class)
        sampled_data = []
        for key in dict_classes:
            sampled_data.extend(random.sample(dict_classes[key], num_samples_per_class))
        for node_idx in sampled_data:
            if node_idx in data.withheld[:,0]:
                label = data.withheld[data.withheld[:,0]==node_idx,1]
            if explanation_type=='one_relation':
                count, ones, id = prediction_with_one_relation(data, model, node_idx,label) 
            if explanation_type=='wrong_if':
                count, ones, id = prediction_wrong_if(data, model, node_idx,label)
            df.loc[str(node_idx)] = count
            df_ones.loc[str(node_idx)] = ones
    print_cpu_utilization()
    df.to_csv(f'chk/{name}_chk/Relation_Selection/Important_{id}_{name}_results_sample_{id_test}.csv', index=False)
    df_ones.to_csv(f'chk/{name}_chk/Relation_Selection/Important_{id}_{name}_results_sample_ones_{id_test}.csv', index=False)




    

if __name__ == '__main__':
    main()

    