from calendar import c
import torch 
import pandas as pd
import numpy as np
import src.kgbench as kg
from rgcn import  RGCN
from src.rgcn_explainer_utils import *
import wandb
import random
from R_explainer import *

from config import * 
import sys
import argparse
print(sys.executable)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name',help='name of the dataset')
    parser.add_argument('init',  help='Mask initialization strategy',default = 'normal', choices =['normal','const','overall_frequency','relative_frequency','inverse_relative_frequency','domain_frequency','range_frequency', 'most_freq_rel','Domain_Knowledge'])
    parser.add_argument('--relation_id', help='relation id to which giving more weight on explanation', type=int,required='Domain_Knowledge' in sys.argv,nargs='+')
    parser.add_argument('--value_for_relations_id', help='weight to give to relations weighted', type=int,required='Domain_Knowledge' in sys.argv)#,nargs='+')
    parser.add_argument('--explain_all', action='store_true', help='if True: explain all nodes')
    parser.add_argument('--random_sample', action='store_true', help='if True: explain sample of nodes')
    parser.add_argument('--explain_one', action='store_true', help='if True: explain one node')
    parser.add_argument('--num_samples_per_class', help='Number of samples per class (Required if --random_sample is used)', type=int)
    parser.add_argument('--sweep', action='store_true', help='if True: sweep over parameters')
    parser.add_argument('--kill_most_freq_rel', action='store_true', help='if True: exclude most frequent relation from explanation')
    parser.add_argument('--size_std_neighbors', action='store_true', help='if True: size std is num neighbors')
    name = parser.parse_args().name
    explain_all = parser.parse_args().explain_all
    random_sample = parser.parse_args().random_sample
    num_samples_per_class = parser.parse_args().num_samples_per_class
    explain_one = parser.parse_args().explain_one
    sweep = parser.parse_args().sweep
    init_strategy = parser.parse_args().init
    kill_most_freq_rel = parser.parse_args().kill_most_freq_rel
    size_std_neighbors = parser.parse_args().size_std_neighbors
    n_hops = 2
    if init_strategy == 'Domain_Knowledge':
        relation_id = parser.parse_args().relation_id
        value_for_relations_id = parser.parse_args().value_for_relations_id
    



    if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre', 'amplus', 'dmg777k']:
        data = kg.load(name, torch=True, final=False)
    if 'IMDb' in name:    
        data = torch.load(f'data/IMDB/finals/{name}.pt')
    if 'dbo' in name:
        data = torch.load(f'data/DBO/finals/{name}.pt')
    if name == 'mdgenre' or 'dbo' in name:
        prune = False
    else:
        prune = True
    if prune:
        data = prunee(data, 2)
    data.triples = torch.Tensor(data.triples).to(int)
    data.withheld = torch.Tensor(data.withheld).to(int)
    data.training = torch.Tensor(data.training).to(int)
    data.hor_graph, data.ver_graph = hor_ver_graph(data.triples, data.num_entities, data.num_relations)

    print(f'Number of entities: {data.num_entities}') 
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') 
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    relations = get_relations(data)

    dict_classes = d_classes(data)
    model = torch.load(f'chk/{name}_chk/models/model_{name}_prune_{prune}')
    torch.set_float32_matmul_precision('medium')
    prediction_model = torch.load(f'chk/{name}_chk/models/prediction_{name}_prune_{prune}')
    prediction_model = nn.Softmax(dim=1)(prediction_model)
    
    if explain_all :
        print('explain_all')
        for target_label in range(len(dict_classes.keys())):
            for node_idx in dict_classes[target_label]:
                num_edges = number_edges(node_idx, data, n_hops)

                config = default_params
                config.update({'explain_all': explain_all})
                config.update({'random_sample': random_sample})
                config.update({'explain_one': explain_one})
                config.update({'kill_most_freq_rel': kill_most_freq_rel})

                if size_std_neighbors:
                    config.update({"size_std": num_edges*0.1})
                config.update({"init_strategy": init_strategy })
                if init_strategy == 'Domain_Knowledge':
                    config.update({'relation_id': relation_id})
                    config.update({'value_for_relations_id': value_for_relations_id})
                main1(n_hops, node_idx, model,prediction_model, data,name,  prune,relations, dict_classes, num_edges,sweep, config)
                wandb.config.update({'experiment': f"RGCNExplainer_{name}"})


    
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
            num_edges = number_edges(node_idx, data, n_hops)
            def wrapped_main1():
                main1(n_hops, node_idx, model,prediction_model, data,name,  prune,relations, dict_classes, num_edges,sweep,config=None )
            if sweep:
                label = int(data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([node_idx])),1])
                sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{label}_{node_idx}" )
                wandb.agent(sweep_id, function= wrapped_main1)
            else:
                config = default_params
                config.update({'explain_all': explain_all})
                config.update({'random_sample': random_sample})
                config.update({'explain_one': explain_one})
                config.update({'kill_most_freq_rel': kill_most_freq_rel})

                
                config.update({"init_strategy": init_strategy })
                if size_std_neighbors:
                    config.update({"size_std": num_edges*0.1})

                if init_strategy == 'Domain_Knowledge':
                    config.update({'relation_id': relation_id})
                    config.update({'value_for_relations_id': value_for_relations_id})
                
                main1(n_hops, node_idx, model,prediction_model, data,name,  prune,relations, dict_classes, num_edges,sweep,  config )
                wandb.config.update({'experiment': f"RGCNExplainer_{name}"})



    if explain_one:
        node_idx = dict_classes[1][0]
        print('explain one node', node_idx)
        num_edges = number_edges(node_idx, data, n_hops)
        # def wrapped_main1():
        #     main1(n_hops, node_idx, model,prediction_model, data,name,  prune,relations, dict_classes, num_edges,sweep,init_strategy,config=None )
        if sweep:
            sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{node_idx}" )
            #wandb.agent(sweep_id, function= wrapped_main1)
            wandb.agent(sweep_id, function=lambda: main1(n_hops, node_idx, model, prediction_model, data, name, prune, relations, dict_classes, num_edges, sweep, config=None))
            config.update({'explain_all': explain_all})
            config.update({'random_sample': random_sample})
            config.update({'explain_one': explain_one})
            config.update({'kill_most_freq_rel': kill_most_freq_rel})

            if size_std_neighbors:
                config.update({"size_std": num_edges*0.1})
            config.update({"init_strategy": init_strategy })
        else:
            config = default_params
            config.update({'explain_all': explain_all})
            config.update({'random_sample': random_sample})
            config.update({'explain_one': explain_one})
            config.update({'kill_most_freq_rel': kill_most_freq_rel})

            if size_std_neighbors:
                config.update({"size_std": num_edges*0.1})
            config.update({"init_strategy": init_strategy })
            if init_strategy == 'Domain_Knowledge':
                config.update({'relation_id': relation_id})
                config.update({'value_for_relations_id': value_for_relations_id})
            
            main1(n_hops, node_idx, model,prediction_model, data,name,  prune,relations, dict_classes, num_edges,sweep,  config )
            wandb.config.update({'experiment': f"RGCNExplainer_{name}"})
    




if __name__ == '__main__':
    main()

