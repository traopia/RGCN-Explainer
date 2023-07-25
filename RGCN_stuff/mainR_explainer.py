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
    parser.add_argument('init',  help='Mask initialization strategy',default = 'normal', choices =['normal','const','overall_frequency','relative_frequency','inverse_relative_frequency','domain_frequency','range_frequency', 'most_freq_rel','More_weight_on_relation'])
    parser.add_argument('--explain_all', action='store_true', help='if True: explain all nodes')
    parser.add_argument('--random_sample', action='store_true', help='if True: explain sample of nodes')
    parser.add_argument('--explain_one', action='store_true', help='if True: explain one node')
    #parser.add_argument('--num_samples_per_class', help='Number of samples per class (Required if --random_sample is used)', required='--random_sample' in sys.argv, type=int)
    parser.add_argument('--num_samples_per_class', help='Number of samples per class (Required if --random_sample is used)', type=int)
    parser.add_argument('--sweep', action='store_true', help='if True: sweep over parameters')
    name = parser.parse_args().name
    explain_all = parser.parse_args().explain_all
    random_sample = parser.parse_args().random_sample
    num_samples_per_class = parser.parse_args().num_samples_per_class
    explain_one = parser.parse_args().explain_one
    sweep = parser.parse_args().sweep
    init_strategy = parser.parse_args().init
    n_hops = 2



    if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre']:
        data = kg.load(name, torch=True, final=False)
    if 'IMDb' in name:    
        data = torch.load(f'data/IMDB/finals/{name}.pt')
    if 'dbo' in name:
        data = torch.load(f'data/DBO/finals/{name}.pt')
    if name == 'mdgenre' or 'dbo' in name:
        prune = False
    else:
        prune = True
    print(prune)
    if prune:
        data = prunee(data, 2)
    data.triples = torch.Tensor(data.triples).to(int)
    data.withheld = torch.Tensor(data.withheld).to(int)
    data.training = torch.Tensor(data.training).to(int)

    print(f'Number of entities: {data.num_entities}') 
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') 
    data.entities = np.append(data.triples[:,0].detach().numpy(),(data.triples[:,2].detach().numpy()))
    relations = get_relations(data)

    dict_classes = d_classes(data)
    model = torch.load(f'chk/{name}_chk/models/model_{name}_prune_{prune}')
    torch.set_float32_matmul_precision('medium')
    pred_label = torch.load(f'chk/{name}_chk/models/prediction_{name}_prune_{prune}')
    
    if explain_all :
        print('explain_all')
        for target_label in range(len(dict_classes.keys())):
            for node_idx in dict_classes[target_label]:
                num_neighbors = number_neighbors(node_idx, data, n_hops)

                config = default_params
                config.update({'explain_all': explain_all})
                config.update({'random_sample': random_sample})
                config.update({'explain_one': explain_one})
                main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,init_strategy, config)
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
            num_neighbors = number_neighbors(node_idx, data, n_hops)
            def wrapped_main1():
                main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,init_strategy,config=None )
            if sweep:
                sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{node_idx}" )
                wandb.agent(sweep_id, function= wrapped_main1)
            else:
                config = default_params
                config.update({'explain_all': explain_all})
                config.update({'random_sample': random_sample})
                config.update({'explain_one': explain_one})
                
                main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,init_strategy,  config )
                wandb.config.update({'experiment': f"RGCNExplainer_{name}"})



    if explain_one:
        node_idx = 5731 # dict_classes[0][0]
        print('explain one node', node_idx)
        num_neighbors = number_neighbors(node_idx, data, n_hops)

        # def wrapped_main1():
        #     main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,init_strategy,config=None )
        if sweep:
            sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{node_idx}" )
            #wandb.agent(sweep_id, function= wrapped_main1)
            wandb.agent(sweep_id, function=lambda: main1(n_hops, node_idx, model, pred_label, data, name, prune, relations, dict_classes, num_neighbors, sweep, init_strategy, config=None))
            config.update({'explain_all': explain_all})
            config.update({'random_sample': random_sample})
            config.update({'explain_one': explain_one})
        else:
            config = default_params
            config.update({'explain_all': explain_all})
            config.update({'random_sample': random_sample})
            config.update({'explain_one': explain_one})
            
            main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,init_strategy,  config )
            wandb.config.update({'experiment': f"RGCNExplainer_{name}"})
    




if __name__ == '__main__':
    main()

