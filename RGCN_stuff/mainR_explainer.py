import torch 
import pandas as pd
import numpy as np
import src.kgbench as kg
from rgcn import  RGCN
from src.rgcn_explainer_utils import *
import wandb

from R_explainer import *

from config import * 
import sys
import argparse
print(sys.executable)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name',help='name of the dataset')
    parser.add_argument('--explain_all', action='store_true', help='if True: explain all nodes')
    parser.add_argument('--sweep', action='store_true', help='if True: sweep over parameters')
    name = parser.parse_args().name
    explain_all = parser.parse_args().explain_all
    sweep = parser.parse_args().sweep
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

    relations = ['label', 'node_idx','number_neighbors', 
                'prediction_explain', 'prediction_full', 'prediction_explain_binary',
                'prediction_inverse_binary', 
                'prediction_random','prediction_sub', 'prediction_threshold',
                'prediction_threshold_lekker',
                'res_random_inverse','res_threshold_lekker_inverse',
                'fidelity_minus', 'fidelity_plus', 'sparsity',
                'fidelity_minus_threshold','fidelity_plus_threshold','sparsity_threshold',
                'fidelity_minus_random','fidelity_plus_random','sparsity_random'] + relations

    dict_classes = d_classes(data)

    node_idx = dict_classes[list(dict_classes.keys())[1]][0]
    model = torch.load(f'chk/{name}_chk/models/model_{name}_prune_{prune}')
    #model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
    pred_label = torch.load(f'chk/{name}_chk/models/prediction_{name}_prune_{prune}')
    print('explain all',explain_all)
    if explain_all == True:
        for target_label in range(len(dict_classes.keys())):
            for node_idx in dict_classes[target_label]:
                num_neighbors = number_neighbors(node_idx, data, n_hops)
                def wrapped_main1():
                    main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,config = None)

                if sweep:
                    sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{node_idx}" )
                    print('sweep_config', sweep_config)
                    wandb.agent(sweep_id, function= wrapped_main1)
                else:
                    config = default_params
                    config.update({'explain_all': explain_all})
                    main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep, config)
                    wandb.config.update({'experiment': f"RGCNExplainer_{name}"})



    if explain_all == False:
        num_neighbors = number_neighbors(node_idx, data, n_hops)

        def wrapped_main1():
            main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,config=None )
        if sweep:
            sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{node_idx}" )
            wandb.agent(sweep_id, function= wrapped_main1)
        else:
            config = default_params
            config.update({'explain_all': explain_all})
            
            main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep, config )
            wandb.config.update({'experiment': f"RGCNExplainer_{name}"})
    




if __name__ == '__main__':
    main()

