import torch 
import numpy as np
import src.kgbench as kg
from rgcn import RGCN
from src.rgcn_explainer_utils import *
import wandb
import random
from R_explainer import *
from config import * 
import sys
import argparse
import multiprocessing

def load_dataset(name):
    if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre', 'amplus', 'dmg777k']:
        data = kg.load(name, torch=True, final=False)
    elif 'IMDb' in name:    
        data = torch.load(f'data/IMDB/finals/{name}.pt')
    elif 'dbo' in name:
        data = torch.load(f'data/DBO/finals/{name}.pt')
    else:
        raise ValueError(f"Dataset '{name}' not recognized or supported.")

    if name == 'mdgenre' or 'dbo' in name:
        prune = False
    else:
        prune = True

    if prune:
        data = prunee(data, 2)

    data.triples = torch.tensor(data.triples, dtype=torch.int64)
    data.withheld = torch.tensor(data.withheld, dtype=torch.int64)
    data.training = torch.tensor(data.training, dtype=torch.int64)

    return data, prune

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of the dataset')
    parser.add_argument('init', help='Mask initialization strategy', default='normal', choices=['normal','const','overall_frequency','relative_frequency','inverse_relative_frequency','domain_frequency','range_frequency','most_freq_rel','Domain_Knowledge'])
    parser.add_argument('--relation_id', help='relation id to which giving more weight on explanation', type=int, nargs='+')
    parser.add_argument('--value_for_relations_id', help='weight to give to relations weighted', type=int)
    parser.add_argument('--explain_all', action='store_true', help='if True: explain all nodes')
    parser.add_argument('--random_sample', action='store_true', help='if True: explain sample of nodes')
    parser.add_argument('--explain_one', action='store_true', help='if True: explain one node')
    parser.add_argument('--num_samples_per_class', help='Number of samples per class (Required if --random_sample is used)', type=int)
    parser.add_argument('--sweep', action='store_true', help='if True: sweep over parameters')

    args = parser.parse_args()
    name = args.name
    explain_all = args.explain_all
    random_sample = args.random_sample
    num_samples_per_class = args.num_samples_per_class
    explain_one = args.explain_one
    sweep = args.sweep
    init_strategy = args.init
    n_hops = 2

    if init_strategy == 'Domain_Knowledge':
        if args.relation_id is None or args.value_for_relations_id is None:
            raise ValueError("Both relation_id and value_for_relations_id are required for Domain_Knowledge initialization.")
        relation_id = args.relation_id
        value_for_relations_id = args.value_for_relations_id

    data, prune = load_dataset(name)

    print(f'Number of entities: {data.num_entities}') 
    print(f'Number of classes: {data.num_classes}')
    print(f'Types of relations: {data.num_relations}') 

    data.entities = np.append(data.triples[:, 0].detach().numpy(), (data.triples[:, 2].detach().numpy()))
    relations = get_relations(data)
    dict_classes = d_classes(data)

    model = torch.load(f'chk/{name}_chk/models/model_{name}_prune_{prune}')
    torch.set_float32_matmul_precision('medium')
    pred_label = torch.load(f'chk/{name}_chk/models/prediction_{name}_prune_{prune}')

    if explain_all:
        print('explain_all')
        processes = []
        for target_label in range(len(dict_classes.keys())):
            for node_idx in dict_classes[target_label]:
                num_neighbors = number_neighbors(node_idx, data, n_hops)
                config = default_params.copy()
                config.update({'explain_all': explain_all})
                config.update({'random_sample': random_sample})
                config.update({'explain_one': explain_one})
                if init_strategy == 'Domain_Knowledge':
                    config.update({'relation_id': relation_id})
                    config.update({'value_for_relations_id': value_for_relations_id})
                args = (n_hops, node_idx, model, pred_label, data, name, prune, relations, dict_classes, num_neighbors, sweep, init_strategy, config)
                process = multiprocessing.Process(target=main1_helper, args=(args,))
                processes.append(process)
                process.start()

        for process in processes:
            process.join()

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
            args = (n_hops, node_idx, model, pred_label, data, name, prune, relations, dict_classes, num_neighbors, sweep, init_strategy, None)
            if sweep:
                label = int(data.withheld[torch.where(data.withheld[:, 0] == torch.tensor([node_idx])),1])
                sweep_id = wandb.sweep(sweep_config, project=f"RGCNExplainer_{name}_{label}_{node_idx}")
                wandb.agent(sweep_id, function=main1_helper, args=(args,))
            else:
                config = default_params
                main1_helper(args)

    if explain_one:
        node_idx = dict_classes[1][0]
        print('explain one node', node_idx)

        num_neighbors = number_neighbors(node_idx, data, n_hops)
        args = (n_hops, node_idx, model, pred_label, data, name, prune, relations, dict_classes, num_neighbors, sweep, init_strategy, None)
        if sweep:
            sweep_id = wandb.sweep(sweep_config, project=f"RGCNExplainer_{name}_{node_idx}")
            wandb.agent(sweep_id, function=main1_helper, args=(args,))
        else:
            config = default_params.copy()
            config.update({'explain_all': explain_all})
            config.update({'random_sample': random_sample})
            config.update({'explain_one': explain_one})
            if init_strategy == 'Domain_Knowledge':
                config.update({'relation_id': relation_id})
                config.update({'value_for_relations_id': value_for_relations_id})
            args = (n_hops, node_idx, model, pred_label, data, name, prune, relations, dict_classes, num_neighbors, sweep, init_strategy, config)
            main1_helper(args)

def main1_helper(args):
    n_hops, node_idx, model, pred_label, data, name, prune, relations, dict_classes, num_neighbors, sweep, init_strategy, config = args
    main1(n_hops, node_idx, model,pred_label, data,name,  prune,relations, dict_classes, num_neighbors,sweep,init_strategy,  config )

if __name__ == '__main__':
    main()
