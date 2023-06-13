import torch 
import pandas as pd
import numpy as np
import kgbench as kg


#rgcn 
from rgcn import  RGCN
from src.rgcn_explainer_utils import *

#params
import wandb

from R_explainer import *

from config import *





if name in ['aifb', 'mutag', 'bgs', 'am', 'mdgenre']:
    data = kg.load(name, torch=True, final=False)
else:    
    data = torch.load(f'data/IMDB/finals/{name}.pt')
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
             'prediction 1-m explain binary', 'prediction_sub', 'prediction_threshold',
             'fidelity_minus', 'fidelity_plus', 'sparsity'] + relations
df = pd.DataFrame(columns=relations)
df_threshold = pd.DataFrame(columns=relations) 

dict_classes = d_classes(data)

node_idx = 5757#dict_classes[list(dict_classes.keys())[0]][0]

model = torch.load(f'chk/{name}_chk/model_{name}_prune_{prune}')
pred_label = torch.load(f'chk/{name}_chk/prediction_{name}_prune_{prune}')
print('explain all',explain_all)
if explain_all == True:
    for target_label in range(len(dict_classes.keys())):
        for node_idx in dict_classes[target_label]:
            num_neighbors = number_neighbors(node_idx, data, n_hops)
            def wrapped_main1():
                main1(n_hops, node_idx, model,pred_label, data,name,  prune,df, dict_classes, num_neighbors,config = None)

            if sweep:
                sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{node_idx}_playground" )
                
                counter, counter_threshold,experiment_name = wandb.agent(sweep_id, function= wrapped_main1)
            else:
                config = default_params
                
                counter, counter_threshold, experiment_name = main1(n_hops, node_idx, model,pred_label, data,name,  prune,df, dict_classes, num_neighbors,config )
                wandb.config.update({'experiment': f"RGCNExplainer_{name}_{node_idx}_playground"})

            directory = f'chk/{name}_chk/{experiment_name}'
            df.loc[str(node_idx)] = counter
            df_threshold.loc[str(node_idx)] = counter_threshold
            if not os.path.exists(directory + f'/Relation_Importance'):
                os.makedirs(directory + f'/Relation_Importance')
                
            # df.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_{node_idx}.csv', index=False)
            # df_threshold.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_{node_idx}_threshold.csv', index=False)
            print('saved results to directory')
    if not os.path.exists(directory + f'/Relation_Importance'):
        os.makedirs(directory + f'/Relation_Importance')
    df.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_full.csv', index=False)
    df_threshold.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_full_threshold.csv', index=False)

if explain_all == False:
    num_neighbors = number_neighbors(node_idx, data, n_hops)

    def wrapped_main1():
        main1(n_hops, node_idx, model,pred_label, data,name,  prune,df,df_threshold, dict_classes, num_neighbors,config = None)

    if sweep:
        sweep_id = wandb.sweep(sweep_config, project= f"RGCNExplainer_{name}_{node_idx}_playground" )
        
        counter, counter_threshold, experiment_name = wandb.agent(sweep_id, function= wrapped_main1)
    else:
        config = default_params
        
        counter, counter_threshold, experiment_name = main1(n_hops, node_idx, model,pred_label, data,name,  prune,df, df_threshold, dict_classes, num_neighbors,config )
        wandb.config.update({'experiment': f"RGCNExplainer_{name}_{node_idx}_playground"})
    

    directory = f'chk/{name}_chk/{experiment_name}'
    df.loc[str(node_idx)] = counter
    df_threshold.loc[str(node_idx)] = counter_threshold
    if not os.path.exists(directory + f'/Relation_Importance'):
        os.makedirs(directory + f'/Relation_Importance')
    df.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_{node_idx}.csv', index=False)
    df_threshold.to_csv(f'{directory}/Relation_Importance/Relations_Important_{name}_{node_idx}_threshold.csv', index=False)
    print('saved results to directory')



    #def main(n_hops, node_idx, model,pred_label, data,name,  prune,df, dict_classes, num_neighbors,config, )




