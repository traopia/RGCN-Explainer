

from cv2 import fastNlMeansDenoisingColored


sweep_config = {
    'method': 'grid', #grid, random
}

metric = {
    'name': 'score_threshold',
    'goal': 'maximize'   
}
sweep_config['metric'] = metric

parameters_dict = {
    'pred': {'values': [1,10]},
    'lr': {
        'values': [0.05,0.1, 0.5]
    },
    'weight_decay': {
        'values': [0.9, 0.1]
    },
    'size': {
        'values': [0.00005,  0.0005,0.005]
    # },
    # 'size_std': {
    #     'values': [10]
    },
    'ent': {
        'values': [ -10,0,1,10]
    },
    'most_freq_rel': {
        'values': [-1,0,1,10]
    },
    'adaptive': {
        'values': [False]
    },
    'kill_most_freq_rel': {
        'values': [True,False]
    },
    'init_strategy': {
        'values': ['normal']#,'const','overall_frequency','relative_frequency','inverse_relative_frequency','domain_frequency','range_frequency', 'most_freq_rel']
    },
    'break_if_wrong_pred': {
        'values': [False]
    },
    'break_on_number_of_high': {
        'values': [False]
    }
}


parameters_dict.update({'epochs': {'value': 30}, 
                        'threshold': {'value': 0.5},
                         'relation_id': {'value': 39}, 
                         'hops': {'value':2},
                        'print': {'value': False},
                        'funky_loss': {'value': False},
                        'num_exp': {'value': 15},
                        'sweep': {'value': True}})
sweep_config['parameters'] = parameters_dict





# default_params={
# "sweep": False,
# "pred": 10,
# "size": 0.00005,  
# #"size_std": num_neighbors, 
# "ent": -1,
# "most_freq_rel": -1,
# "lr": 0.1,
# "weight_decay": 0.9,
# "adaptive": False,
# "epochs": 30,
# "init_strategy": "normal", 
# "threshold": 0.5,
# #"experiment": f"RGCNExplainer_{name}_{node_idx}_playground",
# "hops": 2,
# "try": '', 
# "kill_most_freq_rel": True,
# "relation_id": 39, 
# "break_if_wrong_pred": False,
# "break_on_number_of_high": False,
# "print": False,
# "explain_all": True,
# "dataset_name": 'aifb',
# "prune": False, 
# "funky_loss": False, 
# "num_exp": 15,
# }


# inits = ['overall_frequency','relative_frequency','inverse_relative_frequency','domain_frequency','range_frequency']
# for i in inits:
    # default_params = {
    # "sweep": False,
    # "pred": 1,
    # "size": 5e-05,  
    # "ent": 1,
    # "most_freq_rel": 1,
    # "lr": 0.5,
    # "weight_decay": 0.9,
    # "adaptive": False,
    # "epochs": 30,
    # "init_strategy": i, 
    # "threshold": 0.5,
    # "hops": 2,
    # "try": '', 
    # "kill_most_freq_rel": True,
    # "relation_id": 39, 
    # "break_if_wrong_pred": False,
    # "break_on_number_of_high": False,
    # "print": False,
    # "funky_loss": False, 
    # "num_exp": 15,  
    # }
default_params = {
#"sweep": False,
"pred": 1,
"size": 0.0005,  
"ent": 1,
"most_freq_rel": 1,
"lr": 0.1,
"weight_decay": 0.9,
"adaptive": False,
"epochs": 30,
"init_strategy": 'normal', 
"threshold": 0.3,
"hops": 2,
"try": '', 
"kill_most_freq_rel": False,
"relation_id": 39, 
"break_if_wrong_pred": False,
"break_on_number_of_high": False,
"print": False,
#"explain_all": True ,
#"dataset_name": 'mutag',
#"prune": True, 
"funky_loss": False, 
"num_exp": 15,  
}




