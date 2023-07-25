
sweep_config = {
    'method': 'grid', 
}

metric = {
    'name': 'score_threshold',
    'goal': 'maximize'   
}
sweep_config['metric'] = metric

parameters_dict = {
    'pred': {'values': [1]},
    'lr': {
        'values': [0.1, 0.5]
    },
    'weight_decay': {
        'values': [0.9, 0.1]
    },
    'size': {
        'values': [0.0005,0.005]

    },
    'size_std': {'values': [1380.429*0.01,1380.429*0.1,10]},

    'ent': {
        'values': [1]
    },
    'most_freq_rel': {
        'values': [1,10]
    },
    'adaptive': {
        'values': [False]
    },
    'kill_most_freq_rel': {
        'values': [True,False]
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








# default_params = {
# "pred": 1,
# "size": 0.0005,  
# "ent": 1,
# "most_freq_rel": 1,
# "lr": 0.1,
# "weight_decay": 0.9,
# "epochs": 30,
# "threshold": 0.5,
# "hops": 2,
# "num_exp": 15,   
# "adaptive": False,
# "kill_most_freq_rel": False,
# "relation_id": 39, 
# "break_if_wrong_pred": False,
# "break_on_number_of_high": False,
# "print": True,
# "funky_loss": False, 
# }

default_params = {
"pred": 1,
"size": 0.005,  
"ent": 1,
"size_std": 1380.429*0.1,
"most_freq_rel": 1,
"lr": 0.5,
"weight_decay": 0.9,
"epochs": 30,
"threshold": 0.5,
"hops": 2,
"num_exp": 15,   
"adaptive": False,
"kill_most_freq_rel": False,
"relation_id": 2, 
"break_if_wrong_pred": False,
"break_on_number_of_high": False,
"print": False,
"funky_loss": False, 
}


