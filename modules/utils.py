#%%
import numpy as np
import random
import torch
import yaml
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
#%%
def load_config(config):
    config_path = f'./configs/{config["model"]}.yaml'
    with open(config_path, 'r') as config_file:
        args = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in config.keys():
        if key in args.keys():
            config[key] = args[key]
    return config
#%%