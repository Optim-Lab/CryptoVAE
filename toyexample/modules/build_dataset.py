#%%
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm
#%%
def build_dataset(config, device):
    torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    if config["dataset"] == "uniform":
        data = torch.rand((config["n"], 1)) * 3
        data = data.to(device)
        
    elif config["dataset"] == "mixture":
        samples = 2 + torch.randn((int(config["n"] * 2/3), 1))
        data = torch.cat([
            samples, torch.ones((int(config["n"] * (1/3)), 1)) * 3
        ], dim=0)
        data = data.to(device)
        
    elif config["dataset"] == "truncated":
        loc, scale = 3, 0.5
        component1 = truncnorm.rvs(-1.5, 1.5, loc=loc, scale=scale, size=config["n"]//2)
        loc, scale = -3, 1.5
        component2 = truncnorm.rvs(-1.5, 1.5, loc=loc, scale=scale, size=config["n"]//2)
        
        data = torch.from_numpy(np.concatenate([component1, component2])).to(torch.float)
        data = data.to(device)[:, None]
        
    else:
        raise ValueError('Not valid support option for DATASET.')

    return data
#%%