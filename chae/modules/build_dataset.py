#%%
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm
#%%
# def build_mixture(config, device):
#     torch.random.manual_seed(config["seed"])
#     n = config["n"]
    
#     samples = torch.randn((n, 1))
#     component1 = -3 + 1.5 * samples
#     component2 = 3 + 0.5 * samples
    
#     flag = torch.bernoulli(torch.ones((n, 1)) * 0.5)
#     data = component1 * flag
#     data += component2 * (1. - flag)
#     data = data.to(device)
#     return data
#%%
def build_mixture(config, device):
    n = config["n"]
    
    loc, scale = 3, 0.5
    component1 = truncnorm.rvs(-1.5, 1.5, loc=loc, scale=scale, size=n//2)
    loc, scale = -3, 1.5
    component2 = truncnorm.rvs(-1.5, 1.5, loc=loc, scale=scale, size=n//2)
    
    data = torch.from_numpy(np.concatenate([component1, component2])).to(torch.float)
    data = data.to(device)[:, None]
    return data
#%%