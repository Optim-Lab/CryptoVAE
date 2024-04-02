#%%
import torch
import torch.nn as nn

import numpy as np
from scipy.stats import t
#%%
def build_heavytailed(config, device):
    torch.random.manual_seed(config["seed"])
    n = config["n"]
    
    df = 3 # degrees of freedom
    data = t.rvs(df, size=(n, 1), random_state=0)
    data = np.clip(data, -20, 20)
    data = torch.from_numpy(data).to(device).float()
    return data
#%%
def build_uniform(config, device):
    torch.random.manual_seed(config["seed"])
    n = config["n"]
    data = torch.rand((n, 1)) * 3
    data = data.to(device)
    return data
#%%
def build_mixture(config, device):
    torch.random.manual_seed(config["seed"])
    n = config["n"]
    
    samples = torch.randn((n, 1))
    component1 = -3 + 1.5 * samples
    component2 = 3 + 0.5 * samples
    
    flag = torch.bernoulli(torch.ones((n, 1)) * 0.5)
    data = component1 * flag
    data += component2 * (1. - flag)
    data = data.to(device)
    return data
#%%
#%%
# def build_heavytailed(config, device):
#     torch.random.manual_seed(config["seed"])
#     n = config["n"]
#     z = torch.randn((n, config["latent_dim"]))

#     true_decoder = nn.Sequential(
#         nn.Linear(config["latent_dim"], 128),
#         nn.Sigmoid(),
#         nn.Linear(128, 2),
#     ).to(device)
#     with torch.no_grad():
#         params = true_decoder(z)
    
#     df = 3 # degrees of freedom
#     samples = t.rvs(df, size=(n, 1), random_state=0)
#     data = samples * params[:, [0]].cpu().numpy() + params[:, [1]].cpu().numpy()
#     data = torch.from_numpy(data).to(device)
#     return data
# #%%
# def build_uniform(config, device):
#     torch.random.manual_seed(config["seed"])
#     n = config["n"]
#     z = torch.randn((n, config["latent_dim"]))

#     true_decoder = nn.Sequential(
#         nn.Linear(config["latent_dim"], 128),
#         nn.Sigmoid(),
#         nn.Linear(128, 2),
#     ).to(device)
#     with torch.no_grad():
#         params = true_decoder(z)
    
#     lower = params[:, [0]]
#     upper = params[:, [0]] + (params[:, [1]]).exp()
#     samples = torch.rand((len(z), 1))
#     data = (upper - lower) * samples + lower
#     data = data.to(device)
#     return data
# #%%
# def build_mixture(config, device):
#     torch.random.manual_seed(config["seed"])
#     n = config["n"]
#     z = torch.randn((n, config["latent_dim"]))

#     true_decoder = nn.Sequential(
#         nn.Linear(config["latent_dim"], 128),
#         nn.Sigmoid(),
#         nn.Linear(128, 4),
#     ).to(device)
#     with torch.no_grad():
#         params = true_decoder(z)
    
#     samples = torch.randn((len(z), 1))
    
#     mean1 = -3 * nn.Softplus()(params[:, [0]])
#     logvar1 = params[:, [1]]
#     component1 = mean1 + logvar1.exp().sqrt() * samples
    
#     mean2 = 3 * nn.Softplus()(params[:, [2]])
#     logvar2 = params[:, [3]]
#     component2 = mean2 + logvar2.exp().sqrt() * samples
    
#     flag = torch.bernoulli(nn.Sigmoid()(z[:, [0]]))
#     # flag = (torch.rand((len(z), 1)) > 0.5).float()
    
#     data = component1 * flag
#     data += component2 * (1. - flag)
#     data = data.to(device)
#     return data
# #%%