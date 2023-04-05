#%%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tqdm

import importlib
layers = importlib.import_module('layers')
importlib.reload(layers)

from collections import namedtuple
prior = namedtuple(
    'prior', 
    ['z', 'mean', 'logvar'])
posterior = namedtuple(
    'posterior', 
    ['z', 'mean', 'logvar'])
#%%
class TLAE(nn.Module):
    def __init__(self, config, device):
        super(TLAE, self).__init__()
        self.config = config
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(config["p"], config["d_latent"]),
            nn.ELU(),
            nn.Linear(config["d_latent"], config["d_latent"]),
            nn.ELU(),
            nn.Linear(config["d_latent"], config["d_latent"])).to(device)
        
        """temporal structure"""
        self.lstm = nn.LSTM(config["d_latent"], config["d_latent"], config["num_layer"], batch_first=True).to(self.device)

        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["d_latent"], config["d_latent"]),
            nn.ELU(),
            nn.Linear(config["d_latent"], config["d_latent"]),
            nn.ELU(),
            nn.Linear(config["d_latent"], config["p"])).to(device)
    
    def get_prior(self, context_batch, deterministic=False):
        z = self.encoder(context_batch)
    
        out, (hn, cn) = self.lstm(z)
        
        future_mu = [out[:, [-1], :]]
        future_z = [future_mu[-1] + torch.randn(future_mu[-1].shape)]
        
        for _ in range(self.config["future"]-1):
            if deterministic:
                out, (hn, cn) = self.lstm(future_mu[-1], (hn, cn))
            else:
                out, (hn, cn) = self.lstm(future_z[-1], (hn, cn))
            future_mu.append(out)
            future_z.append(out + torch.randn(out.shape))
        
        future_mu = torch.cat(future_mu, dim=1)
        future_z = torch.cat(future_z, dim=1)
        z = torch.cat([z, future_z], dim=1)
        return z, future_mu, future_z
    
    def forward(self, context_batch, deterministic=False):
        z, future_mu, future_z = self.get_prior(context_batch, deterministic=deterministic)
        xhat = self.decoder(z)
        return xhat, future_mu, future_z
    
    def est_quantile(self, test_context, alphas, MC, disable=False):
        samples = []
        for _ in tqdm.tqdm(range(MC), desc=f"Sampling...", disable=disable):
            with torch.no_grad():
                xhat, _, _ = self.forward(test_context, deterministic=True)
                samples.append(xhat[:, self.config["timesteps"]:, :])
        samples = torch.cat(samples, dim=1)
        
        est_quantiles = []
        for i, a in enumerate(alphas):
            est_quantiles.append(samples.quantile(q=a, dim=1))
        return est_quantiles, samples
#%%