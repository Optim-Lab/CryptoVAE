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
class ProTran(nn.Module):
    def __init__(self, config, device):
        super(ProTran, self).__init__()
        self.config = config
        self.M = config["M"]
        self.device = device
        
        """Generative model"""
        self.fc_C = nn.Linear(config["p"], config["d_model"])
        self.add_posit_C = layers.AddPosition2(config["d_model"], config["timesteps"], device)
        self.prior = layers.PriorModule(self.config, device) 

        """Inference model"""
        self.fc_T = nn.Linear(config["p"], config["d_model"])
        self.add_posit_T = layers.AddPosition2(config["d_model"], config["timesteps"] + config["future"], device)
        self.posterior = layers.PosteriorModule(self.config, self.prior, device) 
        
        self.decoder = nn.ModuleList(
            [nn.Linear(config["d_latent"], config["p"])
             for _ in range(config["timesteps"] + config["future"])])
        # self.decoder = nn.ModuleList(
        #     [nn.Sequential(
        #         nn.Linear(config["d_latent"], 16),
        #         nn.ELU(),
        #         nn.Linear(16, 4 * config["p"])) 
        #     for _ in range(config["timesteps"] + config["future"])])
    
    def get_prior(self, context_batch):
        h_C = self.add_posit_C(self.fc_C(context_batch))
        prior_w, prior_z, prior_mean, prior_logvar = self.prior(h_C)
        return prior_w, prior_z, prior_mean, prior_logvar
    
    def get_decode(self, z):
        xhat = list(map(lambda x, d: d(x.squeeze())[:, None, :], z, self.decoder))
        xhat = torch.cat(xhat, dim=1)
        return xhat
    
    def forward(self, context_batch, target_batch):
        h_C = self.add_posit_C(self.fc_C(context_batch))
        h_T = self.add_posit_T(self.fc_T(target_batch))
        
        prior_z_list = []
        prior_mean_list = []
        prior_logvar_list = []
        
        posterior_z_list = []
        posterior_mean_list = []
        posterior_logvar_list = []
        
        for i in range(self.config["num_layer"]):
            if i == 0:
                prior_w, prior_z, prior_mean, prior_logvar = self.prior(h_C)
                posterior_w, posterior_z, posterior_mean, posterior_logvar = self.posterior(h_C, h_T)
            else:
                prior_w, prior_z, prior_mean, prior_logvar = self.prior(h_C, torch.cat(prior_w, dim=1))
                posterior_w, posterior_z, posterior_mean, posterior_logvar = self.posterior(h_C, h_T, torch.cat(posterior_w, dim=1))

            prior_z_list.append(prior_z)
            prior_mean_list.append(prior_mean)
            prior_logvar_list.append(prior_logvar)
            
            posterior_z_list.append(posterior_z)
            posterior_mean_list.append(posterior_mean)
            posterior_logvar_list.append(posterior_logvar)
            
        xhat = self.get_decode(posterior_z)
        
        return (prior(prior_z_list, prior_mean_list, prior_logvar_list), 
                posterior(posterior_z_list, posterior_mean_list, posterior_logvar_list),
                xhat)
    
    def est_quantile(self, test_context, alphas, MC, test_len, disable=False):
        torch.manual_seed(self.config["seed"])
        if self.config["cuda"]:
            torch.cuda.manual_seed(self.config["seed"])
        
        samples = []
        for _ in tqdm.tqdm(range(MC), desc=f"Sampling...", disable=disable):
            with torch.no_grad():
                _, prior_z, _, _ = self.get_prior(test_context.to(self.device))
                xhat = self.get_decode(prior_z)
                samples.append(xhat[:, self.config["timesteps"]:, :])
        samples = torch.stack(samples)
        
        est_quantiles = []
        for i, a in enumerate(alphas):
            est_quantiles.append(samples.quantile(q=a, dim=0))
            
        samples = samples[:, -test_len:, :, :].reshape(self.config["MC"], -1, self.config["p"]).permute(1, 0, 2)
        return est_quantiles, samples
#%% 