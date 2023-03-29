#%%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import layers

from collections import namedtuple
prior = namedtuple(
    'prior', 
    ['z', 'mean', 'logvar'])
posterior = namedtuple(
    'posterior', 
    ['z', 'mean', 'logvar'])
#%%
class DDM(nn.Module):
    def __init__(self, config):
        super(DDM, self).__init__()
        self.config = config
        self.M = config["M"]
        
        """Generative model"""
        self.fc_C = nn.Linear(config["p"], config["d_model"])
        self.add_posit_C = layers.AddPosition2(config["d_model"], config["timesteps"])
        self.prior = layers.PriorModule(self.config) 

        """Inference model"""
        self.fc_T = nn.Linear(config["p"], config["d_model"])
        self.add_posit_T = layers.AddPosition2(config["d_model"], config["timesteps"] * 2)
        self.posterior = layers.PosteriorModule(self.config, self.prior) 
        
        self.spline = nn.ModuleList(
            [nn.Linear(config["d_latent"], (1 + (config["M"] + 1) + (config["M"])) * config["p"]) 
            for _ in range(config["timesteps"] + config["future"])])
    
    def quantile_parameter(self, h):
        h = torch.split(h, 1 + (self.M + 1) + (self.M), dim=1)
        gamma = [h_[:, [0]] for h_ in h]
        beta = [nn.Softplus()(h_[:, 1:self.M+2]) for h_ in h] # positive constraint
        delta = [torch.cat([
            torch.zeros((h_.size(0), 1)),
            nn.Softmax(dim=1)(h_[:, self.M+2:] / self.config["tau"]).cumsum(dim=1)
        ], dim=1) for h_ in h] # positive constraint
        return gamma, beta, delta
    
    def quantile_inverse(self, x, gamma, beta, delta):
        delta_ = delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(delta.unsqueeze(1) - delta_ > 0,
                            delta.unsqueeze(1) - delta_,
                            torch.zeros(()))
        beta_ = beta.unsqueeze(2).repeat(1, 1, self.M + 1)
        mask = gamma + (beta_ * delta_).sum(dim=1)
        mask = torch.where(mask <= x, 
                        mask, 
                        torch.zeros(())).type(torch.bool).type(torch.float)
        alpha_tilde = x - gamma
        alpha_tilde += (mask * beta * delta).sum(dim=1, keepdims=True)
        alpha_tilde /= (mask * beta).sum(dim=1, keepdims=True) + 1e-6
        alpha_tilde = torch.clip(alpha_tilde, self.config["threshold"], 1) # numerical stability
        return alpha_tilde
    
    def get_prior(self, context_batch):
        h_C = self.add_posit_C(self.fc_C(context_batch))
        _, prior_mean, prior_logvar = self.prior(h_C)
        return prior_mean, prior_logvar
    
    def get_spline(self, z):
        spline_feature = list(map(lambda x, d: d(x.squeeze()), 
            torch.split(z, 1, dim=1), self.spline))
        params = list(map(lambda x: self.quantile_parameter(x), spline_feature))
        return params
    
    def quantile_function(self, alpha, gamma, beta, delta):
        return gamma + (beta * torch.where(
            alpha - delta > 0,
            alpha - delta,
            torch.zeros(()))).sum(dim=1, keepdims=True)
    
    def forward(self, context_batch, target_batch):
        h_C = self.add_posit_C(self.fc_C(context_batch))
        h_T = self.add_posit_T(self.fc_T(target_batch))
        
        prior_z, prior_mean, prior_logvar = self.prior(h_C)
        posterior_z, posterior_mean, posterior_logvar = self.posterior(h_C, h_T)
        
        params = self.get_spline(posterior_z)
            
        return (prior(prior_z, prior_mean, prior_logvar), 
                posterior(posterior_z, posterior_mean, posterior_logvar),
                params)
#%%