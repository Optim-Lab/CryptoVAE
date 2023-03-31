#%%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()
        self.config = config
        self.M = config["M"]
        self.device = device
        
        """Prior"""
        self.fc_C1 = nn.Linear(config["p"], config["d_model"])
        self.fc_C2 = nn.Linear(config["timesteps"] * config["d_model"], config["d_model"])
        self.prior_posit_matrix = layers.positional_encoding(config["timesteps"], config["d_model"]).to(device)
        self.prior = nn.ModuleList([nn.Sequential(
            nn.Linear(config["d_model"] + config["d_latent"], 8),
            nn.ELU(),
            nn.Linear(8, config["d_latent"])
        )
        for i in range(config["timesteps"] + config["future"])])
        
        """Inference"""
        self.fc_T1 = nn.Linear(config["p"], config["d_model"])
        self.fc_T2 = nn.Linear((config["timesteps"] + config["future"]) * config["d_model"], config["d_model"])
        self.posterior_posit_matrix = layers.positional_encoding(config["timesteps"] + config["future"], config["d_model"]).to(device)
        self.posterior = nn.ModuleList([nn.Sequential(
            nn.Linear(config["d_model"] + config["d_latent"], 8),
            nn.ELU(),
            nn.Linear(8, config["d_latent"] * 2)
        )
        for i in range(config["timesteps"] + config["future"])])
        
        """Spline"""
        self.spline = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(config["d_latent"], 8),
                nn.ELU(),
                nn.Linear(8, (1 + (config["M"] + 1)) * config["p"])) 
            for _ in range(config["timesteps"] + config["future"])])
    
    def quantile_parameter(self, h):
        h = torch.split(h, 1 + (self.M + 1), dim=1)
        gamma = [h_[:, [0]] for h_ in h]
        beta = [nn.Softplus()(h_[:, 1:self.M+2]) for h_ in h] # positive constraint
        delta = [torch.linspace(0, 1, self.config["M"] + 1)[None, :].repeat((h_.size(0), 1))
                 for h_ in h]
        # delta = [torch.cat([
        #     torch.zeros((h_.size(0), 1)).to(self.device),
        #     nn.Softmax(dim=1)(h_[:, self.M+2:] / self.config["tau"]).cumsum(dim=1)
        # ], dim=1) for h_ in h] # positive constraint
        return gamma, beta, delta
    
    def quantile_inverse(self, x, gamma, beta, delta):
        delta_ = delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(
            delta.unsqueeze(1) - delta_ > 0,
            delta.unsqueeze(1) - delta_,
            torch.zeros(()).to(self.device))
        beta_ = beta.unsqueeze(2).repeat(1, 1, self.M + 1)
        mask = gamma + (beta_ * delta_).sum(dim=1)
        mask = torch.where(
            mask <= x, 
            mask, 
            torch.zeros(()).to(self.device)
        ).type(torch.bool).type(torch.float).to(self.device)
        alpha_tilde = x - gamma
        alpha_tilde += (mask * beta * delta).sum(dim=1, keepdims=True)
        alpha_tilde /= (mask * beta).sum(dim=1, keepdims=True) + 1e-6
        alpha_tilde = torch.clip(alpha_tilde, self.config["threshold"], 1) # numerical stability
        return alpha_tilde
    
    def quantile_function(self, alpha, gamma, beta, delta):
        return gamma + (beta * torch.where(
            alpha - delta > 0,
            alpha - delta,
            torch.zeros(()).to(self.device)
            )).sum(dim=1, keepdims=True)
    
    def get_prior(self, context_batch):
        h_C = self.fc_C1(context_batch) + self.prior_posit_matrix
        h_C = h_C.reshape(-1, self.config["timesteps"] * self.config["d_model"])
        h_C = self.fc_C2(h_C)
        
        z_ = torch.zeros((h_C.size(0), self.config["d_latent"])).to(self.device)
        z_list = []
        mean_list = []
        logvar_list = []
        for net in self.prior:
            mean = net(torch.cat([h_C, z_], dim=1))
            var = self.config["prior_var"] * torch.ones(mean.shape).to(self.device)
            # mean, logvar = torch.split(net(h_C), self.config["d_latent"], dim=1)
            epsilon = torch.randn(mean.shape).to(self.device)
            z = mean + var.sqrt() * epsilon
            
            z_list.append(z)
            mean_list.append(mean)
            logvar_list.append(var.log())
            
            # h_C = torch.cat([z, h_C], dim=1) # z[1:t-1], x[1:C]
            z_ = 0.1 * z_ + 0.9 * z
        return z_list, mean_list, logvar_list
    
    def get_posterior(self, target_batch):
        h_T = self.fc_T1(target_batch) + self.posterior_posit_matrix
        h_T = h_T.reshape(-1, (self.config["timesteps"] + self.config["future"]) * self.config["d_model"])
        h_T = self.fc_T2(h_T)
        
        z_ = torch.zeros((h_T.size(0), self.config["d_latent"])).to(self.device)
        z_list = []
        mean_list = []
        logvar_list = []
        for net in self.posterior:
            mean, logvar = torch.split(net(torch.cat([h_T, z_], dim=1)), self.config["d_latent"], dim=1)
            epsilon = torch.randn(mean.shape).to(self.device)
            z = mean + (logvar / 2).exp() * epsilon
            
            z_list.append(z)
            mean_list.append(mean)
            logvar_list.append(logvar)
            
            # h_T = torch.cat([z, h_T], dim=1) # z[1:t-1], x[1:T]
            z_ = 0.1 * z_ + 0.9 * z
        return z_list, mean_list, logvar_list
    
    def get_spline(self, z):
        spline_feature = list(map(lambda d, x : d(x), self.spline, z))
        params = list(map(lambda x: self.quantile_parameter(x), spline_feature))
        return params
    
    def forward(self, context_batch, target_batch):
        prior_z, prior_mean, prior_logvar = self.get_prior(context_batch)
        posterior_z, posterior_mean, posterior_logvar = self.get_posterior(target_batch)
        
        params = self.get_spline(posterior_z)
            
        return (prior(prior_z, prior_mean, prior_logvar), 
                posterior(posterior_z, posterior_mean, posterior_logvar),
                params)
#%%