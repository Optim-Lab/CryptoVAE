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
class ExpLog(nn.Module):
    def __init__(self, config, device):
        super(ExpLog, self).__init__()
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
        
        self.spline = nn.ModuleList(
            [nn.Linear(config["d_latent"], 2 * config["p"])
             for _ in range(config["timesteps"] + config["future"])])
        # self.spline = nn.ModuleList(
        #     [nn.Sequential(
        #         nn.Linear(config["d_latent"], 16),
        #         nn.ELU(),
        #         nn.Linear(16, 4 * config["p"])) 
        #     for _ in range(config["timesteps"] + config["future"])])
    
    def quantile_parameter(self, h):
        h = torch.split(h, 2, dim=1)
        theta1 = [nn.Sigmoid()(h_[:, [0]]) for h_ in h]
        theta2 = [nn.Softplus()(h_[:, [1]]) for h_ in h]
        return theta1, theta2
    
    def get_prior(self, context_batch):
        h_C = self.add_posit_C(self.fc_C(context_batch))
        prior_w, prior_z, prior_mean, prior_logvar = self.prior(h_C)
        return prior_w, prior_z, prior_mean, prior_logvar
    
    def get_spline(self, z):
        spline_feature = list(map(lambda x, d: d(x.squeeze()), z, self.spline))
        params = list(map(lambda x: self.quantile_parameter(x), spline_feature))
        return params
    
    def quantile_function(self, tau, theta1, theta2):
        Q = ((1 - theta1 + 1e-6).log() - (1 - theta1 ** (1 - tau) + 1e-6).log()) / theta2
        # Q = ((1 - theta1) / (1 - theta1 ** (1 - tau))).log() / theta2
        return Q
    
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
            
        params = self.get_spline(posterior_z)
        
        return (prior(prior_z_list, prior_mean_list, prior_logvar_list), 
                posterior(posterior_z_list, posterior_mean_list, posterior_logvar_list),
                params)
    
    def est_quantile(self, test_context, alphas, MC, disable=False):
        torch.manual_seed(self.config["seed"])
        if self.config["cuda"]:
            torch.cuda.manual_seed(self.config["seed"])
        
        est_quantiles = []
        for a in alphas:
            Qs = []
            for _ in tqdm.tqdm(range(MC), desc=f"Quantile estimation...(alpha={a})", disable=disable):
                with torch.no_grad():
                    _, prior_z, _, _ = self.get_prior(test_context.to(self.device))
                    params = self.get_spline(prior_z)
                
                theta1 = torch.cat([torch.cat(params[self.config["timesteps"]+t][0], dim=1) for t in range(self.config["future"])])
                theta2 = torch.cat([torch.cat(params[self.config["timesteps"]+t][1], dim=1) for t in range(self.config["future"])])
                
                alpha = (torch.ones(theta1.shape) * a).to(self.device)
                
                Qs_ = self.quantile_function(alpha, theta1, theta2)
                Qs_ = torch.cat([x[:, None, :] for x in torch.split(Qs_, len(test_context), dim=0)], dim=1)
                Qs.append(Qs_[::self.config["future"], :, :].reshape(-1, self.config["p"])[:, None, :])
            Qs = torch.cat(Qs, dim=1).mean(dim=1)
            est_quantiles.append(Qs.cpu())
        return est_quantiles
    
    def sampling(self, test_context, MC, disable=False):
        torch.manual_seed(self.config["seed"])
        if self.config["cuda"]:
            torch.cuda.manual_seed(self.config["seed"])
        
        samples = []
        for _ in tqdm.tqdm(range(MC), desc=f"Data sampling...", disable=disable):
            with torch.no_grad():
                _, prior_z, _, _ = self.get_prior(test_context.to(self.device))
                params = self.get_spline(prior_z)
            
            theta1 = torch.cat([torch.cat(params[self.config["timesteps"]+t][0], dim=1) for t in range(self.config["future"])])
            theta2 = torch.cat([torch.cat(params[self.config["timesteps"]+t][1], dim=1) for t in range(self.config["future"])])
            
            alpha = torch.rand(theta1.shape).to(self.device)
            
            Qs_ = self.quantile_function(alpha, theta1, theta2)
            Qs_ = torch.cat([x[:, None, :] for x in torch.split(Qs_, len(test_context), dim=0)], dim=1)
            samples.append(Qs_[::self.config["future"], :, :].reshape(-1, self.config["p"])[:, None, :])
        samples = torch.cat(samples, dim=1)
        return samples.cpu()
#%%