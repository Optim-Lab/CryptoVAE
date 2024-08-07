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
class LSQF(nn.Module):
    def __init__(self, config, device):
        super(LSQF, self).__init__()
        self.config = config
        self.M = config["M"]
        self.device = device
        
        """Generative model"""
        self.fc_C = nn.Linear(config["p"], config["d_model"])
        self.add_posit_C = layers.AddPosition2(config["d_model"], config["timesteps"], device)
        self.prior = layers.PriorModule(self.config, device) 

        """Inference model"""
        self.fc_T = nn.Linear(config["p"], config["d_model"])
        self.add_posit_T = layers.AddPosition2(config["d_model"], config["future"], device)
        self.posterior = layers.PosteriorModule(self.config, self.prior, device) 
        
        self.spline = nn.ModuleList(
            [nn.Linear(config["d_latent"], (1 + (config["M"] + 1)) * config["p"])
             for _ in range(config["future"])])
        # self.spline = nn.ModuleList(
        #     [nn.Sequential(
        #         nn.Linear(config["d_latent"], 32),
        #         nn.ELU(),
        #         nn.Linear(32, (1 + (config["M"] + 1)) * config["p"])) 
        #     for _ in range(config["future"])])
    
    def quantile_parameter(self, h):
        h = torch.split(h, 1 + (self.M + 1), dim=1)
        gamma = [h_[:, [0]] for h_ in h]
        beta = [nn.Softplus()(h_[:, 1:self.M+2]) for h_ in h] # positive constraint
        delta = [torch.linspace(0, 1, self.config["M"] + 1)[None, :].repeat((h_.size(0), 1)).to(self.device)
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
    
    def get_prior(self, context_batch):
        h_C = self.add_posit_C(self.fc_C(context_batch))
        prior_w, prior_z, prior_mean, prior_logvar = self.prior(h_C)
        return prior_w, prior_z, prior_mean, prior_logvar
    
    def get_spline(self, z):
        spline_feature = list(map(lambda x, d: d(x.squeeze()), z, self.spline))
        params = list(map(lambda x: self.quantile_parameter(x), spline_feature))
        return params
    
    def quantile_function(self, alpha, gamma, beta, delta):
        return gamma + (beta * torch.where(
            alpha - delta > 0,
            alpha - delta,
            torch.zeros(()).to(self.device)
            )).sum(dim=1, keepdims=True)
    
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
                
                Qs_tmp = []
                for j in range(self.config["p"]):
                    gamma = torch.cat([params[t][0][j] for t in range(self.config["future"])], dim=0)
                    beta = torch.cat([params[t][1][j] for t in range(self.config["future"])], dim=0)
                    delta = torch.cat([params[t][2][j] for t in range(self.config["future"])], dim=0)
                    
                    alpha = (torch.ones(gamma.shape) * a).to(self.device)
                    
                    Qs_ = self.quantile_function(alpha, gamma, beta, delta)
                    Qs_ = torch.cat([x[:, None, :] for x in torch.split(Qs_, len(test_context), dim=0)], dim=1)
                    Qs_tmp.append(Qs_)
                Qs.append(torch.cat(Qs_tmp, dim=-1).cpu())
            est_quantiles.append(torch.mean(torch.stack(Qs), dim=0))
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
            
            Qs_tmp = []
            for j in range(self.config["p"]):
                gamma = torch.cat([params[t][0][j] for t in range(self.config["future"])], dim=0)
                beta = torch.cat([params[t][1][j] for t in range(self.config["future"])], dim=0)
                delta = torch.cat([params[t][2][j] for t in range(self.config["future"])], dim=0)
                
                alpha = torch.rand(gamma.shape).to(self.device)
                
                Qs_ = self.quantile_function(alpha, gamma, beta, delta)
                Qs_ = torch.cat([x[:, None, :] for x in torch.split(Qs_, len(test_context), dim=0)], dim=1)
                Qs_tmp.append(Qs_)
            samples.append(torch.cat(Qs_tmp, dim=-1).reshape(-1, self.config["p"])[:, None, :])
        samples = torch.cat(samples, dim=1)
        return samples.cpu()
#%%