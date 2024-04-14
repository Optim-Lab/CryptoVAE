#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
#%%
class GLDDecoder(nn.Module):
    def __init__(self, config, device):
        super(GLDDecoder, self).__init__()
        
        self.config = config
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, config["latent_dim"] * 2),
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        ).to(device)
    
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar):
        noise = torch.randn(mean.size(0), self.config["latent_dim"]).to(self.device) 
        z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar)
        return z, mean, logvar
    
    def forward(self, input):
        z, mean, logvar = self.encode(input)
        
        h = self.decoder(z)
        theta1 = h[:, [0]]
        theta2 = nn.Softplus()(h[:, [1]])
        if self.config["model"] == 'GLD_finite':
            # finite support
            theta3 = (h[:, [2]]).exp()
            theta4 = (h[:, [3]]).exp()
        elif self.config["model"] == 'GLD_infinite':
            # infinite support
            theta3 = -(h[:, [2]]).exp()
            theta4 = -(h[:, [3]]).exp()
        return z, mean, logvar, theta1, theta2, theta3, theta4

    def decode(self, z):
        h = self.decoder(z)
        theta1 = h[:, [0]]
        theta2 = nn.Softplus()(h[:, [1]])
        if self.config["model"] == 'GLD_finite':
            # finite support
            theta3 = (h[:, [2]]).exp()
            theta4 = (h[:, [3]]).exp()
        elif self.config["model"] == 'GLD_infinite':
            # infinite support
            theta3 = -(h[:, [2]]).exp()
            theta4 = -(h[:, [3]]).exp()
        return theta1, theta2, theta3, theta4
    
    """Generalized Lambda distribution"""
    def quantile_function(self, tau, theta1, theta2, theta3, theta4):
        Q = (tau ** theta3 - 1) / theta3
        Q -= ((1 - tau) ** theta4 - 1) / theta4
        return theta1 + 1 / theta2 * Q
#%%
class GaussianDecoder(nn.Module):
    def __init__(self, config, device):
        super(GaussianDecoder, self).__init__()
        
        self.config = config
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, config["latent_dim"] * 2),
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            # nn.Linear(32, 2),
        ).to(device)
    
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar):
        noise = torch.randn(mean.size(0), self.config["latent_dim"]).to(self.device) 
        z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar)
        return z, mean, logvar
    
    def forward(self, input):
        z, mean, logvar = self.encode(input)
        
        xhat = self.decoder(z)
        return z, mean, logvar, xhat
        
        # h = self.decoder(z)
        # xhat, logsigma = h[:, [0]], h[:, [1]]
        # return z, mean, logvar, xhat, logsigma
#%%
class LSQFDecoder(nn.Module):
    def __init__(self, config, device):
        super(LSQFDecoder, self).__init__()
        
        self.config = config
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, config["latent_dim"] * 2),
        ).to(device)
        
        self.delta = torch.arange(0, 1 + config["step"], step=config["step"]).view(1, -1).to(device)
        self.M = self.delta.size(1) - 1
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1 + (self.M + 1)),
        ).to(device)
        
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar):
        noise = torch.randn(mean.size(0), self.config["latent_dim"]).to(self.device) 
        z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar)
        return z, mean, logvar
    
    def forward(self, input):
        z, mean, logvar = self.encode(input)
        
        h = self.decoder(z)
        gamma = h[:, [0]]
        beta = torch.cat([
            torch.zeros_like(gamma),
            nn.ReLU()(h[:, 1:]) # positive constraint
        ], dim=1)
        beta = beta[:, 1:] - beta[:, :-1]
        return z, mean, logvar, gamma, beta

    def decode(self, z):
        h = self.decoder(z)
        gamma = h[:, [0]]
        beta = torch.cat([
            torch.zeros_like(gamma),
            nn.ReLU()(h[:, 1:]) # positive constraint
        ], dim=1)
        beta = beta[:, 1:] - beta[:, :-1]
        return gamma, beta
    
    def quantile_function(self, alpha, gamma, beta):
        return gamma + (beta * torch.where(
            alpha - self.delta > 0,
            alpha - self.delta,
            torch.zeros(()).to(self.device)
        )).sum(dim=1, keepdims=True)
        
    def quantile_inverse(self, x, gamma, beta):
        delta_ = self.delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(
            delta_ - self.delta > 0,
            delta_ - self.delta,
            torch.zeros(()).to(self.device))
        mask = gamma + (beta * delta_.unsqueeze(2)).sum(dim=-1).squeeze(0).t()
        mask = torch.where(
            mask <= x, 
            mask, 
            torch.zeros(()).to(self.device)).type(torch.bool).type(torch.float)
        alpha_tilde = x - gamma
        alpha_tilde += (mask * beta * self.delta).sum(dim=1, keepdims=True)
        alpha_tilde /= (mask * beta).sum(dim=1, keepdims=True) + 1e-6
        alpha_tilde = torch.clip(alpha_tilde, self.config["threshold"], 1) # numerical stability
        return alpha_tilde
#%%