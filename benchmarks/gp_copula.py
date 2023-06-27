import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from statsmodels.distributions.empirical_distribution import ECDF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GPCopula(nn.Module):
    def __init__(self, d_input, d_hidden, tau, beta, rank, n_layers, dr, device, norm="standard"):
        super(GPCopula, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden 
        self.tau = tau
        self.device = device
        self.norm = norm 
        self.lstm_list = nn.ModuleList([nn.LSTM(1, d_hidden, n_layers, dropout=dr, batch_first=True) for _ in range(self.d_input)])
        self.linear_list = nn.ModuleList([nn.Linear(d_hidden, 1) for _ in range(self.d_input)])
        
        self.d_map = nn.Linear(d_hidden, 1, bias=False)
        
        if self.norm == "min-max":
            self.mu_activation = nn.Sigmoid()
        else:
            self.mu_activation = nn.Identity()
        self.v_map = nn.Linear(d_hidden, rank, bias=False)
        self.mu_map = nn.Linear(d_hidden, 1, bias=False)
        self.softplus = nn.Softplus(beta=beta)
         
    def forward(self, x):
        hidden_list = []
        
        for i in range(self.d_input):
            _, (hidden, cell) = self.lstm_list[i](x[..., i:i+1])

            if self.tau > 1: 
                
                tmp_hidden = []
                tmp_hidden.append(hidden[-1:, ...])
                
                for _ in range(self.tau-1):
                    tmp_z = self.linear_list[i](hidden[-1:, ... ].transpose(1, 0))
                    _, (hidden, cell) = self.lstm_list[i](tmp_z, (hidden, cell))
                    tmp_hidden.append(hidden[-1:, ...])
                
                hidden_ = torch.cat(tmp_hidden, dim=0)
                hidden_list.append(hidden_.unsqueeze(0)) # (1, tau, batch_size, d_hidden)
                    
            else: 
                hidden_list.append(hidden[-1:, ...].unsqueeze(0)) # (1, tau, batch_size, d_hidden)

        _hidden = torch.cat(hidden_list, dim=0) # (d_input, tau, batch_size, d_hidden)
        _hidden = _hidden.transpose(2, 0) # (batch_size, tau, d_input, d_hidden)

        mu = self.mu_activation(self.mu_map(_hidden)) # (batch_size, tau, d_input, 1)

        d = self.softplus(self.d_map(_hidden)) # (batch_size, tau, d_input, 1)
        v = self.v_map(_hidden) # (batch_size, tau, d_input, rank)

        sigma_term1 = torch.diag_embed(d.squeeze(-1))
        sigma_term2 = torch.matmul(v, torch.transpose(v, 3, 2))
        sigma = sigma_term1 + sigma_term2

        return mu, sigma

    def sample(self, mu, sigma, n):
        mvn = MultivariateNormal(mu.squeeze(), sigma)
        sample_ = mvn.rsample([n])
        mean = sample_.mean(dim=0)
        std = sample_.std(dim=0)
        
        return mean, std, sample_

def train(model, loader, criterion, optimizer, device):
    
    model.train()
    
    total_loss = []
    
    for batch in loader:
        batch_input, batch_infer = batch 
        
        batch_input = batch_input.to(device)
        batch_infer = batch_infer.to(device)

        pred = model(batch_input)
        
        loss = criterion(batch_infer, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)
    
def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss = []
    
    for batch in loader:
        batch_input, batch_infer = batch 
        
        batch_input = batch_input.to(device)
        batch_infer = batch_infer.to(device)
        
        pred = model(batch_input)
        
        loss = criterion(batch_infer, pred)
        
        total_loss.append(loss)
        
        return sum(total_loss)/len(total_loss)

class GPNegL(nn.Module):
    def __init__(self, ecdf_list, device):
        super(GPNegL, self).__init__()
        self.ecdf_list = ecdf_list 
        self.device = device
        self.norm_dist = Normal(0, 1)
        
    def forward(self, true, params):
        d_input = true.shape[-1]
        
        mu, sigma = params
        L, _ = torch.linalg.cholesky_ex(sigma)
        L_inverse = torch.inverse(L)
        det_L = torch.det(L)
        
        emp_quantile_ = []
        for i in range(d_input):
            emp_quantile_.append(torch.tensor(self.ecdf_list[i](true[..., i:i+1])).float())
        emp_quantile = torch.cat(emp_quantile_, dim=-1).to(self.device)
        gc_output = self.norm_dist.icdf(torch.clip(emp_quantile, min=0.001, max=0.999))
        
        return torch.log(det_L).mean() + (0.5 *  torch.square(L_inverse @ (gc_output.unsqueeze(-1) - mu))).sum(dim=1).mean()    

df = pd.read_csv("../data/crypto.csv", index_col=0)

for tau in [1, 5]:
    train_list, test_list, norm_param_list = build_datasets_gpcopula(df, tau, 200, 3, how="standard")

    for i in range(len(train_list)):
        train_input, train_infer = train_list[i]
        ecdf_list = [ECDF(train_input[::20, ..., i].reshape(-1)) for i in range(10)]
        train_dataset = TensorDataset(train_input, train_infer)
        # test_dataset = TensorDataset(*test_list[i])
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
        # test_loader = DataLoader(test_dataset, shuffle=True, batch_size=10)
        gp_copula = GPCopula(
            d_input=10,
            d_hidden=3,
            tau=5,
            beta=3,
            rank=10,
            n_layers=2,
            dr=0.05,
            device=device
            )
        
        gp_copula.to(device) 
        criterion = GPNegL(ecdf_list, device)
        optimizer = optim.Adam(gp_copula.parameters(), lr=0.001)
        
        pbar = tqdm(range(1000))

        for epoch in pbar:        
            train_loss = train(gp_copula, train_loader, criterion, optimizer, device)
            pbar.set_description("Train Loss: {:.4f}".format(train_loss))
            
        torch.save(gp_copula.state_dict(), './assets/GP-Copula/tau_' + str(tau) + '/GP-Copula_PHASE_{}.pth'.format(i+1))