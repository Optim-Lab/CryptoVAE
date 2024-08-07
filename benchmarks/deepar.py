import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from utils import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, n_layers=3, dr=0.1):
        super(Encoder, self).__init__()
        # self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.lstm = nn.LSTM(d_input, d_model, n_layers, dropout=dr, batch_first=True)
        
    def forward(self, x):
        
        _, (hidden, cell) = self.lstm(x)

        return hidden, cell

class DeepARDecoder(nn.Module):
    def __init__(self, d_enopt, d_model, num_targets, tau, beta=2, n_layers=3, dr=0.1):
        super(DeepARDecoder, self).__init__()
        self.n_layers = n_layers
        self.tau = tau
        self.beta = beta
        self.lstm = nn.LSTM(d_enopt, d_model, n_layers, dropout=dr, batch_first=True)
    
        self.linear1 = nn.Linear(d_model, num_targets)
        self.linear2 = nn.Linear(d_model, num_targets)
        self.dropout = nn.Dropout(dr)
        self.softplus = nn.Softplus(beta=self.beta)
        
    def forward(self, hidden, cell):                   
        lstm_output = []
   
        for _ in range(self.tau):
            output, (hidden, cell) = self.lstm(hidden[self.n_layers-1:self.n_layers].transpose(1, 0), (hidden, cell))
        
            lstm_output.append(output)
        
        lstm_output = torch.cat(lstm_output, axis=1)
        
        mu = self.linear1(lstm_output)
        sigma = self.softplus(self.linear2(lstm_output))
        
        return mu, sigma


class DeepAR(nn.Module):
    def __init__(self, d_input, d_model, num_targets, tau, beta=2, n_layers=2, dr=0.1):
        super(DeepAR, self).__init__()

        self.encoder = Encoder(
                               d_input=d_input,
                               d_model=d_model,
                               n_layers=n_layers,
                               dr=dr
                               )
        self.decoder = DeepARDecoder(
                                     d_enopt=d_model,
                                     d_model=d_model,
                                     num_targets=num_targets,
                                     tau=tau,
                                     beta=beta,
                                     n_layers=n_layers,
                                     dr=dr
                                     )

    def forward(self, x):
        
        encoder_hidden, encoder_cell = self.encoder(x)
        mu, sigma = self.decoder(encoder_hidden, encoder_cell)
        
        return mu, sigma

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

class NegativeGaussianLogLikelihood(nn.Module):
    def __init__(self, device):
        super(NegativeGaussianLogLikelihood, self).__init__()
        import math
        self.pi = torch.tensor(math.pi).float().to(device)
        
    def forward(self, true, pred):
        mu, sigma = pred
        return (torch.square(true - mu)/(2*sigma) + torch.log(2*self.pi*sigma)/2).mean()

df = pd.read_csv("../data/data/crypto.csv", index_col=0)
quanilte_levels = [0.1, 0.5, 0.9]

for tau in [1, 5]:
    train_list, test_list = build_datasets(df, tau, 200, 3)
    for i in range(len(train_list)):
        
        train_dataset = TensorDataset(*train_list[i])
        # test_dataset = TensorDataset(*test_list[i])
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=100)
        # test_loader = DataLoader(test_dataset, shuffle=True, batch_size=10)
        deepar =  DeepAR(d_input=10, d_model=20, num_targets=10, beta=10, tau=tau)
        
        deepar.to(device) 
        criterion = NegativeGaussianLogLikelihood(device)
        optimizer = optim.Adam(deepar.parameters(), lr=0.001)
        
        for epoch in tqdm(range(1000)):        
            train_loss = train(deepar, train_loader, criterion, optimizer, device)
            
        torch.save(deepar.state_dict(), './assets/DeepAR/tau_' + str(tau) + '/DeepAR_PHASE_{}.pth'.format(i+1))