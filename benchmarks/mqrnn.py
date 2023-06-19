import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

class GlobalDecoder(nn.Module):
    def __init__(self, d_hidden:int, d_model:int, tau:int, num_targets:int, dr:float):
        super(GlobalDecoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.linear_layers = nn.ModuleList([nn.Linear(d_hidden, (tau+1) * d_model) for _ in range(num_targets)])
        self.dropout = nn.Dropout(dr)
        
    def forward(self, hidden):                    
        num_layers, batch_size, d_hidden = hidden.size()
        
        assert d_hidden == self.d_model 

        
        tmp_global_context = []
        for l in self.linear_layers:
            tmp_gc = self.dropout(l(hidden[num_layers-1]))
            tmp_global_context.append(tmp_gc.unsqueeze(1))
        
        global_output = torch.cat(tmp_global_context, axis=1)
        
        return global_output # (batch_size, num_targets, (tau+1) * d_model), (tau+1): c_{a} , c_{t+1:t+tau}

class LocalDecoder(nn.Module):
    def __init__(self, d_hidden:int, d_model:int, tau:int, num_targets:int, num_quantiles:int, dr:float):
        super(LocalDecoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.linear_layers = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.Dropout(dr),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dr),
            nn.Linear(d_model, num_quantiles)            
            )
                
    def forward(self, global_output):
        batch_size = global_output.size(0)
        
        c_a = global_output[..., :self.d_model].unsqueeze(-2).repeat(1, 1, self.tau, 1) # (batch_size, num_targets, tau, d_model)
        c_t = global_output[..., self.d_model:].view(batch_size, self.num_targets, self.tau, -1) # (batch_size, num_targets, tau, d_model)
        x = torch.cat([c_a,c_t.view(batch_size, self.num_targets, self.tau, -1)], axis=-1) # (batch_size, num_targets, tau, 2*d_model)        
        
        output = self.linear_layers(x)
        
        return output.transpose(2, 1) # (batch_size, tau, num_targets, num_quantiles)
    

class MQRnn(nn.Module):
    def __init__(self, d_input:int, d_model:int, tau:int, num_targets:int, num_quantiles: int, n_layers:int, dr:float):
        super(MQRnn, self).__init__()
        self.encoder = Encoder(
                               d_input=d_input,
                               d_model=d_model,
                               n_layers=n_layers,
                               dr=dr
                               )
        self.global_decoder = GlobalDecoder(
                                            d_hidden=d_model,
                                            d_model=d_model,
                                            tau=tau,
                                            num_targets=num_targets,
                                            dr=dr
                                            )
        self.local_decoder = LocalDecoder(
                                          d_hidden=d_model,
                                          d_model=d_model,
                                          tau=tau,
                                          num_targets=num_targets,
                                          num_quantiles=num_quantiles,
                                          dr=dr
                                          )
        
    def forward(self, x):
        hidden, _ = self.encoder(x)
        global_output = self.global_decoder(hidden)
        output = self.local_decoder(global_output)
        
        return output

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