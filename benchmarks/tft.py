import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np 
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ContiFeatureEmbedding(nn.Module):
    def __init__(self, d_embedding, num_rv):
        super(ContiFeatureEmbedding, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(1, d_embedding) for _ in range(num_rv)])
        
    def forward(self, x):
        tmp_feature_list = []
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            tmp_feature = l(x[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        return torch.stack(tmp_feature_list, axis=-1).transpose(-1, -2)
    
class GLULN(nn.Module):
    def __init__(self, d_model):
        super(GLULN, self).__init__()
        self.linear1 = nn.LazyLinear(d_model)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.LazyLinear(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, y):
        return self.layer_norm(torch.mul(self.sigmoid(self.linear1(x)), self.linear2(x)) + y)

class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, dr):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.LazyLinear(d_model)
        self.dropout1 = nn.Dropout(dr)
        self.linear2 = nn.LazyLinear(d_model)
        self.dropout2 = nn.Dropout(dr)
        self.elu = nn.ELU()
        self.gluln = GLULN(d_model)
        
    def forward(self, x):
        eta_2 = self.dropout1(self.linear1(x))
        eta_1 = self.elu(self.dropout2(self.linear2(eta_2)))
        grn_output = self.gluln(eta_1, eta_2)
        
        return grn_output

class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, d_input, dr):
        super(VariableSelectionNetwork, self).__init__()
        self.v_grn = GatedResidualNetwork(d_input, dr)
        self.softmax = nn.Softmax(dim=-1)
        self.xi_grn = nn.ModuleList([GatedResidualNetwork(d_model, dr) for _ in range(d_input)])
        
    def forward(self, xi):
        Xi = xi.reshape(xi.size(0), xi.size(1), -1)
        weights = self.softmax(self.v_grn(Xi)).unsqueeze(-1)
        
        tmp_xi_list = []
        for i, l in enumerate(self.xi_grn):
            tmp_xi = l(xi[:, :, i:i+1])
            tmp_xi_list.append(tmp_xi)
        xi_list = torch.cat(tmp_xi_list, axis=-2)
        
        combined = torch.matmul(weights.transpose(3, 2), xi_list).squeeze()
        
        return combined, weights

class TemporalFusionDecoder(nn.Module):
    def __init__(self, d_model, dr, seq_len, device):
        super(TemporalFusionDecoder, self).__init__()
        self.d_model = d_model
        self.dr = dr
        self.seq_len = seq_len
        self.device = device

        self.lstm_obs = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True)

        self.gluln1 = GLULN(self.d_model)
        self.mha = nn.MultiheadAttention(self.d_model, 1, batch_first=True, dropout=self.dr)
        self.gluln2 = GLULN(self.d_model)
        self.temporal_mask = torch.triu(torch.ones([self.seq_len, self.seq_len], dtype=torch.bool), diagonal=1).to(self.device)
        
    def forward(self, vsn_observed):
        time_step = self.seq_len

        future_lstm_output, (_, _) = self.lstm_obs(vsn_observed)
                        
        lstm_hidden = torch.cat([vsn_observed, future_lstm_output], dim=1)
        
        glu_phi_list = []
        
        for t in range(time_step):
            tmp_phi_t = self.gluln1(lstm_hidden[:, t:t+1, :], lstm_hidden[:, t:t+1, :])
            glu_phi_list.append(tmp_phi_t)
        
        glu_phi = torch.cat(glu_phi_list, axis=1)
        B, decoder_attention = self.mha(query=glu_phi, key=glu_phi, value=glu_phi, attn_mask=self.temporal_mask)
        glu_delta_list = [] 

        for j in range(time_step):
            tmp_delta_t = self.gluln2(B[:, j:j+1, :], glu_phi[:, j:j+1, :])
            glu_delta_list.append(tmp_delta_t)

        glu_delta = torch.cat(glu_delta_list, dim=1)
        
        return glu_delta, glu_phi, decoder_attention
    
class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, dr):
        super(PointWiseFeedForward, self).__init__()
        self.grn = GatedResidualNetwork(d_model, dr)
        self.gluln = GLULN(d_model)
        
    def forward(self, delta, phi):
        time_step = delta.size(1)
        
        grn_varphi_list = []
        
        for t in range(time_step):
            tmp_grn_varphi = self.grn(delta[:, t:t+1, :])
            grn_varphi_list.append(tmp_grn_varphi)
            
        grn_varphi = torch.cat(grn_varphi_list, dim=1)
        
        varphi_tilde_list = []
        
        for t in range(time_step):
            tmp_varphi_tilde = self.gluln(grn_varphi[:, t:t+1, :], phi[:, t:t+1, :])
            varphi_tilde_list.append(tmp_varphi_tilde)
            
        varphi = torch.cat(varphi_tilde_list, dim=1)
        
        return varphi        

class TargetFeatureLayer(nn.Module):
    def __init__(self, d_model, num_target):
        super(TargetFeatureLayer, self).__init__()
        self.target_feature_linears = nn.ModuleList([nn.LazyLinear(d_model) for _ in range(num_target)])
    
    def forward(self, varphi):
        target_feature_list = []
        
        for _, l in enumerate(self.target_feature_linears):
            tmp_feature = l(varphi)
            target_feature_list.append(tmp_feature.unsqueeze(-2))
            
        return torch.cat(target_feature_list, dim=-2)

class QuantileOutput(nn.Module):
    def __init__(self, tau, quantile):
        super(QuantileOutput, self).__init__()
        self.tau = tau
        self.quantile_linears = nn.ModuleList([nn.LazyLinear(1) for _ in range(len(quantile))])
        
    def forward(self, varphi):
        total_output_list = []
        
        for _, l in enumerate(self.quantile_linears):
            tmp_quantile_list = []
            
            for t in range(self.tau-1):
                tmp_quantile = l(varphi[:, (-self.tau + t) : (-self.tau + t + 1), ...])
                tmp_quantile_list.append(tmp_quantile)
            
            tmp_quantile_list.append(l(varphi[:, -1:, ...]))
            
            total_output_list.append(torch.cat(tmp_quantile_list, dim=1))
            
        return torch.cat(total_output_list, dim=-1)
    
    
class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        num_var,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(TemporalFusionTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_var)
        self.vsn1 = VariableSelectionNetwork(d_model, num_var, dr)
        
        self.tfd = TemporalFusionDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
    def forward(self, conti_input):
        
        obs_feature = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)

        ### Encoder
        x1, _  = self.vsn1(obs_feature) # (batch_size, seq_len, d_model)
        
        ### Decoder
        delta, glu_phi, _ = self.tfd(x1) # # (batch_size, seq_len+tau, d_model)
        varphi = self.pwff(delta, glu_phi) # (batch_size, seq_len+tau, num_target, d_model)
        tfl_output = self.tfl(varphi)  # (batch_size, seq_len+tau, d_model)
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output

def train(model, loader, criterion, optimizer, device):
    
    model.train()
    
    total_loss = []
    
    for batch in loader:
        conti_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        true_y = true_y.to(device)
        
        pred = model(conti_input)
        
        loss = criterion(true_y, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)
    
def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss = []
    
    for batch in loader:
        conti_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        true_y = true_y.to(device)
        
        pred = model(conti_input)
        
        loss = criterion(true_y, pred)
        
        total_loss.append(loss)
        
        return sum(total_loss)/len(total_loss) 
    
class QuantileRisk(nn.Module):
    def __init__(self, tau, quantile, num_targets, device):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile
        self.device = device
        self.q_arr = torch.tensor(self.quantile).float().unsqueeze(0).unsqueeze(-1).repeat(1, 1, tau).unsqueeze(1).repeat(1, num_targets, 1, 1).to(self.device)
    
    def forward(self, true, pred):
        true_rep = true.unsqueeze(-1).repeat(1, 1, 1, len(self.quantile)).permute(0, 2, 3, 1).to(self.device)
        pred = pred.permute(0, 2, 3, 1)

        ql = torch.maximum(self.q_arr * (true_rep - pred), (1-self.q_arr)*(pred - true_rep))
        
        return ql.mean()
    
df = pd.read_csv("../data/crypto.csv", index_col=0)

quanilte_levels = [0.1, 0.5, 0.9]

for tau in [1, 5]:
    train_list, test_list = build_datasets(df, tau, 200, 3)
    for i in range(len(train_list)):
        train_dataset = TensorDataset(*train_list[i])
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=100)
        tft = TemporalFusionTransformer(
            d_model=20,
            d_embedding=3,
            num_var=10,
            seq_len=20,
            num_targets=10,
            tau=5,
            quantile=quanilte_levels,
            dr=0.1,
            device=device
        )
        
        tft.to(device)  
        criterion = QuantileRisk(1, quanilte_levels, num_targets=10, device=device)  
        optimizer = optim.Adam(tft.parameters(), lr=0.001)
        
        for epoch in tqdm(range(1500)):        
            train_loss = train(tft, train_loader, criterion, optimizer, device)
            
        torch.save(tft.state_dict(), './assets/TFT/tau_' + str(tau) + '/TFT_PHASE_{}.pth'.format(i+1))