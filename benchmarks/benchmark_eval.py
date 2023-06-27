import os
import pandas as pd
import numpy as np 
import torch 
import random

from utils import *

from tft import TemporalFusionTransformer
from mqrnn import MQRnn
from deepar import DeepAR
from gp_copula import GPCopula
from sqf_rnn import SQF_RNN

import argparse 
parser = argparse.ArgumentParser(description='hyperparams')
parser.add_argument('--tau', required=False, default=1, choices=[1, 5], type=int)
parser.add_argument('--model', required=True, choices=["MQRNN", "DeepAR", "TFT", "SQF-RNN", "GP-Coupla"], type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("../data/crypto.csv", index_col=0)

MODEL = args.model
tau = args.tau

if MODEL in ["MQRNN", "DeepAR", "TFT", "SQF-RNN"]:
    train_list, test_list = build_datasets(df, tau, 200, 3)
else: # GP-Coupla 
    train_list, test_list, norm_list = build_datasets_gpcopula(df, tau, 200, 3, how="standard")

seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

alpha_09_quantile_loss = []
alpha_05_quantile_loss = []
alpha_01_quantile_loss = []

estQ1 = []
estQ5 = []
estQ9 = []

if MODEL == "GP-Copula":
    target_ = torch.cat([
        train_list[0][1] * torch.tensor(norm_list[0][1]) + torch.tensor(norm_list[0][0]).unsqueeze(0),
        test_list[0][1] * torch.tensor(norm_list[0][1]) + torch.tensor(norm_list[0][0]).unsqueeze(0),
        test_list[1][1] * torch.tensor(norm_list[1][1]) + torch.tensor(norm_list[1][0]).unsqueeze(0),
        test_list[2][1] * torch.tensor(norm_list[2][1]) + torch.tensor(norm_list[2][0]).unsqueeze(0),
    ], dim=0)
else: 
    target_ = torch.cat([train_list[0][1], test_list[0][1], test_list[1][1], test_list[2][1]], dim=0)

DICR = []
CRPS = []
MI = []
STD = []

dirs = os.listdir("../assets/" + MODEL + "/tau_" + str(tau))
i, model_name = next(enumerate(sorted([dir for dir in dirs if not dir.startswith ('.')])))
for i, model_name in enumerate(sorted([dir for dir in dirs if not dir.startswith ('.')])):
    quanilte_levels = [0.1, 0.5, 0.9]

    if MODEL == "TFT":
        model = TemporalFusionTransformer(
            d_model=20,
            d_embedding=3,
            num_var=10,
            seq_len=20,
            num_targets=10,
            tau=tau,
            quantile=quanilte_levels,
            dr=0.1,
            device=device
        )
    
    elif MODEL == "MQRNN":
        model = MQRnn(
            d_input=10,
            d_model=20,
            tau=tau,
            num_targets=10,
            num_quantiles=3,
            n_layers=3,
            dr=0.1
        )

    elif MODEL == "DeepAR":
        model = DeepAR(
            d_input=10,
            d_model=20,
            num_targets=10,
            beta=10,
            tau=tau
        )
        
    elif MODEL == "GP-Copula":
        model = GPCopula(
           d_input=10,
           d_hidden=3,
           tau=tau,
           beta=3,
           rank=10,
           n_layers=2,
           dr=0.05,
           device=device
        )
        
    elif MODEL == "SQF-RNN":
        model = SQF_RNN(d_input=10,
                        d_model=30,
                        tau=tau,
                        n_layers=3,
                        M=10,
                        device=device)
    
    model.load_state_dict(torch.load(os.getcwd() + "/assets/" + MODEL+ "/tau_" + str(tau) + "/" + model_name, map_location="cpu"))
    model.eval()
    
    test_input, test_infer = test_list[i]

    if MODEL != "GP-Copula":
        maxvalues, _ = test_infer.squeeze().max(dim=0, keepdims=True)
    else: 
        norm_mu, norm_sigma = norm_list[i]
        test_infer_ = test_infer * torch.tensor(norm_sigma).unsqueeze(0) + torch.tensor(norm_mu).unsqueeze(0)
        maxvalues, _ = test_infer_.view(-1, 10).max(dim=0, keepdims=True)

    if MODEL == "DeepAR":
        mu, sigma = model(test_input)
        deepar_output = gaussian_quantile(mu, sigma) 
        estq1 = torch.Tensor(deepar_output)[..., 0].squeeze()
        estq5 = torch.Tensor(deepar_output)[..., 1].squeeze()
        estq9 = torch.Tensor(deepar_output)[..., -1].squeeze()
        
        tmp_alpha_09_quantile_loss = torch.divide(torch.maximum(0.9 * (test_infer.squeeze() - estq9), (1-0.9)*(estq9 - test_infer.squeeze())), maxvalues)
        tmp_alpha_05_quantile_loss = torch.divide(torch.maximum(0.5 * (test_infer.squeeze() - estq5), (1-0.5)*(estq5 - test_infer.squeeze())), maxvalues) 
        tmp_alpha_01_quantile_loss = torch.divide(torch.maximum(0.1 * (test_infer.squeeze() - estq1), (1-0.1)*(estq1 - test_infer.squeeze())), maxvalues)
                
        bn = mu.detach().numpy().shape[0]
        tn = mu.detach().numpy().shape[-1]
        samples = torch.Tensor(np.random.normal(mu.detach().numpy(), sigma.detach().numpy(), (100, bn, tau, tn)))
        term1 = torch.divide((samples - test_infer).abs(), maxvalues).mean()
        term2 = torch.divide((samples.unsqueeze(0) - samples.unsqueeze(1)).abs(), maxvalues).mean() * 0.5
        tmp_CRPS = term1 - term2
        CRPS.append(tmp_CRPS)
        CR = ((torch.Tensor(deepar_output)[..., 0].squeeze() < test_infer.squeeze()) * (test_infer.squeeze() < torch.Tensor(deepar_output)[..., -1].squeeze())).to(torch.float32).mean(dim=0)
        
    elif MODEL == "GP-Copula":
        mu, sigma = model(test_input)
        
        tmp_mu, tmp_sigma, tmp_samples = model.sample(mu, sigma.squeeze(), 100)

        samples = tmp_samples * torch.tensor(norm_sigma).unsqueeze(0) + torch.tensor(norm_mu).unsqueeze(0)
        estq1 = torch.quantile(samples, 0.1, dim=0)
        estq5 = torch.quantile(samples, 0.5, dim=0)
        estq9 = torch.quantile(samples, 0.9, dim=0)

        tmp_alpha_09_quantile_loss = torch.divide(torch.maximum(0.9 * (test_infer_.squeeze() - estq9), (1-0.9)*(estq9 -test_infer_.squeeze())), maxvalues)
        tmp_alpha_05_quantile_loss = torch.divide(torch.maximum(0.5 * (test_infer_.squeeze() - estq5), (1-0.5)*(estq5 -test_infer_.squeeze())), maxvalues)
        tmp_alpha_01_quantile_loss = torch.divide(torch.maximum(0.1 * (test_infer_.squeeze() - estq1), (1-0.1)*(estq1 -test_infer_.squeeze())), maxvalues)

        term1 = torch.divide((samples - test_infer_.squeeze()).abs(), maxvalues).mean()
        term2 = torch.divide((samples.unsqueeze(0) - samples.unsqueeze(1)).abs(), maxvalues).mean() * 0.5
        tmp_CRPS = term1 - term2
        CRPS.append(tmp_CRPS)        
        CR = ((torch.quantile(samples.squeeze(), 0.1, dim=0) < test_infer_.squeeze()) * (test_infer_.squeeze() < torch.quantile(samples.squeeze(), 0.9, dim=0))).to(torch.float32).mean()    
        
    elif MODEL == "SQF-RNN":
        gamma, beta = model(test_input)
        estq1 = model.quantile_function(torch.tensor([[0.1]]), gamma, beta).squeeze()
        estq5 = model.quantile_function(torch.tensor([[0.5]]), gamma, beta).squeeze()
        estq9 = model.quantile_function(torch.tensor([[0.9]]), gamma, beta).squeeze()
        
        tmp_alpha_09_quantile_loss = torch.divide(torch.maximum(0.9 * (test_infer.squeeze() - estq9), (1-0.9)*(estq9 -test_infer.squeeze())), maxvalues)        
        tmp_alpha_05_quantile_loss = torch.divide(torch.maximum(0.5 * (test_infer.squeeze() - estq5), (1-0.5)*(estq5 -test_infer.squeeze())), maxvalues)
        tmp_alpha_01_quantile_loss = torch.divide(torch.maximum(0.1 * (test_infer.squeeze() - estq1), (1-0.1)*(estq1 -test_infer.squeeze())), maxvalues)
        CR = ((estq1 < test_infer.squeeze()) * (test_infer.squeeze() < estq9)).to(torch.float32).mean()
        
        samples_ = []
    
        for i in np.linspace(0.01, 0.99, 100):
            tmp_quantile = model.quantile_function(torch.tensor([[i]]), gamma, beta).unsqueeze(0).squeeze(-1)
            samples_.append(tmp_quantile)

        samples = torch.cat(samples_, axis=0)
        term1 = torch.divide((samples - test_infer.unsqueeze(0)).abs(), maxvalues).mean()
        term2 = torch.divide((samples.unsqueeze(0) - samples.unsqueeze(1)).abs(), maxvalues).mean() * 0.5
        tmp_CRPS = term1 - term2
        CRPS.append(tmp_CRPS)   
        
    else:             
        model_output = model(test_input)
        estq9 = model_output[..., -1].squeeze()
        estq5 = model_output[..., 1].squeeze()
        estq1 = model_output[..., 0].squeeze()
        tmp_alpha_09_quantile_loss = torch.divide(torch.maximum(0.9 * (test_infer.squeeze() - estq9), (1-0.9)*(estq9 -test_infer.squeeze())), maxvalues)
        tmp_alpha_05_quantile_loss = torch.divide(torch.maximum(0.5 * (test_infer.squeeze() - estq5), (1-0.5)*(estq5 -test_infer.squeeze())), maxvalues) 
        tmp_alpha_01_quantile_loss = torch.divide(torch.maximum(0.1 * (test_infer.squeeze() - estq1), (1-0.1)*(estq1 -test_infer.squeeze())), maxvalues) 

        CR = ((estq1 < test_infer.squeeze()) * (test_infer.squeeze() < estq9)).to(torch.float32).mean(dim=0)
    
    alpha_09_quantile_loss.append(tmp_alpha_09_quantile_loss.mean())
    alpha_05_quantile_loss.append(tmp_alpha_05_quantile_loss.mean())
    alpha_01_quantile_loss.append(tmp_alpha_01_quantile_loss.mean())
    estQ1.append(estq1)
    estQ5.append(estq5)
    estQ9.append(estq9)
    
    if MODEL == "GP-Copula":
        STD.append(((torch.divide((samples - test_infer_.squeeze()).abs(), maxvalues).view(-1, 10).mean(dim=0) - torch.divide((samples.unsqueeze(0) - samples.unsqueeze(1)).abs(), maxvalues).view(-1, 10).mean(dim=0) * 0.5).std(),
                    tmp_alpha_01_quantile_loss.view(-1, 10).mean(dim=0).std(),
                    tmp_alpha_05_quantile_loss.view(-1, 10).mean(dim=0).std(),
                    tmp_alpha_09_quantile_loss.view(-1, 10).mean(dim=0).std()))
        
    elif (MODEL != "MQRNN") & (MODEL != "TFT") :
        STD.append(((torch.divide((samples - test_infer.unsqueeze(0)).abs(), maxvalues).view(-1, 10).mean(dim=0) - torch.divide((samples.unsqueeze(0) - samples.unsqueeze(1)).abs(), maxvalues).squeeze().view(-1, 10).mean(dim=0) * 0.5).std(),
                    tmp_alpha_01_quantile_loss.view(-1, 10).mean(dim=0).std(),
                    tmp_alpha_05_quantile_loss.view(-1, 10).mean(dim=0).std(),
                    tmp_alpha_09_quantile_loss.view(-1, 10).mean(dim=0).std()))
    
    else: 
        STD.append((tmp_alpha_01_quantile_loss.view(-1, 10).mean(dim=0).std(),
                    tmp_alpha_05_quantile_loss.view(-1, 10).mean(dim=0).std(),
                    tmp_alpha_09_quantile_loss.view(-1, 10).mean(dim=0).std()))
    
    DICR.append((CR - (0.8)).abs())
    MI.append(torch.round(torch.divide((estq9 - estq1), maxvalues).mean(), decimals=3))