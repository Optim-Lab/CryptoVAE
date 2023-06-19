#%%
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
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
df = pd.read_csv("./data/crypto.csv", index_col=0)
#%%
MODEL = "TFT"
tau = 5

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
# model_name = sorted([dir for dir in dirs if not dir.startswith ('.')])[0]
# i = 0
dirs = os.listdir(os.getcwd() + "/assets/weights/" + MODEL + "/tau_" + str(tau))
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
    
    model.load_state_dict(torch.load(os.getcwd() + "/assets/weights/" + MODEL+ "/tau_" + str(tau) + "/" + model_name, map_location="cpu"))
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
        deepar_output = gaussian_quantile(mu, sigma) # (batch_size, tau, num_targets, num_qunatiles)
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
        # samples.shape
        # torch.tensor(norm_sigma).unsqueeze(0).shape
        samples = tmp_samples * torch.tensor(norm_sigma).unsqueeze(0) + torch.tensor(norm_mu).unsqueeze(0)
        estq1 = torch.quantile(samples, 0.1, dim=0)
        estq5 = torch.quantile(samples, 0.5, dim=0)
        estq9 = torch.quantile(samples, 0.9, dim=0)
        # (100, bn, tau, tn)
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
        model_output = model(test_input) # (batch_size, tau, num_targets, num_quantiles)
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
    # asset_number = 1
    # plt.style.use('default')
    # fig, ax = plt.subplots(figsize=(20, 10))
    # plt.plot(model_output[..., asset_number, 0].squeeze().detach().numpy())
    # plt.plot(model_output[..., asset_number, 1].squeeze().detach().numpy())
    # plt.plot(model_output[..., asset_number, 2].squeeze().detach().numpy())
    # plt.plot(test_infer[..., 0, asset_number].squeeze(), color='k')
# %%
# CRPS
# alpha_01_quantile_loss
# alpha_05_quantile_loss
# alpha_09_quantile_loss
# DICR
# MI
STD 
# %%
# model_output = model(test_input)
# model_output[..., -1].squeeze() # 90% percentile
# model_output[..., 0].squeeze() # 10% percentile 
# model_output[..., 1].squeeze() # Median
#%%
model_output
estQ1 = torch.cat(estQ1, dim=0)
estQ5 = torch.cat(estQ5, dim=0)
estQ9 = torch.cat(estQ9, dim=0)
target_.shape
target_ = target_.detach().cpu().squeeze()
estQ1 = estQ1.detach().cpu().squeeze()
estQ5 = estQ5.detach().cpu().squeeze()
estQ9 = estQ9.detach().cpu().squeeze()

estQ = [estQ1, estQ5, estQ9]
colnames = [col.replace("KRW-", "") for col in df.columns.to_list()]
#%%
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from ing_theme_matplotlib import mpl_style # pip install ing_theme_matplotlib 
import matplotlib as mpl
# %%
def visualize_quantile(target_, estQ, colnames, test_len, config, path, show=False, dark=False):
     # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl.rcParams["figure.dpi"] = 200
    mpl_style(dark=dark)
    SMALL_SIZE = 16
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    if config["future"] == 1:
        start_idx = 1279
        shift = 0
    elif config["future"] == 5:
        start_idx = 1275        
        shift = -4
    
    xticks = [17+shift, 231+shift, 443+shift, 657+shift, 870+shift, 1085+shift, 1297+shift, 1509+shift, 1723+shift]
    xtick_labels = ["2018.03", "2018.10", "2019.05", "2019.12", "2020.07", "2021.02", "2021.09", "2022.04", "2022.11"]
    
    figs = []
    for j in tqdm.tqdm(range(len(colnames)), desc=f"Visualize Quantiles...", disable=show):
        fig = plt.figure(figsize=(12, 7))   
        conf = plt.fill_between(
            np.arange(start_idx, target_.shape[0]), 
            estQ[0][:, j].numpy(), 
            estQ[2][:, j].numpy(), 
            color='blue', alpha=0.3, label=r'80% interval')
        plt.plot(
            target_.numpy()[:, j],
            label=colnames[j], color='black', linestyle='--', linewidth=2)
        plt.plot(
            np.arange(start_idx, target_.shape[0]),
            estQ[1][:, j].numpy(),
            label='Median', color='green', linewidth=2)
        plt.axvline(x=start_idx, color='blue', linewidth=2)
        plt.axvline(x=start_idx + test_len, color='blue', linewidth=2)
        plt.axvline(x=start_idx + test_len * 2, color='blue', linewidth=2)
        # plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=BIGGER_SIZE)
        plt.ylim(0, target_.numpy()[:, j].max()+1.5)
        plt.text(1295+shift, target_.numpy()[:, j].max()+0.3,"Phase 1", color='black', fontsize=19)
        plt.text(1495+shift, target_.numpy()[:, j].max()+0.3,"Phase 2", color='black', fontsize=19)
        plt.text(1695+shift, target_.numpy()[:, j].max()+0.3,"Phase 3", color='black', fontsize=19)
        plt.xticks(xticks, xtick_labels, rotation=20)
        plt.annotate("",
            xy=(1280+shift, target_.numpy()[:, j].max()+0.05),
            xytext=(1480+shift, target_.numpy()[:, j].max()+0.05),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
        plt.annotate("",
            xy=(1480+shift, target_.numpy()[:, j].max()+0.05),
            xytext=(1680+shift, target_.numpy()[:, j].max()+0.05),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
        plt.annotate("",
            xy=(1680+shift, target_.numpy()[:, j].max()+0.05),
            xytext=(1880+shift, target_.numpy()[:, j].max()+0.05),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
        plt.legend(loc = 'upper left')
        plt.tight_layout()
        plt.savefig(f'{path}/{colnames[j]}_{config["model"]}_future{config["future"]}_beta{config["beta"]}_var{config["prior_var"]}.png')
        if show:
            plt.show()
        # plt.show()
        
        plt.close()
        
        figs.append(fig)
    return figs
# %%
colnames = [col.replace("KRW-", "") for col in df.columns.to_list()]
#%%
estQ = [Q[::5, :, :].reshape(-1, 10) for Q in estQ]
#%%
config = {
    "model": MODEL,
    "future": tau,
    "beta": None,
    "prior_var": None
}
path = "/Users/chulhongsung/Desktop/lab/working_paper/DisTran/assets/figure/results/" + MODEL
visualize_quantile(target_[::5, :, :].reshape(-1, 10), estQ, colnames, 200, config, path, show=True)
target_.shape
# %%
