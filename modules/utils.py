#%%
import pandas as pd
import numpy as np
import random
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from ing_theme_matplotlib import mpl_style # pip install ing_theme_matplotlib 
import matplotlib as mpl
import tqdm
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
#%%
def load_config(config, config_path):
    with open(config_path, 'r') as config_file:
        args = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in config.keys():
        if key in args.keys():
            config[key] = args[key]
    return config
#%%
def stock_data_generator(df, C, tau):
    n = df.shape[0] - C - tau
        
    input_data = np.zeros((n, C, df.shape[1]))
    infer_data = np.zeros((n, tau, df.shape[1]))

    for i in range(n):
        input_data[i, :, :] = df.iloc[i : i+C, :]
        infer_data[i, :, :] = df.iloc[i+C : i+C+tau, :]
    
    input_data = torch.from_numpy(input_data).to(torch.float32)
    infer_data = torch.from_numpy(infer_data).to(torch.float32)
    return input_data, infer_data
#%%
def build_datasets(df, test_len, increment, config):
    train_list = []
    test_list = []
    for j in range(increment):
        train_idx_last = len(df) - test_len * (increment - j)
        test_idx_last = len(df) - test_len * (increment - 1 - j)
        train = df.iloc[: train_idx_last]
        test = df.iloc[-(test_len * (increment - j) + config["timesteps"] + config["future"]) : test_idx_last]
        
        """data save"""
        train.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{config["future"]}_phase{j+1}_train.csv')
        test.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{config["future"]}_phase{j+1}_test.csv')
        
        train_context, train_target = stock_data_generator(train, config["timesteps"], config["future"])
        test_context, test_target = stock_data_generator(test, config["timesteps"], config["future"])
        
        assert train_context.shape == (train.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
        assert train_target.shape == (train.shape[0] - config["timesteps"] - config["future"], config["future"], df.shape[1])
        assert test_context.shape == (test.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
        assert test_target.shape == (test.shape[0] - config["timesteps"] - config["future"], config["future"], df.shape[1])
        
        train_list.append((train_context, train_target))
        test_list.append((test_context, test_target))
    return train_list, test_list
#%%
def stock_data_generator2(df, C, tau):
    n = df.shape[0] - C - tau
        
    # T = C + tau
    input_data = np.zeros((n, C, df.shape[1]))
    infer_data = np.zeros((n, C+tau, df.shape[1]))

    for i in range(n):
        input_data[i, :, :] = df.iloc[i : i+C, :]
        infer_data[i, :, :] = df.iloc[i : i+C+tau, :]
    
    input_data = torch.from_numpy(input_data).to(torch.float32)
    infer_data = torch.from_numpy(infer_data).to(torch.float32)
    return input_data, infer_data
#%%
def build_datasets2(df, test_len, increment, config):
    train_list = []
    test_list = []
    for j in range(increment):
        train_idx_last = len(df) - test_len * (increment - j)
        test_idx_last = len(df) - test_len * (increment - 1 - j)
        train = df.iloc[: train_idx_last]
        test = df.iloc[-(test_len * (increment - j) + config["timesteps"] + config["future"]) : test_idx_last]
        
        """data save"""
        train.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{config["future"]}_phase{j+1}_train.csv')
        test.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{config["future"]}_phase{j+1}_test.csv')
        
        train_context, train_target = stock_data_generator2(train, config["timesteps"], config["future"])
        test_context, test_target = stock_data_generator2(test, config["timesteps"], config["future"])
        
        assert train_context.shape == (train.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
        assert train_target.shape == (train.shape[0] - config["timesteps"] - config["future"], config["timesteps"] + config["future"], df.shape[1])
        assert test_context.shape == (test.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
        assert test_target.shape == (test.shape[0] - config["timesteps"] - config["future"], config["timesteps"] + config["future"], df.shape[1])
        
        train_list.append((train_context, train_target))
        test_list.append((test_context, test_target))
    return train_list, test_list
#%%
def visualize_quantile(target_, estQ, start_idx, colnames, test_len, config, path, show=False, dark=False):
    # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl.rcParams["figure.dpi"] = 200
    mpl_style(dark=dark)
    SMALL_SIZE = 10
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
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
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.legend(loc = 'upper left')
        plt.savefig(f'{path}/{colnames[j]}_{config["model"]}_future{config["future"]}_beta{config["beta"]}_var{config["prior_var"]}.png')
        if show:
            plt.show()
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs

def visualize_quantile(target_, estQ, start_idx, colnames, test_len, show=False, dark=False):
    # cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl.rcParams["figure.dpi"] = 200
    mpl_style(dark=dark)
    SMALL_SIZE = 10
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    xticks = [37, 251, 463, 677, 890, 1105, 1317, 1529, 1743, 1894]
    xtick_labels = ["2018.03", "2018.10", "2019.05", "2019.12", "2020.07", "2021.02", "2021.09", "2022.04", "2022.11", "2023.04"]
        
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
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.ylim(0, target_.numpy()[:, j].max()+1)
        plt.text(1300, target_.numpy()[:, j].max()+0.5,"Phase 1", color='black', fontsize=16)
        plt.text(1500, target_.numpy()[:, j].max()+0.5,"Phase 2", color='black', fontsize=16)
        plt.text(1700, target_.numpy()[:, j].max()+0.5,"Phase 3", color='black', fontsize=16)
        plt.xticks(xticks, xtick_labels, rotation=20)
        plt.annotate("",
            xy=(1280, target_.numpy()[:, j].max()+0.35),
            xytext=(1480, target_.numpy()[:, j].max()+0.35),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
        plt.annotate("",
            xy=(1480, target_.numpy()[:, j].max()+0.35),
            xytext=(1680, target_.numpy()[:, j].max()+0.35),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
        plt.annotate("",
            xy=(1680, target_.numpy()[:, j].max()+0.35),
            xytext=(1880, target_.numpy()[:, j].max()+0.35),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
        plt.legend(loc = 'upper left')
        if show:
            plt.show()
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs
#%%