#%%
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
def load_config(config):
    config_path = f'./configs/{config["model"]}.yaml'
    with open(config_path, 'r') as config_file:
        args = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in config.keys():
        if key in args.keys():
            config[key] = args[key]
    return config
#%%
def visualize_quantile(target, test_target, full_est_quantiles, est_quantiles, colnames, config, path, show=False, dark=False):
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
            np.arange(target.shape[0] - test_target.shape[0], target.shape[0]), 
            est_quantiles[0][:, j].numpy(), 
            est_quantiles[2][:, j].numpy(), 
            color='blue', alpha=0.3, label=r'80% interval')
        plt.plot(
            target.numpy()[:, j],
            label=colnames[j], color='black', linestyle='--', linewidth=2)
        plt.plot(
            full_est_quantiles[1][:, j].numpy(),
            label='Median', color='green', linewidth=2)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.legend(loc = 'upper left')
        plt.savefig(f'{path}/{colnames[j]}_beta{config["beta"]}_var{config["prior_var"]}.png')
        if show:
            plt.show()
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs
#%%