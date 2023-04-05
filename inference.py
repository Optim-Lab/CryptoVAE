#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

import sys
sys.path.append('./modules')

import importlib
layers = importlib.import_module('modules.layers')
importlib.reload(layers)
from modules.utils import set_random_seed
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="DDM", 
    entity="anseunghwan",
    tags=['Inference'],
)
#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--model', type=str, default='GLD', 
                        help='Fitting model options: LSQF, GLD, KUMA')
    parser.add_argument("--num", default=8, type=int,
                        help="XXX")
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/DDM/{}:v{}'.format(
        config["model"], config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    df = pd.read_csv(
        './data/df_upbit_top8_krw_181116.csv',
        index_col=0
    )[['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-TRX', 'KRW-ETC']]
    
    """Standardization"""
    test_len = 500
    if config["standardize"]:
        mean_ = df.iloc[:-test_len].mean(axis=0) # train mean
        std_ = df.iloc[:-test_len].std(axis=0) # train std
        df = (df - mean_) / std_
    df.describe()
    df.head()
    
    colnames = df.columns
    config["p"] = df.shape[1]
    #%%
    """train, test split"""
    train = df.iloc[:-test_len]
    test = df.iloc[-(test_len + config["timesteps"] + 1):]
    
    train.to_csv('./data/train.csv')
    test.to_csv('./data/test.csv')
    
    def stock_data_generator(df, C, tau):
        n = df.shape[0] - C - tau
            
        # C = k
        # T = k+tau
        input_data = np.zeros((n, C, df.shape[1]))
        infer_data = np.zeros((n, C+tau, df.shape[1]))

        for i in range(n):
            input_data[i, :, :] = df.iloc[i : i+C, :]
            infer_data[i, :, :] = df.iloc[i : i+C+tau, :]
        
        input_data = torch.from_numpy(input_data).to(torch.float32)
        infer_data = torch.from_numpy(infer_data).to(torch.float32)
        return input_data, infer_data

    train_context, train_target = stock_data_generator(train, config["timesteps"], config["future"])
    test_context, test_target = stock_data_generator(test, config["timesteps"], config["future"])
    
    assert train_context.shape == (train.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
    assert train_target.shape == (train.shape[0] - config["timesteps"] - config["future"], config["timesteps"] + config["future"], df.shape[1])
    assert test_context.shape == (test.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
    assert test_target.shape == (test.shape[0] - config["timesteps"] - config["future"], config["timesteps"] + config["future"], df.shape[1])
    #%%
    model_module = importlib.import_module('modules.{}'.format(config["model"]))
    importlib.reload(model_module)
    model = getattr(model_module, config["model"])(config, device).to(device)
    
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    alphas = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
    est_quantiles, Qs = model.est_quantile(test_context, alphas, config["MC"])
    #%%
    if config["standardize"]:
        test_target_ = (test_target.reshape(-1, config["p"]) * std_.to_numpy()[None, :] + mean_.to_numpy()[None, :]
            ).reshape(test_context.size(0), config["timesteps"] + config["future"], config["p"])
        est_quantiles = [x * std_.to_numpy()[None, :] + mean_.to_numpy()[None, :] for x in est_quantiles]
    
    test_target_ = test_target_[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    #%%
    if not os.path.exists('./assets/{}/plots/'.format(config["model"])):
        os.makedirs('./assets/{}/plots/'.format(config["model"]))
    #%%
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for j in range(len(colnames)):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(test_target_.numpy()[:, j],
                color='black', linestyle='--')
        for i, a in enumerate(alphas):
            plt.plot(est_quantiles[i][:, j].numpy(),
                    label=colnames[j] + f'(alpha={a})', color=cols[j], linewidth=2)
        # plt.legend()
        plt.title(f'{colnames[j]}', fontsize=14)
        plt.ylabel('return', fontsize=12)
        plt.xlabel('days', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./assets/{config["model"]}/plots/quantile_{colnames[j]}.png')
        # plt.show()
        plt.close()
        # wandb.log({f'Quantile': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%