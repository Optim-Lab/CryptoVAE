#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

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
utils = importlib.import_module('modules.utils')
importlib.reload(utils)
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
    
    parser.add_argument("--num", default=0, type=int,
                        help="XXX")
    parser.add_argument('--model', type=str, default='GLD_infinite', 
                        help='Fitting model options: GLD_finite, GLD_infinite, LSQF, ExpLog, TLAE')
    parser.add_argument('--data', type=str, default='crypto', 
                        help='Fitting model options: crypto')
    parser.add_argument("--future", default=5, type=int,
                        help="XXX")
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/DDM/{}_{}_{}:v{}'.format(
        config["data"], config["model"], config["future"], config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)

    utils.set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    df = pd.read_csv(
        './data/df_scaled_crypto_indices_20180123.csv',
        index_col=0
    )
    # [['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-ETC']]
    
    test_len = 500
    # """Standardization"""
    # if config["standardize"]:
    #     mean_ = df.iloc[:-test_len].mean(axis=0) # train mean
    #     std_ = df.iloc[:-test_len].std(axis=0) # train std
    #     df = (df - mean_) / std_
    print(df.describe())
    
    colnames = df.columns
    config["p"] = df.shape[1]
    #%%
    """train, test split"""
    train = df.iloc[:-test_len]
    test = df.iloc[-(test_len + config["timesteps"] + config["future"]):]
    
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
    try:
        model_module = importlib.import_module('modules.{}'.format(config["model"]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"])(config, device).to(device)
    except:
        model_module = importlib.import_module('modules.{}'.format(config["model"].split('_')[0]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"].split('_')[0])(config, device).to(device)
    
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
    """Quantile Estimation"""
    alphas = [0.1, 0.5, 0.9]
    context = torch.cat([train_context, test_context], dim=0)
    target = torch.cat([train_target, test_target], dim=0)
    
    if config["model"] == "TLAE":
        est_quantiles, samples = model.est_quantile(test_context, alphas, config["MC"])
        full_est_quantiles, _ = model.est_quantile(context, alphas, config["MC"])
    else:
        est_quantiles = model.est_quantile(test_context, alphas, config["MC"])
        full_est_quantiles = model.est_quantile(context, alphas, config["MC"])
    
    test_target = test_target[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    target = target[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    #%%
    if not os.path.exists('./assets/{}'.format(config["model"])):
        os.makedirs('./assets/{}'.format(config["model"]))
    if not os.path.exists('./assets/{}/out(future={})/'.format(config["model"], config["future"])):
        os.makedirs('./assets/{}/out(future={})/'.format(config["model"], config["future"]))
    if not os.path.exists('./assets/{}/plots(future={})/'.format(config["model"], config["future"])):
        os.makedirs('./assets/{}/plots(future={})/'.format(config["model"], config["future"]))
    #%%
    """Visualize"""
    figs = utils.visualize_quantile(target, test_target, full_est_quantiles, est_quantiles, colnames, config, show=False, dark=False)
    for j in range(len(colnames)):
        wandb.log({f'Quantile ({colnames[j]})': wandb.Image(figs[j])})
    #%%
    """Volatility"""
    width = est_quantiles[-1] - est_quantiles[0]
    width /= width.max(dim=0).values
    volatility = width.mean()
    print('Volatility(width): {:.3f}'.format(volatility.item()))
    wandb.log({f'Volatility(width)': volatility.item()})
    #%%
    """Vrate and Hit"""
    for i, a in enumerate(alphas):
        vrate = (test_target < est_quantiles[i]).to(torch.float32).mean(dim=0)
        print('Vrate(alpha={}): {:.3f}'.format(a, vrate.mean()), 
              ', Hit(alpha={}): {:.3f}'.format(a, (a - vrate.mean()).abs()))
        wandb.log({f'Vrate(alpha={a})': vrate.mean().item()})
        wandb.log({f'Hit(alpha={a})': (a - vrate.mean()).abs().item()})
        # for j, name in enumerate(colnames):
        #     print('Vrate([{}], alpha={}): {:.3f}'.format(name, a, vrate[j]), 
        #           ', Hit([{}], alpha={}): {:.3f}'.format(name, a, (a - vrate[j]).abs()))
        #     wandb.log({f'Vrate([{name}], alpha={a})': vrate[j].item()})
        #     wandb.log({f'Hit([{name}], alpha={a})': (a - vrate[j]).abs().item()})
        df = pd.DataFrame(
            est_quantiles[i].numpy(),
            columns=colnames     
        )
        df.to_csv(f'./assets/{config["model"]}/out(future={config["future"]})/VaR(alpha={a}).csv')
        wandb.run.summary[f'VaR(alpha={a})'] = wandb.Table(data=df)
    #%%
    """CRPS: Proposal model & TLAE"""
    if config["model"] != "TLAE":
        samples = model.sampling(test_context, config["MC"])
    
    term1 = (samples - test_target[:, None, :]).abs().mean(dim=1)
    term2 = (samples[:, :, None, :] - samples[:, None, :, :]).abs().mean(dim=[1, 2]) * 0.5
    CRPS = term1 - term2
    print('CRPS: {:.3f}'.format(CRPS.mean()))
    wandb.log({f'CRPS': CRPS.mean().item()})
    #%%
    """Quantile loss"""
    if config["model"] != "TLAE":
        tau = torch.linspace(0.01, 0.99, 99)
        est_quantiles = model.est_quantile(test_context, tau, 1, disable=True)
    
        quantile_risk = 0
        for i, a in enumerate(tau):
            residual = test_target - est_quantiles[i]
            quantile_risk += ((a - (residual < 0).to(torch.float32)) * residual).mean()
        quantile_risk /= len(tau)
        print('Quantile Risk: {:.3f}'.format(quantile_risk.item()))
        wandb.log({f'Quantile Risk': quantile_risk.item()})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
# """Expected Shortfall"""
# for i, a in enumerate(alphas):
#     residual = samples - est_quantiles[i][:, None, :]
#     ES = samples.mean(dim=1) - (residual * (a - (residual < 0).to(torch.float32))).mean(dim=1) / a
#     df = pd.DataFrame(
#         ES.numpy(),
#         columns=colnames
#     )
#     df.to_csv(f'./assets/{config["model"]}/out/ES(alpha={a}).csv')
#     wandb.run.summary[f'ES(alpha={a})'] = wandb.Table(data=df)
#%%