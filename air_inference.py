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
    tags=["Inference"],
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
                        help="the number of model")
    parser.add_argument('--model', type=str, default='GLD_infinite', 
                        help='Fitting model options: GLD_finite, GLD_infinite, LSQF, ExpLog, TLAE, ProTran')
    parser.add_argument('--data', type=str, default='air', 
                        help='Fitting model options: air')
    parser.add_argument("--future", default=5, type=int,
                        help="the number of time steps to forecasting")
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
    if not os.path.exists('./assets/{}'.format(config["model"])):
        os.makedirs('./assets/{}'.format(config["model"]))
        
    plots_dir = f'./air_assets/{config["model"]}/plots(future={config["future"]})/beta{config["beta"]}_var{config["prior_var"]}'    
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    #%%
    """train, test split"""
    df_train = pd.read_csv(
        f'./data/df_{config["data"]}_train.csv',
    )
    df_train = df_train.drop(columns=["측정일시"]) * 10
    df_test = pd.read_csv(
        f'./data/df_{config["data"]}_test.csv',
    )
    df_test = df_test.drop(columns=["측정일시"]) * 10
    
    config["p"] = df_train.shape[1]
    if config["model"] in ["TLAE", "ProTran"]: # reconstruct T
        input_train, infer_train = utils.air_data_generator2(df_train, config["timesteps"], config["future"])
        input_test, infer_test = utils.air_data_generator2(df_test, config["timesteps"], config["future"])
    else: # reconstruct only T - C
        input_train, infer_train = utils.air_data_generator(df_train, config["timesteps"], config["future"])
        input_test, infer_test = utils.air_data_generator(df_test, config["timesteps"], config["future"])
    #%%
    try:
        model_module = importlib.import_module('modules.{}'.format(config["model"]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"])(config, device).to(device)
    except:
        model_module = importlib.import_module('modules.{}'.format(config["model"].split('_')[0]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"].split('_')[0])(config, device).to(device)
    #%%
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
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    """Quantile Estimation"""
    # Get maximum for normalization
    maxvalues = infer_test.reshape(-1, config["p"]).max(dim=0, keepdims=True).values
    
    alphas = [0.1, 0.5, 0.9]
    if config["model"] in ["TLAE", "ProTran"]:
        est_quantiles, _ = model.est_quantile(input_test, alphas, config["MC"], config["test_len"])
    else:
        est_quantiles = model.est_quantile(input_test, alphas, config["MC"])
    
    if config["model"] in ["TLAE", "ProTran"]:
        infer_test_ = infer_test[:, config["timesteps"]:, :].reshape(-1, config["p"])
    else:
        infer_test_ = infer_test.reshape(-1, config["p"])
    
    """DICR"""
    est_quantiles_ = [Q[:, :, :].reshape(-1, config["p"]) for Q in est_quantiles]
    CR = ((est_quantiles_[0] < infer_test_) * (infer_test_ < est_quantiles_[-1])).to(torch.float32).mean(dim=0)
    INTERVAL = ((est_quantiles_[-1] - est_quantiles_[0]) / maxvalues).mean(dim=0)
    DICR = (CR - (alphas[-1] - alphas[0])).abs()
    
    print(f'INTERVAL: {INTERVAL.mean():.3f},')
    print(f'DICR: {DICR.mean():.3f},')
    wandb.log({f'INTERVAL': INTERVAL.mean().item()})
    wandb.log({f'DICR': DICR.mean().item()})
    colnames = df_train.columns
    for c, x in zip(colnames, INTERVAL):
        print(f'[{c}] INTERVAL: {x:.3f},')
        wandb.log({f'[{c}] INTERVAL': x.item()})
    print()
    for c, x in zip(colnames, DICR):
        print(f'[{c}] DICR: {x:.3f},')
        wandb.log({f'[{c}] DICR': x.item()})
    print()
    print(f'INTERVAL: {INTERVAL.std():.3f},')
    print(f'DICR: {DICR.std():.3f},')
    wandb.log({f'INTERVAL(STD)': INTERVAL.std().item()})
    wandb.log({f'DICR(STD)': DICR.std().item()})
    
    """Quantile loss"""
    for i, a in enumerate(alphas):
        u = infer_test_ - est_quantiles_[i]
        QL = (((a - (u < 0).to(torch.float32)) * u) / maxvalues).mean(dim=0) # normalized QL
        print(f'QL({a}): {QL.mean():.3f}')
        wandb.log({f'QL({a})': QL.mean().item()})
        for c, q in zip(colnames, QL):
            print(f'[{c}] QL({a}): {q:.3f}')
            wandb.log({f'[{c}] QL({a})': q.item()})
        print()
        print(f'QL({a})(STD): {QL.std():.3f}')
        wandb.log({f'QL({a})(STD)': QL.std().item()})
    #%%
    """CRPS: Proposal model & TLAE"""
    if config["model"] in ["TLAE", "ProTran"]:
        _, samples = model.est_quantile(input_test, alphas, config["MC"], config["test_len"])
        infer_test_ = infer_test[:, config["timesteps"]:, :].reshape(-1, config["p"])
    else:
        samples = model.sampling(input_test, config["MC"])
        infer_test_ = infer_test.reshape(-1, config["p"])
    
    term1 = (samples - infer_test_[:, None, :]).abs().mean(dim=1)
    term2 = (samples[:, :, None, :] - samples[:, None, :, :]).abs().mean(dim=[1, 2]) * 0.5
    CRPS = ((term1 - term2) / maxvalues).mean(dim=0) # normalized CRPS
    print()
    print(f'CRPS: {CRPS.mean():.3f}')
    wandb.log({f'CRPS': CRPS.mean().item()})
    for c, q in zip(colnames, CRPS):
        print(f'[{c}] CRPS: {q:.3f}')
        wandb.log({f'[{c}] CRPS': q.item()})
    print()
    print(f'CRPS(STD): {CRPS.std():.3f}')
    wandb.log({f'CRPS(STD)': CRPS.std().item()})
    #%%
    """Visualize"""
    estQ = [Q[::config["future"], :, :].reshape(-1, config["p"]) for Q in est_quantiles]
    
    target = torch.cat([infer_train, infer_test], dim=0)
    if config["model"] in ["TLAE", "ProTran"]:
        target_ = target[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    else:
        target_ = target[::config["future"], :, :].reshape(-1, config["p"])
    start_idx = input_train.shape[0]
    
    """FIXME"""
    colnames = df_train.columns
    figs = utils.visualize_quantile(
        target_, estQ, start_idx+4, colnames, config["test_len"], config,
        path=plots_dir,
        show=False, dark=False)
    for j in range(len(colnames)):
        wandb.log({f'Quantile ({colnames[j]})': wandb.Image(figs[j])})
    
    # figs = utils.visualize_quantile(
    #     target_, estQ, colnames, config["test_len"], config, plots_dir,
    #     show=False, dark=False)
    # for j in range(len(colnames)):
    #     figs[j].savefig(f'{plots_dir}/{colnames[j]}_{config["model"]}_future{config["future"]}_beta{config["beta"]}_var{config["prior_var"]}.png')
    #     wandb.log({f'Quantile ({colnames[j]})': wandb.Image(figs[j])})
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