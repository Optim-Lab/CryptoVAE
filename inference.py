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
    tags=["Inference", "Incremental procedure"],
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
    parser.add_argument('--data', type=str, default='crypto', 
                        help='Fitting model options: crypto')
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
    artifact = wandb.use_artifact('anseunghwan/DDM/phase_{}_{}_{}:v{}'.format(
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
        
    plots_dir = f'./assets/{config["model"]}/plots(future={config["future"]})/beta{config["beta"]}_var{config["prior_var"]}'    
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    #%%
    """train, test split"""
    df = pd.read_csv(
        f'./data/{config["data"]}.csv',
        index_col=0
    )
    print(df.describe())
    
    colnames = df.columns
    config["p"] = df.shape[1]
    if config["model"] in ["TLAE", "ProTran"]: # reconstruct T
        train_list, test_list = utils.build_datasets2(df, config["test_len"], config["increment"], config)
    else: # reconstruct only T - C
        train_list, test_list = utils.build_datasets(df, config["test_len"], config["increment"], config)
    #%%
    try:
        model_module = importlib.import_module('modules.{}'.format(config["model"]))
        importlib.reload(model_module)
        model = [getattr(model_module, config["model"])(config, device).to(device) for _ in range(config["increment"])]
    except:
        model_module = importlib.import_module('modules.{}'.format(config["model"].split('_')[0]))
        importlib.reload(model_module)
        model = [getattr(model_module, config["model"].split('_')[0])(config, device).to(device) for _ in range(config["increment"])]
    #%%
    if config["cuda"]:
        model_name = sorted([x for x in os.listdir(model_dir) if x.endswith('pth')])
        for n, m in zip(model_name, model):
            m.load_state_dict(
                torch.load(
                    model_dir + '/' + n))
    else:
        model_name = sorted([x for x in os.listdir(model_dir) if x.endswith('pth')])
        for n, m in zip(model_name, model):
            m.load_state_dict(
                torch.load(
                    model_dir + '/' + n, map_location=torch.device('cpu')))
    
    for m in model:
        print(m.eval())
    #%%
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model[0])
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    """Quantile Estimation"""
    # Get maximum for normalization
    maxvalues = [test[1].reshape(-1, config["p"]).max(dim=0, keepdims=True).values for test in test_list]
    
    alphas = [0.1, 0.5, 0.9]
    phaseQ = []
    for j, ((train_context, train_target), (test_context, test_target)) in enumerate(zip(train_list, test_list)):
        print(f"\nPhase {j+1} Quantile Estimation...\n")

        if config["model"] in ["TLAE", "ProTran"]:
            est_quantiles, _ = model[j].est_quantile(test_context, alphas, config["MC"], config["test_len"])
        else:
            est_quantiles = model[j].est_quantile(test_context, alphas, config["MC"])
        phaseQ.append(est_quantiles)
        print()
        
        if config["model"] in ["TLAE", "ProTran"]:
            test_target_ = test_target[:, config["timesteps"]:, :].reshape(-1, config["p"])
        else:
            test_target_ = test_target.reshape(-1, config["p"])
        
        """DICR"""
        est_quantiles_ = [Q[:, :, :].reshape(-1, config["p"]) for Q in est_quantiles]
        CR = ((est_quantiles_[0] < test_target_) * (test_target_ < est_quantiles_[-1])).to(torch.float32).mean(dim=0)
        INTERVAL = ((est_quantiles_[-1] - est_quantiles_[0]) / maxvalues[j]).mean(dim=0)
        DICR = (CR - (alphas[-1] - alphas[0])).abs()
        
        print(f'[Phase{j+1}] INTERVAL: {INTERVAL.mean():.3f},')
        print(f'[Phase{j+1}] DICR: {DICR.mean():.3f},')
        wandb.log({f'[Phase{j+1}] INTERVAL': INTERVAL.mean().item()})
        wandb.log({f'[Phase{j+1}] DICR': DICR.mean().item()})
        for c, x in zip(colnames, INTERVAL):
            print(f'[Phase{j+1}, {c}] INTERVAL: {x:.3f},')
            wandb.log({f'[Phase{j+1}, {c}] INTERVAL': x.item()})
        print()
        for c, x in zip(colnames, DICR):
            print(f'[Phase{j+1}, {c}] DICR: {x:.3f},')
            wandb.log({f'[Phase{j+1}, {c}] DICR': x.item()})
        print()
        print(f'[Phase{j+1}] INTERVAL: {INTERVAL.std():.3f},')
        print(f'[Phase{j+1}] DICR: {DICR.std():.3f},')
        wandb.log({f'[Phase{j+1}] INTERVAL(STD)': INTERVAL.std().item()})
        wandb.log({f'[Phase{j+1}] DICR(STD)': DICR.std().item()})
        
        """Quantile loss"""
        for i, a in enumerate(alphas):
            u = test_target_ - est_quantiles_[i]
            QL = (((a - (u < 0).to(torch.float32)) * u) / maxvalues[j]).mean(dim=0) # normalized QL
            print(f'[Phase{j+1}] QL({a}): {QL.mean():.3f}')
            wandb.log({f'[Phase{j+1}] QL({a})': QL.mean().item()})
            for c, q in zip(colnames, QL):
                print(f'[Phase{j+1}, {c}] QL({a}): {q:.3f}')
                wandb.log({f'[Phase{j+1}, {c}] QL({a})': q.item()})
            print()
            print(f'[Phase{j+1}] QL({a})(STD): {QL.std():.3f}')
            wandb.log({f'[Phase{j+1}] QL({a})(STD)': QL.std().item()})
    #%%
    """CRPS: Proposal model & TLAE"""
    for j, (test_context, test_target) in enumerate(test_list):
        print(f"\nPhase {j+1} CRPS...\n")

        if config["model"] in ["TLAE", "ProTran"]:
            _, samples = model[j].est_quantile(test_context, alphas, config["MC"], config["test_len"])
            test_target_ = test_target[:, config["timesteps"]:, :].reshape(-1, config["p"])
        else:
            samples = model[j].sampling(test_context, config["MC"])
            test_target_ = test_target.reshape(-1, config["p"])
        
        term1 = (samples - test_target_[:, None, :]).abs().mean(dim=1)
        term2 = (samples[:, :, None, :] - samples[:, None, :, :]).abs().mean(dim=[1, 2]) * 0.5
        CRPS = ((term1 - term2) / maxvalues[j]).mean(dim=0) # normalized CRPS
        print()
        print(f'[Phase{j+1}] CRPS: {CRPS.mean():.3f}')
        wandb.log({f'[Phase{j+1}] CRPS': CRPS.mean().item()})
        for c, q in zip(colnames, CRPS):
            print(f'[Phase{j+1}, {c}] CRPS: {q:.3f}')
            wandb.log({f'[Phase{j+1}, {c}] CRPS': q.item()})
        print()
        print(f'[Phase{j+1}] CRPS(STD): {CRPS.std():.3f}')
        wandb.log({f'[Phase{j+1}] CRPS(STD)': CRPS.std().item()})
    #%%
    """Visualize"""
    # remove "KRW-" in column names
    colnames = [x.replace("KRW-", "") for x in colnames]
    
    estQ = []
    for j in range(len(alphas)):
        estQ.append(torch.cat([phaseQ[i][j] for i in range(len(phaseQ))], dim=0))
    estQ = [Q[::config["future"], :, :].reshape(-1, config["p"]) for Q in estQ]
    
    target = torch.cat([train_list[-1][1], test_list[-1][1]], dim=0)
    if config["model"] in ["TLAE", "ProTran"]:
        target_ = target[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    else:
        target_ = target[::config["future"], :, :].reshape(-1, config["p"])
    start_idx = train_list[0][0].shape[0]
    
    figs = utils.visualize_quantile(
        target_, estQ, colnames, config["test_len"], config, plots_dir,
        show=False, dark=False)
    for j in range(len(colnames)):
        figs[j].savefig(f'{plots_dir}/{colnames[j]}_{config["model"]}_future{config["future"]}_beta{config["beta"]}_var{config["prior_var"]}.png')
        wandb.log({f'Quantile ({colnames[j]})': wandb.Image(figs[j])})
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