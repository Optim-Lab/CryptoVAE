#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
    # tags=[""],
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
    parser.add_argument('--model', type=str, default='GLD_infinite', 
                        help='Fitting model options: GLD_finite, GLD_infinite, LSQF, ExpLog, TLAE, ProTran')
    parser.add_argument('--data', type=str, default='air', 
                        help='Fitting model options: air')
    
    parser.add_argument("--d_latent", default=16, type=int,
                        help="size of latent dimension")
    parser.add_argument("--d_model", default=8, type=int,
                        help="size of transformer model dimension")
    parser.add_argument("--num_heads", default=1, type=int,
                        help="the number of heads in transformer")
    parser.add_argument("--num_layer", default=1, type=int,
                        help="the number of layers in transformer")
    parser.add_argument("--M", default=10, type=int,
                        help="the number of knot points")
    parser.add_argument("--K", default=20, type=int,
                        help="the number of quantiles to estimate")
    
    parser.add_argument("--timesteps", default=30, type=int, # equals to C
                        help="the number of conditional time steps")
    parser.add_argument("--future", default=5, type=int, # equals to T - C
                        help="the number of time steps to forecasting")
    parser.add_argument("--test_len", default=365, type=int,
                        help="length of test dataset in each phase")
    
    parser.add_argument("--MC", default=100, type=int,
                        help="the number of samples in Monte Carlo sampling")
    parser.add_argument('--epochs', default=400, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.0025, type=float,
                        help='learning rate')
    parser.add_argument('--threshold', default=1e-8, type=float,
                        help='threshold for clipping alpha_tilde')
    
    parser.add_argument('--prior_var', default=1, type=float,
                        help='variance of prior distribution')
    parser.add_argument('--beta', default=1, type=float,
                        help='scale parameter of asymmetric Laplace distribution')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """load config"""
    if config["model"].startswith("GLD"):
        config_path = f'./configs/{config["model"]}_{config["future"]}.yaml'
        if os.path.isfile(config_path):
            config = utils.load_config(config, config_path)
    else:
        config_path = f'./configs/{config["model"]}.yaml'
        if os.path.isfile(config_path):
            config = utils.load_config(config, config_path)
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if config["cuda"] else torch.device('cpu')
    wandb.config.update(config)

    utils.set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    if not os.path.exists('./assets/{}'.format(config["model"])):
        os.makedirs('./assets/{}'.format(config["model"]))
        
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
    """model"""
    try:
        model_module = importlib.import_module('modules.{}'.format(config["model"]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"])(config, device).to(device)
    except:
        model_module = importlib.import_module('modules.{}'.format(config["model"].split('_')[0]))
        importlib.reload(model_module)
        model = getattr(model_module, config["model"].split('_')[0])(config, device).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    model.train()
    #%%
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    """Training"""
    try:
        train_module = importlib.import_module('modules.{}_train'.format(config["model"]))
    except:
        train_module = importlib.import_module('modules.{}_train'.format(config["model"].split('_')[0]))
    importlib.reload(train_module)
    
    iterations = len(input_train) // config["batch_size"] + 1
        
    for e in range(config["epochs"]):
        logs = train_module.train_function(input_train, infer_train, model, iterations, config, optimizer, device)
        
        if e % 10 == 0 or e == config["epochs"] - 1:
            print_input = "[EPOCH {:03d}]".format(e + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, y.item() / iterations) for x, y in logs.items()])
            print(print_input)
            wandb.log({x : y for x, y in logs.items()})
    #%%
    plots_dir = './air_assets/{}/plots(future={})/'.format(config["model"], config["future"])
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    #%%
    """Quantile Estimation"""
    alphas = [0.1, 0.5, 0.9]
    if config["model"] in ["TLAE", "ProTran"]:
        est_quantiles, _ = model.est_quantile(input_test, alphas, config["MC"], config["test_len"])
    else:
        est_quantiles = model.est_quantile(input_test, alphas, config["MC"])
    #%%
    """CRPS: Proposal model & TLAE"""
    # Get maximum for normalization
    maxvalues = infer_test.reshape(-1, config["p"]).max(dim=0, keepdims=True).values
    
    if config["model"] in ["TLAE", "ProTran"]:
        _, samples = model.est_quantile(input_test, alphas, config["MC"], config["test_len"])
        infer_test_ = infer_test[:, config["timesteps"]:, :].reshape(-1, config["p"])
    else:
        samples = model.sampling(input_test, config["MC"])
        infer_test_ = infer_test.reshape(-1, config["p"])
    
    term1 = (samples - infer_test_[:, None, :]).abs().mean(dim=1)
    term2 = (samples[:, :, None, :] - samples[:, None, :, :]).abs().mean(dim=[1, 2]) * 0.5
    CRPS = ((term1 - term2) / maxvalues).mean(dim=0) # normalized CRPS
    print(f'CRPS: {CRPS.mean():.3f}')
    wandb.log({f'CRPS': CRPS.mean().item()})
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
    #%%
    """model save"""
    if not os.path.exists("./air_assets/models/"):
        os.makedirs("./air_assets/models/")
    artifact = wandb.Artifact(
        f'{config["data"]}_{config["model"]}_{config["future"]}', 
        type='model',
        metadata=config) # description=""
    torch.save(model.state_dict(), f'./air_assets/models/{config["data"]}_{config["model"]}_{config["future"]}.pth')
    artifact.add_file(f'./air_assets/models/{config["data"]}_{config["model"]}_{config["future"]}.pth')
    artifact.add_file('./main.py')
    try:
        artifact.add_file(f'./modules/{config["model"]}.py')
        artifact.add_file(f'./modules/{config["model"]}_train.py')
    except:
        artifact.add_file('./modules/{}.py'.format(config["model"].split('_')[0]))
        artifact.add_file('./modules/{}_train.py'.format(config["model"].split('_')[0]))
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%