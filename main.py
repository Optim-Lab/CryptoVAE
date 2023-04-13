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
    # tags=[''],
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
    parser.add_argument('--model', type=str, default='GLD_finite', 
                        help='Fitting model options: GLD_finite, GLD_infinite, LSQF, ExpLog, TLAE')
    parser.add_argument('--data', type=str, default='crypto', 
                        help='Fitting model options: crypto')
    # parser.add_argument('--standardize', action='store_false')
    
    parser.add_argument("--d_latent", default=16, type=int,
                        help="XXX")
    parser.add_argument("--d_model", default=8, type=int,
                        help="XXX")
    parser.add_argument("--timesteps", default=20, type=int, # equals to C
                        help="XXX")
    parser.add_argument("--future", default=5, type=int, # equals to T - C
                        help="XXX")
    parser.add_argument("--num_heads", default=1, type=int,
                        help="XXX")
    parser.add_argument("--num_layer", default=1, type=int,
                        help="XXX")
    parser.add_argument("--M", default=10, type=int,
                        help="XXX")
    # parser.add_argument("--tau", default=2, type=float,
    #                     help="scaling parameter of softmax")
    parser.add_argument("--K", default=20, type=int,
                        help="XXX")
    
    parser.add_argument("--MC", default=100, type=int,
                        help="XXX")
    parser.add_argument('--epochs', default=200, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.005, type=float,
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
    if os.path.isfile(f'./configs/{config["model"]}.yaml'):
        config = utils.load_config(config)
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if config["cuda"] else torch.device('cpu')
    wandb.config.update(config)

    utils.set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    df = pd.read_csv(
        f'./data/{config["data"]}.csv',
        index_col=0
    )
    
    test_len = 500
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
    
    iterations = len(train_context) // config["batch_size"] + 1
    
    for e in range(config["epochs"]):
        logs = train_module.train_function(train_context, train_target, model, iterations, config, optimizer, device)
        
        print_input = "[EPOCH {:03d}]".format(e + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, y.item() / iterations) for x, y in logs.items()])
        print(print_input)
        wandb.log({x : y for x, y in logs.items()})
    #%%
    """Quantile Estimation"""
    alphas = [0.1, 0.5, 0.9]
    context = torch.cat([train_context, test_context], dim=0)
    target = torch.cat([train_target, test_target], dim=0)
    
    if config["model"] == "TLAE":
        full_est_quantiles, _ = model.est_quantile(context, alphas, config["MC"], test_len)
    else:
        full_est_quantiles = model.est_quantile(context, alphas, config["MC"])
    
    est_quantiles = [Q[-test_len:, ...] for Q in full_est_quantiles]
    #%%
    if not os.path.exists('./assets/{}'.format(config["model"])):
        os.makedirs('./assets/{}'.format(config["model"]))
    if not os.path.exists('./assets/{}/out(future={})/'.format(config["model"], config["future"])):
        os.makedirs('./assets/{}/out(future={})/'.format(config["model"], config["future"]))
    if not os.path.exists('./assets/{}/plots(future={})/'.format(config["model"], config["future"])):
        os.makedirs('./assets/{}/plots(future={})/'.format(config["model"], config["future"]))
    #%%
    """Vrate and Hit"""
    test_target_ = test_target[:, config["timesteps"]:, :].reshape(-1, config["p"])
    est_quantiles_ = [Q[:, :, :].reshape(-1, config["p"]) for Q in est_quantiles]
    for i, a in enumerate(alphas):
        vrate = (test_target_ < est_quantiles_[i]).to(torch.float32).mean(dim=0)
        hit = (a - vrate).mean().abs()
        print('Vrate(alpha={}): {:.3f}'.format(a, vrate.mean()), 
                ', Hit(alpha={}): {:.3f}'.format(a, hit))
        wandb.log({f'Vrate(alpha={a})': vrate.mean().item()})
        wandb.log({f'Hit(alpha={a})': hit.item()})
    #%%
    """Visualize"""
    target_ = target[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    test_target_ = test_target[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    full_est_quantiles_ = [Q[::config["future"], :, :].reshape(-1, config["p"]) for Q in full_est_quantiles]
    est_quantiles_ = [Q[::config["future"], :, :].reshape(-1, config["p"]) for Q in est_quantiles]
    
    figs = utils.visualize_quantile(target_, test_target_, full_est_quantiles_, est_quantiles_, colnames, config, show=False, dark=False)
    for j in range(len(colnames)):
        wandb.log({f'Quantile ({colnames[j]})': wandb.Image(figs[j])})
    #%%
    """data save"""
    train.to_csv(f'./assets/{config["model"]}/{config["data"]}_train.csv')
    test.to_csv(f'./assets/{config["model"]}/{config["data"]}_test.csv')
    #%%
    """model save"""
    torch.save(model.state_dict(), f'./assets/{config["data"]}_{config["model"]}_{config["future"]}.pth')
    artifact = wandb.Artifact(f'{config["data"]}_{config["model"]}_{config["future"]}', 
                            type='model',
                            metadata=config) # description=""
    artifact.add_file(f'./assets/{config["data"]}_{config["model"]}_{config["future"]}.pth')
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