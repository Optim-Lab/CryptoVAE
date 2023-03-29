#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
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
models = importlib.import_module('modules.models')
importlib.reload(models)
train = importlib.import_module('modules.train')
importlib.reload(train)
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
    
    parser.add_argument("--d_model", default=32, type=int,
                        help="XXX")
    parser.add_argument("--d_latent", default=2, type=int,
                        help="XXX")
    parser.add_argument("--timesteps", default=10, type=int, # equals to C
                        help="XXX")
    parser.add_argument("--future", default=10, type=int,
                        help="XXX")
    parser.add_argument("--num_heads", default=4, type=int,
                        help="XXX")
    parser.add_argument("--num_layers", default=2, type=int,
                        help="XXX")
    parser.add_argument("--M", default=10, type=int,
                        help="XXX")
    parser.add_argument("--tau", default=2, type=float,
                        help="scaling parameter of softmax")
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--threshold', default=1e-6, type=float,
                        help='threshold for clipping alpha_tilde')
    
    parser.add_argument('--beta', default=1, type=float,
                        help='scale parameter of asymmetric Laplace distribution')
  
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    df = pd.read_csv(
        './data/' + os.listdir('./data')[0],
        index_col=0
    )
    df.head()

    config["p"] = df.shape[1]
    #%%
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

    context, target = stock_data_generator(df, config["timesteps"], config["future"])
    assert context.shape == (df.shape[0] - config["timesteps"] - config["future"], config["timesteps"], df.shape[1])
    assert target.shape == (df.shape[0] - config["timesteps"] - config["future"], config["timesteps"] + config["future"], df.shape[1])
    #%%
    model = models.DDM(config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )

    model.train()

    iterations = len(context) // config["batch_size"] + 1
    #%%
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    for e in range(config["epochs"]):
        logs = train.train_function(context, target, model, iterations, config, optimizer, device)
        
        print_input = "[EPOCH {:03d}]".format(e + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, y.item()) for x, y in logs.items()])
        print(print_input)
        wandb.log({x : y for x, y in logs.items()})
    #%%
    prior_mean, prior_logvar = model.get_prior(context[-10:])
    epsilon = torch.randn(prior_mean.shape)
    z = prior_mean + (prior_logvar / 2).exp() * epsilon
    params = model.get_spline(z)

    gamma = torch.cat([torch.cat(params[i][0], dim=0) for i in range(len(params))], dim=0)
    beta = torch.cat([torch.cat(params[i][1], dim=0) for i in range(len(params))], dim=0)
    delta = torch.cat([torch.cat(params[i][2], dim=0) for i in range(len(params))], dim=0)

    alpha = torch.ones(gamma.shape) * 0.5

    xhat = model.quantile_function(alpha, gamma, beta, delta)

    xhat.reshape(10, 20, 13)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%