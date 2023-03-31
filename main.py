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
    parser.add_argument('--model', type=str, default='GLD', 
                        help='Fitting model options: LSQF, GLD, KUMA')
    
    parser.add_argument("--d_model", default=4, type=int,
                        help="XXX")
    parser.add_argument("--d_latent", default=4, type=int,
                        help="XXX")
    parser.add_argument("--timesteps", default=20, type=int, # equals to C
                        help="XXX")
    parser.add_argument("--future", default=1, type=int,
                        help="XXX")
    parser.add_argument("--num_heads", default=1, type=int,
                        help="XXX")
    parser.add_argument("--M", default=10, type=int,
                        help="XXX")
    # parser.add_argument("--tau", default=2, type=float,
    #                     help="scaling parameter of softmax")
    parser.add_argument("--K", default=20, type=int,
                        help="XXX")
    
    parser.add_argument('--epochs', default=500, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='learning rate')
    parser.add_argument('--threshold', default=1e-8, type=float,
                        help='threshold for clipping alpha_tilde')
    
    parser.add_argument('--prior_var', default=1, type=float,
                        help='variance of prior distribution')
    parser.add_argument('--beta', default=2, type=float,
                        help='scale parameter of asymmetric Laplace distribution')
  
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
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
    colnames = df.columns

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
    test_len = 100
    test_context = context[-test_len:]
    test_target = target[-test_len:]
    context = context[:-test_len]
    target = target[:-test_len]
    #%%
    model_module = importlib.import_module('modules.{}'.format(config["model"]))
    importlib.reload(model_module)
    model = getattr(model_module, config["model"])(config, device).to(device)
    
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
    train = importlib.import_module('modules.{}_train'.format(config["model"]))
    importlib.reload(train)
    
    if config["model"] == "KUMA": config["epochs"] = 200
    iterations = len(context) // config["batch_size"] + 1
    
    for e in range(config["epochs"]):
        logs = train.train_function(context, target, model, iterations, config, optimizer, device)
        
        print_input = "[EPOCH {:03d}]".format(e + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, y.item() / iterations) for x, y in logs.items()])
        print(print_input)
        wandb.log({x : y for x, y in logs.items()})
    #%%
    MC = 100
    alphas = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    est_quantiles = model.est_quantile(test_context, alphas, MC)
    #%%
    for i, a in enumerate(alphas):
        vrate = (test_target[:, -1, :] < est_quantiles[i]).to(torch.float32).mean()
        print('Vrate(alpha={}): {:.3f}'.format(a, vrate), ', Hit(alpha={}): {:.3f}'.format(a, a - vrate))
        wandb.log({f'Vrate(alpha={a})': vrate.item()})
        wandb.log({f'Hit(alpha={a})': (a - vrate).item()})
    #%%
    if not os.path.exists('./assets/{}'.format(config["model"])):
        os.makedirs('./assets/{}'.format(config["model"]))
        
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # for i, a in enumerate(alphas):
    #     fig, axs = plt.subplots(len(colnames), 1, sharex=True, figsize=(6, 24))
    #     for j in range(len(colnames)):
    #         axs[j].plot(test_target[::config["future"], config["timesteps"]:, j].reshape(-1, ).numpy(),
    #                 color='black', linestyle='--')
    #         axs[j].plot(est_quantiles[i][:, j].numpy(),
    #                 label=colnames[j] + f'(alpha={a})', color=cols[j])
    #         axs[j].legend(loc='upper right')
    #         # axs[j].set_ylim(-0.2, 0.2)
    #         # axs[j].set_ylabel('return', fontsize=12)
    #     plt.xlabel('days', fontsize=12)
    #     plt.tight_layout()
    #     plt.savefig(f'./assets/{config["model"]}/quantile_alpha({a}).png')
    #     # plt.show()
    #     plt.close()
    #     wandb.log({f'Quantile Estimation:alpha({a})': wandb.Image(fig)})
    #%%
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(12, 6))
    for i, a in enumerate(alphas):
        for j in range(len(colnames)):
            plt.plot(test_target[::config["future"], config["timesteps"]:, j].reshape(-1, ).numpy(),
                    color='black', linestyle='--')
            plt.plot(est_quantiles[i][:, j].numpy(),
                    label=colnames[j] + f'(alpha={a})', color=cols[j])
            # plt.legend(loc='upper right')
    plt.ylabel('return', fontsize=12)
    plt.xlabel('days', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'./assets/{config["model"]}/all_quantile.png')
    # plt.show()
    plt.close()
    wandb.log({f'Estimations All': wandb.Image(fig)})
    #%%
    # fig = plt.figure(figsize=(18, 6))
    # ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    # for j in range(len(colnames)):
    #     ax.plot(
    #         np.arange(test_context.size(0)), 
    #         est_quantiles[1][::config["future"], config["timesteps"]:, j].reshape(-1, ).numpy(), 
    #         zs=j, zdir='x', label=colnames[j])
    #     ax.plot(
    #         np.arange(test_context.size(0)), 
    #         test_target[::config["future"], config["timesteps"]:, j].reshape(-1, ).numpy(),
    #         zs=j, zdir='x', color='black', linewidth=2)
    # ax.set_xticks(range(len(colnames)))
    # ax.set_xticklabels(list(colnames))
    # ax.view_init(45, 45)
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'./assets/test.png')
    # # plt.show()
    # plt.close()
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%