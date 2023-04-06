#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import yaml
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
                        help='Fitting model options: LSQF, GLD, KUMA, TLAE')
    parser.add_argument('--standardize', action='store_false')
    
    parser.add_argument("--d_model", default=8, type=int,
                        help="XXX")
    parser.add_argument("--d_latent", default=4, type=int,
                        help="XXX")
    parser.add_argument("--timesteps", default=20, type=int, # equals to C
                        help="XXX")
    parser.add_argument("--future", default=1, type=int,
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
    
    parser.add_argument('--prior_var', default=0.1, type=float,
                        help='variance of prior distribution')
    parser.add_argument('--beta', default=1, type=float,
                        help='scale parameter of asymmetric Laplace distribution')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
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
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """load config"""
    if os.path.isfile(f'./configs/{config["model"]}.yaml'):
        config = load_config(config)
    
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
    """Training"""
    train_module = importlib.import_module('modules.{}_train'.format(config["model"]))
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
    # alphas = [0.025, 0.05]
    alphas = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
    if config["model"] == "TLAE":
        est_quantiles, samples = model.est_quantile(test_context, alphas, config["MC"])
    else:
        est_quantiles, Qs = model.est_quantile(test_context, alphas, config["MC"])
        samples = model.sampling(test_context, config["MC"])
    #%%
    if config["standardize"]:
        test_target_ = (test_target.reshape(-1, config["p"]) * std_.to_numpy()[None, :] + mean_.to_numpy()[None, :]
            ).reshape(test_context.size(0), config["timesteps"] + config["future"], config["p"])
        est_quantiles = [x * std_.to_numpy()[None, :] + mean_.to_numpy()[None, :] for x in est_quantiles]
        samples = samples * std_.to_numpy()[None, :] + mean_.to_numpy()[None, :]
    else:
        test_target_ = test_target.clone()
    #%%
    if not os.path.exists('./assets/{}'.format(config["model"])):
        os.makedirs('./assets/{}'.format(config["model"]))
    if not os.path.exists('./assets/{}/out/'.format(config["model"])):
        os.makedirs('./assets/{}/out/'.format(config["model"]))
    #%%
    """Vrate and Hit"""
    test_target_ = test_target_[::config["future"], config["timesteps"]:, :].reshape(-1, config["p"])
    df = pd.DataFrame(
        test_target_.numpy(),
        columns=colnames
    )
    df.to_csv(f'./assets/{config["model"]}/out/test_data.csv')
    wandb.run.summary['test_data'] = wandb.Table(data=df)
    
    for i, a in enumerate(alphas):
        vrate = (test_target_ < est_quantiles[i]).to(torch.float32).mean(dim=0)
        for j, name in enumerate(colnames):
            print('Vrate([{}], alpha={}): {:.3f}'.format(name, a, vrate[j]), 
                  ', Hit([{}], alpha={}): {:.3f}'.format(name, a, (a - vrate[j]).abs()))
            wandb.log({f'Vrate([{name}], alpha={a})': vrate[j].item()})
            wandb.log({f'Hit([{name}], alpha={a})': (a - vrate[j]).abs().item()})
        df = pd.DataFrame(
            est_quantiles[i].numpy(),
            columns=colnames     
        )
        df.to_csv(f'./assets/{config["model"]}/out/VaR(alpha={a}).csv')
        wandb.run.summary[f'VaR(alpha={a})'] = wandb.Table(data=df)
    #%%
    """Expected Shortfall"""
    for i, a in enumerate(alphas):
        residual = samples - est_quantiles[i][:, None, :]
        ES = samples.mean(dim=1) - (residual * (a - (residual < 0).to(torch.float32))).mean(dim=1) / a
        df = pd.DataFrame(
            ES.numpy(),
            columns=colnames
        )
        df.to_csv(f'./assets/{config["model"]}/out/ES(alpha={a}).csv')
        wandb.run.summary[f'ES(alpha={a})'] = wandb.Table(data=df)
    #%%
    """Visualize"""
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(12, 6))
    for i, a in enumerate(alphas):
        for j in range(len(colnames)):
            plt.plot(test_target_.numpy(),
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
    """data save"""
    train.to_csv(f'./assets/{config["model"]}/train.csv')
    test.to_csv(f'./assets/{config["model"]}/test.csv')
    #%%
    """model save"""
    torch.save(model.state_dict(), './assets/coin_{}.pth'.format(config["model"]))
    artifact = wandb.Artifact('coin_{}'.format(config["model"]), 
                            type='model',
                            metadata=config) # description=""
    artifact.add_file(f'./assets/coin_{config["model"]}.pth')
    artifact.add_file('./main.py')
    artifact.add_file(f'./modules/{config["model"]}.py')
    artifact.add_file(f'./modules/{config["model"]}_train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
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