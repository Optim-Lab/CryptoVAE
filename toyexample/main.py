#%%
import os
import importlib
import matplotlib.pyplot as plt

import torch
#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='mixture', 
                        help='Dataset options: heavytailed, uniform, mixture')
    parser.add_argument('--model', type=str, default='Gaussian', 
                        help='Model options: Gaussian, GLD_finite, GLD_infinite, LSQF')
    
    parser.add_argument("--latent_dim", default=1, type=int,
                        help="the latent dimension size")
    parser.add_argument("--n", default=5000, type=int,
                        help="the number of observations")
    
    parser.add_argument('--epochs', default=500, type=int,
                        help='the number of epochs')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    
    parser.add_argument("--step", default=0.05, type=float,
                        help="interval size of quantile levels")
    parser.add_argument('--threshold', default=1e-8, type=float,
                        help='threshold for clipping alpha_tilde')
  
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
    
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    dataset_module = importlib.import_module('modules.build_dataset')
    importlib.reload(dataset_module)
    if config["dataset"] == "heavytailed":
        data = dataset_module.build_heavytailed(config, device)
    elif config["dataset"] == "uniform":
        data = dataset_module.build_uniform(config, device)
    elif config["dataset"] == "mixture":
        data = dataset_module.build_mixture(config, device)
    else:
        raise ValueError('Not valid support option for DATASET.')
    plt.hist(data.numpy(), density=True, bins="scott")
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    if config["model"] in ["GLD_finite", "GLD_infinite"]:
        model = model_module.GLDDecoder(config, device).to(device)
    elif config["model"] == "Gaussian":
        model = model_module.GaussianDecoder(config, device).to(device)
    elif config["model"] == "LSQF":
        model = model_module.LSQFDecoder(config, device).to(device)
    else:
        raise ValueError('Not valid support option for MODEL.')

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'])
    model.train()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params:d}")
    #%%
    for epoch in range(config["epochs"]+1):
        optimizer.zero_grad()
        
        z_ = torch.randn((len(data), config["latent_dim"]))
        
        if config["model"] in ["GLD_finite", "GLD_infinite"]:
            tau = torch.linspace(0.01, 0.99, 100).unsqueeze(0).to(device)
            theta1, theta2, theta3, theta4 = model(z_)
            Q = model.quantile_function(tau, theta1, theta2, theta3, theta4)
            residual = data - Q
            loss = (residual * (tau - (residual < 0).to(torch.float32))).mean()
            
        elif config["model"] == "Gaussian":
            mean, logvar = model(z_)
            residual = data - mean
            loss = (residual.pow(2) / (2 * logvar.exp()) + 0.5 * logvar).mean()
        
        elif config["model"] == "LSQF":
            gamma, beta = model(z_)
            
            alpha_tilde = model.quantile_inverse(data, gamma, beta)
            term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde, model.delta).pow(2)
            term += 2 * torch.maximum(alpha_tilde, model.delta) * model.delta
            
            loss = (2 * alpha_tilde) * data
            loss += (1 - 2 * alpha_tilde) * gamma
            loss += (beta * term).sum(dim=1, keepdims=True)
            loss *= 0.5
            loss = loss.mean()
            
        else:
            raise ValueError('Not valid support option for MODEL.')
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print_input = "[epoch {:03d}]".format(epoch)
            print_input += ', {}: {:.4f}'.format("loss", loss)
            print(print_input)
    #%%
    with torch.no_grad():
        if config["model"] in ["GLD_finite", "GLD_infinite"]:
            z_ = torch.randn((len(data), config["latent_dim"]))
            theta1, theta2, theta3, theta4 = model(z_)
            alpha = torch.rand((len(data), 1))
            syndata = model.quantile_function(alpha, theta1, theta2, theta3, theta4)
            
        elif config["model"] == "Gaussian":
            z_ = torch.randn((len(data), config["latent_dim"]))
            mean, logvar = model(z_)
            syndata = mean + logvar.exp().sqrt() * torch.randn((len(z_), 1))
        
        elif config["model"] == "LSQF":
            z_ = torch.randn((len(data), config["latent_dim"]))
            alpha = torch.rand((len(data), 1))
            gamma, beta = model(z_)
            syndata = model.quantile_function(alpha, gamma, beta)
            
        else:
            raise ValueError('Not valid support option for MODEL.')
    
    asset_dir = f"./assets/"
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
    plt.hist(
        data.numpy(), 
        density=True, bins="scott", label=f"true({config['dataset']})", alpha=0.7)
    plt.hist(
        syndata.numpy(), 
        density=True, bins="scott", label=f"synthetic({config['model']})", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./{asset_dir}/{config['dataset']}_{config['model']}.png")
    plt.close()
    #%%
    if config["model"] == "LSQF":
        with torch.no_grad():
            alpha = torch.linspace(0, 1, 100)[:, None]
            z_ = torch.randn((len(alpha), config["latent_dim"]))
            gamma, beta = model(z_)
            syndata = model.quantile_function(alpha, gamma, beta)
            
            plt.plot(syndata.numpy(), alpha.numpy())
#%%
if __name__ == '__main__':
    main()
#%%    