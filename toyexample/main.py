#%%
import importlib
import numpy as np
import matplotlib.pyplot as plt

import torch
#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='mixture', 
                        help='Dataset options: uniform, mixture, truncated')
    parser.add_argument('--model', type=str, default='Gaussian', 
                        help='Model options: GLD_finite, GLD_infinite, Gaussian, LSQF')
    
    parser.add_argument("--latent_dim", default=2, type=int,
                        help="the latent dimension size")
    parser.add_argument("--n", default=10000, type=int,
                        help="the number of observations")
    
    parser.add_argument('--epochs', default=500, type=int,
                        help='the number of epochs')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    
    parser.add_argument("--beta", default=0.1, type=float,
                        help="weight of KL-Divergence term")
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
    """dataset"""
    dataset_module = importlib.import_module('modules.build_dataset')
    importlib.reload(dataset_module)
    data = dataset_module.build_dataset(config, device)
    
    plt.figure(figsize=(5, 3))
    # plt.hist(data.numpy(), density=True, bins="scott")
    plt.plot(np.sort(data.squeeze().numpy()), np.linspace(0, 1, len(data), endpoint=False))
    plt.ylabel("CDF", fontsize=14)
    plt.xlabel("x", fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.close()
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.DistVAE(config, device).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'])
    model.train()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params:d}")
    #%%
    """training"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    
    if config["model"] in ["GLD_finite", "GLD_infinite"]:
        train_module.GLD_train(model, optimizer, data, config, device)
    elif config["model"] == "Gaussian":
        train_module.Gaussian_train(model, optimizer, data, config, device)
    elif config["model"] == "LSQF":
        train_module.LSQF_train(model, optimizer, data, config, device)
    else:
        raise ValueError('Not valid support option for MODEL.')
    #%%
    """visualization"""
    viz_module = importlib.import_module('modules.viz')
    importlib.reload(viz_module)
    syndata = viz_module.viz_synthetic(model, data, config, show=False)
#%%
if __name__ == '__main__':
    main()
#%%    