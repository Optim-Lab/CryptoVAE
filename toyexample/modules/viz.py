#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
#%%
def viz_synthetic(model, data, config, show=False):
    torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    z_ = torch.randn((len(data), config["latent_dim"]))
    
    with torch.no_grad():
        if config["model"] in ["GLD_finite", "GLD_infinite"]:
            alpha = torch.rand((len(data), 1))
            theta1, theta2, theta3, theta4 = model.GLD_decode(z_)
            syndata = model.GLD_quantile_function(alpha, theta1, theta2, theta3, theta4)
            
        elif config["model"] == "Gaussian":
            xhat = model.Gaussian_decode(z_)
            syndata = xhat + np.sqrt(config["beta"]) * torch.randn((len(data), 1))
            # h = model.Gaussian_decode(z_)
            # xhat, logsigma = h[:, [0]], h[:, [1]]
            # syndata = xhat + logsigma.exp().sqrt() * torch.randn((len(data), 1))
        
        elif config["model"] == "LSQF":
            alpha = torch.rand((len(data), 1))
            gamma, beta = model.LSQF_decode(z_)
            syndata = model.LSQF_quantile_function(alpha, gamma, beta)
            
        else:
            raise ValueError('Not valid support option for MODEL.')
    
    asset_dir = f"./assets/"
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(
        data.numpy(), 
        density=True, bins="scott", label=f"observed({config['dataset']})", alpha=0.7)
    ax[0].hist(
        syndata.numpy(), 
        density=True, bins="scott", label=f"synthetic({config['model']})", alpha=0.7)
    ax[0].set_ylabel("density", fontsize=18)
    ax[0].set_xlabel("x", fontsize=18)
    
    ax[1].plot(
        np.sort(data.squeeze().numpy()), np.linspace(0, 1, len(data), endpoint=False),
        label=f"observed({config['dataset']})")
    ax[1].plot(
        np.sort(syndata.squeeze().numpy()), np.linspace(0, 1, len(syndata), endpoint=False),
        label=f"synthetic({config['model']})")
    ax[1].set_ylabel("CDF", fontsize=16)
    ax[1].set_xlabel("x", fontsize=16)
    
    # ax[0].legend(fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(
        handles, labels, 
        ncol=2,
        loc='upper center', fontsize=14, bbox_to_anchor=(0.5, 1.2))
    
    plt.tight_layout()
    plt.savefig(f"./{asset_dir}/synthetic_{config['dataset']}_{config['model']}.png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    return syndata
#%%