#%%
import os
import importlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

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
    
    parser.add_argument("--latent_dim", default=2, type=int,
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
        
        if config["model"] in ["GLD_finite", "GLD_infinite"]:
            tau = torch.linspace(0.01, 0.99, 20).unsqueeze(0).to(device)
            z, mean, logvar, theta1, theta2, theta3, theta4 = model(data)
            Q = model.quantile_function(tau, theta1, theta2, theta3, theta4)
            residual = data - Q
            loss = (residual * (tau - (residual < 0).to(torch.float32))).mean()
            
            """KL-Divergence"""
            KL = torch.pow(mean, 2).sum(dim=1)
            KL -= logvar.sum(dim=1)
            KL += torch.exp(logvar).sum(dim=1)
            KL -= config["latent_dim"]
            KL *= 0.5
            KL = KL.mean()
            
            beta_ = 0.1
            loss += beta_ * KL
            
        elif config["model"] == "Gaussian":
            z, mean, logvar, xhat = model(data)
            residual = data - xhat
            loss = 0.5 * (residual.pow(2)).mean()
            
            """KL-Divergence"""
            KL = torch.pow(mean, 2).sum(dim=1)
            KL -= logvar.sum(dim=1)
            KL += torch.exp(logvar).sum(dim=1)
            KL -= config["latent_dim"]
            KL *= 0.5
            KL = KL.mean()
            
            beta_ = 0.1
            loss += beta_ * KL
            
            # z, mean, logvar, xhat, logsigma = model(data)
            # residual = data - xhat
            # loss = (residual.pow(2) / (2 * logsigma.exp()) + 0.5 * logsigma).squeeze()
            
            # """KL-Divergence"""
            # KL = torch.pow(mean, 2).sum(dim=1)
            # KL -= logvar.sum(dim=1)
            # KL += torch.exp(logvar).sum(dim=1)
            # KL -= config["latent_dim"]
            # KL *= 0.5
            
            # loss = (loss + KL).mean()
            # KL = KL.mean()
        
        elif config["model"] == "LSQF":
            z, mean, logvar, gamma, beta = model(data)
            
            alpha_tilde = model.quantile_inverse(data, gamma, beta)
            term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde, model.delta).pow(2)
            term += 2 * torch.maximum(alpha_tilde, model.delta) * model.delta
            
            loss = (2 * alpha_tilde) * data
            loss += (1 - 2 * alpha_tilde) * gamma
            loss += (beta * term).sum(dim=1, keepdims=True)
            loss *= 0.5
            loss = loss.mean()
            
            """KL-Divergence"""
            KL = torch.pow(mean, 2).sum(dim=1)
            KL -= logvar.sum(dim=1)
            KL += torch.exp(logvar).sum(dim=1)
            KL -= config["latent_dim"]
            KL *= 0.5
            KL = KL.mean()
            
            beta_ = 0.1
            loss += beta_ * KL
            
        else:
            raise ValueError('Not valid support option for MODEL.')
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print_input = "[epoch {:03d}]".format(epoch)
            print_input += ', {}: {:.4f}, {}: {:.4f}'.format("loss", loss, "KL", KL)
            print(print_input)
    #%%
    with torch.no_grad():
        if config["model"] in ["GLD_finite", "GLD_infinite"]:
            z_ = torch.randn((len(data), config["latent_dim"]))
            alpha = torch.rand((len(data), 1))
            theta1, theta2, theta3, theta4 = model.decode(z_)
            syndata = model.quantile_function(alpha, theta1, theta2, theta3, theta4)
            
        elif config["model"] == "Gaussian":
            z_ = torch.randn((len(data), config["latent_dim"]))
            xhat = model.decoder(z_)
            syndata = xhat + np.sqrt(beta_) * torch.randn((len(data), 1))
            # h = model.decoder(z_)
            # xhat, logsigma = h[:, [0]], h[:, [1]]
            # syndata = xhat + logsigma.exp().sqrt() * torch.randn((len(data), 1))
        
        elif config["model"] == "LSQF":
            z_ = torch.randn((len(data), config["latent_dim"]))
            alpha = torch.rand((len(data), 1))
            gamma, beta = model.decode(z_)
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
    plt.savefig(f"./{asset_dir}/aggregated_{config['dataset']}_{config['model']}.png")
    # plt.show()
    plt.close()
    #%%
    if config["model"] in ["GLD_finite", "GLD_infinite"]:
        with torch.no_grad():
            z_ = torch.randn((30, config["latent_dim"]))
            theta1, theta2, theta3, theta4 = model.decode(z_)

        for i in range(len(z_)):
            alpha = torch.linspace(0, 1, 100)[:, None]
            syndata = model.quantile_function(alpha, theta1[[i]], theta2[[i]], theta3[[i]], theta4[[i]])
            plt.plot(syndata.numpy(), alpha.numpy())
        plt.plot(
            np.sort(data.squeeze().numpy()), 
            np.linspace(0, 1, len(data), endpoint=False))
        plt.fill_between(
            np.sort(data.squeeze().numpy()), 
            np.linspace(0, 1, len(data), endpoint=False), 
            color='blue', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"./{asset_dir}/latent_{config['dataset']}_{config['model']}.png")
        # plt.show()
        plt.close()
    #%%
    if config["model"] == "LSQF":
        with torch.no_grad():
            z_ = torch.randn((30, config["latent_dim"]))
            gamma, beta = model.decode(z_)

        for i in range(len(z_)):
            alpha = torch.linspace(0, 1, 100)[:, None]
            syndata = model.quantile_function(alpha, gamma[[i]], beta[[i]])
            plt.plot(syndata.numpy(), alpha.numpy())
        plt.plot(
            np.sort(data.squeeze().numpy()), 
            np.linspace(0, 1, len(data), endpoint=False))
        plt.fill_between(
            np.sort(data.squeeze().numpy()), 
            np.linspace(0, 1, len(data), endpoint=False), 
            color='blue', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"./{asset_dir}/latent_{config['dataset']}_{config['model']}.png")
        # plt.show()
        plt.close()
    #%%
    if config["model"] == "Gaussian":
        with torch.no_grad():
            z_ = torch.randn((30, config["latent_dim"])) # prior distribution
            xhat = model.decoder(z_)
            # h = model.decoder(z_)
            # xhat, logsigma = h[:, [0]], h[:, [1]]
        
        weight = stats.norm.pdf(z_.numpy(), 0, 1)
        mu = xhat.numpy()
        sigma = np.ones((len(z_), 1)) * np.sqrt(beta_)
        # sigma = logsigma.exp().sqrt().numpy()
        
        for i in range(len(z_)):
            x = np.linspace(mu[i] - 3*sigma[i], mu[i] + 3*sigma[i], 100)
            plt.plot(x, weight[i] * stats.norm.pdf(x, mu[i], sigma[i]))
        plt.hist(data.numpy(), density=True, bins="scott", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"./{asset_dir}/latent_{config['dataset']}_{config['model']}.png")
        # plt.show()
        plt.close()
    #%%
    # if config["model"] == "LSQF":
    #     with torch.no_grad():
    #         alpha = torch.linspace(0, 1, 100)[:, None]
    #         z_ = torch.randn((len(alpha), config["latent_dim"]))
    #         gamma, beta = model(z_)
    #         syndata = model.quantile_function(alpha, gamma, beta)
            
    #         plt.plot(syndata.numpy(), alpha.numpy())
#%%
if __name__ == '__main__':
    main()
#%%    