#%%
import numpy as np
import tqdm
import torch
# import wandb
#%%
def train_function(context, target, model, iterations, config, optimizer, device):
    logs = {
        'loss': 0,
        'quantile': 0,
        'KL': 0,
        'active': 0,
    }
    for i in range(iterations):
        idx = np.random.choice(
            range(len(context)), config["batch_size"], replace=False)
        context_batch = context[idx, :].to(device)
        target_batch = target[idx, :].to(device)
        
        optimizer.zero_grad()
        
        prior, posterior, params = model(context_batch, target_batch)
        target_batch_ = target_batch.reshape(-1, 1)
        
        gamma = torch.cat([torch.cat(params[i][0], dim=0) for i in range(len(params))], dim=0)
        beta = torch.cat([torch.cat(params[i][1], dim=0) for i in range(len(params))], dim=0)
        delta = torch.cat([torch.cat(params[i][2], dim=0) for i in range(len(params))], dim=0)
        
        """alpha_tilde"""
        alpha_tilde_list = model.quantile_inverse(target_batch_, gamma, beta, delta)
        
        """CRPS loss"""
        term = (1 - delta.pow(3)) / 3 - delta - torch.maximum(alpha_tilde_list, delta).pow(2)
        term += 2 * torch.maximum(alpha_tilde_list, delta) * delta
        quantile = (2 * alpha_tilde_list - 1) * target_batch_
        quantile += (1 - 2 * alpha_tilde_list) * gamma
        quantile += (beta * term).sum(dim=1, keepdims=True)
        quantile = quantile.sum() / context_batch.size(0)
        logs["quantile"] = logs.get("quantile") + quantile
        
        """KL-divergence"""
        prior_mean = prior.mean.reshape(-1, config["d_latent"])
        prior_logvar = prior.logvar.reshape(-1, config["d_latent"])
        posterior_mean = posterior.mean.reshape(-1, config["d_latent"])
        posterior_logvar = posterior.logvar.reshape(-1, config["d_latent"])
        
        KL = ((posterior_mean - prior_mean).pow(2) / prior_logvar.exp()).sum(dim=1)
        KL += (prior_logvar - posterior_logvar).sum(dim=1)
        KL += (posterior_logvar.exp() / prior_logvar.exp()).sum(dim=1)
        KL -= config["d_latent"]
        KL *= 0.5
        KL = KL.sum() / context_batch.size(0)
        logs["KL"] = logs.get("KL") + KL
        
        loss = quantile + config["beta"] * KL
        logs["loss"] = logs.get("loss") + loss
        
        active = (posterior.logvar.exp().mean(dim=0) < 0.1).sum()
        logs["active"] = logs.get("active") + active
        
        loss.backward()
        optimizer.step()
        
    return logs
#%%