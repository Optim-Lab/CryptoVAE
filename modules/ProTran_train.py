#%%
import numpy as np
import tqdm
import torch
# from joblib import Parallel, delayed
#%%
def train_function(context, target, model, iterations, config, optimizer, device):
    
    logs = {
        'loss': 0,
        'recon': 0,
        'KL': 0,
        'active': 0,
    }
    for i in range(iterations):
        # context, target = train_context, train_target
        idx = np.random.choice(
            range(len(context)), config["batch_size"], replace=False)
        context_batch = context[idx, :].to(device)
        target_batch = target[idx, :].to(device)
        
        optimizer.zero_grad()
        
        prior, posterior, xhat = model(context_batch, target_batch)
        
        """Reconstruction"""
        recon = (target_batch - xhat).abs().sum(dim=-1).mean()
        logs["recon"] = logs.get("recon") + recon
        
        """KL-divergence"""
        prior_mean = torch.cat([torch.cat(prior.mean[i], dim=0) for i in range(len(prior.mean))], dim=0)
        prior_logvar = torch.cat([torch.cat(prior.logvar[i], dim=0) for i in range(len(prior.logvar))], dim=0)
        posterior_mean = torch.cat([torch.cat(posterior.mean[i], dim=0) for i in range(len(posterior.mean))], dim=0)
        posterior_logvar = torch.cat([torch.cat(posterior.logvar[i], dim=0) for i in range(len(posterior.logvar))], dim=0)
        
        KL = ((posterior_mean - prior_mean).pow(2) / prior_logvar.exp()).sum(dim=1)
        KL += (prior_logvar - posterior_logvar).sum(dim=1)
        KL += (posterior_logvar.exp() / prior_logvar.exp()).sum(dim=1)
        KL -= config["d_latent"]
        KL *= 0.5
        KL = KL.sum() / context_batch.size(0)
        logs["KL"] = logs.get("KL") + KL
        
        loss = recon + config["beta"] * KL
        logs["loss"] = logs.get("loss") + loss
        
        active = (posterior_logvar.exp().mean(dim=0) < 0.1).to(torch.float32).mean()
        logs["active"] = logs.get("active") + active
        
        loss.backward()
        optimizer.step()
        
    return logs
#%%