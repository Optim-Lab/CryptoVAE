#%%
import numpy as np
import tqdm
import torch
# import wandb
#%%
def train_function(context, target, model, iterations, config, optimizer, device):
    
    logs = {
        'loss': 0,
        'recon': 0,
        'KL': 0,
    }
    for i in range(iterations):
        idx = np.random.choice(
            range(len(context)), config["batch_size"], replace=False)
        context_batch = context[idx, :].to(device)
        target_batch = target[idx, :].to(device)
        
        optimizer.zero_grad()
        
        xhat, future_mu, future_z = model(context_batch)
        
        """Reconstruction"""
        recon = (target_batch - xhat).abs().sum(dim=-1).mean()
        logs["recon"] = logs.get("recon") + recon
        
        """KL-divergence"""
        KL = (future_mu - future_z).pow(2).sum(dim=-1).mean() * 0.5
        logs["KL"] = logs.get("KL") + KL
        
        loss = recon + config["beta"] * KL
        logs["loss"] = logs.get("loss") + loss
        
        loss.backward()
        optimizer.step()
        
    return logs
#%%
# gamma = torch.cat([torch.cat(params[i][0], dim=0) for i in range(len(params))], dim=0)
# beta = torch.cat([torch.cat(params[i][1], dim=0) for i in range(len(params))], dim=0)
# delta = torch.cat([torch.cat(params[i][2], dim=0) for i in range(len(params))], dim=0)
#%%