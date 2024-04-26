#%%
import torch
#%%
def GLD_train(model, optimizer, data, config, device): 
    for epoch in range(config["epochs"]+1): # full-batch training
        tau = torch.linspace(0.01, 0.99, 20).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        
        z, mean, logvar, theta1, theta2, theta3, theta4 = model(data)
        
        """reconstruction"""
        Q = model.GLD_quantile_function(tau, theta1, theta2, theta3, theta4)
        residual = data - Q
        recon = (residual * (tau - (residual < 0).to(torch.float32))).mean()
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(dim=1)
        KL -= logvar.sum(dim=1)
        KL += torch.exp(logvar).sum(dim=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        
        loss = recon + config["beta"] * KL
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print_input = "[epoch {:03d}]".format(epoch)
            print_input += f', Loss: {loss:.4f}, recon: {recon:.4f}, KL: {KL:.4f}'
            print(print_input)
    return None
#%%
def Gaussian_train(model, optimizer, data, config, device): 
    for epoch in range(config["epochs"]+1): # full-batch training
        
        optimizer.zero_grad()
        
        z, mean, logvar, xhat = model(data)
        
        """reconstruction"""
        residual = data - xhat
        recon = 0.5 * (residual.pow(2)).mean()
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(dim=1)
        KL -= logvar.sum(dim=1)
        KL += torch.exp(logvar).sum(dim=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        
        loss = recon + config["beta"] * KL
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print_input = "[epoch {:03d}]".format(epoch)
            print_input += f', Loss: {loss:.4f}, recon: {recon:.4f}, KL: {KL:.4f}'
            print(print_input)
    return None
#%%
def LSQF_train(model, optimizer, data, config, device): 
    for epoch in range(config["epochs"]+1): # full-batch training
        
        optimizer.zero_grad()
        
        z, mean, logvar, gamma, beta = model(data)
        
        """reconstruction"""
        alpha_tilde = model.quantile_inverse(data, gamma, beta)
        term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde, model.delta).pow(2)
        term += 2 * torch.maximum(alpha_tilde, model.delta) * model.delta
        
        recon = (2 * alpha_tilde) * data
        recon += (1 - 2 * alpha_tilde) * gamma
        recon += (beta * term).sum(dim=1, keepdims=True)
        recon *= 0.5
        recon = recon.mean()
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(dim=1)
        KL -= logvar.sum(dim=1)
        KL += torch.exp(logvar).sum(dim=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        
        loss = recon + config["beta"] * KL
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print_input = "[epoch {:03d}]".format(epoch)
            print_input += f', Loss: {loss:.4f}, recon: {recon:.4f}, KL: {KL:.4f}'
            print(print_input)
    return None
#%%