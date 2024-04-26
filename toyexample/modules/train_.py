#%%

#%%
def train(config): 
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