import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class NegativeGaussianLogLikelihood(nn.Module):
    def __init__(self, device):
        super(NegativeGaussianLogLikelihood, self).__init__()
        import math
        self.pi = torch.tensor(math.pi).float().to(device)
        
    def forward(self, true, pred):
        mu, sigma = pred
        return (torch.square(true - mu)/(2*sigma) + torch.log(2*self.pi*sigma)/2).mean()
    

class QuantileRisk(nn.Module):
    def __init__(self, tau, quantile, num_targets, device):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile
        self.device = device
        self.q_arr = torch.tensor(self.quantile).float().unsqueeze(0).unsqueeze(-1).repeat(1, 1, tau).unsqueeze(1).repeat(1, num_targets, 1, 1).to(self.device)
    
    def forward(self, true, pred):
        true_rep = true.unsqueeze(-1).repeat(1, 1, 1, len(self.quantile)).permute(0, 2, 3, 1).to(self.device)
        pred = pred.permute(0, 2, 3, 1)

        ql = torch.maximum(self.q_arr * (true_rep - pred), (1-self.q_arr)*(pred - true_rep))
        
        return ql.mean()

class GPNegL(nn.Module):
    def __init__(self, ecdf_list, device):
        super(GPNegL, self).__init__()
        self.ecdf_list = ecdf_list 
        self.device = device
        self.norm_dist = Normal(0, 1)
        
    def forward(self, true, params):
        d_input = true.shape[-1]
        
        mu, sigma = params
        L, _ = torch.linalg.cholesky_ex(sigma)
        L_inverse = torch.inverse(L)
        det_L = torch.det(L)
        
        emp_quantile_ = []
        for i in range(d_input):
            emp_quantile_.append(torch.tensor(self.ecdf_list[i](true[..., i:i+1])).float())
        emp_quantile = torch.cat(emp_quantile_, dim=-1).to(self.device)
        gc_output = self.norm_dist.icdf(torch.clip(emp_quantile, min=0.001, max=0.999))
        
        return torch.log(det_L).mean() + (0.5 *  torch.square(L_inverse @ (gc_output.unsqueeze(-1) - mu))).sum(dim=1).mean()    

class LinearSplineCRPS(nn.Module):
    def __init__(self, d_input, device):
        super(LinearSplineCRPS, self).__init__()
        self.device = device
        self.d_input = d_input 
    
    def forward(self, true, gamma, beta, delta, alpha_tilde):
        crps = []
        for i in range(self.d_input):
            tmp_crps = (2 * alpha_tilde[:, :, i, :] - 1) * true[..., i:i+1] + (1 - 2 * alpha_tilde[:, :, i, :]) * gamma[:, :, i]
            tmp_crps += (beta[:, :, i] * ((1 - delta**3)/ 3 - delta - torch.max(alpha_tilde[:, :, i, :], delta.unsqueeze(-2))**2)).sum(-1, keepdims=True)
            tmp_crps += (beta[:, :, i] * (2 * torch.max(alpha_tilde[:, :, i, :], delta.unsqueeze(-2)) * delta.unsqueeze(-2))).sum(-1, keepdims=True)
            crps.append(tmp_crps.mean())

        return  sum(crps)/len(crps)