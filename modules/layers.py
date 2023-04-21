#%%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
#%%
def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / d_model)
    return pos * angle_rates
#%%
def positional_encoding(timesteps, d_model):
    angle_rads = get_angles(
        torch.arange(timesteps)[:, None],
        torch.arange(d_model)[None, :],
        d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[None, ...]
    return pos_encoding
#%%
class AddPosition(nn.Module):
    def __init__(self, d_model, timesteps, device):
        super(AddPosition, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.posit_matrix = positional_encoding(timesteps, d_model).to(device)
        
    def forward(self, x, t):
        return self.layer_norm(x + self.posit_matrix[:, t:t+1 :])
#%%
class AddPosition2(nn.Module):
    def __init__(self, d_model, timesteps, device):
        super(AddPosition2, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.posit_matrix = positional_encoding(timesteps, d_model).to(device)
        
    def forward(self, x):
        return self.layer_norm(x + self.posit_matrix)
#%%
def scaled_dot_product_attention(q, k, v, d_model, mask, device):
    matmul_qk = q.matmul(k.permute(0, 1, 3, 2)) # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    sqrt_d_model = torch.from_numpy(np.sqrt(d_model)[None, ]).to(torch.float32).to(device)
    scaled_attention_logits = matmul_qk / sqrt_d_model

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

    output = attention_weights.matmul(v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, config, device):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.num_heads = config["num_heads"]
        self.d_model = config["d_model"]
        self.device = device

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = nn.Linear(config["d_model"], config["d_model"])
        self.wk = nn.Linear(config["d_model"], config["d_model"])
        self.wv = nn.Linear(config["d_model"], config["d_model"])

        self.dense = nn.Linear(config["d_model"], config["d_model"])

    def split_heads(self, x, batch_size):
        x = x.view((batch_size, -1, self.config["num_heads"], self.depth)).contiguous()
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        depth = self.config["d_model"] // self.config["num_heads"]

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        def split_heads(x, batch_size):
            x = x.reshape((batch_size, -1, self.config["num_heads"], depth)).contiguous()
            return x.permute(0, 2, 1, 3)

        q = split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.config["d_model"], mask, self.device)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)
        
        concat_attention = scaled_attention.reshape(batch_size, -1, self.config["d_model"]).contiguous()  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
#%%
class PriorModule(nn.Module):
    def __init__(self, config, device):
        super(PriorModule, self).__init__()
        self.config = config
        self.device = device
        
        self.w0 = nn.Parameter(
            torch.randn((1, 1, config["d_model"])) * 0.1, # (batch_size, timestep, d_model)
            requires_grad=True)

        self.mha1 = MultiHeadAttention(config, device)
        self.mha2 = MultiHeadAttention(config, device)
        self.mha3 = MultiHeadAttention(config, device)

        self.fc1 = nn.Linear(config["d_model"], config["d_latent"])
        self.fc2 = nn.Linear(config["d_latent"], config["d_model"])
        self.fc3 = nn.Linear(config["d_latent"], config["d_model"])

        self.layer_norm1 = nn.LayerNorm(config["d_model"])
        self.layer_norm2 = nn.LayerNorm(config["d_model"])
        self.layer_norm3 = nn.LayerNorm(config["d_model"])

        self.add_posit = AddPosition(config["d_model"], config["future"], device)
        
    def forward(self, h_C, prior_W=None):
        w = self.w0.repeat(h_C.size(0), 1, 1)
        
        w_list = []
        z_list = []
        mean_list = []
        logvar_list = []
        
        for i in range(self.config["future"]):
            if prior_W == None:
                w_bar = self.layer_norm1(w[:, i:i+1, :] + self.mha1(w[:, i:i+1, :], w[:, :i+1, :], w[:, :i+1, :]))
            else:
                w_tilde = self.layer_norm3(w[:, i:i+1, :] + self.mha3(w[:, i:i+1, :], prior_W, prior_W))
                w_bar = self.layer_norm1(w_tilde + self.mha1(w_tilde, w[:, :i+1, :], w[:, :i+1, :]))
                
            w_hat = self.layer_norm2(w_bar + self.mha2(w_bar, h_C, h_C))

            mean = self.fc1(w_hat)
            var = self.config["prior_var"] * torch.ones(mean.shape).to(self.device)
            # mean, logvar = torch.split(self.fc1(w_hat), self.config["d_latent"], dim=2)
            epsilon = torch.randn(mean.shape).to(self.device)
            z = mean + var.sqrt() * epsilon
            
            w_hat = self.add_posit(w_hat + self.fc3(z), i)
            w = torch.cat([w, w_hat], dim=1)

            w_list.append(w_hat)
            z_list.append(z.squeeze(1)) 
            mean_list.append(mean.squeeze(1))
            logvar_list.append(var.log().squeeze(1))
        
        return w_list, z_list, mean_list, logvar_list
#%%
class PosteriorModule(nn.Module):
    def __init__(self, config, prior, device):
        super(PosteriorModule, self).__init__()
        self.config = config
        self.device = device
        
        self.prior = prior
        self.mha = MultiHeadAttention(config, device)
        self.fc1 = nn.Linear(config["d_model"] * 2, config["d_latent"] * 2)
        self.fc2 = nn.Linear(config["d_latent"], config["d_model"])
        
    def forward(self, h_C, h_T, prior_W=None):
        w = self.prior.w0.repeat(h_T.size(0), 1, 1)
        
        w_list = []
        z_list = []
        mean_list = []
        logvar_list = []
        
        k = self.mha(h_T, h_T, h_T)
        
        for i in range(self.config["future"]):
            if prior_W == None:
                w_bar = self.prior.layer_norm1(w[:, i:i+1, :] + self.prior.mha1(w[:, i:i+1, :], w[:, :i+1, :], w[:, :i+1, :]))
            else:
                w_tilde = self.prior.layer_norm3(w[:, i:i+1, :] + self.prior.mha3(w[:, i:i+1, :], prior_W, prior_W))
                w_bar = self.prior.layer_norm1(w_tilde + self.prior.mha1(w_tilde, w[:, :i+1, :], w[:, :i+1, :]))
                
            w_hat = self.prior.layer_norm2(w_bar + self.prior.mha2(w_bar, h_C, h_C))
            
            w_hat_k = torch.cat([w_hat, k[:, [i], :]], dim=2)
            # k = self.mha(h_T[:, i:i+1, :], h_T[:, :i+1, :], h_T[:, :i+1, :])
            # w_hat_k = torch.cat([w_hat, k], dim=2)

            mean, logvar = torch.split(self.fc1(w_hat_k), self.config["d_latent"], dim=2)
            epsilon = torch.randn(mean.shape).to(self.device)
            z = mean + torch.exp(logvar / 2) * epsilon
            
            w_hat = self.prior.add_posit(w_hat + self.fc2(z), i)
            w = torch.cat([w, w_hat], dim=1)

            w_list.append(w_hat)
            z_list.append(z.squeeze(1)) 
            mean_list.append(mean.squeeze(1))
            logvar_list.append(logvar.squeeze(1))
                           
        return w_list, z_list, mean_list, logvar_list
#%%