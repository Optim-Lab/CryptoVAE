import torch
import numpy as np 
import torch.nn as nn

from scipy.stats import norm

def stock_data_generator2(df, C, tau, input_min=None, input_max=None):
    n = df.shape[0] - C - tau
    
    if input_min is None and input_max is None:
        input_min = df.min(axis=0)
        input_max = df.max(axis=0)
    
    # C = k
    # T = k+tau
    input_data = np.zeros((n, C, df.shape[1]))
    infer_data = np.zeros((n, tau, df.shape[1]))

    for i in range(n):
        input_data[i, :, :] = (df.iloc[i : i+C, :] - input_min)/input_max
        infer_data[i, :, :] = df.iloc[i+C:i+C+tau, :]
    
    input_data = torch.from_numpy(input_data).to(torch.float32)
    infer_data = torch.from_numpy(infer_data).to(torch.float32)
    return input_data, infer_data, input_min, input_max

def build_datasets(df, tau, test_len, increment):
    train_list = []
    test_list = []
    
    for j in range(increment):
        train_idx_last = len(df) - test_len * (increment - j)
        test_idx_last = len(df) - test_len * (increment - 1 - j)
        train = df.iloc[: train_idx_last]
        test = df.iloc[-(test_len * (increment - j) + 20 + tau) : test_idx_last]
        
        """data save"""
        # train.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{1}_phase{j+1}_train.csv')
        # test.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{1}_phase{j+1}_test.csv')
        train_context, train_target, input_min, input_max = stock_data_generator2(train, C=20, tau=tau)
        test_context, test_target, _, _ = stock_data_generator2(test, C=20, tau=tau, input_min=input_min, input_max=input_max)
        
        assert train_context.shape == (train.shape[0] - 20 - tau, 20, df.shape[1])
        assert train_target.shape == (train.shape[0] - 20 - tau, tau, df.shape[1])
        assert test_context.shape == (test.shape[0] - 20 - tau, 20, df.shape[1])
        assert test_target.shape == (test.shape[0] - 20 - tau, tau, df.shape[1])
        
        train_list.append((train_context, train_target))
        test_list.append((test_context, test_target))
        
    return train_list, test_list

def stock_data_generator_for_gpcopula(df, C, tau, how=None, loc=None, scale=None):
    n = df.shape[0] - C - tau
    
    if how is not None:
        if how == "min-max":
            loc = df.min(axis=0)
            scale = df.max(axis=0)
        elif how == "standard": 
            loc = df.mean(axis=0)
            scale = df.std(axis=0)
        else:
            raise Exception("Choose correct normalization method!") 
        
    else: 
        if loc is None and scale is None:
            raise Exception("Give location and scale parameters for nomarlization!") 
    # C = k
    # T = k+tau
    input_data = np.zeros((n, C, df.shape[1]))
    infer_data = np.zeros((n, tau, df.shape[1]))

    for i in range(n):
        input_data[i, :, :] = (df.iloc[i : i+C, :] - loc)/scale
        infer_data[i, :, :] = (df.iloc[i+C:i+C+tau, :] - loc)/scale
    
    input_data = torch.from_numpy(input_data).to(torch.float32)
    infer_data = torch.from_numpy(infer_data).to(torch.float32)
    
    return input_data, infer_data, loc, scale

def build_datasets_gpcopula(df, tau, test_len, increment, how):
    train_list = []
    test_list = []
    train_norm_param_lsit = []
    
    for j in range(increment):
        train_idx_last = len(df) - test_len * (increment - j)
        test_idx_last = len(df) - test_len * (increment - 1 - j)
        train = df.iloc[: train_idx_last]
        test = df.iloc[-(test_len * (increment - j) + 20 + tau) : test_idx_last]
        
        """data save"""
        # train.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{1}_phase{j+1}_train.csv')
        # test.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{1}_phase{j+1}_test.csv')
        train_context, train_target, loc, scale = stock_data_generator_for_gpcopula(train, C=20, tau=tau, how=how)
        test_context, test_target, _, _ = stock_data_generator_for_gpcopula(test, C=20, tau=tau, loc=loc, scale=scale)
        
        assert train_context.shape == (train.shape[0] - 20 - tau, 20, df.shape[1])
        assert train_target.shape == (train.shape[0] - 20 - tau, tau, df.shape[1])
        assert test_context.shape == (test.shape[0] - 20 - tau, 20, df.shape[1])
        assert test_target.shape == (test.shape[0] - 20 - tau, tau, df.shape[1])
        
        train_list.append((train_context, train_target))
        test_list.append((test_context, test_target))
        train_norm_param_lsit.append((loc, scale))
        
    return train_list, test_list, train_norm_param_lsit

def stock_data_generator(df, C, tau):
    n = df.shape[0] - C - tau

    # C = k
    # T = k+tau
    input_data = np.zeros((n, C, df.shape[1]))
    infer_data = np.zeros((n, tau, df.shape[1]))

    for i in range(n):
        input_data[i, :, :] = df.iloc[i : i+C, :]
        infer_data[i, :, :] = df.iloc[i+C:i+C+tau, :]
    
    input_data = torch.from_numpy(input_data).to(torch.float32)
    infer_data = torch.from_numpy(infer_data).to(torch.float32)
    return input_data, infer_data

def build_datasets2(df, tau, test_len, increment):
    train_list = []
    test_list = []
    for j in range(increment):
        train_idx_last = len(df) - test_len * (increment - j)
        test_idx_last = len(df) - test_len * (increment - 1 - j)
        train = df.iloc[: train_idx_last]
        test = df.iloc[-(test_len * (increment - j) + 20 + tau) : test_idx_last]
        
        """data save"""
        # train.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{1}_phase{j+1}_train.csv')
        # test.to_csv(f'./assets/{config["model"]}/{config["data"]}_future{1}_phase{j+1}_test.csv')
        train_context, train_target = stock_data_generator(train, C=20, tau=tau)
        test_context, test_target = stock_data_generator(test, C=20, tau=tau)
        
        assert train_context.shape == (train.shape[0] - 20 - tau, 20, df.shape[1])
        assert train_target.shape == (train.shape[0] - 20 - tau, tau, df.shape[1])
        assert test_context.shape == (test.shape[0] - 20 - tau, 20, df.shape[1])
        assert test_target.shape == (test.shape[0] - 20 - tau, tau, df.shape[1])
        
        train_list.append((train_context, train_target))
        test_list.append((test_context, test_target))
    
    return train_list, test_list

def gaussian_quantile(mu, sigma):

    batch_size, _, _ = mu.shape

    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()

    total_output = []
    
    for q in [0.1, 0.5, 0.9]:
        tmp_output = []

        for i in range(batch_size):
            tmp_output.append(norm.ppf(q, loc=mu[i], scale=sigma[i])[np.newaxis, ...])
            
        total_output.append(np.concatenate(tmp_output, axis=0)[..., np.newaxis])
    
    return np.concatenate(total_output, axis=-1)

def scaled_dot_product_attention(q, k, v, d_model, mask, device):
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))
    dk = torch.tensor(d_model).to(device)

    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + ((1 - mask) * -1e9)

    attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)
    
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights



