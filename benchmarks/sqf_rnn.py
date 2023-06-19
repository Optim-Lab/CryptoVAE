import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, n_layers=3, dr=0.1):
        super(Encoder, self).__init__()
        # self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.lstm = nn.LSTM(d_input, d_model, n_layers, dropout=dr, batch_first=True)
        
    def forward(self, x):
        
        _, (hidden, cell) = self.lstm(x)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, d_model, tau, n_layers=3, dr=0.1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, n_layers, dropout=dr, batch_first=True)
        self.tau = tau
        self.n_layers = n_layers
        
    def forward(self, hidden, cell):
        lstm_output = []
        
        for _ in range(self.tau):
            output, (hidden, cell) = self.lstm(hidden[self.n_layers-1:self.n_layers].transpose(1, 0), (hidden, cell))
        
            lstm_output.append(output)
        
        lstm_output = torch.cat(lstm_output, axis=1)
        
        return lstm_output 

class SQF_RNN(nn.Module):
    def __init__(self, d_input, d_model, tau, n_layers, M, device, dr=0.05):
        super(SQF_RNN, self).__init__()
        
        self.encoder = Encoder(d_input, d_model, n_layers, dr)
        self.decoder = Decoder(d_model, tau, n_layers, dr)
        self.M = M
        self.beta_map = nn.Linear(d_model, d_input * (M+1))
        self.softplus = nn.Softplus()
        self.delta = torch.linspace(0, 1, M + 1)[None, :]
        self.gamma_map = nn.Linear(d_model, d_input * 1)
        self.device = device 
        self.tau = tau 
        self.d_input = d_input 
        
    def forward(self, x):
        en_hidden, en_cell = self.encoder(x)
        dec_output = self.decoder(en_hidden, en_cell)
        beta = self.softplus(self.beta_map(dec_output))
        gamma = self.gamma_map(dec_output)    

        return gamma.view(-1, self.tau, self.d_input, 1), beta.view(-1, self.tau, self.d_input, self.M+1)
    
    def quantile_inverse(self, y, gamma, beta):
        delta_ = self.delta.unsqueeze(2) - self.delta.unsqueeze(1) 
        q_mask = torch.where(delta_ >= 0., delta_, torch.zeros(()).to(self.device))
        q_delta = gamma + (beta.unsqueeze(-2) * q_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        mask = torch.where(y.unsqueeze(-1) >= q_delta, torch.ones(()).to(self.device), torch.zeros(()).to(self.device))

        alpha_tilde = (y.unsqueeze(-1) - gamma) # alpha_tilde.shape (batch_size, tau, d_input, 1)
        alpha_tilde += (mask * beta * self.delta).sum(dim=-1, keepdims=True) # alpha_tilde.shape (batch_size, tau, d_input, 1)
        alpha_tilde /= (mask * beta).sum(dim=-1, keepdims=True) + 1e-6
        alpha_tilde = torch.clip(alpha_tilde, 0.00001, 1)
        
        return alpha_tilde
    
    def quantile_function(self, alpha, gamma, beta):
        return gamma + (beta * torch.where(
            alpha.unsqueeze(-1) - self.delta > 0,
            alpha.unsqueeze(-1) - self.delta,
            torch.zeros(()).to(self.device)
            )).sum(dim=-1, keepdims=True)