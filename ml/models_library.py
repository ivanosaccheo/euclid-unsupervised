import torch.nn as nn
import torch


class BaseDense(nn.Module):
    
    def __init__(self, input_dim, output_dim, 
        hidden_dims, activation_func = nn.LeakyReLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation_func())
        for i,j in zip(hidden_dims[:-1],hidden_dims[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(activation_func())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x_hat = self.model(x)
        return x_hat

class VEncoder(BaseDense):

    def __init__(self, input_dim, latent_dim, 
              hidden_dims, activation_func = nn.LeakyReLU):
        super().__init__(input_dim = input_dim, 
                        output_dim = 2*latent_dim, 
                        hidden_dims = hidden_dims,
                        activation_func=activation_func)

class VDecoder(BaseDense):

    def __init__(self, latent_dim, output_dim, 
                 hidden_dims, activation_func = nn.LeakyReLU):
                super().__init__(input_dim = latent_dim, 
                        output_dim = output_dim, 
                        hidden_dims = hidden_dims,
                        activation_func=activation_func)
                

class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim, 
                 hidden_dims, activation_func = nn.LeakyReLU):
      super().__init__()
      self.encoder = VEncoder(input_dim, latent_dim, hidden_dims,
                              activation_func = activation_func)
      self.decoder  = VDecoder(latent_dim, input_dim, hidden_dims,
                              activation_func = activation_func) 
      self.latent_dim = latent_dim

    def reparametrization_trick(self, mu, log_var):
       log_var = torch.clamp(log_var, min=-10, max=10)
       std = torch.exp(0.5*log_var)
       epsilon = torch.randn_like(std)
       return mu + epsilon*std

    def forward(self, x):
       y = self.encoder(x)
       
       mu, log_var = torch.chunk(y, chunks =2, dim = -1)
       z = self.reparametrization_trick(mu, log_var)
       x_hat = self.decoder(z)
       return x_hat, mu, log_var
    