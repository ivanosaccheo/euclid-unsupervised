import torch.nn as nn
import torch
from torch.distributions.uniform import Uniform


class BaseDense(nn.Module):
    
    def __init__(self, 
                input_dim, 
                output_dim, 
                hidden_dims, 
                activation_func = nn.LeakyReLU,
                dropout = 0.0,
                batch_norm = False
                ):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation_func())
            if dropout > 0:
               layers.append(nn.Dropout(p = dropout))
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
    

class SCARF(nn.Module):
    """
    Adapted from https://github.com/clabrugere/pytorch-scarf with 
    addition of fine-tuning head
    """
    
    def __init__(self, 
                 input_dim : int, 
                 features_low : float,
                 features_high : float,
                 encoder_output_dim : int,
                 pre_train_head_output_dim : int,
                 class_head_output_dim : int,
                 encoder_hidden_dims : list,
                 pre_train_head_hidden_dims : list,
                 class_head_hidden_dims : list,
                 corruption_rate : float = 0.6,
                 dropout : float = 0.0,
                 batch_norm = True,
                 activation_func = nn.LeakyReLU
                ):
        
        super().__init__()
        self.encoder = BaseDense(input_dim, 
                                 encoder_output_dim,
                                 encoder_hidden_dims,
                                 activation_func=activation_func,
                                 dropout=dropout,
                                 batch_norm=batch_norm
                                 )
        
        self.pretraining_head = BaseDense(encoder_output_dim, 
                                         pre_train_head_output_dim,
                                         pre_train_head_hidden_dims, 
                                         activation_func=activation_func,
                                         dropout=dropout,
                                         batch_norm=batch_norm
                                         )

        self.classification_head = BaseDense(encoder_output_dim,
                                             class_head_output_dim,
                                             class_head_hidden_dims,
                                             activation_func=activation_func,
                                             dropout=dropout,
                                             batch_norm=batch_norm
                                             )
        self.marginals = Uniform(torch.as_tensor(features_low), torch.as_tensor(features_high))
        self.corruption_rate = corruption_rate
    

    def corrupt(self, x):
        batch_size, _ = x.size() 
        corruption_mask = (torch.rand_like(x, device = x.device) < self.corruption_rate)
        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)       
        return x_corrupted

    def pretraining_forward(self, x):
        x_corrupted = self.corrupt(x)
        encoded = self.encoder(x)
        encoded_corrupted = self.encoder(x_corrupted)
        embeddings = self.pretraining_head(encoded)
        embeddings_corrupted = self.pretraining_head(encoded_corrupted)
        return embeddings, embeddings_corrupted
    
    def classification_forward(self, x):
        encoded = self.encoder(x)
        x_hat = self.classification_head(encoded)
        return x_hat

    def forward(self, x):
        return self.classification_forward(x)

    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
    



        




                 
                 
                 
                 
                 