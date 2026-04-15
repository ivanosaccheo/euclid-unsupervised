import torch
import torch.nn as nn


def train_routine(dataloader, model, loss_fn, optimizer, beta = 1, verbose = True):
    
    model.train()
    tot_loss_epoch = 0.0
    mse_loss_epoch = 0.0
    kl_loss_epoch = 0.0
    
    num_batches = len(dataloader)
    for (data,) in dataloader:
        x_hat, mu, log_var = model(data)
        tot_loss, mse, kl = loss_fn(x_hat, data, mu, log_var, beta = beta)
        
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        
        tot_loss_epoch += tot_loss.item()
        mse_loss_epoch += mse.item()
        kl_loss_epoch += kl.item()

    tot_loss_epoch /= num_batches
    mse_loss_epoch /= num_batches
    kl_loss_epoch /= num_batches

    if verbose:
        print(f"Train: tot_loss = {tot_loss_epoch:>7f} -- MSE = {mse_loss_epoch:>7f} -- KL = {kl_loss_epoch:>7f}")

    return tot_loss_epoch, mse_loss_epoch, kl_loss_epoch 



def validation_routine(dataloader, model, loss_fn, beta=1, verbose=True):
    model.eval()

    tot_loss_epoch = 0.0
    mse_loss_epoch = 0.0
    kl_loss_epoch = 0.0

    num_batches = len(dataloader)

    with torch.no_grad():
        for (data,) in dataloader:
            x_hat, mu, log_var = model(data)
            tot_loss, mse, kl = loss_fn(x_hat, data, mu, log_var, beta=beta)

            tot_loss_epoch += tot_loss.item()
            mse_loss_epoch += mse.item()
            kl_loss_epoch += kl.item()

    tot_loss_epoch /= num_batches
    mse_loss_epoch /= num_batches
    kl_loss_epoch /= num_batches

    if verbose:
        print(f"Validation: tot_loss = {tot_loss_epoch:>7f} -- MSE = {mse_loss_epoch:>7f} -- KL = {kl_loss_epoch:>7f}")

    return tot_loss_epoch, mse_loss_epoch, kl_loss_epoch



def get_kl_loss(mu, log_var, beta = 1):
    kl = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
    kl = torch.sum(kl, dim =-1) 
    kl = torch.mean(kl)
    return beta * kl

def VAE_loss_function(x_hat, x, mu, log_var, beta = 1):
    mse = nn.functional.mse_loss(x_hat, x, reduction ="mean")
    kl = get_kl_loss(mu, log_var, beta = beta)
    tot_loss = mse + kl
    return tot_loss, mse, kl 
