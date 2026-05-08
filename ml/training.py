import torch
import torch.nn as nn


def VAE_train_routine(dataloader, model, loss_fn, optimizer, beta = 1, 
                  device = "cpu",
                  verbose = True):
    
    model.train()
    tot_loss_epoch = 0.0
    mse_loss_epoch = 0.0
    kl_loss_epoch = 0.0
    
    num_batches = len(dataloader)
    for (data,) in dataloader:
        data = data.to(device)
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
        print(f"Train: tot_loss = {tot_loss_epoch:>7f} -- MSE = {mse_loss_epoch:>7f} -- KL = {kl_loss_epoch:>7f}", flush = True)

    return tot_loss_epoch, mse_loss_epoch, kl_loss_epoch 



def VAE_validation_routine(dataloader, model, loss_fn, beta=1, 
                       device = "cpu",
                       verbose=True):
    model.eval()

    tot_loss_epoch = 0.0
    mse_loss_epoch = 0.0
    kl_loss_epoch = 0.0

    num_batches = len(dataloader)

    with torch.no_grad():
        for (data,) in dataloader:
            data = data.to(device)
            x_hat, mu, log_var = model(data)
            tot_loss, mse, kl = loss_fn(x_hat, data, mu, log_var, beta=beta)

            tot_loss_epoch += tot_loss.item()
            mse_loss_epoch += mse.item()
            kl_loss_epoch += kl.item()

    tot_loss_epoch /= num_batches
    mse_loss_epoch /= num_batches
    kl_loss_epoch /= num_batches

    if verbose:
        print(f"Validation: tot_loss = {tot_loss_epoch:>7f} -- MSE = {mse_loss_epoch:>7f} -- KL = {kl_loss_epoch:>7f}", flush = True)

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


def update_history_VAE(epoch, train_losses, val_losses, **kwargs):
    train_total, train_mse, train_kl = train_losses
    val_total, val_mse, val_kl = val_losses
    history ={}
    history["epoch"] = epoch
    history["train_total"] = train_total
    history["train_mse"] = train_mse
    history["train_kl"] = train_kl
    history["validation_total"] = val_total
    history["validation_mse"] = val_mse
    history["validation_kl"] =  val_kl
    history.update(**kwargs)
    return history


def get_beta(epoch, Nepochs, splits = [0.33, 0.55], floor_val =0.0, ceil_val =1.0):
    if splits[0] < 1:
       e0 = splits[0]*Nepochs
       e1 = splits[1]*Nepochs
    else:
        e0, e1 = splits[0], splits[1]
    if epoch<=e0:
        return floor_val
    elif epoch> e1:
        return ceil_val
    else:
        dy = ceil_val-floor_val
        dx = e1-e0
        return dy/dx*(epoch-e0)
    


def SCARF_epoch_pretraining(dataloader,
                            model,
                            loss_fn,
                            optimizer = None,
                            device = "cpu",
                            verbose = True,
                            ):
    training = optimizer is not None
    epoch_loss = 0.0
    num_batches = len(dataloader)

    if training:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()
    with context:
        for data, _ in dataloader:
            data = data.to(device)
            emb_anchor, emb_positive = model.pretraining_forward(data)
            loss = loss_fn(emb_anchor, emb_positive)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
    
    
    epoch_loss = epoch_loss/num_batches
    if verbose:
        mode = "Train" if training else "Validation"
        print(f"{mode}: tot_loss = {epoch_loss:>7f}", flush = True)
    
    return epoch_loss
    

def SCARF_epoch_classifier(dataloader,
                            model,
                            loss_fn = nn.CrossEntropyLoss(),
                            optimizer = None,
                            device = "cpu",
                            verbose = True,
                            ):
    training = optimizer is not None
    epoch_loss = 0.0
    num_batches = len(dataloader)

    if training:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()
    with context:
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            x_hat = model.classification_forward(data)
            loss = loss_fn(x_hat, label)
            
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()

    epoch_loss = epoch_loss/num_batches
    
    if verbose:
        mode = "Train" if training else "Validation"
        print(f"{mode}: tot_loss = {epoch_loss:>7f}", flush = True)
    
    return epoch_loss


def update_history(epoch, train_loss, val_loss, **kwargs):
    
    history = {}
    history["epoch"] = epoch
    history["train_loss"] = train_loss
    history["validation_loss"] = val_loss
    history.update(**kwargs)
   
    return history



class EarlyStopper:

    def __init__(self, min_delta = 0.001, patience = 10):
        
        self.min_val_loss = float("inf")
        self.epochs_without_improving = 0
        self.patience = patience
        self.min_delta = min_delta

    def check_early_stopping(self, val_loss):

        if self.min_val_loss == float("inf"):
            self.min_val_loss = val_loss
            return False

        relative_improvement = (self.min_val_loss - val_loss) / self.min_val_loss
        improved = relative_improvement > self.min_delta

        if improved:
            self.min_val_loss = val_loss
            self.epochs_without_improving = 0
            return False

        else:
            self.epochs_without_improving += 1
            if self.epochs_without_improving >= self.patience:
                return True

            return False




            
    