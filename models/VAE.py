import numpy as np
import polars as pl
import os
import datetime
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utility import load_utility as lu
from ml import models_library, training 

filename = f"VAE"

with open("exploration/output/ranked_features_multilabel.dat", "r") as f:
    features_to_use = [line.strip() for line in f]


config = {
    "NEPOCHS" : 200,
    "FILL_NAN_VALUES" : True,
    "LEARNING_RATE" : 0.0001,
    "BATCH_SIZE" : 1000,
    "SPLIT_SEED" : 26052013,
    "input_dim" : 40,
    "hidden_dims" : [30, 15, 7],
    "latent_dim" : 5,
    "features_to_use" : features_to_use,
}


base_dir = os.path.dirname(os.path.abspath(__file__))
save_directory = os.path.join(base_dir, "saved_models")
os.makedirs(save_directory, exist_ok=True)
today = datetime.datetime.now().strftime("%Y-%m-%d")
filename = filename + "_" + today
with open(os.path.join(save_directory, filename + ".json"), "w") as f:
     json.dump(config, f, indent=4)


df, features = lu.load_data(file_path="/scratch/extra/ELSA/ivano.saccheo2/DR1/EDFN_teresa_parquet")
df = df.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(float("nan")) for c in features_to_use])

if config["FILL_NAN_VALUES"]:
    df = df.fill_nan(None).with_columns([pl.col(c).fill_null(pl.col(c).mean()) for c in features_to_use])
else:
    df = df.fill_nan(None).drop_nulls(subset = features_to_use)

data = df.select(features_to_use)
if not config["input_dim"] == data.shape[1]:
    raise ValueError("Model and data have conflicting dimensions")

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
train_data, validation_data = train_test_split(data, test_size = 0.3, random_state=config["SPLIT_SEED"])

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = TensorDataset(torch.from_numpy(train_data.astype("float32")))
validation_dataset = TensorDataset(torch.from_numpy(validation_data.astype("float32")))
train_dataloader = DataLoader(train_dataset, batch_size= config["BATCH_SIZE"], shuffle = True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size = config["BATCH_SIZE"], pin_memory = True)

model = models_library.VAE(input_dim=config["input_dim"], latent_dim=config["latent_dim"], 
                           hidden_dims = config["hidden_dims"]).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr = config["LEARNING_RATE"])

history = []

for epoch in range(config["NEPOCHS"]):
    beta = training.get_beta(epoch, config["NEPOCHS"], splits = (0.33,0.66))
    train_losses = training.train_routine(train_dataloader, 
                                          model = model,
                                          loss_fn = training.VAE_loss_function,
                                          optimizer = optimizer,
                                          beta = beta,
                                          device = device,
                                          verbose = False)
    val_losses = training.validation_routine(validation_dataloader, 
                                                                model = model,
                                                                loss_fn = training.VAE_loss_function,
                                                                beta = beta,                                                          
                                                                device = device,
                                                                verbose = (epoch%10==0))
    history.append(training.update_history(epoch, train_losses, val_losses, beta = beta))


torch.save(model.state_dict(), os.path.join(save_directory, filename + ".pt"))

history = pl.DataFrame(history)
history.write_csv(os.path.join(save_directory, filename + ".csv"))
