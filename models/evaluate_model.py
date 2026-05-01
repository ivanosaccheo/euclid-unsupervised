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
from ml import models_library, training, preprocessing, plot_library

model_filename = "VAE_2026-04-29"


base_dir = os.path.dirname(os.path.abspath(__file__))
save_directory = os.path.join(base_dir, "saved_models")

with open(os.path.join(save_directory, model_filename + ".json"), "r") as f:
     config = json.load(f)


df, features = lu.load_data(file_path="/scratch/extra/ELSA/ivano.saccheo2/DR1/EDFN_teresa_parquet",
                            extra_columns = ["TARGETID_desi", "SPECTYPE_desi"])

features_to_use = config["features_to_use"]
df = preprocessing.replace_infs(df, features_to_use)
if config["FILL_NAN_VALUES"]:
    df = preprocessing.fill_nans(df, features_to_use)
else:
    df = preprocessing.drop_nans(df, features_to_use)

train_df, validation_df= preprocessing.split_train_test(df, 
                                                        test_size = 0.3,
                                                        force_label=False,
                                                        split_seed=config["SPLIT_SEED"])
train_data  = train_df.select(features_to_use)
validation_data  = validation_df.select(features_to_use)

validation_label = validation_df.select(pl.col("SPECTYPE_desi")).with_columns(lu._get_label_column_expr("multilabel"))
validation_label = validation_label.select("label").to_numpy().ravel()

train_data, validation_data = preprocessing.scale_data(train_data, validation_data)

device = "cuda" if torch.cuda.is_available() else "cpu"
validation_dataset = TensorDataset(torch.from_numpy(validation_data.astype("float32")))
validation_dataloader = DataLoader(validation_dataset, batch_size = config["BATCH_SIZE"], pin_memory = True)
model = models_library.VAE(input_dim=config["input_dim"], latent_dim=config["latent_dim"], 
                           hidden_dims = config["hidden_dims"]).to(device) 

state_dict = torch.load(os.path.join(save_directory,model_filename + ".pt"), map_location=device)
model.load_state_dict(state_dict)

mu_list = []
log_var_list = []
model.eval()
with torch.no_grad():
    for data, in validation_dataloader:
        data = data.to(device)
        y = model.encoder(data)
        mu, log_var = torch.chunk(y, chunks =2, dim =-1)
        mu_list.append(mu.to("cpu"))
        log_var_list.append(log_var.to("cpu"))

mu = torch.cat(mu_list, dim =0).numpy()
log_var = torch.cat(log_var_list, dim=0).numpy()


save_directory_plot = os.path.join(base_dir, "plot")
os.makedirs(save_directory_plot, exist_ok=True)
plot_name = f"latent_{model_filename}.png"
fig = plot_library.plot_latent_space(mu, labels = validation_label)
fig.savefig(os.path.join(save_directory_plot, plot_name))
