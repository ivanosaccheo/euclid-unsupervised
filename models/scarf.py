import numpy as np
import polars as pl
import os
import datetime
from torch.utils.data import TensorDataset, DataLoader
import torch
import json


from scarf.loss import NTXent
from scarf.model import SCARF
from scarf.dataset import SCARFDataset



from utility import load_utility as lu
from ml import preprocessing

filename = f"SCARF"

with open("exploration/output/ranked_features_multilabel.dat", "r") as f:
    features_to_use = [line.strip() for line in f]


config = {
    "SNR_min" : 3,
    "FILL_NAN_VALUES" : True,
    "features_to_use" : features_to_use,
    "LEARNING_RATE" : 0.0001,
    "SPLIT_SEED" : 26052013,
    "FORCE_LABEL" : False, 
    "TEST_SIZE" : 0.3,
    "NEPOCHS" : 200,
    "input_dim" : 40,
    "hidden_dims" : [30, 15, 7],
    "latent_dim" : 5,
}


df, features =  lu.load_data(extra_columns = ["TARGETID_desi", "SPECTYPE_desi"])
df = preprocessing.clean_data(df, config = config, verbose = True)

train_data, validation_data = preprocessing.split_train_test(df, 
                                                             test_size=config["TEST_SIZE"],
                                                             split_seed=config["SPLIT_SEED"],
                                                             force_label=config["FORCE_LABEL"])

train_target =                                            
train_data  = train_data.select(config["features_to_use"])
validation_data  = validation_data.select(config["features_to_use"])
train_data, validation_data = preprocessing.scale_data(train_data, validation_data)



device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = SCARFDataset(train_data)
validation_dataset = SCARFDataset(validation_data)
train_dataloader = DataLoader(train_dataset, batch_size= config["BATCH_SIZE"], shuffle = True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size = config["BATCH_SIZE"], pin_memory = True)




model = SCARF(
              input_dim = config["input_dim"],
              features_low = train_dataset.features_low,
              features_high = train_dataset.features_high,
              dim_hidden_encoder = config["dim_hidden_encoder"],
              num_hidden_encoder = config["num_hidden_encoder"],
              dim_hidden_head = config["dim_hidden_head"],
              num_hidden_head =  config["dim_hidden_head"],
              corruption_rate = 0.6,
              dropout = 0.1,)

optimizer = torch.optim.Adam(model.parameters(), lr = config["LEARNING_RATE"], weight_decay=1e-5)
ntxent_loss = NTXent()


