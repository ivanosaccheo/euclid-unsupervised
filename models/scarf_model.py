import numpy as np
import polars as pl
import os
import datetime
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
import matplotlib.pyplot as plt


from scarf.loss import NTXent




from utility import load_utility as lu
from utility import save_utility as su
from ml import preprocessing, training, models_library, plot_library


debug = False
filename = "SCARF"

with open("exploration/output/ranked_features_multilabel.dat", "r") as f:
    features_to_use = [line.strip() for line in f]


config = {
    "SNR_min" : 50,
    "FILL_NAN_VALUES" : True,
    "features_to_use" : features_to_use,
    "LEARNING_RATE" : 0.0001,
    "SPLIT_SEED" : 26052013,
    "FORCE_LABEL" : False, 
    "TEST_SIZE" : 0.3,
    "NEPOCHS" : 1000,
    "BATCH_SIZE" : 200,
    "input_dim" : 40,
    "encoder_output_dim" : 30,
    "pre_train_head_output_dim" : 30,
    "class_head_output_dim" : 3,
    "encoder_hidden_dims" : [30, 30, 30],
    "pre_train_head_hidden_dims" : [30],
    "class_head_hidden_dims" : [30, 10],
    "corruption_rate" :  0.6,
    "batch_norm" : True,
    "dropout" : 0.04,
    }




save_directory = su.get_save_directory(__file__, "saved_models")
save_directory_plot = su.get_save_directory(__file__, "plot")
filename = su.get_date_filename(filename)


if debug:
    config["NEPOCHS"] = 2
else: 
    su.save_config(config, filename + ".json", directory = save_directory)


df, features =  lu.load_data(extra_columns = ["TARGETID_desi", "SPECTYPE_desi"],
                             debug = debug)
df = preprocessing.clean_data(df, config = config, verbose = True)
df = df.with_columns(lu._get_label_column_expr("multilabel"))

train_data, validation_data = preprocessing.split_train_test(df, 
                                                             test_size=config["TEST_SIZE"],
                                                             split_seed=config["SPLIT_SEED"],
                                                             force_label=config["FORCE_LABEL"])



train_target = train_data.select("label").to_numpy()      
validation_target = validation_data.select("label").to_numpy()           

train_data  = train_data.select(config["features_to_use"])
validation_data  = validation_data.select(config["features_to_use"])
train_data, validation_data = preprocessing.scale_data(train_data, validation_data)

train_target = torch.tensor(train_target, dtype = torch.long).squeeze()
validation_target = torch.tensor(validation_target, dtype = torch.long).squeeze()
train_data = torch.tensor(train_data, dtype = torch.float32)
validation_data = torch.tensor(validation_data, dtype = torch.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(train_data, train_target)
validation_dataset = TensorDataset(validation_data, validation_target)

train_dataloader = DataLoader(train_dataset, batch_size= config["BATCH_SIZE"], shuffle = True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size = config["BATCH_SIZE"], pin_memory = True)



model = models_library.SCARF(
              input_dim = config["input_dim"],
              features_low = train_data.min(dim=0).values,
              features_high = train_data.max(dim=0).values,
              encoder_output_dim =  config["encoder_output_dim"],
              pre_train_head_output_dim = config["pre_train_head_output_dim"],
              class_head_output_dim = config["class_head_output_dim"],
              encoder_hidden_dims = config["encoder_hidden_dims"],
              pre_train_head_hidden_dims =  config["pre_train_head_hidden_dims"],
              class_head_hidden_dims = config["class_head_hidden_dims"],
              corruption_rate = config["corruption_rate"],
              dropout = config["dropout"],
              batch_norm = config["batch_norm"]).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = config["LEARNING_RATE"], weight_decay=1e-5)
ntxent_loss = NTXent()

earlystopper = training.EarlyStopper(patience = 100)

history =[]
for epoch in range(config["NEPOCHS"]):
    
    train_loss = training.SCARF_epoch_pretraining(train_dataloader, 
                                                model,
                                                loss_fn = ntxent_loss,
                                                optimizer = optimizer, 
                                                device = device, 
                                                verbose = False)
    val_loss = training.SCARF_epoch_pretraining(validation_dataloader, 
                                                model,
                                                loss_fn = ntxent_loss,
                                                device = device, 
                                                verbose = (epoch%10==0))
    history.append(training.update_history(epoch, train_loss, val_loss))

    early_stop_flag = earlystopper.check_early_stopping(val_loss=val_loss)
    
    need_saving = ((epoch%200 == 0) or (epoch == config["NEPOCHS"]-1) or early_stop_flag)
    need_saving = need_saving and (not debug)

    if need_saving:
        su.save_training_state(model, 
                               history = history, 
                               filename = filename,
                               save_directory= save_directory,
                               save_directory_plot=save_directory_plot)
    if early_stop_flag:
        print(f"Pre-Training early stop at Epoch: {epoch}", flush = True)
        break
    

    
########################################
################# Fine tuning with classifier

filename = filename.replace("SCARF", "SCARF_finetuned")

train_data = train_data[train_target >= 0]
validation_data = validation_data[validation_target >=0]
train_target = train_target[train_target >= 0]
validation_target = validation_target[validation_target >=0]

train_dataset = TensorDataset(train_data, train_target)
validation_dataset = TensorDataset(validation_data, validation_target)

train_dataloader = DataLoader(train_dataset, batch_size= config["BATCH_SIZE"], shuffle = True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size = config["BATCH_SIZE"], pin_memory = True)

optimizer = torch.optim.Adam(model.parameters(), lr = config["LEARNING_RATE"]/10, weight_decay=1e-5)

earlystopper = training.EarlyStopper(patience = 15)
history =[]
for epoch in range(config["NEPOCHS"]):
    train_loss = training.SCARF_epoch_classifier(train_dataloader, 
                                                model,
                                                loss_fn = nn.CrossEntropyLoss(),
                                                optimizer = optimizer, 
                                                device = device, 
                                                verbose = False)
    val_loss = training.SCARF_epoch_classifier(validation_dataloader, 
                                                model,
                                                loss_fn = nn.CrossEntropyLoss(),
                                                device = device, 
                                                verbose = (epoch%10==0))
    history.append(training.update_history(epoch, train_loss, val_loss))
    

    early_stop_flag = earlystopper.check_early_stopping(val_loss=val_loss)
    
    need_saving = ((epoch%200 == 0) or (epoch == config["NEPOCHS"]-1) or early_stop_flag)
    need_saving = need_saving and (not debug)

    if need_saving:
        su.save_training_state(model, 
                               history = history, 
                               filename = filename,
                               save_directory= save_directory,
                               save_directory_plot=save_directory_plot)
    if early_stop_flag:
        print(f"Classification head early stop at Epoch: {epoch}", flush = True)
        break



