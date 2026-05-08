import json
import os
import datetime
import polars as pl
import torch

from ml import plot_library


def get_base_dir(caller_file):
    base_dir = os.path.dirname(os.path.abspath(caller_file))
    return base_dir

def get_save_directory(caller_file, save_directory):
    base_dir = os.path.dirname(os.path.abspath(caller_file))
    save_directory = os.path.join(base_dir, save_directory)
    os.makedirs(save_directory, exist_ok=True)
    return save_directory

def get_date_filename(filename):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return filename + "_" + today

def save_config(config, filename, directory = None):
    if directory is not None:
        filename = os.path.join(directory, filename) 
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)



def save_training_state(model, 
                        history, 
                        filename,
                        save_directory,
                        save_directory_plot = None):
    torch.save(model.state_dict(), os.path.join(save_directory, filename + ".pt"))
    history_df = pl.DataFrame(history)
    history_df.write_csv(os.path.join(save_directory, filename + ".csv"))
    if save_directory_plot is not None:
        plot_library.plot_training(history_df["train_loss"], history_df["validation_loss"],
                                   filename = filename + ".png", directory=save_directory_plot)
