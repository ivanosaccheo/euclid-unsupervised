from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np 
import polars as pl 
from utility import astro_utility


_SCALERS = {"StandardScaler" : StandardScaler,
            "MinMaxScaler" : MinMaxScaler,
            }


def replace_infs(df, columns_required):
    return df.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(float("nan")) for c in columns_required])

def drop_nans(df, columns_required):
    return  df.fill_nan(None).drop_nulls(subset = columns_required)

def fill_nans(df, columns_required):
    return df.fill_nan(None).with_columns([pl.col(c).fill_null(pl.col(c).mean()) for c in columns_required])

def filter_low_SNR(df, err_mag_columns, snr_min = 3):
    if snr_min <= 0:
        return df 
    factor = 2.5/np.log(10) 
    expression = [pl.col(e) <= (factor/snr_min) for e in err_mag_columns]
    return df.filter(pl.all_horizontal(expression))

def scale_data(training_data, *args, 
               scaler = "StandardScaler",
               return_scaler = False):
    scaler_obj = _SCALERS[scaler]()
    scaler_obj.fit(training_data)
    scaled_training = scaler_obj.transform(training_data)
    scaled_args = [scaler_obj.transform(arg) for arg in args]
    if return_scaler:
        return (scaler_obj, scaled_training, *scaled_args)
    return (scaled_training, *scaled_args)


def split_train_test(df, 
                    test_size = 0.3,
                    force_label = False, 
                    has_label_expression = pl.col("TARGETID_desi") > 0,
                    split_seed = 26052013):
    """          
    Split sample into test and train samples using sklearn. 
    df = polars.DataFrame()
    force_desi : if True all sources with label (from DESI) are moved into the validation sample
    """
    if not force_label:
        return train_test_split(df, test_size=test_size, random_state=split_seed)
    n = len(df)
    n_test = int(np.ceil(test_size*n)) if test_size < 1 else int(test_size)
    

    df_label = df.filter(has_label_expression)
    df_no_label = df.filter(~has_label_expression)

    n_labeled = len(df_label)
    n_extra = n_test - n_labeled
    
    if n_extra < 0:  
        #Some labeled sources are moved into the train sample
        labeled_train, labeled_test = train_test_split(
                                        df_label,
                                        test_size=n_test,
                                        random_state=split_seed,)

        train_df = pl.concat([df_no_label, labeled_train], how="vertical")
        test_df = labeled_test

    elif n_extra == 0:
        train_df = df_no_label
        test_df = df_label

    else:
        no_label_train, no_label_test = train_test_split(
                                          df_no_label,
                                          test_size=n_extra,
                                          random_state=split_seed,)

        train_df = no_label_train
        test_df = pl.concat([df_label, no_label_test], how="vertical")
    
    return train_df, test_df



def clean_data(df, config, verbose = True):
    snr_min = config.get("SNR_min", None)
    if snr_min is not None:
        len_in = df.height
        err_mag_columns = [c for c in df.columns if "magerr_" in c]
        df = filter_low_SNR(df, err_mag_columns, snr_min = snr_min)
        if verbose:
            print(f"Keeping {df.height} sources with SNR >= {snr_min}")
            print(f"Removed {len_in - df.height} sources with SNR < {snr_min}")
            print("-----")
    columns_required = config.get(columns_required, df.columns)
    len_in = df.height
    df = replace_infs(df, columns_required)
    if config.get("FILL_NAN_VALUES", False):
        df = fill_nans(df, columns_required)
    else:
        df = drop_nans(df, columns_required)
        if verbose:
            print(f"Keeping {df.height} sources without NaN values")
            print(f"Removed {len_in - df.height} sources with NaN values")
            print("-----")

    return df