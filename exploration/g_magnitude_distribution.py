import polars as pl 
import os
import matplotlib.pyplot as plt
import numpy as np

from ml import preprocessing
from utility import load_utility as lu, save_utility as su


save_directory_plot = su.get_save_directory(__file__, "plot")
snr_min = 50

df, features = lu.load_data(fwhm_values=(1,2), extra_columns=["right_ascension_euclid", "declination_euclid"])
err_g_mag_col = [c for c in features if ("magerr_g_hsc" in c)]
g_mag_col = [i.replace("magerr_","mag_") for i in err_g_mag_col]
df  = preprocessing.filter_low_SNR(df, err_g_mag_col, snr_min = snr_min)

bins = np.arange(16, 25, 0.25)
fig, ax = plt.subplots()

for band in ["u", "r", "i", "z", "y", "h"]:
     err_band_mag_col = [c for c in features if (f"magerr_{band}_" in c)]
     print(err_band_mag_col)
     temp_df  = preprocessing.filter_low_SNR(df, err_band_mag_col, snr_min = snr_min)
     ax.hist(temp_df.select(g_mag_col), bins = bins)
     #ax.set_yscale("log")

     plt.savefig(os.path.join(save_directory_plot, f"_g_mag_distribution_{band}.png"), bbox_inches = "tight")


