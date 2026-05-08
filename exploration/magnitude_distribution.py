import polars as pl 
import os
import matplotlib.pyplot as plt
import numpy as np

from ml import preprocessing
from utility import load_utility as lu, save_utility as su


save_directory_plot = su.get_save_directory(__file__, "plot")


df, features = lu.load_data(fwhm_values=(1,2), extra_columns=["SPECTYPE_desi"])
snr_min = 50
err_mag_cols = [c for c in features if "magerr_" in c]

print(err_mag_cols)

mag_cols = [i.replace("magerr_","mag_") for i in err_mag_cols]

df = preprocessing.filter_low_SNR(df, err_mag_cols, snr_min = snr_min)
df = df.with_columns(lu._get_label_column_expr("multilabel"))

desi = df.filter(pl.col("label")>0)


ncols = 3
nrows = int(np.ceil(len(mag_cols)/ncols))
fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), 
                       gridspec_kw = {"wspace": 0.05, "hspace":0.05}, sharex=True)
for i, mag_col in enumerate(lu._sort_flux_columns_by_wavelength(mag_cols)):
    row = i//ncols
    col = i % ncols
    name = mag_col.split("_total_")[0]
    ax[row, col].hist(df[mag_col], color = "b", weights = np.ones(len(df))/len(df),
                      bins = np.arange(16, 25, 0.25), label ="SNR>50")
    ax[row, col].hist(desi[mag_col], color = "r", weights = np.ones(len(desi))/len(desi),
                      bins = np.arange(16, 25, 0.25), lw =2, histtype = "step", label = "has DESI")
    ax[row, col].set_ylim(0,0.2)
    ax[row,col].annotate(name, (0.1, 0.85), xycoords = "axes fraction")
    if col !=0:
        ax[row,col].set_yticklabels([])
for j in range(i + 1, nrows * ncols):
    row = j // ncols
    col = j % ncols
    ax[row, col].set_axis_off()
ax[2,1].legend()
fig.supylabel("N sources")
fig.supxlabel("AB mag")
fig.subplots_adjust(left=0.08, bottom=0.08)
plt.savefig(os.path.join(save_directory_plot, "mag_hist_SNR50.png"), bbox_inches = "tight")


