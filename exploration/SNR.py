import os
import polars as pl
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from utility import astro_utility as au
from utility import load_utility as lu
from utility import utility



def plot_snr(ax, mag, snr,  
            name = "",
            snr_min = 1, snr_max = 30, mag_min = 18, mag_max = 28,
            **kwargs):
    mask = (snr >= snr_min) & (snr <=snr_max) &  (mag >= mag_min) & (mag <=mag_max)
    try: 
        x = mag[mask]
        y = snr[mask]
    except TypeError:
        x = mag.filter(mask)
        y = snr.filter(mask)
    
    hb =ax.hexbin(x, y, **kwargs)
    ax.annotate(name, (0.1, 0.85), xycoords = "axes fraction")
    return hb


base_dir = os.path.dirname(os.path.abspath(__file__))
save_directory_plot = os.path.join(base_dir, "plot")
save_directory = os.path.join(base_dir, "output")
os.makedirs(save_directory_plot, exist_ok=True)
os.makedirs(save_directory, exist_ok=True)



df, features = lu.load_data()
magerrs_cols = [i for i in features if "magerr_" in i]
mag_cols = [i.replace("magerr_","mag_") for i in magerrs_cols]


df = df.with_columns([pl.col(err).map_batches(au.get_SNR, is_elementwise=True, returns_scalar=True).alias(f"snr_{mag}")
                      for (err, mag) in zip(magerrs_cols, mag_cols)])



ncols = 3
nrows = int(np.ceil(len(mag_cols)/ncols))
fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10,10), 
                       gridspec_kw = {"wspace": 0.05, "hspace":0.05}, sharex=True)
for i, mag_col in enumerate(lu._sort_flux_columns_by_wavelength(mag_cols)):
    snr_col = "snr_" +  mag_col
    row = i//ncols
    col = i % ncols
    hb = plot_snr(ax[row, col], df[mag_col], df[snr_col], 
             name = mag_col.split("_total_")[0],
             snr_min = 0,
             mag_max =30,
        gridsize=300, mincnt =2, cmap = "jet", norm = LogNorm(vmin = 1, vmax = 10000))
    ax[row, col].set_ylim(0,10)
    if col !=0:
        ax[row,col].set_yticklabels([])
for j in range(i + 1, nrows * ncols):
    row = j // ncols
    col = j % ncols
    ax[row, col].set_axis_off()
fig.supylabel("SNR")
fig.supxlabel("AB mag")
fig.subplots_adjust(left=0.08, bottom=0.08)
cbar = fig.colorbar(hb, ax=ax, location="right", pad=0.02)
cbar.set_label("Log counts")
plt.savefig(os.path.join(save_directory_plot, "SNR_mag.png"), bbox_inches = "tight")


mag_bins = np.arange(16, 30, 0.1)
snr_dict ={}
snr_dict["mag"] = 0.5*(mag_bins[:-1]+mag_bins[1:])
for i, mag_col in enumerate(lu._sort_flux_columns_by_wavelength(mag_cols)):
    snr_col = "snr_" +  mag_col
    x = df[mag_col].to_numpy()
    y = df[snr_col].to_numpy()
    edges, vals = utility.get_binned_quantiles(x, y, bins = mag_bins, quantiles = [0.16, 0.5, 0.84])

    snr_dict[f"{mag_col}_16"] = vals[:,0]
    snr_dict[f"{mag_col}_50"] = vals[:,1]
    snr_dict[f"{mag_col}_84"] = vals[:,2]

snr_df = pl.DataFrame(snr_dict)
snr_df.write_csv(os.path.join(save_directory, "SNR_quantiles.csv"))