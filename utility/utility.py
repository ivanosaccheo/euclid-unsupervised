import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os


def my_binned_statistic(x, y, 
                        func,
                        bins = 10, 
                        include_counts=False
                        ):
    #same as scipy.stats.binned_statistic, but func can return also non scalar
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isscalar(bins):
        bins = np.linspace(x.min(), x.max(), bins+1)
    else:
        bins = np.asarray(bins)
    
    Nbins = len(bins) - 1
    bin_idx = np.digitize(x, bins = bins) - 1
    values = [None for _ in range(Nbins)]
    counts = np.zeros(Nbins, dtype=int)              
    for i in range(Nbins):
        select = bin_idx == i
        if np.any(select):
            values[i] = func(y[select])
            counts[i] = np.sum(select)
        else:
            values[i] = func(np.full(y.shape, np.nan))        
    if include_counts:
        return bins, np.asarray(values), counts
    else:
        return bins, np.asarray(values)
    

def my_binned_statistic_2d(x, y, z, func, 
                           xbins =10, ybins =10,
                           include_counts=False,
                           ):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if np.isscalar(xbins):
        xedges = np.linspace(x.min(), x.max(), xbins+1)
    else:
        xedges = np.asarray(xbins, dtype=float)

    if np.isscalar(ybins):
        yedges = np.linspace(y.min(), y.max(), ybins+1)
    else:
        yedges = np.asarray(ybins, dtype=float)

    Nx, Ny = len(xedges)-1, len(yedges)-1

    xi = np.digitize(x, bins=xedges) - 1
    yi = np.digitize(y, bins=yedges) - 1

    values = [[None for _ in range(Ny)] for _ in range(Nx)]
    counts = np.zeros((Nx, Ny), dtype=int)

    for i in range(Nx):
        for j in range(Ny):
            select= (xi == i) & (yi == j)
            if np.any(select):
                values[i][j] = func(z[select])
                counts[i, j] = np.sum(select)
            else:
                values[i][j] = func(np.full(z.shape, np.nan))

    if include_counts:
        return xedges, yedges, np.asarray(values), counts
    else:
        return xedges, yedges, np.asarray(values)


def get_binned_quantiles(x, y, 
                         bins = 10, 
                         quantiles = [0.05, 0.16, 0.5, 0.84, 0.95],
                         include_counts=False):
    
    quantile_func = lambda x:  np.quantile(x, quantiles)
    return my_binned_statistic(x, y, func=quantile_func,
                               bins = bins,
                               include_counts=include_counts)