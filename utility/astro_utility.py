import numpy as np

def flux_to_mag(flux):
    return -2.5 * np.log10(flux) + 23.9

def fluxes_to_color(flux1, flux2):
    return -2.5 * np.log10(flux1/flux2)