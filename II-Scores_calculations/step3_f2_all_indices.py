#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Generate F2 Score Maps for Meteorological Variables
Usage: Compute and map F2 scores across multiple meteorological variables
Authors: Jonathan Durand and Mathias Ponton
Created on 2024/04/01

Description:
This script calculates and generates F2 score maps for various meteorological 
variables including CAPE, CP, CAPE x CP, Kx, and TTX. The F2 scores are computed 
for a range of thresholds and are stored in NetCDF files. The script loops through 
each variable, calculates precision, recall, and F2 score, and saves the results 
in a structured format.

Libraries required:
- matplotlib
- cartopy
- numpy
- xarray
- seaborn
- mpl_toolkits
- matplotlib.colors

How to use:
The script iterates over the specified meteorological variables, computes the F2 
scores for each variable at different thresholds, and saves the resulting datasets 
as NetCDF files. Adjust the input and output paths as necessary.

Example:
The script will save F2 score datasets for each variable at the specified 
output path.
"""

#######################################################################
### LIBRAIRIES
import xarray as xr
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#######################################################################
#######################################################################
### PATHS
path_in = '/path_in/score_data_step1/'
path_out = '/path_out/best_f2score_step4/'

### VARIABLES
variables = ["cape", "cp", "capecp", "LI", "kx", "totalx"]

### OPEN FILES (JUSTE FOR THE SHAPE, THE VAR IS NOT IMPORTANT)
ds_shape = xr.open_mfdataset(path_in + "all_scores_cape_date.nc4")

### CREATE A NEW FILE
data = xr.Dataset(coords={'longitude': ('longitude', ds_shape.variables['longitude'][:]),
                          'latitude': ('latitude', ds_shape.variables['latitude'][:])})

### LOOP FOR THE ALL VARIABLES
datasets = {}  # Dictionnaire pour stocker les datasets par variable

for var in variables:
    # Threshold selection based on the variable
    if var == "totalx":   
        seuil = np.arange(40, 60, 0.1)
    elif var == "kx":   
        seuil = np.arange(15, 40, 0.2)
    elif var == "LI":
        seuil = np.arange(-7.5, 7.5, 0.25)
    elif var == "cape":
        seuil = np.arange(0, 2500, 50)
    elif var == "capecp":
        seuilcenter = np.arange(0.00001, 1, 0.01).tolist()
        seuilend = np.arange(2, 130, 10).tolist()
        seuil = seuilcenter + seuilend
    elif var == "cp":
        seuil = np.linspace(0, 0.005, 150)

    # Open files of the different scores
    ds_all = xr.open_mfdataset(f"{path_in}all_scores_{var}_date.nc4")

    # Initialize a data structure for F2 scores with a new dimension for thresholds
    f2_scores = np.full((len(ds_all.latitude), len(ds_all.longitude), len(seuil)), np.nan)

    ### LOOP TO CALCULATE THE METRICS
    for i in range(len(ds_all.longitude)):
        for y in range(len(ds_all.latitude)):
            tp = ds_all.tp[:, y, i].compute()
            fp = ds_all.fp[:, y, i].compute()
            fn = ds_all.fn[:, y, i].compute()

            # Calculate precision
            precision = tp / (tp + fp)
            precision[np.isnan(precision)] = 0
            precision[precision == 1] = 0
            
            # Calculate recall
            recall = tp / (tp + fn)
            recall[np.isnan(recall)] = 0
            recall[recall == 1] = 0
            
            # Calculate F2 score
            f2 = 5 * (precision * recall) / (4 * precision + recall)
            f2_scores[y, i, :] = f2

    # Create a new dataset for each variable
    data_var = xr.Dataset({
        f"{var}_f2": (("latitude", "longitude", "threshold"), f2_scores)
    }, coords={
        'latitude': ds_all.latitude,
        'longitude': ds_all.longitude,
        'threshold': seuil
    })

    datasets[var] = data_var  # Stocker chaque dataset dans un dictionnaire

### OPTIONS OF THE FILES
comp = dict(zlib=True, complevel=1)
for var, data in datasets.items():
    encoding = {var: comp for var in data.data_vars}
    # Save the dataset, adjust the filename as needed
    data.to_netcdf(f"{path_out}{var}_all_f2_scores.nc4", mode='w', format='NETCDF4',
                   engine='netcdf4', encoding=encoding)
