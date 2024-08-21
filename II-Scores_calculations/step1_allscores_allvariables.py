#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Calculations for All Scores of Cross-Validation for Meteorological Variables

Usage: This script computes various classification metrics (e.g., true positive rate, false positive rate, accuracy) 
for different meteorological variables used in the prediction of lightning occurrences across Canada or Quebec. 
The results are generated as two NetCDF files: one containing all the scores (all_scores) and another with 
the Area Under the Curve (AUC) metrics.

Authors: Jonathan Durand and Mathias Ponton
Created on 2024/03/25

Description: 
This script processes meteorological reanalysis data to calculate classification metrics across different 
thresholds for variables such as CAPE, CP, LI, K-index, Total Totals Index, and CAPE*CP. These metrics evaluate 
the ability of each variable to predict lightning events using cross-validation techniques. The script handles 
data loading, subset selection, metric calculation, and result storage in NetCDF format for subsequent analysis.

Key Features:
- Variables: The script analyzes a set of specified meteorological variables and their predictive power.
- Geographic Area: The area of interest can be set to either Canada or Quebec using latitude and longitude boundaries.
- Time Period: The analysis covers a user-defined time range and focuses on the months from May to October.
- Output: Results are stored in NetCDF format, with detailed scores for all thresholds and AUC metrics.

How to Use:
1. Set the variables to analyze in the `variables` list.
2. Define the time range by adjusting `time_start` and `time_end`.
3. Specify the geographic boundaries with `lat_min`, `lat_max`, `lon_min`, and `lon_max`.
4. Set the `target_path` where the output files will be stored.

Example: The script will calculate cross-validation scores for CAPE, CP, LI, K-index, and Total Totals Index 
for the period from 2015 to 2023 over the specified geographic area, and store the results in the target directory.

Dependencies:
- Python libraries: glob, xarray, numpy, os, matplotlib, sklearn, warnings
- Ensure all dependencies are installed and the required data files are accessible.
"""

#######################################################################
### LIBRARIES
import glob
import xarray as xr
import numpy as np
import os.path
import os
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings

#######################################################################
### ADJUST (TO SKIP WARNING MESSAGES)
warnings.filterwarnings("ignore")

#######################################################################
### PARAMETERS (TO MODIFY)
# Variables to be analyzed
variables = ['cape', 'cp', 'capecp', 'LI', 'kx', 'totalx']
# Time range for analysis
time_start = "2015-01-01"
time_end = "2023-12-31"
# Geographic boundaries (Canada or Quebec)
lat_min = 84 #55
lat_max = 40 #45
lon_min = 212 #280
lon_max = 310 #305
# Path to store results
target_path = '/target_path/score_data_step1/'

#######################################################################
### MAIN SCRIPT
# Iterate over each variable
for var in variables:
    print(f"* Calculating scores for: {var}...")

    #######################################################################
    ### THRESHOLDS FOR VARIABLES
    if var == "cape":   
        seuil = np.arange(0, 2500, 50)

    if var == "cp":
        seuil = np.linspace(0, 0.005, 150)

    if var == "capecp":
        seuilcenter = np.arange(0.00001, 1, 0.01).tolist()
        seuilend = np.arange(2, 130, 10).tolist()
        seuil = seuilcenter + seuilend

    if var == "kx":   
        seuil = np.arange(15, 40, 0.2)

    if var == "totalx":   
        seuil = np.arange(40, 60, 0.1)

    if var == "LI":
        seuil = np.arange(-7.5, 7.5, 0.25)

    #######################################################################
    ### LOAD FILES
    filename_var = glob.glob(f"/filename_var/{var}_canada/{var}_era5_*")
    ds_var = xr.open_mfdataset(filename_var)
    ds_foudre = xr.open_mfdataset('/ds_foudre/foudre.nc4')
    ds_tz = xr.open_mfdataset('/tornado/ponton/mask_data/shp.nc4')

    #######################################################################
    #######################################################################
    ##### PROCESS & SUBSET VAR FILES
    # Resample to daily maximum values
    ds_var = ds_var.resample(time='d').max()
    ds_var = ds_var.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max),
                    time=slice(time_start, time_end))
    ds_foudre = ds_foudre.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min-360, lon_max-360),
                          Time=slice(time_start, time_end))
    ds_tz = ds_tz.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min-360, lon_max-360))
    ds_tz = ds_tz.region.values
    region = [1]

    # Select data for the months May to October
    summer_ds_var = ds_var.time.dt.month.isin(range(5, 11))
    ds_var = ds_var.sel(time=summer_ds_var)
    summer_ds_foudre = ds_foudre.Time.dt.month.isin(range(5, 11))
    ds_foudre = ds_foudre.sel(Time=summer_ds_foudre)

    # Initialize datasets for all scores and AUC
    data_all = xr.Dataset(coords={'longitude': ([ 'longitude'], ds_var.variables['longitude'][:]),
                                          'latitude': (['latitude',], ds_var.variables['latitude'][:]),
                                          'seuil': seuil})
    data_auc = xr.Dataset(coords={'longitude': ([ 'longitude'], ds_var.variables['longitude'][:]),
                                          'latitude': (['latitude',], ds_var.variables['latitude'][:])})

    # Initialize lists for false positive rate (FPR) and true positive rate (TPR)
    fpr_list = []
    tpr_list = []

    # Initialize arrays for metrics
    tp = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    tp[tp==0] = np.nan
    tn = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    tn[tn==0] = np.nan
    fp = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    fp[fp==0] = np.nan
    fn = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    fn[fn==0] = np.nan  
    auc = np.zeros((len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    auc[auc==0] = np.nan
    fpr = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    fpr[fpr==0] = np.nan
    tpr = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    tpr[tpr==0] = np.nan
    tnr = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    tnr[tnr==0] = np.nan
    ppv = np.zeros((len(seuil),len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    ppv[ppv==0] = np.nan
    npv = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    npv[npv==0] = np.nan
    fnr = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    fnr[fnr==0] = np.nan 
    fdr = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    fdr[fdr==0] = np.nan
    acc = np.zeros((len(seuil), len(ds_var.latitude), len(ds_var.longitude)), dtype=float)
    acc[acc==0] = np.nan

    # Select variable values
    ds_var = ds_var.variables[var].values
    ds_foudre = ds_foudre.F.values
    ds_foudre = np.where(ds_foudre > 1, 1, 0)

    # Calculate metrics for each grid point
    for i in range(len(ds_var[0][0])):
        print("Grid point #", i, ", size :", len(ds_var[0][0]))

        for y in range(len(ds_var[0])):
            if ds_tz[y, i] in region:
                fpr_list = []
                tpr_list = []

                for s in range(len(seuil)):
                    if var == "LI":
                        ds_var_mat = np.where(ds_var <= seuil[s], 1, 0)
                    else:
                        ds_var_mat = np.where(ds_var >= seuil[s], 1, 0)

                    predictions = ds_var_mat[:, y, i]
                    actuals = ds_foudre[:, y, i]
                    cf_train_matrix = confusion_matrix(actuals, predictions, labels=[1, 0])

                    TP = cf_train_matrix[0][0]

                    if np.array_equal(predictions, actuals):
                        FN = 0
                        FP = 0
                        TN = 0
                        fn[s, y, i] = FN
                        tp[s, y, i] = TP
                        fp[s, y, i] = FP
                        tn[s, y, i] = TN
                        TPR = TP / (TP + FN)
                        PPV = TP / (TP + FP)
                        ACC = (TP + TN) / (TP + FP + FN + TN)
                        fn[s, y, i] = FN
                        tp[s, y, i] = TP
                        fp[s, y, i] = FP
                        tn[s, y, i] = TN
                        TPR = TP / (TP + FN)
                        FPR = 0
                        tpr[s, y, i] = TP / (TP + FN)
                        tnr[s, y, i] = 0
                        ppv[s, y, i] = TP / (TP + FP)
                        npv[s, y, i] = 0
                        fpr[s, y, i] = 0
                        fnr[s, y, i] = 0
                        fdr[s, y, i] = 0
                        acc[s, y, i] = (TP + TN) / (TP + FP + FN + TN)
                    else:  
                        FN = cf_train_matrix[0][1]
                        FP = cf_train_matrix[1][0]
                        TN = cf_train_matrix[1][1]
                        fn[s, y, i] = FN
                        tp[s, y, i] = TP
                        fp[s, y, i] = FP
                        tn[s, y, i] = TN

                        TPR = TP / (TP + FN)
                        FPR = FP / (FP + TN)

                        fpr_list.append(FPR)
                        tpr_list.append(TPR)
                        fn[s, y, i] = FN
                        tp[s, y, i] = TP
                        fp[s, y, i] = FP
                        tn[s, y, i] = TN
                        tpr[s, y, i] = TP / (TP + FN)
                        tnr[s, y, i] = TN / (TN + FP)
                        ppv[s, y, i] = TP / (TP + FP)
                        npv[s, y, i] = TN / (TN + FN)
                        fpr[s, y, i] = FP / (FP + TN)
                        fnr[s, y, i] = FN / (TP + FN)
                        fdr[s, y, i] = FP / (TP + FP)
                        acc[s, y, i] = (TP + TN) / (TP + FP + FN + TN)

                #######################################################################
                ### CALCULATE AUC
                fpr_list_sorted = np.array(fpr_list)
                sorted_index = np.argsort(fpr_list_sorted)
                fpr_list_sorted = np.array(fpr_list)[sorted_index]
                tpr_list_sorted = np.array(tpr_list)[sorted_index]
                auc[y, i] = metrics.auc(fpr_list_sorted, tpr_list_sorted)

    #######################################################################
    ### ADD SCORES AND AUC TO XARRAY DATASETS
    data_all["TP"] = (('seuil', 'latitude', 'longitude'), tp)
    data_all["FP"] = (('seuil', 'latitude', 'longitude'), fp)
    data_all["TN"] = (('seuil', 'latitude', 'longitude'), tn)
    data_all["FN"] = (('seuil', 'latitude', 'longitude'), fn)
    data_all["TPR"] = (('seuil', 'latitude', 'longitude'), tpr)
    data_all["TNR"] = (('seuil', 'latitude', 'longitude'), tnr)
    data_all["FPR"] = (('seuil', 'latitude', 'longitude'), fpr)
    data_all["FNR"] = (('seuil', 'latitude', 'longitude'), fnr)
    data_all["FDR"] = (('seuil', 'latitude', 'longitude'), fdr)
    data_all["PPV"] = (('seuil', 'latitude', 'longitude'), ppv)
    data_all["NPV"] = (('seuil', 'latitude', 'longitude'), npv)
    data_all["ACC"] = (('seuil', 'latitude', 'longitude'), acc)
    data_auc["AUC"] = (('latitude', 'longitude'), auc)

    #######################################################################
    ### METRICS FILES
    # Définit des paramètres de compression à appliquer lors de l'écriture dans le fichier NetCDF
    comp = dict(zlib=True, complevel=1)

    # Créer un dictionnaire d'encodage où les clés sont les variables de data_all et les valeurs 
    # sont les paramètres de compression définis ci-dessus
    encoding = {var: comp for var in data_all.data_vars}

    ### ALL_SCORE
    # Définit un chemin du fichier de sortie NetCDF en fonction de la variable (var) 
    # et de l'horodatage (time_start)
    outfile = target_path + 'all_scores_' + str(var) + '_' + time_start + '.nc4'

    # Vérifie l'existence du fichier de sortie, et s'il existe, suppression pour éviter les conflits
    if os.path.exists(outfile):
        os.remove(outfile)

    # Écrit des données contenues dans data_all dans le fichier NetCDF spécifié par outfile,
    # avec le format "NETCDF4_CLASSIC" et l'encodage défini par le dictionnaire encoding
    data_all.to_netcdf(path=outfile, format="NETCDF4_CLASSIC", encoding=encoding)

    ### AUC FILE
    # Créer un dictionnaire d'encodage où les clés sont les variables de data_auc et les valeurs
    # sont les compressions à appliquer
    encoding = {var: comp for var in data_auc.data_vars}

    outfile = target_path + 'auc_' + str(var) + "_" + time_start + ".nc4"

    if os.path.exists(outfile):
        os.remove(outfile)
    data_auc.to_netcdf(path=outfile, format="NETCDF4_CLASSIC", encoding=encoding)
    print(f"*** The files scores for : {var} are calculated.")