#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Create a map with the score calculated in step 1 (the difference at. step2_map_scores is just the plot)
Usage: Generate map for all calculations in grid
Authors: Jonathan Durand and Mathias Ponton
Created on 2024/03/26

Description:
This script generates maps based on scores calculated in step 1 for various meteorological variables 
such as CAPE, CP, CAPE x CP, LI, Kx, and TTX. The maps display different metrics such as best F1 score, 
accuracy, threshold, and precision.

Libraries required:
- matplotlib
- cartopy
- numpy
- warnings
- xarray
- seaborn
- mpl_toolkits
- matplotlib.colors

How to use:
The `map_scores` function is the main function to generate maps for different variables and scores. 
Adjust the paths to your data and save locations as needed.

Example:
map_scores(var='cape', score='f1', mode='bestscore', label='F1 best score')
"""

#######################################################################
### LIBRARIES
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import warnings
import xarray as xr
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

sns.set(style="white", color_codes=True)
sns.set(rc={'figure.figsize':(28, 16)})

#######################################################################
### ADJUST (TO SKIP WARNING MESSAGES)
warnings.filterwarnings("ignore")

#######################################################################
#######################################################################
##### DISPLAY
# All variables
plot_subplot(var='var', score='f1', mode='bestscore', label='F1 best score')

#######################################################################
#######################################################################
### FUNCTION
def plot_subplot(var, score, mode, label):
    """
    Generates maps for different variables and scores.

    Parameters:
    var (str): Variable name (e.g., 'cape', 'cp', 'capecp', 'LI', 'kx', 'totalx')
    score (str): Score type (e.g., 'f1', 'f2')
    mode (str): Mode type (e.g., 'bestscore', 'accuracy', 'threshold', 'precision')
    label (str): Label for the colorbar

    Returns:
    Graphics
    """
    ### PARAMETERS
    # Paths
    path_in = '/path_in/score_data_step1/'
    path_out = f'/path_out/{score}_score_step2/{mode}/'
    
    ### DEFINE AREA OF WORK
    lon_min = -140 #-80
    lon_max = -50 #-55
    lat_min = 35 #45
    lat_max = 75 #55
    lon_inc = 20 #5
    lat_inc = 10 #5

    ### THRESHOLD OF VARIABLES
    if var == "cape":   
        seuil = np.arange(0, 2500, 50)
    elif var == "cp":
        seuil = np.linspace(0, 0.005, 150)
    elif var == "capecp":
        seuilcenter = np.arange(0.00001, 1, 0.01).tolist()
        seuilend = np.arange(2, 130, 10).tolist()
        seuil = seuilcenter + seuilend
    elif var == "kx":   
        seuil = np.arange(15, 40, 0.2)
    elif var == "totalx":   
        seuil = np.arange(40, 60, 0.1)
    elif var == "LI":
        seuil = np.arange(-7.5, 7.5, 0.25)
        
    ### OPEN DATASETS
    ds_all = xr.open_mfdataset(path_in + 'all_scores_' + str(var) + '_date.nc4')
    ds_auc = xr.open_mfdataset(path_in + 'auc_' + str(var) + '_date.nc4')

    ### VARIABLES TO PLOT
    var_toplot = ds_auc.auc  # Area Under the Curve (measure of separability)
    tpr = ds_all.tpr.values  # True Positive Rate
    fpr = ds_all.fpr.values  # False Positive Rate
    tp = ds_all.tp.values    # True Positive
    fp = ds_all.fp.values    # False Positive
    fn = ds_all.fn.values    # False Negative
    tn = ds_all.tn.values    # True Negative
    acc = ds_all.acc.values  # Accuracy
    
    ### INITIALISATION OF TABLES FOR F1_BEST, THRESHOLD AND ACC
    fbest = np.zeros((len(ds_all.latitude), len(ds_all.longitude)), dtype=float)
    fbest[fbest == 0] = np.nan

    threshold = np.zeros((len(ds_all.latitude), len(ds_all.longitude)), dtype=float)
    threshold[threshold == 0] = np.nan

    acc_array = np.zeros((len(ds_all.latitude), len(ds_all.longitude)), dtype=float)
    acc_array[acc_array == 0] = np.nan
    
    prec_array = np.zeros((len(ds_all.latitude), len(ds_all.longitude)), dtype=float)
    prec_array[prec_array == 0] = np.nan
    
    ### LOOP TO CALCULATE F1 SCORE AND SELECT BEST F1 SCORE/THRESHOLD AND ACCURACY
    for i in range(len(fbest[0])):
        for y in range(len(fbest)):
            # Calculate precision
            precision = tp[:, y, i] / (tp[:, y, i] + fp[:, y, i])
            precision[np.isnan(precision)] = 0  # Replace nan values with 0
            precision[precision == 1] = 0  # Replace 1 values with 0

            # Calculate recall
            recall = tp[:, y, i] / (tp[:, y, i] + fn[:, y, i])
            recall[np.isnan(recall)] = 0  # Replace nan values with 0
            recall[recall == 1] = 0  # Replace 1 values with 0
            
            if score == 'f1':
                # Calculate F1 score
                f = 2 * (precision * recall) / (precision + recall)  
            elif score == 'f2':
                # Calculate F2 score
                f = 5 * (precision * recall) / (4 * precision + recall)

            # Check if F score is not entirely nan
            if np.isnan(f).all() == False:
                # Select maximum F1 score
                fbest[y, i] = np.nanmax(f)
                # Record the threshold corresponding to the maximum F1 score
                threshold[y, i] = seuil[np.nanargmax(f)]
                # Record the accuracy associated with the maximum F1 score
                acc_array[y, i] = acc[np.nanargmax(f), y, i]
                # Record the precision associated with the maximum F1 score
                prec_array[y, i] = precision[np.nanargmax(f)]
            else:
                # If all values are nan, record nan in the corresponding arrays
                fbest[y, i] = np.nan
                threshold[y, i] = np.nan 
                acc_array[y, i] = np.nan
                prec_array[y, i] = np.nan
                                          
                                          
    #####################################################################################
    ### MAIN IMAGE SETUP ###
    ### PARAMETERS
    # Create a custom color map
    colors = ["blue", "cyan", "green", "yellow", "orange", "red"]
    n_bins = 30
    custom_cmap = LinearSegmentedColormap.from_list("custom1", colors, N=n_bins)
    
        ### DEFINE EXTREMUMS (MODIFY AS YOU WISH)
    if mode == 'bestscore':
        data_min, data_max, data_increment, colorbar_increment = 0, 1, 0.1, 0.1
    elif mode == 'accuracy':
        data_min, data_max, data_increment, colorbar_increment = 0, 1, 0.1, 0.1
    elif mode == 'threshold':
        if var == 'cape':
            data_min, data_max, data_increment, colorbar_increment = 0, 800, 10, 10
        elif var == 'cp':
            data_min, data_max, data_increment, colorbar_increment = 0, 0.002, 0.0001, 0.0005
        elif var == 'capecp':
            data_min, data_max, data_increment, colorbar_increment = 0, 0.6, 0.1, 0.1
        elif var == 'kx':
            data_min, data_max, data_increment, colorbar_increment = 40, 15, 1, 1
        elif var == 'totalx':
            data_min, data_max, data_increment, colorbar_increment = 45, 55, 5, 5
        elif var == 'LI':
            data_min, data_max, data_increment, colorbar_increment = 40, 60, 2, 2
    elif mode == 'precision':
        data_min, data_max, data_increment, colorbar_increment = 0, 1, 0.1, 0.1
    # LAYERS
    states_provinces = cfeature.NaturalEarthFeature(category='cultural', 
                                                    name='admin_1_states_provinces_lines',
                                                    scale='50m', facecolor='none')
    land = cartopy.feature.NaturalEarthFeature('physical', 'land',
                                                  scale='50m', edgecolor=None,
                                                  facecolor=cfeature.COLORS['land'])
    ax = axes[idx]
    ax.set_extent([-145, -50, 30, 75])

    ax.add_feature(land, facecolor='beige',zorder=0) 
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.75)
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(states_provinces, edgecolor='dimgrey')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth =0.75, edgecolor = 'black')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth =0.75, edgecolor = 'black')

    ### DATAS
    if mode == 'bestscore':
        mm = ax.pcolormesh(ds_auc.longitude, ds_auc.latitude, fbest, vmin=data_min, vmax=data_max, cmap=plt.cm.jet,
                      transform=ccrs.PlateCarree())
    elif mode == 'accuracy':
        mm = ax.pcolormesh(ds_auc.longitude, ds_auc.latitude, acc_array, vmin=data_min, vmax=data_max, cmap=plt.cm.jet,
                      transform=ccrs.PlateCarree())
    elif mode == 'threshold':
        mm = ax.pcolormesh(ds_auc.longitude, ds_auc.latitude, threshold, vmin=data_min, vmax=data_max, cmap=plt.cm.jet,
                      transform=ccrs.PlateCarree())
    elif mode == 'precision':
        mm = ax.pcolormesh(ds_auc.longitude, ds_auc.latitude, prec_array, vmin=data_min, vmax=data_max, cmap=plt.cm.jet,
                      transform=ccrs.PlateCarree())

    ### GRIDLINES
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=2, color='gray', alpha=0.35, linestyle='--',
                      x_inline=False, y_inline=False)

    # Define the position of lines for lat/lon
    gl.xlocator = mticker.FixedLocator(np.arange(start=-140, stop=-50, step=20))
    gl.ylocator = mticker.FixedLocator(np.arange(start=35, stop=75, step=10))

    # Desactivate the rotating labels
    gl.rotate_labels = False

    # Configuration of labels
    gl.ylabels_left = False
    gl.xlabels_left = False
    gl.ylabels_top = False
    gl.xlabels_top = False
    gl.ylabels_right = True
    gl.xlabels_right = False
    gl.ylabels_bottom = False
    gl.xlabels_bottom = True

    # Labels format
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Labels size and color
    gl.xlabel_style = {'size': 25, 'color': 'black'}
    gl.ylabel_style = {'size': 25, 'color': 'black'}

    # Ajoute une boîte de texte avec le nom de la variable
    var_name = "Variable: " + var
    ax.text(0.05, 0.95, var_name, transform=ax.transAxes, fontsize=25,
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

    ### COLORBAR
    cbar = fig.colorbar(mm, ax=axes, orientation='horizontal', shrink=0.50, pad=0.05, extend='max')
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label(label, size=25)                                          
    
    ### PARAMETERS
    plt.tight_layout()
    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0.15)

    ### SAVE THE FIGURE
    fig.savefig(f'{score}_{mode}_all_vars.pdf', bbox_inches='tight', dpi=200)
    
    return plt.show()