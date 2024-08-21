#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Generate Comparative Maps for Meteorological Indices Based on F1 and F2 Scores
Usage: This script generates comparative maps for F1 and F2 scores across various meteorological indices.
Authors: Jonathan Durand and Mathias Ponton
Created on 2024/04/25

Description:
This script generates maps that compare different meteorological variables such as Kx, Totalx, CAPE, CP, 
and CAPExCP based on their F1 and F2 scores. The maps display the dominant variable for each grid point 
and highlight the regions where each variable achieves the highest score.

Libraries required:
- matplotlib
- cartopy
- numpy
- xarray
- seaborn
- warnings
- mpl_toolkits.axes_grid1
- matplotlib.colors

How to use:
The `map_bestproxys` function is the main function to generate the comparative maps. The function will display and save the maps comparing the F1 and F2 scores 
for different meteorological indices.

Example:
map_bestproxys()
"""

#######################################################################
#######################################################################
### LIBRAIRIES
import matplotlib
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import xarray as xr
import seaborn as sns; 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

#######################################################################
#######################################################################
### SET SEARBORNS PARAMETERS
sns.set(style="white", color_codes=True)
sns.set(rc={'figure.figsize':(28, 16)})

#######################################################################
#######################################################################
##### DISPLAY
map_bestproxys()

#######################################################################
#######################################################################
### FUNCTION (SIN LI)
def map_bestproxys():
    """
    Generates comparative maps for different meteorological indices based on F1 and F2 scores.

    Parameters:

    Returns:
    Graphics: Displays and saves comparative maps for F1 and F2 scores across different meteorological indices.
    """

    mode = ['f1', 'f2']
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), subplot_kw={'projection': ccrs.LambertConformal()})

    for idx, f in enumerate(mode):
        # OPEN FILE
        ds_all = xr.open_mfdataset(f"/ds_all/best_{f}score_step4/{f}best_all_variables.nc4")

        # EXTRACT VARIABLES
        kx_array = ds_all["kx"]
        totalx_array = ds_all["totalx"]
        cape_array = ds_all["cape"]
        cp_array = ds_all["cp"]
        capecp_array = ds_all["capecp"]

        # INITIALISATION OF THE DATASETS
        color = np.full((len(ds_all.latitude), len(ds_all.longitude)), np.nan)
        fvalue = np.full((len(ds_all.latitude), len(ds_all.longitude)), np.nan)

        # LOOP THROUGH EACH ELEMENT OF THE COLOR MATRICES
        for y in range(len(color)):
            for i in range(len(color[0])):
                # Extract the values of the various meteorological variables at the position (y, i)
                kx = np.array(kx_array[y, i]).item()
                totalx = np.array(totalx_array[y, i]).item()
                cape = np.array(cape_array[y, i]).item()
                cp = np.array(cp_array[y, i]).item()
                capecp = np.array(capecp_array[y, i]).item()
                # Create a dictionary to link each value to its variable name.
                maxvar = {kx: "kx", totalx: "totalx", cape: "cape", cp: "cp", capecp: "capecp"}
                # Find the maximum value among the variables.
                maxx = max(maxvar)
                # Get the name of the variable that has the maximum value.
                var_max = maxvar.get(max(maxvar))
                # Assign specific values to the 'color' and 'fvalue' matrices based on the maximum variable.
                if var_max == "kx":
                    color[y, i] = 1
                    fvalue[y, i] = maxx
                if var_max == "totalx":
                    color[y, i] = 2
                    fvalue[y, i] = maxx
                if var_max == "cape":
                    color[y, i] = 3
                    fvalue[y, i] = maxx
                if var_max == "cp":
                    color[y, i] = 4
                    fvalue[y, i] = maxx
                if var_max == "capecp":
                    color[y, i] = 5
                    fvalue[y, i] = maxx
                    
                # Handle NaN values: if the maximum value is NaN, assign NaN to 'color' and 'fvalue' matrices.
                if np.isnan(maxx):
                    color[y, i] = np.nan
                    fvalue[y, i] = np.nan
                
        #####################################################################################
        ### MAIN IMAGE SETUP ###
        ### COLOR MAP
        col_dict = {1: "tomato",
                    2: "pink",
                    3: "firebrick",
                    4: "aqua",
                    5: "greenyellow"}
        cm = ListedColormap([col_dict[x] for x in sorted(col_dict.keys())])
        labels = np.array(["kx", "totalx", "cape", "cp", "capecp"])
        len_lab = len(labels)

        ### NORMALIZER
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[int(x)-1])

        ### PARAMETERS
        projection = ccrs.LambertConformal()
        map_projection = ccrs.PlateCarree()

        ### DATAS
        mm = axes[idx].pcolormesh(ds_all.longitude, ds_all.latitude, color, vmin=1, vmax=5, cmap=cm, norm=norm,
                                  transform=ccrs.PlateCarree())

        ### BOUNDS COORDS
        axes[idx].set_extent([-90, -50, 45, 55])

        ### LAYERS
        states_provinces = cfeature.NaturalEarthFeature(category='cultural', 
                                                        name='admin_1_states_provinces_lines',
                                                        scale='50m', facecolor='none')
        land = cartopy.feature.NaturalEarthFeature('physical', 'land',
                                                   scale='50m', edgecolor=None,
                                                   facecolor=cfeature.COLORS['land'])
        axes[idx].add_feature(land, facecolor='beige', zorder=0) 
        axes[idx].add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.75)
        axes[idx].add_feature(cfeature.LAKES.with_scale('50m'))
        axes[idx].add_feature(cfeature.RIVERS.with_scale('50m'))
        axes[idx].add_feature(states_provinces, edgecolor='dimgrey')
        axes[idx].add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.75, edgecolor='black')
        axes[idx].add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, edgecolor='black')

        ### GRIDLINES
        gl = axes[idx].gridlines(crs=map_projection, draw_labels=False,
                                 linewidth=2, color='gray', alpha=0.35, linestyle='--',
                                 x_inline=False, y_inline=False)

        # Define the position of lines for lat/lon
        gl.xlocator = mticker.FixedLocator(np.arange(start=-85, stop=-55, step=10))
        gl.ylocator = mticker.FixedLocator(np.arange(start=40, stop=55, step=5))

        # Deactivate the rotating labels
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

        # Label formatting
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Labels size and color
        gl.xlabel_style = {'size': 20, 'color': 'black'}
        gl.ylabel_style = {'size': 20, 'color': 'black'}

        ## Create a divider for the axes to manage the layout of the colorbar.
        divider = make_axes_locatable(axes[idx])
        ax_cb = divider.new_horizontal(size="2.5%", pad=1, axes_class=plt.Axes)
        
        mode_maj = f.upper()
        axes[idx].text(0.05, 0.95, f'Score: {mode_maj}', transform=axes[idx].transAxes, fontsize=20,
                verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

    # Create a divider for the axes to manage the layout of the colorbar.
    divider = make_axes_locatable(axes[1])
    ax_cb = divider.new_horizontal(size="2.5%", pad=1, axes_class=plt.Axes)

    # Create normalizer and formatter
    norm_bins = np.array(norm_bins)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[int(x)-1])

    # Calculate tick positions for the colorbar based on the norm_bins
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(mm, cax=ax_cb, format=fmt, ticks=tickz, shrink=0.60)
    cb.ax.tick_params(labelsize=20)
    # Adjust the position of the colorbar to the right
    cb.ax.set_position([0.9, 0.1, 0.8, 0.8])
    # Add the colorbar axis to the figure.
    fig.add_axes(ax_cb)
    
    ### EXTEND PARAMETERS
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.30, hspace=0.05, wspace=0.10)
    
    ### SAVE
    fig.savefig('all_indices_quebec.pdf', bbox_inches='tight', dpi=200)
    
    return plt.show()