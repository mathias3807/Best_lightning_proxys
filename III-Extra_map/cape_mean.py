#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Generate Maps CAPE mean from ERA5 Data
Usage: This script processes ERA5 data to generate maps showing the CAPE mean
Authors: Mathias Ponton
Created on 2024/06/12

Description:
This script processes ERA5 reanalysis data and applies geographic masks to create visualizations of lightning strike density. The script uses Cartopy and Matplotlib to generate maps, applying custom colormaps, grids, and colorbars based on the input data.

Libraries required:
- matplotlib
- cartopy
- numpy
- xarray
- pandas
- seaborn
- mpl_toolkits.axes_grid1
- warnings
- matplotlib.colors

How to use:
The main functions in this script allow you to load and process data, apply geographic masks, and generate maps that display the density of lightning strikes or other meteorological data. The `create_map` function can be used to customize the appearance of the maps, including the color scheme, geographic extent, and title.

Example:
mask = mask_data('/mask.nc4', 'bdns1')
ds = subset_data('/ds.nc', 'bdns1')
lon, lat, masked_data = final_dataset(mask, ds)
create_map(lon, lat, lon_min, lon_max, lat_min, lat_max, masked_data, 'density', 0, 5, 0.5, 0.5, 
           'Density of strikes on km², May to October', output_path='cape_mean_20152023.pdf')
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
import pandas as pd
import seaborn as sns; 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap

#######################################################################
# WARMING IGNORES
warnings.filterwarnings("ignore")

#######################################################################
#######################################################################
### DISPLAY (ACCORDING BNDS AS YOU FILES)

# Define latitude and longitude
lon_min = -148 #-80
lon_max = -50 #-55
lat_min = 40 #45
lat_max = 84 #55
lon_inc = 20 #5
lat_inc = 10 #5

#######################################################################
#######################################################################
### FUNCTIONS
### FUNCTIONS

def mask_data(filepath, lat_min, lat_max, lon_min, lon_max, boundary):
    """
    Load a set of NetCDF files and applies a geographic mask based on provided bounds.

    Args:
        filepath (str): Path to the NetCDF file(s).
        boundary (str): Type of boundary conditions.

    Returns:
        xarray.Dataset: Dataset masked according to the specified geographic bounds.
    """
    # Load the NetCDF file or files
    dataset = xr.open_mfdataset(filepath)
    
    # Check if 'lat' and 'lon' are in the dataset
    if 'lat' in dataset.coords and 'lon' in dataset.coords:
        dataset = dataset.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    # Boundaries
    if boundary == 'bdns1' :
        lat_bnd = [lat_max, lat_min]
        lon_bnd = [lon_min, lon_max]
    elif boundary == 'bdns2' :
        lat_bnd = [lat_min, lat_max]
        lon_bnd = [360+lon_min, 360+lon_max]
    
    # Select the subset of data based on the geographic bounds
    masked_dataset = dataset.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd))
    
    return masked_dataset

#######################################################################
def subset_data(filepath, lat_min, lat_max, lon_min, lon_max, boundary):
    """
    Load a NetCDF file, optionally rename coordinates, adjust longitudes, and subset the data by geographic bounds.

    Args:
        filepath (str): Path to the NetCDF file.
        boundary (str): Type of boundary conditions.

    Returns:
        xarray.Dataset: Dataset subsetted and adjusted according to specified geographic bounds.
    """
    # Load the NetCDF file
    dataset = xr.open_mfdataset(filepath)

    # Check if 'lat' and 'lon' are in the dataset
    if 'lat' in dataset.coords and 'lon' in dataset.coords:
        dataset = dataset.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    # Adjust longitudes if needed to ensure they are in a -180 to 180 range
    # dataset = dataset.assign_coords(lon=(((dataset.lon + 180) % 360) - 180)).sortby('lon')
    
    # Boundaries
    if boundary == 'bdns1' :
        lat_bnd = [lat_min, lat_max]
        lon_bnd = [lon_min, lon_max]
    elif boundary == 'bdns2' :
        lat_bnd = [lat_max, lat_min]
        lon_bnd = [360+lon_min, 360+lon_max]
    
    # Subset the data based on geographic bounds
    subset_dataset = dataset.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd))
    
    return subset_dataset

#######################################################################
def final_dataset(mask, ds):
    """
    Load data and a mask file, then apply an ecozone-based mask to the data.

    Args:
        mask (str): Path to the NetCDF file containing the reults of the function mask_data.
        ds (str): Path to the NetCDF file containing the reults of the function subset_data.

    Returns:
        xarray.DataArray: Data masked according to specified ecozones.
    """
    
    # Extract the variable of interest and calculate its mean along the first axis
    years = np.arange(2015, 2023, 1)
    data = ds.sel(time=ds['time'].dt.year.isin(years))
    data = data.sel(time=data['time'].dt.month.isin([5, 6, 7, 8, 9, 10]))
    data = ds.variables['cape'][:].mean(axis=0)
    
    # Ensure 'lat' and 'lon' are in the dataset
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
    else:
        print("Dataset does not contain 'latitude' and 'longitude' coordinates.")
        return None
    
    # Regions
    region = [1]
    
    # Apply the mask for selected ecozones
    condition = mask.region.isin(region)
    masked_data = data.where(condition)

    return lon, lat, masked_data

#######################################################################
def create_map(lon, lat, lon_min, lon_max, lat_min, lat_max, masked_data, option, data_min, data_max, data_increment, colorbar_increment, units,
               fig_title='', output_path='output_map.png'):
    """
    Creates a geographical visualization using Cartopy with custom settings.

    Args:
        lon (array): Array of longitudes.
        lat (array): Array of latitudes.
        lon_min (float): Minimum longitude for the map extent.
        lon_max (float): Maximum longitude for the map extent.
        lat_min (float): Minimum latitude for the map extent.
        lat_max (float): Maximum latitude for the map extent.
        utis (str) : Title of the colorbar.
        masked_data (array): Data array that matches the lat and lon dimensions.
        fig_title (str): Title of the figure.
        output_path (str): File path to save the figure.
    """
    
    ### PARAMETERS
    projection = ccrs.LambertConformal()
    map_projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(28, 10))
    ax = plt.subplot(111, projection=projection)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    
    ### DEFINE CMAP
    # Create a custom color map
    colors = ["blue", "cyan", "green", "yellow", "orange", "red"]
    n_bins = 30
    custom_cmap = LinearSegmentedColormap.from_list("custom1", colors, N=n_bins)

    # LAYERS
    states_provinces = cfeature.NaturalEarthFeature(category='cultural', 
                                                    name='admin_1_states_provinces_lines',
                                                    scale='50m', facecolor='none')
    land = cartopy.feature.NaturalEarthFeature('physical', 'land',
                                               scale='50m', edgecolor=None,
                                               facecolor=cfeature.COLORS['land'])
    ax.add_feature(land, facecolor='beige',zorder=0) 
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.75)
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(states_provinces, edgecolor='dimgrey')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth =0.75, edgecolor = 'black')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth =0.75, edgecolor = 'black')

    ### DATAS
    #mm = ax.contourf(lon,\
    #                   lat,\
    #                   masked_data,\
    #                   vmin=data_min,\
    #                   vmax=data_max, \
    #                   transform=map_projection,\
    #                   levels=np.arange(data_min, data_max, data_increment),\
    #                   cmap=plt.cm.jet,\
    #                   extend='max' )
    
    mm = ax.pcolormesh(lon, lat, masked_data, vmin=data_min, vmax=data_max, cmap=custom_cmap,
                      transform=map_projection)



    ### GRIDLINES
    gl = ax.gridlines(crs=map_projection, draw_labels=False,
                      linewidth=2, color='gray', alpha=0.35, linestyle='--',
                      x_inline=False, y_inline=False)

    # Define the position of lines for lat/lon
    gl.xlocator = mticker.FixedLocator(np.arange(start=lon_min, stop=lon_max, step=20))
    gl.ylocator = mticker.FixedLocator(np.arange(start=lat_min, stop=lat_max, step=10))

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

    # Labels fromatage
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Labels size and color
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}



    ### COLORBAR
    # extend = 'max' permet d'avoir la fléche à droite
    # drawedges est la discrétisation
    if option == 'number' :
        cbar = plt.colorbar(mm, orientation='horizontal', shrink=0.30, drawedges='True', extend='max',
                        ticks=np.arange(data_min, data_max, colorbar_increment), pad=0.08)
    elif option == 'density':
        cbar = plt.colorbar(mm, orientation='horizontal', shrink=0.30, drawedges=True, extend='max',
                        ticks=np.arange(data_min, data_max, colorbar_increment), pad=0.08)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        cbar.ax.xaxis.set_major_formatter(formatter)
        
        # Customizing exponent text
        cbar.ax.xaxis.get_offset_text().set_fontsize(15)  # Adjust the size of the exponent
        cbar.ax.xaxis.get_offset_text().set_position((0.96, 0))  # Adjust the position

    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(units, size = 15)
    
    ax.text(0.05, 0.95, 'Period: 2010-2019', transform=ax.transAxes, fontsize=15,
        verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))
    
    # Set title and save the plot
    plt.title(fig_title, size='xx-large')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()