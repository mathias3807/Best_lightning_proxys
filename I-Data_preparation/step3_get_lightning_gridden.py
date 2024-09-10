#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Process and Grid Lightning Strikes Data on ERA5 Grid
Usage: This script processes lightning strike data from Hydro-Québec (Oragélect) and grids it onto the ERA5 grid.
Authors: Mathias Ponton
Created on 2024/09/10

Description:
This script reads CSV data of lightning strikes over Québec, filters the data within a specific time range (1996-2024), and assigns each lightning strike to a grid point based on ERA5 0.25° grid. The script then counts the strikes per grid point and day, and generates a netCDF dataset containing the results. Finally, the dataset is sliced to focus on the Québec region and saved to a file.

Libraries required:
- pandas
- xarray
- numpy
- dask
- datetime
- pytz

How to use:
1. Ensure that the CSV file with lightning strike data is available in the specified directory.
2. Adjust the `time_start` and `time_end` variables to match the desired time range.
3. The `create_grid_coordinates` function creates grid parameters based on ERA5 lat/lon values.
4. The script assigns each lightning strike to a grid point, counts the strikes, and stores the data in a dataset.
5. The result is saved as a netCDF file (`quebec_hq_0.25deg_dataset_1996_2024.nc4`), which can be used for further analysis.

Example:
The script reads lightning data, grids it onto the ERA5 grid, counts occurrences, and creates a dataset. It can be modified to adjust the time range or region of interest.

Steps:
1. Load the lightning strike data using `pd.read_csv`.
2. Filter the data within the specified time range.
3. Assign grid coordinates for each strike using ERA5 grid.
4. Group the data by time and grid points to count strikes.
5. Create an xarray dataset and save it to a netCDF file.

"""

#######################################################################
#######################################################################
### LIBRAIRIES
import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da
import datetime
import pytz

#######################################################################
#######################################################################
### FUNCTION
def create_grid_coordinates(ds):
    """
    Create grid coordinates based on the min/max values of latitude and longitude in the dataset.
    
    Args:
        ds: xarray Dataset containing 'lon' (longitude) and 'lat' (latitude) variables.
        
    Returns:
        dict: A dictionary with grid parameters including:
            - xsize: Number of points in the x (longitude) direction.
            - ysize: Number of points in the y (latitude) direction.
            - xfirst: The starting longitude value (minimum longitude).
            - xinc: The increment in the longitude direction.
            - yfirst: The starting latitude value (maximum latitude).
            - yinc: The increment in the latitude direction (typically negative).
    """
    
    # Extract the minimum and maximum values of longitudes and latitudes
    lon_min = ds['lon'].min().item()
    lon_max = ds['lon'].max().item()
    lat_min = ds['lat'].min().item()
    lat_max = ds['lat'].max().item()
    
    # Define xfirst, yfirst, xinc, and yinc
    xfirst = lon_min
    yfirst = lat_max
    
    xinc = 0.25  # Increment in the longitude direction
    yinc = -0.25  # Increment in the latitude direction (typically negative)
    
    # Calculate grid size based on min and max values of lon/lat
    xsize = int((lon_max - lon_min) / xinc) + 1
    ysize = int((lat_max - lat_min) / abs(yinc)) + 1

    return {
        "xsize": xsize,
        "ysize": ysize,
        "xfirst": xfirst,
        "xinc": xinc,
        "yfirst": yfirst,
        "yinc": yinc
    }

#######################################################################
#######################################################################
### MAIN CODE
### LOAD FILES
# Columns names
columns = ['Time', 'lat', 'lon', 'polarity']
df = pd.read_csv(f'/tornado/ponton/foudre_data/ressources/FoudreHQ_1996-Sept2024.csv', sep=';', header=None, names=columns)

time_start = '1996-04-05'
time_end = '2024-09-01'

df['Time'] = pd.to_datetime(df['Time'])

df = df[(df['Time'] >= time_start) & (df['Time'] <= time_end)]

#######################################################################
### RUN grid_params
grid_params = create_grid_coordinates(ds)

# Extraction of parameters
xfirst = grid_params['xfirst']
xsize = grid_params['xsize']
xinc = grid_params['xinc']
yfirst = grid_params['yfirst']
ysize = grid_params['ysize']
yinc = grid_params['yinc']

#######################################################################
### VECTORS OF LAT/LON
lon = np.arange(xfirst, xfirst + xsize * xinc, xinc)
lat = np.arange(yfirst, yfirst + ysize * yinc, yinc)

#######################################################################
### ASSIGN EACH LIGHTNING STRIKE TO A GRID POINT
df['lon_idx'] = np.floor((df['lon'] - xfirst) / xinc).astype(int)
df['lat_idx'] = np.floor((df['lat'] - yfirst) / yinc).astype(int)

#######################################################################
### COUNT STRIKES PER POINTS/DAYS
# Group for dates, grid index and occurences
grouped = df.groupby([df['Time'].dt.date, 'lat_idx', 'lon_idx']).size()

#######################################################################
### CREATE DATASET
f = xr.DataArray(np.zeros((len(pd.date_range(time_start, time_end)), len(lat), len(lon))),
                 coords=[pd.date_range(time_start, time_end), lat, lon],
                 dims=['Time', 'lat', 'lon'])

# Fill in the dataset with the group datas
for ((date, lat_idx, lon_idx), count) in grouped.items():
    f.loc[str(date), lat[lat_idx], lon[lon_idx]] = count

ds = xr.Dataset({'F': f})

# Current datetime creation
current_datetime_gmt = datetime.datetime.now(pytz.timezone('GMT')).strftime('%Y-%m-%d %H:%M:%S %Z')

# AAdd attributes
ds.attrs['Title'] = 'Lightning strikes on Québec.'
ds.attrs['Description'] = 'Dataset with counts of strikes on the ERA5 grid cell (0.25°) on Québec.'
ds.attrs['Source'] = f'Data of the Hydro-Québec (Oragélect) link, gridded on {current_datetime_gmt} \
                       by Mathias Ponton, Université du Québec À Montréal.'
ds['lat'].attrs['units'] = '°'
ds['lat'].attrs['long_name'] = 'latitude'
ds['lon'].attrs['units'] = '°'
ds['lon'].attrs['long_name'] = 'longitude'
ds['F'].attrs['units'] = 'counts/grid cell'
ds['F'].attrs['long_name'] = 'counts of lightning strikes'

#######################################################################
### SLICING
lat_bnd = [65, 40]
lon_bnd = [-85, -50]

ds_sliced = ds.sel(lat=slice(lat_bnd[0], lat_bnd[1]), lon=slice(lon_bnd[0], lon_bnd[1]))
ds_sliced

#######################################################################
ds_sliced.to_netcdf('/tornado/ponton/foudre_data/quebec_hq_0.25deg_dataset_1996_2024.nc4')