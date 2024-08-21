#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: ERA5 Data Downloader for Canada
Usage: This script automates the process of downloading ERA5 reanalysis data for specified meteorological variables and a defined geographical area (Canada). The data is stored in NetCDF format.

Author: Mathias Ponton
Created on 2024/03/26

Description:
This script downloads ERA5 reanalysis data for specific meteorological variables over a defined time period and geographical area. It uses the CDSAPI client to connect to the Copernicus Climate Data Store (CDS) and retrieve the data. The downloaded data is saved in NetCDF format in a specified directory.

Key Features:
- Variables: The script can handle multiple meteorological variables (e.g., CAPE, CP, K-index, Total Totals Index).
- Time Period: Users can define the range of years and months for which data should be downloaded.
- Geographic Area: The area of interest is defined in terms of its north, west, south, and east bounds, focusing on Canada.
- Automated Download: The script checks if a file already exists before attempting to download it, avoiding duplicate downloads.

How to Use:
1. Define the variables to download in the `variables` list.
2. Set the desired time period by adjusting `year_min`, `year_max`, `month_min`, and `month_max`.
3. Specify the area of interest with the `area` list, formatted as [N, W, S, E].
4. Set the path to the target directory where the data will be saved with `start_target`.

Example:
The script will download ERA5 reanalysis data for CAPE, CP, K-index, and Total Totals Index from 1943 to 2009, covering the area of Canada, and store the data in the specified directory.

Dependencies:
- Python libraries: datetime, pandas, numpy, os
- ERA5 data download module: cdsapi

Note: Ensure you have access to the Copernicus Climate Data Store (CDS) and have installed the `cdsapi` library.
"""


############################################################################################
# IMPORT MODULES

# Modules for Python
from datetime import date
import calendar
import pandas as pd
import numpy as np
import os.path
import os

# Modules for downloading ERA5 data
import cdsapi
c = cdsapi.Client()

###########################################################################################
# MODIFICATION AREA

# Definition of the variables to be downloaded
variables = ["cape", "cp", "k_index", "total_totals_index"]

# Definition of the time period to be used
year_min = 1943
year_max = 2009
month_min = 1
month_max = 12

# Definition of the area of interest (in N/W/S/E), here it's for Canada
area = [84, -148, 40, -50]

# Start of the target
start_target = "/your_path_target/"

###########################################################################################
# MAIN SCRIPT

for variable in variables:
    for year in range(year_min, year_max + 1):
        year_str = str(year)
        for month in range(month_min, month_max + 1):
            month_str = str("{:02}".format(month))
            target = start_target + f"{variable}_canada/{variable}_era5_" + year_str + month_str + ".nc4"
            if not os.path.exists(target):
                print(f"* The file for {variable} in {year_str}/{month_str} is going to be downloaded...")
                c.retrieve('reanalysis-era5-single-levels',
                           {'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': variable,
                            'year': [year_str],
                            'month': [month_str],
                            'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                                    '25', '26', '27', '28', '29', '30', '31'],
                            'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00',
                                     '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                                     '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                            'area': area},
                           target)
                print(f"*** The file for {variable} in {year_str}/{month_str} has been downloaded.")
                
            else:
                print(f"/!\ The file for {variable} in {year_str}/{month_str} already exists and has not been downloaded.")
