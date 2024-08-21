# !/bin/bash

# Created on 2024 Mar 14
# Author: Mathias Ponton
# Subject: Bilinear interpolation for ERA5 grid from 0.25° to WWLLN grid at 1°
# Usage: Change the grid to 1° and reposition it to match the WWLLN grid points
# /!\ BEWARE OF FILE EXTENSIONS THAT MAY VARY .nc or .nc4

####################################################################################
### INPUT
# Change according to the needs of the grid usage
path_WWLLN="/path_wwlln/WWLLN_1deg1d_dataset_2010-2019_canada.nc"
output_grid="/output_grid/grid_WWLLN_1deg.txt"

### TARGET VARIABLES
variables=("cape" "cp" "k_index" "total_totals_index")

####################################################################################
### BOUNDARIES
year_min=2010
year_max=2023
month_min=01
month_max=12

####################################################################################
### SCRIPT
# Step 1: Create a reference grid description file
cdo griddes $path_WWLLN > $output_grid

### LOOP
for var in "${variables[@]}"; do
    target_dir="/target_dir/${var}/"
    for year in $(seq $year_min $year_max); do
        for month in $(seq -w $month_min $month_max); do
            # Define the input path for the variable in the current iteration
            path_var="/path_var/${var}_canada/${var}_era5_${year}${month}.nc4"
            
            # Define the output path for the variable in the current iteration
            output_var="${target_dir}${var}_era5_${year}${month}_1deg.nc4"
            
            # Display a message indicating the file being processed
            echo "Processing file: $path_var"
            
            # Step 2: Modify longitude and latitude attributes in the NetCDF file
            # - a: add a new attribute
            ncatted -a units,longitude,o,c,"degreeE" -a units,latitude,o,c,"degreeN" $path_var
            
            # Step 3: Resample the file to the WWLLN reference grid
            cdo remapbil,$output_grid $path_var $output_var
        done
    done
done