# Best_lightning_proxys
Codes for a cross validation method to give a result for the best proxys of lightning.
============
This project aims to improve forest fire forecasting in relation to severe thunderstorm prediction using convection indices. It includes scripts for data preparation, score calculation, and map generation.

Installation
------------
Requirements: Python 3.8+, NumPy, Matplotlib, Cartopy, xarray

Usage
-----
1. I-Data_preparation:
    - Download ERA5 data:
      python I-Data_preparation/download_era5.py --input raw_data.nc --output prepared_data.nc
    - Bilinear interpolation to WWLLN grid:
      bash I-Data_preparation/interpolation.sh

2. II-Scores_calculations:
    - Calculate metrics and scores from ERA5 data and observations:
      python II-Scores_calculations/step1_allscores_allvariables.py --input prepared_data.nc --output scores.nc
    - Generate maps from calculated scores:
      python III-Extra_map/step2_mapscores.py --input scores.nc --output map.png
    - Generate additional maps for each variable and score calculated:
      python II-Scores_calculations/step2_mapscores_grid.py --input scores.nc --output grid_map.png
    - Generate F1 score maps for multiple meteorological indices:
      python II-Scores_calculations/step3_f1_all_indices.py --input score_data_step1/ --output best_f1score_step4/
    - Generate F2 score maps for multiple meteorological indices:
      python II-Scores_calculations/step3_f2_all_indices.py --input score_data_step1/ --output best_f2score_step4/
    - Generate comparative maps for meteorological indices based on F1 and F2 scores:
      python II-Scores_calculations/step4_trace_bestf_following_indices.py --input best_f1score_step4/ --output comparative_maps/

3. III-Extra_map:
    - Generate lightning density maps for Canada and Quebec:
      python III-Extra_map/map_lightning.py --input era5_data.nc --output lightning_density_map.png
    - Generate mean CAPE maps from ERA5 data:
      python III-Extra_map/cape_mean.py --input era5_data.nc --output cape_mean_20152023.pdf

Folder Structure
----------------
1. I-Data_preparation:
   - download_era5.py: Downloads ERA5 data for specified meteorological variables.
   - interpolation.sh: Performs bilinear interpolation of ERA5 data to a 1° grid.

2. II-Scores_calculations:
  - step1_allscores_allvariables.py: Calculates scores based on convection indices and performs cross-validation.
  - step2_mapscores.py: Generates additional maps from calculated scores.
  - step2_mapscores_grid.py: Creates maps for each variable and score calculated using data from step 1. Maps show the best F1 score values, precision, threshold, and accuracy.
  - step3_f1_all_indices.py: Calculates and generates F1 score maps for various meteorological variables, including CAPE, CP, CAPE x CP, Kx, and TTX. F1 scores are calculated for a range of thresholds and saved in NetCDF files. The script loops through each variable, calculates precision, recall, and F1 score, and saves results in a structured format.
  - step3_f2_all_indices.py: Calculates and generates F2 score maps for various meteorological variables, including CAPE, CP, CAPE x CP, Kx, and TTX. F2 scores are calculated for a range of thresholds and saved in NetCDF files. The script loops through each variable, calculates precision, recall, and F2 score, and saves results in a structured format.
  - step4_trace_bestf_following_indices.py: Generates comparative maps for meteorological indices based on F1 and F2 scores. Maps display the dominant variable for each grid point and highlight regions where each variable achieves the highest score. The script creates maps with a legend for each meteorological index, using specific colors for each variable and displaying associated maximum values.

3. III-Extra_map:
   - map_lightning.py: Generates lightning density maps from ERA5 data for Canada and Quebec. The script applies geographic masks to create lightning density visualizations, using Cartopy and Matplotlib for visualization. It includes options for customizing maps, such as colormaps, grids, and colorbars.
   - cape_mean.py: Generates mean CAPE maps from ERA5 data. This script processes ERA5 data, applies geographic masks, and creates visualizations showing mean CAPE for the years 2015 to 2023. It uses Cartopy and Matplotlib to customize map appearance, including options for colormaps and colorbars.

Contributing
------------
Contributions are welcome! Please submit a pull request or open an issue.

License
-------
This project is licensed under the MIT License - see the LICENSE file for details.

