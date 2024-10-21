# Codes to (1) download test and validation ERA5 data and (2) run Pangu-Weather and GraphCast models

## To run Pangu-Weather model for the year 2023

1. Modify and run file `ERA5_raw_input_download.py` to download ERA5 input data

2. Modify and run file `Pangu_input_data_processing.py` to process raw ERA5 data into required input format by Pangu 

3. Modify and run file `Pangu_Inference.py` to generate results file for the year 2023. Pangu pretrained models are from https://drive.google.com/file/d/1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S/view?usp=share_link

## To run GraphCast model for the year 2023

1. Modify and run file `ERA5_raw_input_download.py` to download ERA5 input data

2. Modify and run file `GraphCast_input_data_processing.py` to process raw ERA5 data into required input format by GraphCast 

3. Modify and run file `GraphCast_Inference.py` to generate results file for the year 2023. GraphCast pretrained models, stat files are from https://console.cloud.google.com/storage/browser/dm_graphcast
	We use the GraphCast_operational model

## To validate Pangu-Weather outputs for the year 2023

1. Modify and run file `ERA5_raw_validation_download.py` to download ERA5 validation data

2. Modify and run file `ERA5_validation_data_processing.py` to process raw ERA5 data into validation numpy files

3. Modify and run file `Pangu_validation_6h.py` to generate results netcdf file for the year 2023.

## To validate GraphCast outputs for the year 2023

1. Modify and run file `ERA5_raw_validation_download.py` to download ERA5 validation data

2. Modify and run file `ERA5_validation_data_processing.py` to process raw ERA5 data into validation numpy files

3. Modify and run file `GraphCast_validation_6h.py` to generate results netcdf file for the year 2023.


## System requirement

- Python3.10 and up
- score.py file is obtained and modified from the original WeatherBench repository: https://github.com/pangeo-data/WeatherBench/blob/master/src/score.py


