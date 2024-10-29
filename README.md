## ERA5_PredRNN

This is the repository for the paper: "Mid-range hourly weather forecasting using PredRNN with image preprocessing".

This repository contains codes to download and preprocess input data, inference scripts and the modified predrnn-pytorch for using in this application.

## Installation


If you use a GPU environment, please run:
```
pip install -r requirements_gpu.txt
```

## Global weather forecasting (inference) using the trained models

#### Downloading trained models

Please download the hourly pre-trained models (~6.2GB) from Figshare:

The 1-hour model (model_final.ckpt): [Figshare](https://figshare.com/ndownloader/files/50050722)

These models are stored using the ckpt format for being used via pytorch.

#### Input data preparation using Python

Inside the `era5_download_process` directory contains:

1. `era5_download.py` download the ERA5 dataset provided by ECMWF
2. `era5_process.py` process the download .grib files into input files required by PredRNN. For input data, it is a numpy array shaped (ii, 5,721,1440) where the first dimension represents the number of batch, the second dimension represents the 5 surface variables (U10, V10, MSLP, PRECIP, T2M **in the exact order**).

#### Inference

After the above steps are finished, please check `era5_script` for example of training and testing PredRNN. The .slurm scripts are for configuring and submitting jobs. The bash scripts (.sh) are for configuring PredRNN model. The main run script of PredRNN is located in `predrnn-pytorch/run1.py`

## PredRNN-pytorch

We have modified the original code from https://github.com/thuml/predrnn-pytorch to apply for weather forecasting. More details about the changes can be found after the under-reviewing manuscript got published.

## License

ERA5_PredRNN was released by PNNL.

The trained parameters of ERA5_PredRNN were made available under the terms of the BY-NC-SA 4.0 license. You can find details [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**The commercial use of these models is forbidden.**

Also, please note that all models were trained using the ERA5 dataset provided by ECMWF. Please do follow [their policy](https://apps.ecmwf.int/datasets/licences/copernicus/).

## References

Will be updated soon!