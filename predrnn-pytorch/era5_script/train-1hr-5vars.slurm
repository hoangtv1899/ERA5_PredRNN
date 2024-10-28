#!/bin/bash
#SBATCH --job-name=train_WO_Exp        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH -A m4513
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

cd /global/u2/h/hvtran/haoli/ERA5_PredRNN/predrnn-pytorch/era5_script

module load pytorch/2.0.1
pip install xarray,lpips,pytorch_wavelets,PyWavelets 
sh predrnn_era5_train_1hr.sh
