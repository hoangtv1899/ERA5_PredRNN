#!/bin/bash
#SBATCH --job-name=train        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=64G               # memory per cpu-core (4G is default)
#SBATCH -p gpu-a100
#SBATCH -A DMS21038
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=hoang.tran@pnnl.gov

cd /work/08589/hvtran/ls6/ERA5_PredRNN-main/predrnn-pytorch/era5_script

sh predrnn_era5_train_wind.sh
