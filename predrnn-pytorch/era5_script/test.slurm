#!/bin/bash
#SBATCH --job-name=test        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=64G               # memory per cpu-core (4G is default)
#SBATCH -A ADCIRC
#SBATCH -p gpu-a100
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=hoang.tran@pnnl.gov

cd /work/08589/hvtran/ls6/ERA5_PredRNN-main/predrnn-pytorch/era5_script

#sh predrnn_era5_test_precip.sh

#sh predrnn_era5_test_wind.sh

#sh predrnn_era5_test_sea_press.sh

sh predrnn_era5_test.sh
