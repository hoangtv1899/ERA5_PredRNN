#!/bin/bash
#SBATCH --job-name=test        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G               # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=hvtran@princeton.edu

module load parflow-ml/tf2.2
cd /scratch/network/hvtran/era5/predrnn-pytorch/era5_script/

sh predrnn_era5_test.sh
