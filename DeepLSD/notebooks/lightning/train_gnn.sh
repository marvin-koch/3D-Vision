#!/bin/bash
#SBATCH --time=24:00:00 # total time allocated
#SBATCH --mem-per-cpu=16000 # memory per CPU core in MB
#SBATCH --output=gnn_job.out # specify a file to direct standard output stream,

#SBATCH --account=3dv # mandatory course tag
#SBATCH --error=gnn_job.err # specify a file to direct standard error stream
#SBATCH --open-mode=truncate # truncate overwrites output and error files, append just appends
#SBATCH --mail-user=maurdu@ethz.ch
#SBATCH --mail-type=START,END,FAIL

# ./etc/profile.d/modules.sh
# start using module command, for example:
module add cuda/12.6
eval "$(conda shell.bash hook)"
conda activate deeplsd
cd "${HOME}/3D-Vision/DeepLSD/notebooks/lightning" # path to your project folder
# your command
python train.py