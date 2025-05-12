#!/bin/bash
#SBATCH --time=24:00:00 # total time allocated
#SBATCH --mem-per-cpu=16000 # memory per CPU core in MB
#SBATCH --output=gnn_job.out # specify a file to direct standard output stream,
#SBATCH --account=dslab_jobs # mandatory course tag
#SBATCH --error=gnn_job.err # specify a file to direct standard error stream
#SBATCH --open-mode=truncate # truncate overwrites output and error files, append just appends
#SBATCH --mail-type=START,END,FAIL
# ./etc/profile.d/modules.sh
# start using module command, for example:
module add cuda/12.6
eval "$(conda shell.bash hook)"
conda activate deeplsd
cd "${HOME}/3D-Vision/DeepLSD/notebooks/lightning_tools" # path to your project folder
# your command
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
srun python train_merged.py --config=config_merged.yaml