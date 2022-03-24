#!/bin/bash
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 6 # number of cores
#           # SBATCH -w g001   # Specific node
#SBATCH --gres=gpu:1
#       #SBATCH --mem 128G # memory pool for all cores   # Removed due to bug in Slurm 20.02.5
#SBATCH -t 1-1:00 # time (D-HH:MM)
#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR

ulimit -s 10240
mkdir -p ~/output/g001

module purge
module load slurm/20.02.7

module load cuda11.2/blas/11.2.2
module load cuda11.2/fft/11.2.2
module load cuda11.2/nsight/11.2.2
module load cuda11.2/profiler/11.2.2
module load cuda11.2/toolkit/11.2.2
module load pytorch-extra-py37-cuda11.2-gcc8/1.9.1  

source $HOME/.venv/bin/activate

srun nvidia-smi
hostname
# Example of sourcing prebuilt python venv
# . /home/<username>/py38-venv/bin/activate
srun python3 /home/feliciaj/PolypSegmentation/src/deep_ensemble.py
