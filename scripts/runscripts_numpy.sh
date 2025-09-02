#!/bin/bash
#SBATCH -p gpu
#SBATCH -o slurm_gpu.out

module load qibo

export CUDA_VISIBLE_DEVICES=0

python3 scripts/scripts_executor.py --device numpy
