#!/bin/bash
#SBATCH -p sinq20
#SBATCH -o slurm_sinq20.out

module load qibo

export CUDA_VISIBLE_DEVICES=0

python3 scripts/scripts_executor.py --device sinq20

