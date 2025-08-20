#!/bin/bash
#SBATCH -p gpu
#SBATCH -o slurm_gpu.out


module load qibo

export CUDA_VISIBLE_DEVICES=0

# for string in ["sinq20", "numpy"]
# sbatch  -p string
python3 scripts/runscripts.py --device numpy

echo "done"
