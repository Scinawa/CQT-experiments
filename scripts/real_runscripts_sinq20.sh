#!/bin/bash
#SBATCH -p sinq20
#SBATCH -o slurm_sinq20.out


module load qibo

export CUDA_VISIBLE_DEVICES=0

#python3 scripts/template/main.py --device sinq20
python3 scripts/runscripts.py --device sinq20

echo "done"
