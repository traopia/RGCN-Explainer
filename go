#!/bin/bash
#SBATCH --job-name=rgcn_expl_mdgenre_1node
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --mem=164000M
#SBATCH --gpus-per-task=4
#SBATCH --output=rgcn_expl_mdgenre_1node

python3 RGCN_stuff/r_exp.py
#python3 rgcn_torch.py
