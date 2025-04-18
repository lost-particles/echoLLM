#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -p gpu --gres=gpu:1
python -u ./'Reintegrating AI'/script1.py
