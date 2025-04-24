#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -p gpu --gres=gpu:1

# Install requirements in the virtual environment
# pip install -r requirements.txt
# pip install huggingface_hub
# pip install transformers
# pip install torch
python -u ./script1.py --num_eps 1000 --save_every 100
