#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -p gpu --gres=gpu:1

# Install requirements in the virtual environment
pip install -r requirements.txt
pip install huggingface_hub
pip install transformers
pip install torch
python -u ./Persistent_LLM_Rewards.py