#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -p gpu --gres=gpu:1
# Create and activate a virtual environment named reai-venv
python -m venv reai-venv
source reai-venv/bin/activate

# Install requirements in the virtual environment
pip install -r requirements.txt
pip install huggingface_hub
pip install transformers
pip install torch
python -u ./script1.py
