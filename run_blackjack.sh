#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -p gpu --gres=gpu:1

pip install -r requirements.txt
python -u ./blackjack.py