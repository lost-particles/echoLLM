#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -p gpu --gres=gpu:1

pip install -r requirements.txt
python -u ./Persistent_LLM_Rewards.py --num_eps 100 --save_every 20 --max_dynamic_tokens 4096 --sliding_summary_window 10