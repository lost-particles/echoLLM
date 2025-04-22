#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -p gpu --gres=gpu:1

set -e  # Stop on error

# Load modules if needed
# module load python/3.10

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install packages
pip install --upgrade pip
pip install -r requirements.txt

# Optional: check what's installed
pip list

# Run your script
python -u ./reai_langchain_llm.py
