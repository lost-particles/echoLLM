
#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -p gpu --gres=gpu:1
pip install -r requirements.txt
pip install huggingface_hub
pip install transformers
pip install torch
pip install langchain_huggingface
python -u ./reai_langchain_llm.py
