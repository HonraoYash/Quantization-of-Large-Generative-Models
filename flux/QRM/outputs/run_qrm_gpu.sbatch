#!/bin/bash
#SBATCH -p gpu
#SBATCH -J qrm
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module purge
module load Miniconda3/23.10.0-1
eval "$(micromamba shell hook --shell bash)"
micromamba activate /scratch/user/dishant.zaveri/genai/env/qrm

python /scratch/user/dishant.zaveri/genai/qrm/run_llm_quant_experiment.py \
  --model_dir /scratch/user/dishant.zaveri/hf_models/Qwen2.5-3B-Instruct \
  --out_dir   /scratch/user/dishant.zaveri/genai/qrm/outputs/test1 \
  --prompt "Explain rectified flow in FLUX to a CS undergrad." \
  --max_new_tokens 128
