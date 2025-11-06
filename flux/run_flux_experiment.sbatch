#!/bin/bash
#SBATCH -p faster
#SBATCH -J flux-exp
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module purge
module load Miniconda3/23.10.0-1
eval "$(micromamba shell hook --shell bash)"
micromamba activate /scratch/user/dishant.zaveri/genai/env/bentodiff

# use the cache you already set up
export HF_HOME=/scratch/user/dishant.zaveri/.cache/huggingface

cd /scratch/user/dishant.zaveri/genai/flux-timestep-distilled

python flux_quantization_experiment.py \
  --model_dir /scratch/user/dishant.zaveri/hf_models/FLUX.1-schnell \
  --prompt "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe" \
  --output_dir /scratch/user/dishant.zaveri/genai/flux-timestep-distilled/outputs \
  --height 512 --width 512 --steps 4
