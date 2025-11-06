#!/bin/bash
#SBATCH -p faster
#SBATCH -J flux-bento
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module purge
module load Miniconda3/23.10.0-1
eval "$(micromamba shell hook --shell bash)"
micromamba activate /scratch/user/dishant.zaveri/genai/env/bentodiff

export HF_HOME=/scratch/user/dishant.zaveri/.cache/huggingface

cd /scratch/user/dishant.zaveri/genai/flux-timestep-distilled

python - << 'PY'
from quantize import FluxTimestepDistilled
svc = FluxTimestepDistilled()
svc.setup_pipeline()
img = svc.txt2img(prompt="A cinematic shot of a baby raccoon wearing an intricate Italian priest robe")
img.save("flux_out.png")
print("saved flux_out.png")
PY
