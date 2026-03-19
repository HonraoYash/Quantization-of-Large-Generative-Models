# Quantization of Large Generative Models

This repository contains the code, experiments, and artifacts for our course project on **quantizing large generative models**. We study how far we can push quantization before models like SDXL, Kandinsky, FLUX.1-schnell, and ParlerTTS “break” visually or perceptually, and we propose a lightweight **Timestep-Aware Quantization (TAQ)** strategy that adjusts activation bit-width across diffusion / flow steps.

The repo is meant to be readable and runnable both on a **single GPU locally** and on **TAMU HPRC Grace**.

---

## Problem Statement

Modern text-to-image and generative speech models are extremely capable, but their FP16/FP32 footprints routinely overflow consumer GPUs and even free-tier cloud GPUs, leading to out-of-memory errors for simple prompts. Our goals are:

1. **Quantify** how standard post-training quantization (FP32→FP16/INT8, GGUF variants, etc.) affects quality and speed.
2. **Design a simple, practical scheme** (TAQ) that varies activation precision over diffusion timesteps to reduce memory and latency while minimizing perceptual degradation.
3. **Demonstrate end-to-end workflows** for running these quantized models both on local machines and on HPRC Grace.

---

## Repository Structure

Each top-level folder corresponds to one model family and contains its own notebooks / scripts:

```text
.
├── flux/             # FLUX.1-schnell quantization + GGUF experiments (Grace + local)
├── kandinsky/        # Kandinsky 2.2 post-training + timestep-aware quantization
├── parlertts/        # ParlerTTS audio generation + quantization experiments
├── sdxl/             # SDXL 1.0 PTQ and TAQ experiments
├── README.md         # This file
├── FinalReport.pdf   # Full research-style writeup
└── HowToRun.pdf      # Detailed run instructions (local + HPRC)
```

Inside each model folder you will typically find:

- `*_baseline.ipynb` – original FP16 / FP32 pipeline.
- `*_quantization.ipynb` – post-training INT8 / GGUF weight quantization.
- `*_timestep_aware.ipynb` – Timestep-Aware Quantization experiments and evaluation.
- `results/` – saved metrics, plots, and example images (e.g., the “frog” comparison grid for FLUX).

---

## Documentation

- **`FinalReport.pdf`** – research-style report explaining motivation, methodology (including TAQ), experiments, and results across all four models.
- **`HowToRun.pdf`** – step-by-step instructions for:
  - setting up Python environments and Hugging Face access,
  - running each model **locally** (e.g., single-GPU workstation / Colab),
  - running FLUX / SDXL / Kandinsky / ParlerTTS on **HPRC Grace** using Slurm (including example job scripts).

If you are trying to reproduce results or just generate images/audio, *start with `HowToRun.pdf`* and then open the corresponding notebook in the model folder.

---

## Experiments (High-Level)

Across the four model families we implement and evaluate:

- **Baseline precision**  
  FP32 / FP16 reference pipelines.

- **Post-training quantization (PTQ)**  
  - Weight-only INT8 / INT4 (where supported).  
  - GGUF variants (Q2_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q8_0) for FLUX.1-schnell.

- **Timestep-Aware Quantization (TAQ)**  
  - PyTorch-hook based activation quantization.  
  - More aggressive precision in early noise-dominated steps.  
  - Higher precision restored in late refinement steps.

We report FID, LPIPS, PSNR, SSIM, CLIP alignment (where applicable), as well as approximate **memory reduction vs. FP16** and **speed-ups** on realistic hardware (Colab T4, Grace A100).

---

## Quick Start (Very Short)

1. **Clone the repo**

   ```bash
   git clone <PRIVATE-URL>/Quantization-of-Large-Generative-Models.git
   cd Quantization-of-Large-Generative-Models
   ```

2. **Read `HowToRun.pdf`**

   - Choose the model you care about (`flux/`, `sdxl/`, `kandinsky/`, `parlertts/`).
   - Follow the corresponding environment + run instructions.

3. **Open the notebook**

   Example: FLUX (quantized GGUF + Grace run):

   - `flux/FLUX1_schnell_quantization.ipynb`
   - `flux/FLUX1_schnell_grace_slurm_example.ipynb`

The notebooks are written to be self-contained: each one loads the model, applies the quantization strategy, runs a small set of prompts, and saves sample outputs.

---

## Contributors

- **Yash Honrao** – repository structure, Kandinsky experiments, metrics logging, SDXL experiments, evaluation scripts 
- **Hitha Magadi Vijayanand** – diffusion literature review 
- **Dishant Parag Zaveri** – FLUX.1-schnell GGUF pipeline, HPRC integration, FLUX “frog’’ study  
- **Waris Quamer** – ParlerTTS quantization, audio evaluation, report integration  

Texas A&M University — CSCE 689 / GenAI project on *Quantization of Large Generative Models*.

---

## Acknowledgements

We thank the TAMU **HPRC (Grace)** team for compute resources and support, and the authors of SDXL, Kandinsky, FLUX, and ParlerTTS for releasing high-quality open models and tooling that made this project possible.
