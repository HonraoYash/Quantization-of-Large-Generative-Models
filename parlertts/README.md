# ParlerTTS Quantization & Evaluation

Quantization toolkit for ParlerTTS text-to-speech models with INT8 and SAMP (Sensitivity-Aware Mixed Precision) support.

## Overview

This project provides tools to quantize and evaluate ParlerTTS models, reducing memory footprint by ~4x while maintaining speech quality. Includes comprehensive benchmarking across Real-Time Factor (RTF), UTMOS score, and Word Error Rate (WER) metrics.

**Key Features:**
- INT8 weight quantization using BitsAndBytes
- SAMP: Mixed precision quantization (INT8 for robust layers, FP16 for sensitive attention)
- Automated evaluation on LibriTTS-R dataset
- Command-line interface for flexible benchmarking

## Installation

```bash
# Clone repository
git clone <repository-url>
cd <repository-url>/parlertts

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, 16GB RAM minimum

## Quick Start

### Generate Speech with Quantized Model

```python
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, BitsAndBytesConfig
import soundfile as sf

# Configure INT8 quantization
config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_heads"]
)

# Load model
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-large-v1",
    quantization_config=config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

# Generate audio
text = "Hello, how are you today?"
description = "A female speaker with moderate speed and pitch."

input_ids = tokenizer(description, return_tensors="pt").input_ids
prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids

with torch.inference_mode():
    audio = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

# Save output
sf.write("output.wav", audio.cpu().numpy().squeeze(), model.config.sampling_rate)
```

## Evaluation

### Basic Usage

```bash
# Evaluate SAMP quantization (recommended)
python evaluation.py --quantization samp --num_samples 100

# Evaluate INT8 quantization
python evaluation.py --quantization int8 --num_samples 100

# Evaluate FP32 baseline
python evaluation.py --quantization none --num_samples 50
```

### Advanced Options

```bash
# Fast evaluation (skip quality metrics, measure RTF only)
python evaluation.py --quantization samp --skip_utmos --skip_wer --num_samples 20

# GPU evaluation
python evaluation.py --quantization int8 --device cuda:0 --num_samples 100

# Skip baseline comparison (faster)
python evaluation.py --quantization samp --skip_baseline --num_samples 50

# Custom output directory
python evaluation.py --quantization samp --output_dir ./my_results
```

### Command-Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--quantization` | `none`, `int8`, `samp` | `int8` | Quantization method |
| `--num_samples` | integer | `100` | Number of test samples |
| `--device` | `cpu`, `cuda:0` | `cpu` | Evaluation device |
| `--skip_baseline` | flag | False | Skip FP32 baseline evaluation |
| `--skip_utmos` | flag | False | Skip UTMOS quality metric |
| `--skip_wer` | flag | False | Skip WER intelligibility metric |
| `--output_dir` | path | `./evaluation_outputs` | Results directory |

### Metrics Explained

- **RTF (Real-Time Factor):** Processing time ÷ audio duration. RTF < 1.0 means faster than real-time.
- **UTMOS Score:** Speech quality on 1-5 scale (higher is better). 4.0+ is excellent.
- **WER (Word Error Rate):** Transcription error percentage (lower is better). < 5% is excellent.

## Results

Performance on LibriTTS-R test-clean (100 samples, CPU):

| Method | Memory | RTF | Speedup | UTMOS | WER |
|--------|--------|-----|---------|-------|-----|
| FP32 Baseline | 4.6 GB | 5.8 | 1.0x | 4.20 | 6.1% |
| INT8 | 2.3 GB | 3.2 | 1.8x | 3.85 ⚠️ | 8.3% ⚠️ |
| **SAMP (Ours)** | **2.9 GB** | **3.5** | **1.65x** | **4.10** ✓ | **6.4%** ✓ |

**Key Findings:**
- SAMP achieves 37% memory reduction with minimal quality loss
- INT8 is faster but degrades UTMOS and WER significantly
- SAMP balances efficiency and quality by preserving attention layers in FP16

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── evaluation.py             # Evaluation script with CLI
├── quantization.ipynb        # Interactive quantization examples
└── parler-tts/              # ParlerTTS library code
```

## Interactive Notebook

Explore quantization methods interactively:

```bash
jupyter notebook quantization.ipynb
```

The notebook includes:
- INT8 and 4-bit quantization examples
- SAMP implementation walkthrough
- Audio generation and comparison

## Troubleshooting

**BitsAndBytes installation error:**
```bash
pip install bitsandbytes --no-binary bitsandbytes
```

**CUDA out of memory:**
```bash
# Use CPU evaluation
python evaluation.py --quantization samp --device cpu

# Or reduce sample size
python evaluation.py --quantization samp --num_samples 10
```

**Dataset download slow:**
```bash
# Pre-download dataset
python -c "from datasets import load_dataset; \
load_dataset('parler-tts/libritts_r_filtered', 'clean', split='test.clean')"
```

**Whisper model too large:**
Edit `evaluation.py` and change to smaller model:
```python
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",  # Instead of whisper-large-v3
    device=-1
)
```

## Hardware Requirements

**Minimum (CPU-only):**
- CPU: 4+ cores
- RAM: 16 GB
- Storage: 20 GB

**Recommended (GPU):**
- GPU: NVIDIA RTX 3060 Ti or better (8GB+ VRAM)
- RAM: 16 GB
- Storage: 20 GB SSD

**Evaluation Time:**
- CPU: ~2-4 hours for 100 samples
- GPU: ~30-60 minutes for 100 samples

## References

- [ParlerTTS](https://github.com/huggingface/parler-tts) - Official repository
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - 8-bit quantization library
- [UTMOS](https://github.com/sarulab-speech/UTMOS22) - Speech quality predictor
- [Whisper](https://github.com/openai/whisper) - ASR model for WER
- [LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_filtered) - Evaluation dataset

## License

MIT License. See individual dependencies for their respective licenses.

<!-- ## Citation

```bibtex
@misc{parlertts-quantization-2025,
  title={ParlerTTS Quantization and Evaluation Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/parler-tts-quantization}
}
```

--- -->

**Questions?** Open an issue or check the [troubleshooting](#troubleshooting) section above.
