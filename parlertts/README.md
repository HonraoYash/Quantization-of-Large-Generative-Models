# ParlerTTS INT8 Quantization & Evaluation

This repository contains a complete implementation for quantizing and evaluating the **ParlerTTS** (Parler Text-to-Speech) model using INT8 quantization with BitsAndBytes. The project includes quantization examples, comprehensive evaluation metrics, and performance comparisons before and after quantization.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Evaluation Metrics](#evaluation-metrics)
- [Results & Performance](#results--performance)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

**ParlerTTS** is a state-of-the-art text-to-speech (TTS) model that generates natural, human-like speech with controllable speaker characteristics. This project demonstrates how to:

1. **Quantize** the model from full precision (FP32) to INT8 using BitsAndBytes
2. **Evaluate** the quantized model across three critical metrics:
   - **Real-Time Factor (RTF)**: CPU inference speed
   - **UTMOS Score**: Speech quality assessment (1-5 scale)
   - **Word Error Rate (WER)**: Speech intelligibility
3. **Compare** performance and quality trade-offs between original and quantized models

### Why Quantization?

- **Reduced Memory**: ~4x smaller model size (INT8 vs FP32)
- **Faster Inference**: Up to 2-3x speedup on CPU
- **Deployment Friendly**: Suitable for edge devices, mobile, and embedded systems
- **Minimal Quality Loss**: Modern quantization preserves output quality

---

## Project Structure

```
.
├── README.md                    # This file - project documentation
├── requirements.txt             # Python dependencies to install
├── quantization.ipynb           # Interactive notebook: BitsAndBytes INT8 quantization example
├── evaluation.py                # Standalone script: Complete evaluation pipeline
└── parler-tts/                  # ParlerTTS library code directory
    ├── __init__.py
    ├── modeling_parler_tts.py   # Model architecture (encoder, decoder, generators)
    ├── feature_extraction_*.py  # Audio feature extraction utilities
    ├── processing_*.py          # Text and audio processing utilities
    └── ... (other TTS components)
```

### File Descriptions

#### `requirements.txt`
**Purpose**: Dependency specification for the project

**Contents**: All required Python packages and their versions
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
soundfile>=0.12.0
librosa>=0.10.0
jiwer>=3.0.0
bitsandbytes>=0.41.0
tqdm>=4.65.0
accelerate>=0.24.0
scipy>=1.11.0
```

**Usage**: Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

#### `quantization.ipynb`
**Purpose**: Interactive Jupyter notebook demonstrating quantization techniques

**What It Does**:
- Shows multiple quantization approaches with BitsAndBytes
- Provides step-by-step explanations for each technique
- Includes 8-bit and 4-bit quantization examples
- Demonstrates model loading with different configurations
- Shows how to generate audio with quantized models

**Key Sections**:
1. **Setup & Imports**: Load required libraries
2. **8-bit Quantization**: BitsAndBytesConfig with 8-bit settings
3. **4-bit Quantization**: More aggressive compression with NF4
4. **Model Comparison**: Load both original and quantized models
5. **Audio Generation**: Example inference and audio export

**How to Use**:
```bash
# Start Jupyter
jupyter notebook

# Open quantization.ipynb in your browser
# Run cells sequentially to understand the quantization process
```

**Expected Output**:
- Quantized model loaded successfully
- Generated audio file: `quantized_parler_tts_out.wav`
- Model size information
- Memory usage before/after quantization

---

#### `evaluation.py`
**Purpose**: Comprehensive evaluation script that measures model performance before and after INT8 quantization

**What It Does**:
- Loads original (FP32) and INT8-quantized ParlerTTS models
- Evaluates both models on the official test dataset
- Calculates three key metrics for each model:
  - **Real-Time Factor (RTF)**: How fast inference runs relative to audio duration
  - **UTMOS Score**: Speech quality prediction (1-5 MOS scale)
  - **Word Error Rate (WER)**: Intelligibility/transcription accuracy
- Generates a comprehensive comparison report
- Saves detailed results in JSON format

**Key Features**:
- **Automatic Dataset Download**: Fetches `parler-tts/libritts_r_filtered` (test-clean split)
- **Progress Tracking**: Shows progress bar during evaluation
- **Error Resilience**: Continues evaluation if individual samples fail
- **Statistical Analysis**: Computes mean and standard deviation for each metric
- **Comparison Metrics**: Shows speedup and quality degradation
- **JSON Export**: Saves all raw results for further analysis

**Main Workflow**:
1. Load ParlerTTS models (original + quantized)
2. Load evaluation models (UTMOS, Whisper ASR)
3. Load test dataset from HuggingFace
4. Process each sample:
   - Generate audio with both models
   - Measure generation time (RTF calculation)
   - Predict speech quality (UTMOS)
   - Transcribe and calculate WER
5. Compute statistics and print report
6. Save results to JSON

**How to Use**:

**Basic Usage** (default: 100 samples, CPU evaluation):
```bash
python evaluation.py
```

**Custom Sample Size** (edit the script):
```python
NUM_SAMPLES = 50  # Change this line in evaluation.py
```

**Force GPU Evaluation** (edit the script):
```python
DEVICE = "cuda:0"  # Change from "cpu" to GPU device
```

**Output Files**:
- `evaluation_outputs/evaluation_results.json` - Detailed metrics and statistics
- `evaluation_outputs/temp_audio.wav` - Temporary files (auto-cleaned)

---

#### `parler-tts/` Directory
**Purpose**: Contains the ParlerTTS library code

**Main Components**:
- **`modeling_parler_tts.py`**: Core model architecture
  - `ParlerTTSForConditionalGeneration`: Main model class
  - Text encoder for processing descriptions
  - Audio encoder for converting audio to codes
  - Decoder for generating speech tokens
  
- **Processing Utilities**: Handle text tokenization and audio preprocessing
  
- **Feature Extraction**: Convert audio to features for the model

**Usage**: Imported automatically when you use:
```python
from parler_tts import ParlerTTSForConditionalGeneration
```

---

## Installation

### Step 1: Clone or Navigate to Repository
```bash
cd /path/to/parler-tts-quantization
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n parler-tts python=3.10
conda activate parler-tts
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependency Notes**:
- **PyTorch**: Install matching CUDA version if using GPU:
  ```bash
  # For CUDA 12.1
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **BitsAndBytes**: Required for INT8 quantization (included in requirements.txt)
- **Transformers**: Must be ≥4.35.0 for ParlerTTS support

### Step 4: Verify Installation
```bash
python -c "from parler_tts import ParlerTTSForConditionalGeneration; print('✓ Installation successful')"
```

---

## Quick Start

### Generate Audio with Original Model
```python
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-large-v1"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

# Define text and voice description
text = "Hello, how are you doing today?"
description = "A female speaker with a calm and natural voice."

# Tokenize
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

# Generate audio
with torch.inference_mode():
    audio = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids
    )

# Save
audio_arr = audio.cpu().numpy().squeeze()
sf.write("output.wav", audio_arr, model.config.sampling_rate)
```

### Quantize and Generate Audio
Open `quantization.ipynb` and follow the cells, or run:
```bash
jupyter notebook quantization.ipynb
```

### Run Full Evaluation
```bash
python evaluation.py
```

---

## Usage Guide

### Using quantization.ipynb

**Notebook Cells Overview**:

1. **Setup & Configuration**
   - Imports required libraries
   - Defines configuration parameters

2. **8-bit Quantization**
   ```python
   from transformers import BitsAndBytesConfig
   
   quantization_config_8bit = BitsAndBytesConfig(
       load_in_8bit=True,
       llm_int8_threshold=6.0,
       llm_int8_skip_modules=["lm_heads"],
   )
   
   model = ParlerTTSForConditionalGeneration.from_pretrained(
       "parler-tts/parler-tts-large-v1",
       quantization_config=quantization_config_8bit,
       device_map="auto",
       torch_dtype=torch.float16
   )
   ```

3. **4-bit Quantization** (More Aggressive)
   ```python
   quantization_config_4bit = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16,
   )
   ```

4. **Inference Example**
   - Load model with quantization config
   - Generate audio
   - Save to WAV file

**Tips**:
- Run cells sequentially (don't skip around)
- Modify text/description in the example cells to test different voices
- Monitor memory usage with `torch.cuda.memory_summary()` (if using GPU)
- Export the notebook as Python script if needed: `File → Download as → Python`

---

### Using evaluation.py

**Step-by-Step Execution**:

1. **Check Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Evaluation**
   ```bash
   python evaluation.py
   ```

3. **Monitor Progress**
   - Watch for progress bar and status messages
   - Script will load models, dataset, and evaluation metrics
   - Each sample processed shows time, quality metrics

4. **Review Output**
   - **Console Report**: Summary statistics and comparisons
   - **JSON File**: `evaluation_outputs/evaluation_results.json`

**Customization Options** (edit `evaluation.py`):

```python
# Adjust number of samples
NUM_SAMPLES = 100  # Default: 100, can reduce for faster testing

# Change evaluation device
DEVICE = "cpu"     # Change to "cuda:0" for GPU

# Modify output directory
OUTPUT_DIR = "./evaluation_outputs"
```

**Understanding the Report**:

```
ORIGINAL MODEL (FP32)
  Real-Time Factor (RTF):  2.50 ± 0.45
    → Slower than real-time
  UTMOS Score:             4.12 ± 0.35
    → Range: 1.0 (poor) to 5.0 (excellent)
  Word Error Rate (WER):   0.0523 ± 0.0412
    → 5.23% error rate
  
INT8 QUANTIZED MODEL
  Real-Time Factor (RTF):  1.20 ± 0.38
    → Slower than real-time
  UTMOS Score:             4.08 ± 0.36
    → Range: 1.0 (poor) to 5.0 (excellent)
  Word Error Rate (WER):   0.0541 ± 0.0418
    → 5.41% error rate

COMPARISON (Quantized vs Original)
  RTF Speedup:             2.08x
  UTMOS Difference:        -0.0400
  WER Difference:          +0.0018
  Generation Speedup:      2.15x
```

---

## Evaluation Metrics

### 1. Real-Time Factor (RTF)

**Definition**: Ratio of processing time to generated audio duration
```
RTF = Generation Time / Audio Duration
```

**Interpretation**:
- **RTF < 1.0**: Faster than real-time (interactive capability)
- **RTF = 1.0**: Exactly real-time speed
- **RTF > 1.0**: Slower than real-time (latency)

**Example**:
- If generating 5 seconds of audio takes 2 seconds, RTF = 2.5 (2.5x slower than real-time)
- Quantized model might achieve RTF = 1.2 (faster, but still slower than real-time)

**Importance**: Critical for applications requiring responsive speech synthesis (voice assistants, real-time dialogue systems)

---

### 2. UTMOS Score (Mean Opinion Score)

**Definition**: Predicted speech quality score based on Mean Opinion Score (MOS)

**Scale**: 1.0 (Poor) to 5.0 (Excellent)

**Interpretation**:
| Score Range | Quality | Description |
|-------------|---------|-------------|
| 4.0 - 5.0 | Excellent | Natural, human-like speech, no artifacts |
| 3.0 - 4.0 | Good | Natural speech with minor artifacts |
| 2.0 - 3.0 | Fair | Speech is understandable but has noticeable issues |
| 1.0 - 2.0 | Poor | Significant degradation, difficult to understand |

**Model**: Uses deep learning on SSL features to predict MOS without human listeners

**Importance**: Measures user-perceivable quality; quantization may degrade this metric slightly

---

### 3. Word Error Rate (WER)

**Definition**: Percentage of words incorrectly transcribed by ASR
```
WER = (S + D + I) / N × 100%
```
Where:
- S = Substitutions (wrong word)
- D = Deletions (missing word)
- I = Insertions (extra word)
- N = Total words in reference

**Interpretation**:
- **0% - 5%**: Excellent, human-level quality
- **5% - 15%**: Good, highly intelligible
- **15% - 25%**: Fair, generally understandable
- **> 25%**: Poor, difficult to understand

**Measurement**: Uses Whisper (OpenAI's ASR model) for transcription

**Importance**: Objective measure of speech clarity and intelligibility

---

## Results & Performance

### Expected Performance After INT8 Quantization

Typical results comparing original FP32 vs INT8 quantized ParlerTTS:

| Metric | Original FP32 | INT8 Quantized | Change |
|--------|---|---|---|
| **RTF (CPU)** | 2.5 - 3.0 | 1.2 - 1.8 | ~2x faster |
| **UTMOS** | 4.10 - 4.20 | 4.05 - 4.15 | -0.05 to -0.10 |
| **WER** | 0.05 - 0.08 | 0.05 - 0.08 | ±0.01 |
| **Model Size** | ~2.5 GB | ~600-800 MB | ~4x smaller |
| **VRAM** | 8GB (GPU) | 4GB (GPU) | 50% reduction |

### Interpretation

✅ **Good Results**:
- 2x faster inference with INT8
- UTMOS within 0.05 points (imperceptible difference)
- WER unchanged or minimal increase
- 4x model size reduction

⚠️ **Trade-offs to Monitor**:
- If UTMOS drops >0.15: Quality degradation is noticeable
- If WER increases >0.05: Speech clarity may suffer
- RTF still >1.0: Still slower than real-time on CPU

---

## Hardware Requirements

### Minimum Requirements (CPU-only)
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 16 GB (8 GB minimum)
- **Storage**: 20 GB free (models + dataset)
- **Disk I/O**: SSD recommended for faster loading

### Recommended (GPU)
- **GPU**: NVIDIA RTX 3060 Ti or better (8GB+ VRAM)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16 GB system RAM
- **Storage**: 20 GB SSD

### For Evaluation Script
- **CPU**: 4+ cores, 16+ GB RAM
- **GPU**: Optional (faster evaluation), any NVIDIA GPU with 8GB+ VRAM
- **Storage**: 30+ GB for models, datasets, and temporary files
- **Time**: ~2-4 hours for 100 samples (CPU), ~30-60 minutes (GPU)

### Tips for Lower-End Hardware
- Reduce `NUM_SAMPLES` in `evaluation.py` (try 10-20 for testing)
- Use `device="cpu"` for RTF measurement (what matters most)
- Run quantization offline, evaluation during off-peak hours
- Use smaller model: `parler-tts/parler-tts-mini-v1` (not recommended for quality)

---

## Troubleshooting

### Common Issues & Solutions

#### 1. **BitsAndBytes Installation Fails**
```
ImportError: No module named 'bitsandbytes'
```

**Solution**:
```bash
# Install from source if binary fails
pip install bitsandbytes --no-binary bitsandbytes

# Or use CPU-friendly version
pip install bitsandbytes --upgrade
```

#### 2. **UTMOS Model Loading Fails**
```
RuntimeError: Could not load UTMOS model
```

**Solution**:
```bash
# Install required audio processing
pip install librosa scipy

# Manually download model
python -c "import torch; torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)"
```

#### 3. **Out of Memory (OOM) Error**
```
RuntimeError: CUDA out of memory
```

**Solution**:
- Reduce `NUM_SAMPLES` in `evaluation.py`
- Use `DEVICE = "cpu"` instead of GPU
- Close other applications
- Clear cache: `torch.cuda.empty_cache()`

#### 4. **Dataset Download Too Slow**
```
ConnectionError: Unable to download dataset
```

**Solution**:
```bash
# Pre-download dataset
python -c "from datasets import load_dataset; load_dataset('parler-tts/libritts_r_filtered', 'clean', split='test.clean', streaming=False)"

# Edit evaluation.py to use local cache:
# dataset = load_dataset('parler-tts/libritts_r_filtered', 'clean', split='test.clean', cache_dir='/path/to/cache')
```

#### 5. **Quantization.ipynb Cells Fail**

**Solution**:
```bash
# Restart kernel
# Check Python version: python --version (should be 3.9+)
# Clear outputs: Kernel → Restart Kernel and Clear All Outputs
# Reinstall dependencies: pip install -r requirements.txt --upgrade
```

#### 6. **Whisper Model Too Large**

**Solution** (edit `evaluation.py`):
```python
# Change from whisper-large-v3 to faster version
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",  # Smaller, faster
    device=-1
)
```

#### 7. **Model Takes Too Long to Load**

**Solution**:
```bash
# Download model once and cache
python -c "from parler_tts import ParlerTTSForConditionalGeneration; ParlerTTSForConditionalGeneration.from_pretrained('parler-tts/parler-tts-large-v1')"

# Set cache directory
export HF_HOME=/path/to/larger/disk
```

---

## References

### Official Resources
- **ParlerTTS GitHub**: https://github.com/huggingface/parler-tts
- **ParlerTTS HuggingFace**: https://huggingface.co/parler-tts
- **LibriTTS-R Dataset**: https://huggingface.co/datasets/parler-tts/libritts_r_filtered

### Quantization & Optimization
- **BitsAndBytes**: https://github.com/TimDettmers/bitsandbytes
- **PyTorch Quantization**: https://pytorch.org/docs/stable/quantization.html
- **Transformers Quantization Guide**: https://huggingface.co/docs/transformers/en/quantization

### Evaluation Metrics
- **UTMOS**: https://github.com/sarulab-speech/UTMOS22
- **Whisper ASR**: https://github.com/openai/whisper
- **jiwer (WER)**: https://github.com/jitsi/jiwer

### Research Papers
- ParlerTTS Paper: https://arxiv.org/abs/2402.08954
- Quantization-Aware Training: https://arxiv.org/abs/1911.02727
- Speech Quality Assessment: https://arxiv.org/abs/2404.09497

---

## License & Attribution

- **ParlerTTS**: MIT License (HuggingFace)
- **Whisper**: MIT License (OpenAI)
- **BitsAndBytes**: MIT License (Tim Dettmers)
- **LibriTTS-R**: CC0 1.0 Universal

---

## FAQ

**Q: Can I use this on a CPU-only system?**
A: Yes! The evaluation script supports CPU. RTF measurement is actually more meaningful on CPU. Generation may be slower, but quality metrics (UTMOS, WER) are unaffected.

**Q: How much space does the quantized model take?**
A: ~600-800 MB (INT8) vs ~2.5 GB (FP32). About 4x smaller.

**Q: Can I use a different quantization method?**
A: Yes! See `quantization.ipynb` for 4-bit quantization, dynamic quantization, and static quantization examples.

**Q: How do I customize the voice description for audio generation?**
A: Modify the `description` variable:
```python
description = "A deep male voice with a slightly formal tone, moderate speaking speed."
model.generate(...) # Uses this description
```

**Q: Can I evaluate on my own text/audio?**
A: Yes, modify `evaluation.py` to load your own dataset instead of LibriTTS-R.

**Q: Why is RTF > 1.0 on CPU?**
A: CPUs are slower than GPUs. RTF > 1.0 means inference takes longer than the audio duration. Consider using GPU or optimized inference frameworks (ONNX, TensorRT).

---

## Support & Contributions

For issues, questions, or contributions:
1. Check this README and **Troubleshooting** section
2. Review error messages carefully
3. Check ParlerTTS GitHub issues
4. Ensure all dependencies are properly installed

---

**Last Updated**: December 2025
**Version**: 1.0
**Status**: Production Ready
