import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import load_dataset
import soundfile as sf
import librosa
import time
import numpy as np
from jiwer import wer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ParlerTTS INT8 Quantization Evaluation")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "parler-tts/parler-tts-large-v1"
DATASET_NAME = "parler-tts/libritts_r_filtered"
DATASET_SPLIT = "test.clean"
NUM_SAMPLES = 100  # Adjust based on your compute budget
DEVICE = "cpu"  # Force CPU for RTF measurement
OUTPUT_DIR = "./evaluation_outputs"

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_utmos_model():
    """Load UTMOS strong model for speech quality assessment"""
    print("\nLoading UTMOS model...")
    try:
        utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", 
            "utmos22_strong", 
            trust_repo=True
        )
        print("UTMOS model loaded successfully")
        return utmos_predictor
    except Exception as e:
        print(f"Error loading UTMOS: {e}")
        print("Attempting alternative UTMOS loading method...")
        return None

def load_whisper_asr():
    """Load Whisper ASR model for WER calculation"""
    print("\nLoading Whisper ASR model...")
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=-1  # CPU
        )
        print("Whisper ASR loaded successfully")
        return asr_pipeline
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        return None

def calculate_rtf(audio_array, sampling_rate, generation_time):
    """
    Calculate Real-Time Factor
    RTF = Processing Time / Audio Duration
    RTF < 1.0 means faster than real-time
    """
    audio_duration = len(audio_array) / sampling_rate
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    return rtf

def calculate_utmos(audio_array, sampling_rate, utmos_predictor):
    """Calculate UTMOS score for audio quality"""
    if utmos_predictor is None:
        return None
    
    try:
        # UTMOS expects 16kHz
        if sampling_rate != 16000:
            audio_16k = librosa.resample(
                audio_array.astype(np.float32), 
                orig_sr=sampling_rate, 
                target_sr=16000
            )
        else:
            audio_16k = audio_array.astype(np.float32)
        
        # Convert to tensor
        wave_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
        
        # Predict MOS
        with torch.no_grad():
            score = utmos_predictor(wave_tensor, 16000)
        
        return score.item()
    except Exception as e:
        print(f"Warning: UTMOS calculation failed: {e}")
        return None

def calculate_wer_score(audio_array, sampling_rate, reference_text, asr_pipeline):
    """Calculate Word Error Rate using Whisper ASR"""
    if asr_pipeline is None:
        return None
    
    try:
        # Save audio temporarily for ASR
        temp_audio_path = os.path.join(OUTPUT_DIR, "temp_audio.wav")
        sf.write(temp_audio_path, audio_array, sampling_rate)
        
        # Transcribe
        result = asr_pipeline(temp_audio_path)
        hypothesis = result["text"].strip()
        
        # Calculate WER
        wer_score = wer(reference_text, hypothesis)
        
        # Clean up
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return wer_score
    except Exception as e:
        print(f"  Warning: WER calculation failed: {e}")
        return None

# ============================================================================
# LOAD MODELS
# ============================================================================

print("\nLoading ParlerTTS models...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load ORIGINAL model (on CPU)
print("Loading original model...")
original_model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(DEVICE)
original_model.eval()
print("Original model loaded")

# Load QUANTIZED model (INT8)
print("Loading INT8 quantized model...")
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_heads"],
)

quantized_model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config_8bit,
    device_map="cpu",
    torch_dtype=torch.float16
)
quantized_model.eval()
print("INT8 quantized model loaded")

# Load evaluation models
utmos_predictor = load_utmos_model()
asr_pipeline = load_whisper_asr()

# ============================================================================
# LOAD DATASET
# ============================================================================

print(f"\nLoading dataset: {DATASET_NAME} ({DATASET_SPLIT})...")
try:
    dataset = load_dataset(DATASET_NAME, "clean", split=DATASET_SPLIT, streaming=True)
    # Take subset
    dataset_subset = list(dataset.take(NUM_SAMPLES))
    print(f"Loaded {len(dataset_subset)} samples from {DATASET_SPLIT}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using fallback minimal dataset...")
    dataset_subset = []

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, model_name, dataset_subset, tokenizer, utmos_predictor, asr_pipeline):
    """Evaluate a ParlerTTS model on RTF, UTMOS, and WER"""
    
    print(f"\n[Evaluating {model_name}]")
    print("-" * 60)
    
    results = {
        "rtf_scores": [],
        "utmos_scores": [],
        "wer_scores": [],
        "generation_times": [],
        "audio_durations": []
    }
    
    for idx, sample in enumerate(tqdm(dataset_subset, desc=f"Processing {model_name}")):
        try:
            # Get text and create description
            text = sample["text_normalized"]
            
            # Create a generic description (you can customize this)
            description = "A clear speaker with moderate speed and pitch."
            
            # Prepare inputs
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(DEVICE)
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
            
            # Generate audio and measure time
            start_time = time.time()
            
            with torch.inference_mode():
                generation = model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    do_sample=True,
                    temperature=1.0,
                    max_length=2048  # Limit generation length
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Get audio
            audio_arr = generation.cpu().numpy().squeeze()
            sampling_rate = model.config.sampling_rate
            
            # Calculate metrics
            rtf = calculate_rtf(audio_arr, sampling_rate, generation_time)
            utmos_score = calculate_utmos(audio_arr, sampling_rate, utmos_predictor)
            wer_score = calculate_wer_score(audio_arr, sampling_rate, text, asr_pipeline)
            
            # Store results
            results["rtf_scores"].append(rtf)
            results["generation_times"].append(generation_time)
            results["audio_durations"].append(len(audio_arr) / sampling_rate)
            
            if utmos_score is not None:
                results["utmos_scores"].append(utmos_score)
            if wer_score is not None:
                results["wer_scores"].append(wer_score)
                
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    # Calculate statistics
    stats = {
        "rtf_mean": np.mean(results["rtf_scores"]) if results["rtf_scores"] else None,
        "rtf_std": np.std(results["rtf_scores"]) if results["rtf_scores"] else None,
        "utmos_mean": np.mean(results["utmos_scores"]) if results["utmos_scores"] else None,
        "utmos_std": np.std(results["utmos_scores"]) if results["utmos_scores"] else None,
        "wer_mean": np.mean(results["wer_scores"]) if results["wer_scores"] else None,
        "wer_std": np.std(results["wer_scores"]) if results["wer_scores"] else None,
        "avg_generation_time": np.mean(results["generation_times"]) if results["generation_times"] else None,
        "avg_audio_duration": np.mean(results["audio_durations"]) if results["audio_durations"] else None,
    }
    
    return stats, results

# ============================================================================
# RUN EVALUATION
# ============================================================================

print("\nEvaluating ORIGINAL model...")
original_stats, original_results = evaluate_model(
    original_model, 
    "Original FP32", 
    dataset_subset, 
    tokenizer, 
    utmos_predictor, 
    asr_pipeline
)

print("\nEvaluating QUANTIZED model...")
quantized_stats, quantized_results = evaluate_model(
    quantized_model, 
    "INT8 Quantized", 
    dataset_subset, 
    tokenizer, 
    utmos_predictor, 
    asr_pipeline
)

# ============================================================================
# GENERATE REPORT
# ============================================================================

print("\nGenerating Evaluation Report...")
print("\n" + "=" * 80)
print("EVALUATION REPORT: ParlerTTS INT8 Quantization")
print("=" * 80)

print(f"\nDataset: {DATASET_NAME} ({DATASET_SPLIT})")
print(f"Number of Samples: {len(dataset_subset)}")
print(f"Device: {DEVICE}")

print("\n" + "-" * 80)
print("ORIGINAL MODEL (FP32)")
print("-" * 80)
if original_stats["rtf_mean"] is not None:
    print(f"Real-Time Factor (RTF):  {original_stats['rtf_mean']:.4f} ± {original_stats['rtf_std']:.4f}")
    print(f"{'Faster' if original_stats['rtf_mean'] < 1.0 else 'Slower'} than real-time")
else:
    print("Real-Time Factor (RTF):  N/A")

if original_stats["utmos_mean"] is not None:
    print(f"UTMOS Score:             {original_stats['utmos_mean']:.4f} ± {original_stats['utmos_std']:.4f}")
    print(f"Range: 1.0 (poor) to 5.0 (excellent)")
else:
    print("UTMOS Score: N/A")

if original_stats["wer_mean"] is not None:
    print(f"Word Error Rate (WER):   {original_stats['wer_mean']:.4f} ± {original_stats['wer_std']:.4f}")
    print(f"{(original_stats['wer_mean'] * 100):.2f}% error rate")
else:
    print("  Word Error Rate (WER):   N/A")

print(f"  Avg Generation Time:     {original_stats['avg_generation_time']:.3f}s")
print(f"  Avg Audio Duration:      {original_stats['avg_audio_duration']:.3f}s")

print("\n" + "-" * 80)
print("INT8 QUANTIZED MODEL")
print("-" * 80)
if quantized_stats["rtf_mean"] is not None:
    print(f"Real-Time Factor (RTF):  {quantized_stats['rtf_mean']:.4f} ± {quantized_stats['rtf_std']:.4f}")
    print(f"{'Faster' if quantized_stats['rtf_mean'] < 1.0 else 'Slower'} than real-time")
else:
    print("Real-Time Factor (RTF):  N/A")

if quantized_stats["utmos_mean"] is not None:
    print(f"UTMOS Score:             {quantized_stats['utmos_mean']:.4f} ± {quantized_stats['utmos_std']:.4f}")
    print(f"Range: 1.0 (poor) to 5.0 (excellent)")
else:
    print("  UTMOS Score:             N/A")

if quantized_stats["wer_mean"] is not None:
    print(f"Word Error Rate (WER):   {quantized_stats['wer_mean']:.4f} ± {quantized_stats['wer_std']:.4f}")
    print(f"{(quantized_stats['wer_mean'] * 100):.2f}% error rate")
else:
    print("  Word Error Rate (WER):   N/A")

print(f"Avg Generation Time: {quantized_stats['avg_generation_time']:.3f}s")
print(f"Avg Audio Duration: {quantized_stats['avg_audio_duration']:.3f}s")

print("\n" + "-" * 80)
print("COMPARISON (Quantized vs Original)")
print("-" * 80)

if original_stats["rtf_mean"] and quantized_stats["rtf_mean"]:
    rtf_speedup = original_stats["rtf_mean"] / quantized_stats["rtf_mean"]
    print(f"RTF Speedup: {rtf_speedup:.2f}x")
    
if original_stats["utmos_mean"] and quantized_stats["utmos_mean"]:
    utmos_diff = quantized_stats["utmos_mean"] - original_stats["utmos_mean"]
    print(f"UTMOS Difference: {utmos_diff:+.4f}")
    
if original_stats["wer_mean"] and quantized_stats["wer_mean"]:
    wer_diff = quantized_stats["wer_mean"] - original_stats["wer_mean"]
    print(f"WER Difference: {wer_diff:+.4f}")

gen_time_speedup = original_stats["avg_generation_time"] / quantized_stats["avg_generation_time"]
print(f"Generation Speedup: {gen_time_speedup:.2f}x")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

# Save detailed results
import json
results_dict = {
    "original": {
        "statistics": original_stats,
        "raw_results": {
            "rtf_scores": [float(x) for x in original_results["rtf_scores"]],
            "utmos_scores": [float(x) for x in original_results["utmos_scores"]],
            "wer_scores": [float(x) for x in original_results["wer_scores"]],
        }
    },
    "quantized": {
        "statistics": quantized_stats,
        "raw_results": {
            "rtf_scores": [float(x) for x in quantized_results["rtf_scores"]],
            "utmos_scores": [float(x) for x in quantized_results["utmos_scores"]],
            "wer_scores": [float(x) for x in quantized_results["wer_scores"]],
        }
    }
}

output_file = os.path.join(OUTPUT_DIR, "evaluation_results.json")
with open(output_file, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nDetailed results saved to: {output_file}")
