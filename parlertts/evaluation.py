"""
ParlerTTS Quantization Evaluation Script
Supports: FP32 baseline, INT8, and SAMP (Sensitivity-Aware Mixed Precision)
"""

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
import argparse
import json
import os
from typing import Set
warnings.filterwarnings('ignore')

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ParlerTTS with different quantization methods"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "samp"],
        default="int8",
        help="Quantization method: none (FP32), int8 (BitsAndBytes INT8), samp (Sensitivity-Aware Mixed Precision)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for evaluation (default: cpu for RTF measurement)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_outputs",
        help="Output directory for results (default: ./evaluation_outputs)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="parler-tts/parler-tts-large-v1",
        help="HuggingFace model name (default: parler-tts/parler-tts-large-v1)"
    )
    
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip FP32 baseline evaluation (faster, only evaluate quantized model)"
    )
    
    parser.add_argument(
        "--skip_utmos",
        action="store_true",
        help="Skip UTMOS evaluation (faster)"
    )
    
    parser.add_argument(
        "--skip_wer",
        action="store_true",
        help="Skip WER evaluation (faster)"
    )
    
    return parser.parse_args()


# ============================================================================
# SAMP QUANTIZATION IMPLEMENTATION
# ============================================================================

def apply_sensitivity_aware_quantization(model, sensitive_modules: Set[str], verbose: bool = True):
    """
    Apply Sensitivity-Aware Mixed Precision (SAMP) quantization.
    
    Args:
        model: The ParlerTTS model to quantize
        sensitive_modules: Set of module name patterns to keep in FP16
        verbose: Print detailed quantization progress
    
    Returns:
        Quantized model with mixed precision
    """
    from bitsandbytes.nn import Linear8bitLt
    import torch.nn as nn
    
    total_params = 0
    quantized_params = 0
    skipped_layers = []
    quantized_layers = []
    
    def is_sensitive_module(name: str) -> bool:
        """Check if module name matches any sensitive pattern"""
        for pattern in sensitive_modules:
            if pattern in name.lower():
                return True
        return False
    
    # Collect all linear layers first
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module))
            total_params += module.weight.numel()
    
    if verbose:
        print(f"\nFound {len(linear_modules)} linear layers to process")
        print(f"Total parameters: {total_params:,}\n")
    
    # Quantize non-sensitive layers
    for name, module in linear_modules:
        # Check if this is a sensitive module
        if is_sensitive_module(name):
            skipped_layers.append(name)
            if verbose:
                print(f"  [SKIP] {name} (sensitive)")
            continue
        
        # Quantize this layer to INT8
        try:
            # Navigate to parent module
            *parent_path, attr_name = name.rsplit('.', 1) if '.' in name else ([], name)
            parent = model
            for p in parent_path:
                for part in p.split('.'):
                    parent = getattr(parent, part)
            
            # Create 8-bit linear layer
            quantized_layer = Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0
            )
            
            # Copy weights
            quantized_layer.weight = nn.Parameter(module.weight.data.clone())
            if module.bias is not None:
                quantized_layer.bias = nn.Parameter(module.bias.data.clone())
            
            # Replace module
            setattr(parent, attr_name, quantized_layer)
            quantized_params += module.weight.numel()
            quantized_layers.append(name)
            
            if verbose:
                print(f"  [INT8] {name}")
                
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Failed to quantize {name}: {e}")
            skipped_layers.append(name)
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("SAMP QUANTIZATION SUMMARY")
        print("="*70)
        print(f"Total linear layers: {len(linear_modules)}")
        print(f"Quantized to INT8: {len(quantized_layers)}")
        print(f"Kept in FP16: {len(skipped_layers)}")
        print(f"Parameters quantized: {quantized_params:,} / {total_params:,} "
              f"({100*quantized_params/total_params:.1f}%)")
        print("="*70 + "\n")
    
    return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_utmos_model(skip: bool = False):
    """Load UTMOS strong model for speech quality assessment"""
    if skip:
        print("\n[SKIP] UTMOS evaluation disabled")
        return None
    
    print("\n[1/3] Loading UTMOS model...")
    try:
        utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", 
            "utmos22_strong", 
            trust_repo=True
        )
        print("✓ UTMOS model loaded successfully")
        return utmos_predictor
    except Exception as e:
        print(f"✗ Error loading UTMOS: {e}")
        return None


def load_whisper_asr(skip: bool = False):
    """Load Whisper ASR model for WER calculation"""
    if skip:
        print("\n[SKIP] WER evaluation disabled")
        return None
    
    print("\n[2/3] Loading Whisper ASR model...")
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=-1  # CPU
        )
        print("✓ Whisper ASR loaded successfully")
        return asr_pipeline
    except Exception as e:
        print(f"✗ Error loading Whisper: {e}")
        return None


def calculate_rtf(audio_array, sampling_rate, generation_time):
    """Calculate Real-Time Factor"""
    audio_duration = len(audio_array) / sampling_rate
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    return rtf


def calculate_utmos(audio_array, sampling_rate, utmos_predictor):
    """Calculate UTMOS score for audio quality"""
    if utmos_predictor is None:
        return None
    
    try:
        if sampling_rate != 16000:
            audio_16k = librosa.resample(
                audio_array.astype(np.float32), 
                orig_sr=sampling_rate, 
                target_sr=16000
            )
        else:
            audio_16k = audio_array.astype(np.float32)
        
        wave_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
        
        with torch.no_grad():
            score = utmos_predictor(wave_tensor, 16000)
        
        return score.item()
    except Exception as e:
        print(f"  Warning: UTMOS calculation failed: {e}")
        return None


def calculate_wer_score(audio_array, sampling_rate, reference_text, asr_pipeline, output_dir):
    """Calculate Word Error Rate using Whisper ASR"""
    if asr_pipeline is None:
        return None
    
    try:
        temp_audio_path = os.path.join(output_dir, "temp_audio.wav")
        sf.write(temp_audio_path, audio_array, sampling_rate)
        
        result = asr_pipeline(temp_audio_path)
        hypothesis = result["text"].strip()
        
        wer_score = wer(reference_text, hypothesis)
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return wer_score
    except Exception as e:
        print(f"  Warning: WER calculation failed: {e}")
        return None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name, quantization_method, device):
    """
    Load ParlerTTS model with specified quantization method.
    
    Args:
        model_name: HuggingFace model identifier
        quantization_method: 'none', 'int8', or 'samp'
        device: Target device
    
    Returns:
        model, tokenizer, method_description
    """
    print(f"\n{'='*70}")
    print(f"LOADING MODEL: {quantization_method.upper()}")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if quantization_method == "none":
        # FP32 baseline
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)
        description = "FP32 Baseline"
        
    elif quantization_method == "int8":
        # Standard INT8 quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["lm_heads"],
        )
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        description = "INT8 Quantized (BitsAndBytes)"
        
    elif quantization_method == "samp":
        # Sensitivity-Aware Mixed Precision
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Define sensitive modules
        SENSITIVE_MODULES = {
            "self_attn",
            "cross_attn", 
            "encoder_attn",
            "lm_head",
            "final_proj",
        }
        
        print("\nApplying SAMP quantization...")
        print("Strategy: INT8 for encoder/FFN, FP16 for attention layers\n")
        model = apply_sensitivity_aware_quantization(
            model, 
            SENSITIVE_MODULES,
            verbose=True
        )
        model = model.to(device)
        description = "SAMP (Sensitivity-Aware Mixed Precision)"
        
    else:
        raise ValueError(f"Unknown quantization method: {quantization_method}")
    
    model.eval()
    print(f"✓ Model loaded: {description}\n")
    
    return model, tokenizer, description


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, model_name, dataset_subset, tokenizer, utmos_predictor, 
                   asr_pipeline, device, output_dir):
    """Evaluate a ParlerTTS model on RTF, UTMOS, and WER"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}\n")
    
    results = {
        "rtf_scores": [],
        "utmos_scores": [],
        "wer_scores": [],
        "generation_times": [],
        "audio_durations": []
    }
    
    for idx, sample in enumerate(tqdm(dataset_subset, desc=f"Processing {model_name}")):
        try:
            text = sample["text_normalized"]
            description = "A clear speaker with moderate speed and pitch."
            
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            
            start_time = time.time()
            
            with torch.inference_mode():
                generation = model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    do_sample=True,
                    temperature=1.0,
                    max_length=2048
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            audio_arr = generation.cpu().numpy().squeeze()
            sampling_rate = model.config.sampling_rate
            
            # Calculate metrics
            rtf = calculate_rtf(audio_arr, sampling_rate, generation_time)
            utmos_score = calculate_utmos(audio_arr, sampling_rate, utmos_predictor)
            wer_score = calculate_wer_score(audio_arr, sampling_rate, text, asr_pipeline, output_dir)
            
            # Store results
            results["rtf_scores"].append(rtf)
            results["generation_times"].append(generation_time)
            results["audio_durations"].append(len(audio_arr) / sampling_rate)
            
            if utmos_score is not None:
                results["utmos_scores"].append(utmos_score)
            if wer_score is not None:
                results["wer_scores"].append(wer_score)
                
        except Exception as e:
            print(f"\n✗ Error processing sample {idx}: {e}")
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
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    print("=" * 80)
    print("ParlerTTS Quantization Evaluation")
    print("=" * 80)
    print(f"Quantization Method: {args.quantization.upper()}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluation models
    utmos_predictor = load_utmos_model(skip=args.skip_utmos)
    asr_pipeline = load_whisper_asr(skip=args.skip_wer)
    
    # Load dataset
    print(f"\n[3/3] Loading dataset...")
    try:
        dataset = load_dataset(
            "parler-tts/libritts_r_filtered", 
            "clean", 
            split="test.clean", 
            streaming=True
        )
        dataset_subset = list(dataset.take(args.num_samples))
        print(f"✓ Loaded {len(dataset_subset)} samples\n")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        dataset_subset = []
    
    # Evaluate models
    results_dict = {}
    
    # Baseline evaluation (optional)
    if not args.skip_baseline and args.quantization != "none":
        baseline_model, baseline_tokenizer, baseline_desc = load_model(
            args.model_name, "none", args.device
        )
        baseline_stats, baseline_results = evaluate_model(
            baseline_model, baseline_desc, dataset_subset, baseline_tokenizer,
            utmos_predictor, asr_pipeline, args.device, args.output_dir
        )
        results_dict["baseline"] = {
            "method": "FP32 Baseline",
            "statistics": baseline_stats,
            "raw_results": {
                "rtf_scores": [float(x) for x in baseline_results["rtf_scores"]],
                "utmos_scores": [float(x) for x in baseline_results["utmos_scores"]],
                "wer_scores": [float(x) for x in baseline_results["wer_scores"]],
            }
        }
        del baseline_model  # Free memory
    
    # Quantized model evaluation
    quant_model, quant_tokenizer, quant_desc = load_model(
        args.model_name, args.quantization, args.device
    )
    quant_stats, quant_results = evaluate_model(
        quant_model, quant_desc, dataset_subset, quant_tokenizer,
        utmos_predictor, asr_pipeline, args.device, args.output_dir
    )
    results_dict["quantized"] = {
        "method": quant_desc,
        "statistics": quant_stats,
        "raw_results": {
            "rtf_scores": [float(x) for x in quant_results["rtf_scores"]],
            "utmos_scores": [float(x) for x in quant_results["utmos_scores"]],
            "wer_scores": [float(x) for x in quant_results["wer_scores"]],
        }
    }
    
    # Generate report
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    for key, data in results_dict.items():
        stats = data["statistics"]
        print(f"\n{data['method'].upper()}")
        print("-" * 80)
        
        if stats["rtf_mean"] is not None:
            print(f"  Real-Time Factor (RTF):  {stats['rtf_mean']:.4f} ± {stats['rtf_std']:.4f}")
            print(f"    → {'Faster' if stats['rtf_mean'] < 1.0 else 'Slower'} than real-time")
        
        if stats["utmos_mean"] is not None:
            print(f"  UTMOS Score:             {stats['utmos_mean']:.4f} ± {stats['utmos_std']:.4f}")
        
        if stats["wer_mean"] is not None:
            print(f"  Word Error Rate (WER):   {stats['wer_mean']:.4f} ± {stats['wer_std']:.4f}")
            print(f"    → {(stats['wer_mean'] * 100):.2f}% error rate")
        
        print(f"  Avg Generation Time:     {stats['avg_generation_time']:.3f}s")
        print(f"  Avg Audio Duration:      {stats['avg_audio_duration']:.3f}s")
    
    # Comparison
    if len(results_dict) > 1:
        print("\n" + "-" * 80)
        print("COMPARISON (Quantized vs Baseline)")
        print("-" * 80)
        
        baseline_stats = results_dict["baseline"]["statistics"]
        quant_stats = results_dict["quantized"]["statistics"]
        
        if baseline_stats["rtf_mean"] and quant_stats["rtf_mean"]:
            speedup = baseline_stats["rtf_mean"] / quant_stats["rtf_mean"]
            print(f"  RTF Speedup:             {speedup:.2f}x")
        
        if baseline_stats["utmos_mean"] and quant_stats["utmos_mean"]:
            diff = quant_stats["utmos_mean"] - baseline_stats["utmos_mean"]
            print(f"  UTMOS Difference:        {diff:+.4f}")
        
        if baseline_stats["wer_mean"] and quant_stats["wer_mean"]:
            diff = quant_stats["wer_mean"] - baseline_stats["wer_mean"]
            print(f"  WER Difference:          {diff:+.4f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, f"evaluation_{args.quantization}.json")
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
