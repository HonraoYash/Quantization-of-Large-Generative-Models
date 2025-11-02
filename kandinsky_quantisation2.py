import torch
import psutil
import os
import sys
import time
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from PIL import Image
import numpy as np

PROMPT = "A sunset over mountains"
OUTPUT_DIR = "comparison_outputs"
IMAGE_SIZE = 512
NUM_STEPS = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_system_memory_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def get_model_size_gb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 3)

def get_device():
    print("\nDEVICE DIAGNOSTICS:")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA compiled version: {torch.version.cuda if torch.version.cuda else 'Not compiled with CUDA'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("CUDA_VISIBLE_DEVICES: Not set")
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        device = "cpu"
        print("\nWARNING: Using CPU - No CUDA devices found!")
        print("\nPossible reasons:")
        print("1. PyTorch not installed with CUDA support")
        print("2. Running on login node (not compute node)")
        print("3. No GPU allocated by SLURM")
        print("\nTo fix:")
        print("- Reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("- Or submit as SLURM job with --gres=gpu:1")
    
    print("="*70)
    return device

def calculate_image_metrics(image1, image2=None):
    metrics = {}
    
    img_array = np.array(image1)
    
    metrics['mean_brightness'] = np.mean(img_array)
    metrics['std_brightness'] = np.std(img_array)
    metrics['min_value'] = np.min(img_array)
    metrics['max_value'] = np.max(img_array)
    metrics['dynamic_range'] = np.max(img_array) - np.min(img_array)
    
    if len(img_array.shape) == 3:
        metrics['mean_red'] = np.mean(img_array[:, :, 0])
        metrics['mean_green'] = np.mean(img_array[:, :, 1])
        metrics['mean_blue'] = np.mean(img_array[:, :, 2])
    
    if image2 is not None:
        img2_array = np.array(image2)
        
        mse = np.mean((img_array.astype(float) - img2_array.astype(float)) ** 2)
        metrics['mse'] = mse
        
        if mse > 0:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            metrics['psnr'] = psnr
        else:
            metrics['psnr'] = float('inf')
        
        mean_diff = np.abs(np.mean(img_array) - np.mean(img2_array))
        metrics['mean_difference'] = mean_diff
        
        diff_pixels = np.sum(np.abs(img_array - img2_array) > 5)
        total_pixels = img_array.size
        metrics['different_pixels_pct'] = (diff_pixels / total_pixels) * 100
    
    return metrics

def print_metrics(title, metrics, generation_time=None):
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    
    if generation_time:
        print(f"Generation Time: {generation_time:.2f}s ({generation_time/60:.2f} min)")
        print("-"*70)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<30} {value:>15.2f}")
        else:
            print(f"{key:<30} {value:>15}")
    print("="*70)

def try_load_full_model(device):
    print("\n" + "STEP 1: ATTEMPTING TO LOAD FULL MODEL (float32)".center(70))
    print("="*70)
    
    try:
        start_mem = get_system_memory_gb()
        print(f"\nStarting memory: {start_mem:.2f} GB")
        
        if device == "cuda":
            print(f"Starting GPU memory: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        print("\nLoading prior model (float32)...")
        
        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch.float32
        )
        
        prior_size = get_model_size_gb(pipe_prior.prior)
        print(f"Prior loaded: {prior_size:.2f} GB")
        
        print("\nLoading decoder model (float32)...")
        pipe = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            torch_dtype=torch.float32
        )
        
        decoder_size = get_model_size_gb(pipe.unet)
        print(f"Decoder loaded: {decoder_size:.2f} GB")
        
        print(f"\nMoving models to {device}...")
        pipe_prior.to(device)
        pipe.to(device)
        
        current_mem = get_system_memory_gb()
        memory_used = current_mem - start_mem
        
        print("\n" + "="*70)
        print("SUCCESS: Full model loaded!".center(70))
        print("="*70)
        print(f"Prior model size:     {prior_size:.2f} GB")
        print(f"Decoder model size:   {decoder_size:.2f} GB")
        print(f"Total model size:     {prior_size + decoder_size:.2f} GB")
        print(f"System memory used:   {memory_used:.2f} GB")
        print(f"Current memory usage: {current_mem:.2f} GB")
        
        if device == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
            print(f"GPU memory reserved:  {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
        
        print("="*70)
        
        print("\nGenerating image with FULL MODEL...")
        start_time = time.time()
        
        with torch.inference_mode():
            image_embeds, negative_image_embeds = pipe_prior(
                prompt=PROMPT,
                num_inference_steps=10
            ).to_tuple()
            
            image = pipe(
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=NUM_STEPS
            ).images[0]
        
        generation_time = time.time() - start_time
        
        output_path = os.path.join(OUTPUT_DIR, "full_model_output.png")
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        metrics = calculate_image_metrics(image)
        print_metrics("FULL MODEL IMAGE METRICS", metrics, generation_time)
        
        del pipe_prior, pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True, image, metrics, generation_time
        
    except Exception as e:
        print("\n" + "="*70)
        print("FAILED: Could not load full model".center(70))
        print("="*70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*70)
        return False, None, None, None

def load_quantized_model(device):
    print("\n" + "STEP 2: LOADING QUANTIZED MODEL (float16)".center(70))
    print("="*70)
    
    try:
        start_mem = get_system_memory_gb()
        print(f"\nStarting memory: {start_mem:.2f} GB")
        
        if device == "cuda":
            print(f"Starting GPU memory: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        print("\nLoading prior model (float16)...")
        
        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch.float16
        )
        
        prior_size = get_model_size_gb(pipe_prior.prior)
        print(f"Prior loaded: {prior_size:.2f} GB")
        
        print("\nLoading decoder model (float16)...")
        pipe = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            torch_dtype=torch.float16
        )
        
        decoder_size = get_model_size_gb(pipe.unet)
        print(f"Decoder loaded: {decoder_size:.2f} GB")
        
        print(f"\nMoving models to {device}...")
        pipe_prior.to(device)
        pipe.to(device)
        
        current_mem = get_system_memory_gb()
        memory_used = current_mem - start_mem
        
        print("\n" + "="*70)
        print("SUCCESS: Quantized model loaded!".center(70))
        print("="*70)
        print(f"Prior model size:     {prior_size:.2f} GB")
        print(f"Decoder model size:   {decoder_size:.2f} GB")
        print(f"Total model size:     {prior_size + decoder_size:.2f} GB")
        print(f"System memory used:   {memory_used:.2f} GB")
        print(f"Current memory usage: {current_mem:.2f} GB")
        print(f"Memory reduction:     ~50% compared to float32")
        
        if device == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
            print(f"GPU memory reserved:  {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
        
        print("="*70)
        
        print("\nGenerating image with QUANTIZED MODEL...")
        start_time = time.time()
        
        with torch.inference_mode():
            image_embeds, negative_image_embeds = pipe_prior(
                prompt=PROMPT,
                num_inference_steps=10
            ).to_tuple()
            
            image = pipe(
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=NUM_STEPS
            ).images[0]
        
        generation_time = time.time() - start_time
        
        output_path = os.path.join(OUTPUT_DIR, "quantized_model_output.png")
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        metrics = calculate_image_metrics(image)
        print_metrics("QUANTIZED MODEL IMAGE METRICS", metrics, generation_time)
        
        return True, image, metrics, generation_time, pipe_prior, pipe
        
    except Exception as e:
        print("\n" + "="*70)
        print("FAILED: Could not load quantized model".center(70))
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*70)
        return False, None, None, None, None, None

def compare_images(full_image, quant_image, full_metrics, quant_metrics, full_time, quant_time):
    print("\n" + "="*70)
    print("COMPARISON: FULL vs QUANTIZED".center(70))
    print("="*70)
    
    print("\nGeneration Time Comparison:")
    print("-"*70)
    print(f"Full model (float32):       {full_time:>10.2f}s ({full_time/60:>6.2f} min)")
    print(f"Quantized model (float16):  {quant_time:>10.2f}s ({quant_time/60:>6.2f} min)")
    time_diff = full_time - quant_time
    time_diff_pct = (time_diff / full_time) * 100 if full_time > 0 else 0
    print(f"Time difference:            {time_diff:>10.2f}s ({time_diff_pct:>6.1f}% faster/slower)")
    
    print("\nImage Quality Comparison:")
    print("-"*70)
    
    comparison_metrics = calculate_image_metrics(quant_image, full_image)
    
    print(f"{'Metric':<30} {'Value':>15}")
    print("-"*70)
    print(f"{'MSE (lower is better)':<30} {comparison_metrics['mse']:>15.2f}")
    print(f"{'PSNR (higher is better)':<30} {comparison_metrics['psnr']:>15.2f} dB")
    print(f"{'Mean difference':<30} {comparison_metrics['mean_difference']:>15.2f}")
    print(f"{'Different pixels':<30} {comparison_metrics['different_pixels_pct']:>15.2f}%")
    
    print("\n" + "="*70)
    print("QUALITY ASSESSMENT".center(70))
    print("="*70)
    
    psnr = comparison_metrics['psnr']
    if psnr > 40:
        quality = "EXCELLENT - Nearly identical"
    elif psnr > 30:
        quality = "GOOD - Minor differences"
    elif psnr > 20:
        quality = "ACCEPTABLE - Noticeable differences"
    else:
        quality = "POOR - Significant differences"
    
    print(f"PSNR Score: {psnr:.2f} dB")
    print(f"Quality Rating: {quality}")
    print(f"\nNote: PSNR > 30 dB is generally considered good quality")
    print(f"      PSNR > 40 dB means images are perceptually very similar")
    
    print("\nCreating side-by-side comparison...")
    comparison_img = Image.new('RGB', (IMAGE_SIZE * 2, IMAGE_SIZE))
    comparison_img.paste(full_image, (0, 0))
    comparison_img.paste(quant_image, (IMAGE_SIZE, 0))
    comparison_path = os.path.join(OUTPUT_DIR, "side_by_side_comparison.png")
    comparison_img.save(comparison_path)
    print(f"Comparison saved to: {comparison_path}")
    
    print("="*70)

def main():
    print("\n" + "KANDINSKY QUANTIZATION EXPERIMENT".center(70))
    print("="*70)
    print(f"Test prompt: {PROMPT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Inference steps: {NUM_STEPS}")
    print("="*70)
    
    device = get_device()
    
    if device == "cpu":
        print("\n" + "!"*70)
        print("WARNING: Running on CPU - this will be VERY SLOW!".center(70))
        print("!"*70)
        user_input = input("\nContinue anyway? (yes/no): ").strip().lower()
        if user_input != 'yes':
            print("Exiting. Please reinstall PyTorch with CUDA or submit as SLURM job.")
            sys.exit(0)
    
    print("="*70)
    
    full_success, full_image, full_metrics, full_time = try_load_full_model(device)
    
    quant_success, quant_image, quant_metrics, quant_time, pipe_prior, pipe = load_quantized_model(device)
    
    if not quant_success:
        print("\nQuantized model failed to load. This is unexpected!")
        return
    
    if full_success and quant_success:
        compare_images(full_image, quant_image, full_metrics, quant_metrics, full_time, quant_time)
    else:
        print("\n" + "="*70)
        print("COMPARISON SKIPPED".center(70))
        print("="*70)
        print("Full model could not be loaded")
        print("Only quantized model results are available")
        print("="*70)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE".center(70))
    print("="*70)
    print("\nSummary:")
    print(f"  Full model (float32):      {'Loaded' if full_success else 'Failed'}")
    print(f"  Quantized model (float16): {'Loaded' if quant_success else 'Failed'}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    
    if quant_success:
        print("\nQuantized model works! You can use it for image generation.")
        print(f"   Memory usage: ~50% less than full model")
        if full_success:
            comparison_metrics = calculate_image_metrics(quant_image, full_image)
            print(f"   Image quality: PSNR = {comparison_metrics['psnr']:.2f} dB")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()