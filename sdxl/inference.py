"""
Generate images using your quantized SDXL-Lightning model
Simple script to test your quantized model with custom prompts
"""

import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path
import time

# Configuration
MODEL_PATH = "quantized_models/sdxl_lightning_2step_mps_fp16"
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Your prompts - customize these!
PROMPTS = [
    "A serene Japanese garden with cherry blossoms and a koi pond",
    "A futuristic city at sunset with flying cars",
    "A cozy coffee shop on a rainy day, warm lighting",
    "An astronaut riding a horse on Mars",
    "A magical forest with glowing mushrooms and fireflies",
]

# Settings
NUM_STEPS = 2  # For 2-step model
GUIDANCE_SCALE = 0  # Lightning models use 0
WIDTH = 768
HEIGHT = 768

def load_model():
    """Load the quantized model"""
    print("Loading quantized model...")
    print(f"Model path: {MODEL_PATH}")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    
    print("✓ Model loaded successfully!")
    return pipe, device

def generate_image(pipe, prompt, device, index=0):
    """Generate a single image"""
    print(f"\n{'='*60}")
    print(f"Generating image {index + 1}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    # Clean memory
    if device == "mps" and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    # Generate
    start_time = time.time()
    
    with torch.inference_mode():
        image = pipe(
            prompt,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            width=WIDTH,
            height=HEIGHT
        ).images[0]
    
    elapsed = time.time() - start_time
    
    # Save
    filename = f"image_{index:03d}_{prompt[:30].replace(' ', '_')}.png"
    filepath = OUTPUT_DIR / filename
    image.save(filepath)
    
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"✓ Saved to: {filepath}")
    
    return image, elapsed

def batch_generate():
    """Generate all images in the prompt list"""
    print("="*60)
    print("SDXL-Lightning Image Generation")
    print("="*60)
    
    # Load model once
    pipe, device = load_model()
    
    # Generate all images
    times = []
    for i, prompt in enumerate(PROMPTS):
        try:
            image, elapsed = generate_image(pipe, prompt, device, i)
            times.append(elapsed)
        except Exception as e:
            print(f"✗ Failed to generate image {i+1}: {e}")
            continue
    
    # Summary
    if times:
        print(f"\n{'='*60}")
        print("GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total images: {len(times)}")
        print(f"Average time: {sum(times)/len(times):.2f}s")
        print(f"Total time: {sum(times):.2f}s")
        print(f"Output folder: {OUTPUT_DIR.absolute()}")
        print(f"{'='*60}")

def interactive_mode():
    """Interactive prompt entry"""
    print("="*60)
    print("Interactive Mode - Enter your prompts!")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)
    
    pipe, device = load_model()
    
    counter = 0
    while True:
        prompt = input("\nEnter prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not prompt:
            print("Please enter a prompt!")
            continue
        
        try:
            generate_image(pipe, prompt, device, counter)
            counter += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")

def compare_resolutions(prompt="A beautiful landscape"):
    """Compare generation at different resolutions"""
    print("="*60)
    print("Resolution Comparison")
    print("="*60)
    
    pipe, device = load_model()
    
    resolutions = [
        (512, 512),
        (768, 768),
        (1024, 1024),
    ]
    
    for width, height in resolutions:
        print(f"\nGenerating at {width}×{height}...")
        
        start_time = time.time()
        try:
            with torch.inference_mode():
                if device == "mps" and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                image = pipe(
                    prompt,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    width=width,
                    height=height
                ).images[0]
            
            elapsed = time.time() - start_time
            filename = f"resolution_{width}x{height}.png"
            image.save(OUTPUT_DIR / filename)
            
            print(f"✓ {width}×{height}: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"✗ {width}×{height} failed: {e}")

def benchmark_variants():
    """Compare different model variants if you have them"""
    variants = [
        ("quantized_models/sdxl_lightning_1step_mps_fp16", 1),
        ("quantized_models/sdxl_lightning_2step_mps_fp16", 2),
        ("quantized_models/sdxl_lightning_4step_mps_fp16", 4),
    ]
    
    prompt = "A serene mountain landscape"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("="*60)
    print("Model Variant Benchmark")
    print("="*60)
    
    for model_path, steps in variants:
        if not Path(model_path).exists():
            print(f"✗ Skipping {model_path} (not found)")
            continue
        
        print(f"\nTesting {steps}-step variant...")
        
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "mps" else torch.float32
            ).to(device)
            
            start_time = time.time()
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=0,
                    width=512,
                    height=512
                ).images[0]
            elapsed = time.time() - start_time
            
            image.save(OUTPUT_DIR / f"variant_{steps}step.png")
            print(f"✓ {steps}-step: {elapsed:.2f}s")
            
            del pipe
            if device == "mps":
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"✗ Failed: {e}")

if __name__ == "__main__":
    import sys
    
    print("\nSDXL-Lightning Image Generator")
    print("Choose a mode:")
    print("1. Batch generate (predefined prompts)")
    print("2. Interactive mode (enter your own prompts)")
    print("3. Resolution comparison")
    print("4. Benchmark variants")
    print()
    
    choice = input("Enter choice (1-4) or press Enter for batch: ").strip()
    
    if choice == "2":
        interactive_mode()
    elif choice == "3":
        compare_resolutions()
    elif choice == "4":
        benchmark_variants()
    else:
        batch_generate()