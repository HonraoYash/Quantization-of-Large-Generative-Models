"""
SDXL-Lightning Model Quantization Script - macOS Optimized
Fixed for Apple Silicon MPS and CPU fallback
"""

import torch
import os
from pathlib import Path
import argparse
from typing import Optional
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import gc
import warnings
warnings.filterwarnings('ignore')

class SDXLLightningQuantizer:
    """Quantization utilities for SDXL-Lightning models - macOS Compatible"""
    
    def __init__(self, model_variant: str = "4step", output_dir: str = "./quantized_models"):
        self.model_variant = model_variant
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        self.repo = "ByteDance/SDXL-Lightning"
        
        # Smart device selection
        self.device = self._select_device()
        print(f"Using device: {self.device}")
        
        # Set checkpoint filename based on variant
        if model_variant == "1step":
            self.ckpt_name = "sdxl_lightning_1step_unet_x0.safetensors"
        else:
            self.ckpt_name = f"sdxl_lightning_{model_variant}_unet.safetensors"
    
    def _select_device(self) -> str:
        """Smart device selection for macOS"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                # Test MPS functionality
                test_tensor = torch.randn(1).to("mps")
                return "mps"
            except Exception as e:
                print(f"MPS test failed: {e}, falling back to CPU")
                return "cpu"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_base_pipeline(self) -> StableDiffusionXLPipeline:
        """Load the base SDXL-Lightning pipeline optimized for MPS/CPU"""
        print(f"Loading SDXL-Lightning {self.model_variant} model on {self.device}...")
        
        # Use float32 for CPU/MPS to avoid compatibility issues
        dtype = torch.float32 if self.device in ["cpu", "mps"] else torch.float16
        
        # Load UNet config
        print("Loading UNet...")
        unet = UNet2DConditionModel.from_config(
            self.base_model, 
            subfolder="unet"
        )
        
        # Download and load Lightning weights
        print(f"Downloading {self.ckpt_name}...")
        lightning_weights = load_file(
            hf_hub_download(self.repo, self.ckpt_name), 
            device="cpu"
        )
        unet.load_state_dict(lightning_weights)
        
        # Move to device
        print(f"Moving UNet to {self.device}...")
        unet = unet.to(self.device, dtype=dtype)
        
        # Create pipeline
        print("Loading full pipeline...")
        if dtype == torch.float32:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model, 
                unet=unet, 
                torch_dtype=torch.float32
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model, 
                unet=unet, 
                torch_dtype=torch.float16,
                variant="fp16"
            )
        
        # Move pipeline to device
        pipe = pipe.to(self.device)
        
        # Configure scheduler
        if self.model_variant == "1step":
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, 
                timestep_spacing="trailing", 
                prediction_type="sample"
            )
        else:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, 
                timestep_spacing="trailing"
            )
        
        print("Pipeline loaded successfully!")
        return pipe
    
    def quantize_dynamic_int8(self) -> Optional[StableDiffusionXLPipeline]:
        """Apply PyTorch's dynamic int8 quantization (CPU optimized)"""
        print("Applying PyTorch dynamic int8 quantization (CPU-friendly)...")
        
        try:
            # Load base pipeline
            pipe = self.load_base_pipeline()
            
            # Move to CPU for quantization
            print("Moving model to CPU for quantization...")
            if self.device == "mps":
                # Keep track of original device
                original_device = self.device
                pipe = pipe.to("cpu")
            
            # Apply dynamic quantization to UNet (main compute bottleneck)
            print("Quantizing UNet layers...")
            pipe.unet = torch.quantization.quantize_dynamic(
                pipe.unet,
                {torch.nn.Linear},  # Only quantize Linear layers for stability
                dtype=torch.qint8
            )
            
            # Save quantized model
            save_path = self.output_dir / f"sdxl_lightning_{self.model_variant}_int8"
            pipe.save_pretrained(save_path)
            print(f"Quantized model saved to: {save_path}")
            
            return pipe
            
        except Exception as e:
            print(f"Dynamic quantization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def quantize_for_mps(self) -> Optional[StableDiffusionXLPipeline]:
        """MPS-optimized quantization using FP16 and memory optimization"""
        if self.device != "mps":
            print("This method is only for MPS devices")
            return None
        
        print("Creating MPS-optimized FP16 model...")
        
        try:
            # Load UNet with FP16
            print("Loading UNet with FP16...")
            unet = UNet2DConditionModel.from_config(
                self.base_model, 
                subfolder="unet"
            )
            
            lightning_weights = load_file(
                hf_hub_download(self.repo, self.ckpt_name), 
                device="cpu"
            )
            unet.load_state_dict(lightning_weights)
            
            # Convert to FP16 and move to MPS
            unet = unet.half().to("mps")
            
            # Create pipeline with FP16
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model, 
                unet=unet, 
                torch_dtype=torch.float16,
                variant="fp16"
            ).to("mps")
            
            # Configure scheduler
            if self.model_variant == "1step":
                pipe.scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config, 
                    timestep_spacing="trailing", 
                    prediction_type="sample"
                )
            else:
                pipe.scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config, 
                    timestep_spacing="trailing"
                )
            
            # Enable attention slicing for memory efficiency
            pipe.enable_attention_slicing(slice_size=1)
            
            # Save model
            save_path = self.output_dir / f"sdxl_lightning_{self.model_variant}_mps_fp16"
            pipe.save_pretrained(save_path)
            print(f"MPS-optimized model saved to: {save_path}")
            
            return pipe
            
        except Exception as e:
            print(f"MPS optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_lightweight_model(self) -> Optional[StableDiffusionXLPipeline]:
        """Create a lightweight version with optimizations"""
        print("Creating lightweight optimized model...")
        
        try:
            pipe = self.load_base_pipeline()
            
            # Enable memory-efficient attention
            print("Enabling memory-efficient attention...")
            try:
                pipe.enable_attention_slicing(slice_size=1)
                pipe.enable_vae_slicing()
            except:
                print("Some optimizations not available, continuing...")
            
            # Save optimized model
            save_path = self.output_dir / f"sdxl_lightning_{self.model_variant}_lightweight"
            pipe.save_pretrained(save_path)
            print(f"Lightweight model saved to: {save_path}")
            
            return pipe
            
        except Exception as e:
            print(f"Lightweight model creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_model(self, pipe: StableDiffusionXLPipeline, method: str):
        """Test the model with sample generation"""
        print(f"\n{'='*50}")
        print(f"Testing {method} model on {self.device}")
        print(f"{'='*50}")
        
        try:
            # Get inference steps
            steps = 1 if self.model_variant == "1step" else int(self.model_variant.replace("step", ""))
            
            # Use appropriate resolution
            if self.device == "cpu":
                height, width = 512, 512  # Smaller for CPU
            else:
                height, width = 768, 768  # Larger for MPS/GPU
            
            print(f"Generating {height}x{width} image with {steps} steps...")
            
            # Clean memory before generation
            self._clean_memory()
            
            # Generate test image
            with torch.inference_mode():
                image = pipe(
                    "A serene mountain landscape at sunset with a crystal clear lake",
                    num_inference_steps=steps,
                    guidance_scale=0,
                    height=height,
                    width=width
                ).images[0]
            
            # Save test image
            test_path = self.output_dir / f"test_{method}_{self.model_variant}_{height}x{width}.png"
            image.save(test_path)
            print(f"✓ Test image saved: {test_path}")
            
        except Exception as e:
            print(f"✗ Model testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    def benchmark_model(self, pipe: StableDiffusionXLPipeline, method: str, iterations: int = 3):
        """Benchmark model performance"""
        import time
        
        print(f"\n{'='*50}")
        print(f"Benchmarking {method} model")
        print(f"{'='*50}")
        
        steps = 1 if self.model_variant == "1step" else int(self.model_variant.replace("step", ""))
        height = width = 512
        times = []
        
        # Warmup
        print("Warming up...")
        try:
            self._clean_memory()
            with torch.inference_mode():
                pipe("warmup test", num_inference_steps=steps, guidance_scale=0, 
                     height=height, width=width)
            print("Warmup complete")
        except Exception as e:
            print(f"Warmup failed: {e}")
            return None
        
        # Benchmark iterations
        print(f"Running {iterations} benchmark iterations...")
        for i in range(iterations):
            try:
                self._clean_memory()
                
                start_time = time.time()
                with torch.inference_mode():
                    pipe(
                        f"A beautiful landscape {i}",
                        num_inference_steps=steps,
                        guidance_scale=0,
                        height=height,
                        width=width
                    )
                end_time = time.time()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                print(f"  Iteration {i+1}/{iterations}: {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  Iteration {i+1} failed: {e}")
                continue
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\nResults:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")
            return avg_time
        else:
            print("Benchmark failed - no successful iterations")
            return None
    
    def _clean_memory(self):
        """Clean up memory"""
        gc.collect()
        if self.device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Quantize SDXL-Lightning models for macOS")
    parser.add_argument("--variant", choices=["1step", "2step", "4step", "8step"], 
                       default="4step", help="Model variant to quantize")
    parser.add_argument("--method", choices=["int8", "mps_fp16", "lightweight", "all"], 
                       default="all", help="Quantization method")
    parser.add_argument("--output_dir", default="./quantized_models", 
                       help="Output directory")
    parser.add_argument("--test", action="store_true", 
                       help="Test the model after quantization")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Benchmark the model")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't delete models after testing (keeps them in memory)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SDXL-Lightning Quantization for macOS")
    print("="*60)
    
    quantizer = SDXLLightningQuantizer(args.variant, args.output_dir)
    
    # Determine methods to run
    if args.method == "all":
        if quantizer.device == "mps":
            methods = ["mps_fp16", "lightweight"]
        else:
            methods = ["int8", "lightweight"]
    else:
        methods = [args.method]
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print(f"{'='*60}")
        
        pipe = None
        
        try:
            if method == "int8":
                pipe = quantizer.quantize_dynamic_int8()
            elif method == "mps_fp16":
                pipe = quantizer.quantize_for_mps()
            elif method == "lightweight":
                pipe = quantizer.create_lightweight_model()
            
            if pipe is not None:
                if args.test:
                    quantizer.test_model(pipe, method)
                
                if args.benchmark:
                    avg_time = quantizer.benchmark_model(pipe, method)
                    if avg_time:
                        results[method] = avg_time
                
                # Cleanup unless disabled
                if not args.no_cleanup:
                    del pipe
                    quantizer._clean_memory()
                    
        except Exception as e:
            print(f"Failed to process method {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if results:
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for method, time in results.items():
            print(f"{method:15s}: {time:.2f}s average")
    
    print(f"\n{'='*60}")
    print("Quantization complete!")
    print(f"Models saved to: {quantizer.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()