"""
Original SDXL-Lightning Implementation
Full precision model exactly as provided by ByteDance on HuggingFace
No quantization, no optimization - just the pure model
"""

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pathlib import Path
import time
import argparse
from typing import Optional
import gc

class OriginalSDXLLightning:
    """
    Original SDXL-Lightning implementation without modifications
    Follows the exact specification from HuggingFace
    """
    
    def __init__(self, variant: str = "4step", use_lora: bool = False):
        """
        Initialize SDXL-Lightning
        
        Args:
            variant: "1step", "2step", "4step", or "8step"
            use_lora: If True, use LoRA weights; if False, use full UNet
        """
        self.variant = variant
        self.use_lora = use_lora
        
        # Model identifiers
        self.base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        self.lightning_repo = "ByteDance/SDXL-Lightning"
        
        # Device selection
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        # Set checkpoint filename
        self._set_checkpoint_name()
        
        # Pipeline
        self.pipe = None
    
    def _get_device(self) -> str:
        """Smart device selection"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _set_checkpoint_name(self):
        """Set the correct checkpoint filename based on variant and type"""
        if self.use_lora:
            # LoRA checkpoints
            if self.variant == "1step":
                raise ValueError("1-step LoRA not available, use full UNet")
            self.ckpt_name = f"sdxl_lightning_{self.variant}_lora.safetensors"
        else:
            # Full UNet checkpoints
            if self.variant == "1step":
                self.ckpt_name = "sdxl_lightning_1step_unet_x0.safetensors"
            else:
                self.ckpt_name = f"sdxl_lightning_{self.variant}_unet.safetensors"
        
        print(f"Checkpoint: {self.ckpt_name}")
    
    def load_model_full_unet(self) -> StableDiffusionXLPipeline:
        """
        Load SDXL-Lightning with full UNet replacement
        This is the recommended method for best quality
        
        Official code from HuggingFace:
        https://huggingface.co/ByteDance/SDXL-Lightning
        """
        print("\n" + "="*60)
        print("Loading Original SDXL-Lightning (Full UNet)")
        print("="*60)
        print(f"Variant: {self.variant}")
        print(f"Base model: {self.base_model}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        # Step 1: Load base UNet configuration
        print("Step 1/5: Loading UNet configuration...")
        unet = UNet2DConditionModel.from_config(
            self.base_model, 
            subfolder="unet"
        )
        print(f"  ✓ UNet config loaded")
        print(f"  - Input channels: {unet.config.in_channels}")
        print(f"  - Output channels: {unet.config.out_channels}")
        print(f"  - Attention head dim: {unet.config.attention_head_dim}")
        
        # Step 2: Download Lightning weights
        print("\nStep 2/5: Downloading SDXL-Lightning weights...")
        checkpoint_path = hf_hub_download(
            self.lightning_repo, 
            self.ckpt_name
        )
        print(f"  ✓ Downloaded to: {checkpoint_path}")
        
        # Step 3: Load Lightning weights into UNet
        print("\nStep 3/5: Loading Lightning weights into UNet...")
        state_dict = load_file(checkpoint_path, device="cpu")
        unet.load_state_dict(state_dict)
        print(f"  ✓ Loaded {len(state_dict)} weight tensors")
        
        # Calculate model size
        total_params = sum(p.numel() for p in unet.parameters())
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Model size: {total_params * 4 / (1024**3):.2f} GB (FP32)")
        
        # Step 4: Move UNet to device
        print(f"\nStep 4/5: Moving UNet to {self.device}...")
        if self.device == "cuda":
            unet = unet.to("cuda", torch.float16)
            torch_dtype = torch.float16
            variant = "fp16"
            print(f"  ✓ Using FP16 precision (recommended for CUDA)")
        elif self.device == "mps":
            unet = unet.to("mps", torch.float16)
            torch_dtype = torch.float16
            variant = "fp16"
            print(f"  ✓ Using FP16 precision (recommended for MPS)")
        else:
            unet = unet.to("cpu", torch.float32)
            torch_dtype = torch.float32
            variant = None
            print(f"  ✓ Using FP32 precision (required for CPU)")
        
        # Step 5: Load full SDXL pipeline with Lightning UNet
        print("\nStep 5/5: Loading complete SDXL pipeline...")
        if variant:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                unet=unet,
                torch_dtype=torch_dtype,
                variant=variant
            ).to(self.device)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                unet=unet,
                torch_dtype=torch_dtype
            ).to(self.device)
        
        print("  ✓ Pipeline loaded")
        print("  - Text Encoder")
        print("  - Text Encoder 2")
        print("  - VAE")
        print("  - UNet (Lightning)")
        
        # Configure scheduler for Lightning
        print("\nConfiguring scheduler for Lightning...")
        if self.variant == "1step":
            # 1-step uses "sample" prediction type
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
                prediction_type="sample"
            )
            print(f"  ✓ Euler scheduler (sample prediction, trailing timesteps)")
        else:
            # 2-step, 4-step, 8-step use "epsilon" prediction
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing"
            )
            print(f"  ✓ Euler scheduler (epsilon prediction, trailing timesteps)")
        
        # Print memory usage
        self._print_memory_usage(pipe)
        
        print("\n" + "="*60)
        print("✓ Model loaded successfully!")
        print("="*60 + "\n")
        
        self.pipe = pipe
        return pipe
    
    def load_model_lora(self) -> StableDiffusionXLPipeline:
        """
        Load SDXL-Lightning with LoRA weights
        Use only if applying to non-SDXL base models
        
        Official code from HuggingFace
        """
        print("\n" + "="*60)
        print("Loading Original SDXL-Lightning (LoRA)")
        print("="*60)
        print(f"Variant: {self.variant}")
        print(f"Base model: {self.base_model}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        # Step 1: Load base SDXL pipeline
        print("Step 1/4: Loading base SDXL-XL pipeline...")
        if self.device in ["cuda", "mps"]:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32
            ).to(self.device)
        print("  ✓ Base pipeline loaded")
        
        # Step 2: Download LoRA weights
        print("\nStep 2/4: Downloading Lightning LoRA weights...")
        lora_path = hf_hub_download(
            self.lightning_repo,
            self.ckpt_name
        )
        print(f"  ✓ Downloaded to: {lora_path}")
        
        # Step 3: Load and fuse LoRA
        print("\nStep 3/4: Loading and fusing LoRA weights...")
        pipe.load_lora_weights(lora_path)
        print("  ✓ LoRA weights loaded")
        
        pipe.fuse_lora()
        print("  ✓ LoRA weights fused into model")
        
        # Step 4: Configure scheduler
        print("\nStep 4/4: Configuring scheduler...")
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing"
        )
        print("  ✓ Euler scheduler configured")
        
        self._print_memory_usage(pipe)
        
        print("\n" + "="*60)
        print("✓ Model loaded successfully!")
        print("="*60 + "\n")
        
        self.pipe = pipe
        return pipe
    
    def _print_memory_usage(self, pipe):
        """Print estimated memory usage"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"\nGPU Memory:")
            print(f"  - Allocated: {allocated:.2f} GB")
            print(f"  - Reserved: {reserved:.2f} GB")
        elif self.device == "mps":
            print(f"\nMPS Memory: Apple Silicon unified memory")
        else:
            print(f"\nCPU Memory: System RAM")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
        output_path: Optional[Path] = None
    ):
        """
        Generate an image using the loaded model
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            num_inference_steps: Number of denoising steps (None = use variant default)
            guidance_scale: CFG scale (Lightning uses 0)
            height: Image height in pixels
            width: Image width in pixels
            seed: Random seed for reproducibility
            output_path: Where to save the image
        
        Returns:
            PIL Image and generation time
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model_full_unet() or load_model_lora() first")
        
        # Set default steps based on variant
        if num_inference_steps is None:
            if self.variant == "1step":
                num_inference_steps = 1
            elif self.variant == "2step":
                num_inference_steps = 2
            elif self.variant == "4step":
                num_inference_steps = 4
            elif self.variant == "8step":
                num_inference_steps = 8
        
        print("\n" + "="*60)
        print("Generating Image")
        print("="*60)
        print(f"Prompt: {prompt}")
        print(f"Steps: {num_inference_steps}")
        print(f"Resolution: {width}×{height}")
        print(f"Guidance scale: {guidance_scale}")
        if seed is not None:
            print(f"Seed: {seed}")
        print("="*60 + "\n")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)
        
        # Clean memory
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # Generate
        start_time = time.time()
        
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
        
        image = result.images[0]
        elapsed = time.time() - start_time
        
        print(f"\n✓ Generation complete in {elapsed:.2f} seconds")
        print(f"  - Time per step: {elapsed/num_inference_steps:.2f}s")
        
        # Save if path provided
        if output_path:
            image.save(output_path)
            print(f"✓ Saved to: {output_path}")
        
        return image, elapsed
    
    def benchmark(
        self,
        prompt: str = "A serene landscape with mountains and lake",
        iterations: int = 3,
        height: int = 1024,
        width: int = 1024
    ):
        """Benchmark the model performance"""
        if self.pipe is None:
            raise RuntimeError("Model not loaded")
        
        print("\n" + "="*60)
        print("BENCHMARK")
        print("="*60)
        print(f"Variant: {self.variant}")
        print(f"Resolution: {width}×{height}")
        print(f"Iterations: {iterations}")
        print("="*60 + "\n")
        
        # Get steps
        steps = int(self.variant.replace("step", "")) if self.variant != "1step" else 1
        
        times = []
        
        # Warmup
        print("Warmup run...")
        try:
            self.generate_image(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps
            )
        except Exception as e:
            print(f"Warmup failed: {e}")
        
        # Benchmark iterations
        print(f"\nRunning {iterations} benchmark iterations...\n")
        for i in range(iterations):
            try:
                _, elapsed = self.generate_image(
                    prompt=f"{prompt} {i}",
                    height=height,
                    width=width,
                    num_inference_steps=steps
                )
                times.append(elapsed)
                print(f"Iteration {i+1}/{iterations}: {elapsed:.2f}s\n")
            except Exception as e:
                print(f"Iteration {i+1} failed: {e}\n")
        
        # Results
        if times:
            print("="*60)
            print("RESULTS")
            print("="*60)
            print(f"Average: {sum(times)/len(times):.2f}s")
            print(f"Min: {min(times):.2f}s")
            print(f"Max: {max(times):.2f}s")
            print(f"Total: {sum(times):.2f}s")
            print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Original SDXL-Lightning Implementation"
    )
    parser.add_argument(
        "--variant",
        choices=["1step", "2step", "4step", "8step"],
        default="4step",
        help="Model variant (default: 4step)"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA weights instead of full UNet"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over mountains with a crystal clear lake",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: variant default)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output filename (default: output.png)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark instead of single generation"
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Initialize model
    model = OriginalSDXLLightning(
        variant=args.variant,
        use_lora=args.lora
    )
    
    # Load model
    if args.lora:
        model.load_model_lora()
    else:
        model.load_model_full_unet()
    
    # Run benchmark or generate single image
    if args.benchmark:
        model.benchmark(
            prompt=args.prompt,
            iterations=args.benchmark_iterations,
            height=args.height,
            width=args.width
        )
    else:
        model.generate_image(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            seed=args.seed,
            output_path=Path(args.output)
        )

if __name__ == "__main__":
    main()