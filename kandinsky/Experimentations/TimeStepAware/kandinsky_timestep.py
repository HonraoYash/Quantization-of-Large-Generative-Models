import torch
import numpy as np
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from PIL import Image
import os
from collections import defaultdict

OUTPUT_DIR = "timestep_aware_quantization_v2"
SEED = 42
PROMPT = "A sunset over mountains"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ImprovedTimestepQuantizer:
    """Improved timestep-aware quantizer with proper tracking"""
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.timestep_stats = defaultdict(lambda: defaultdict(list))
        self.scales = {}
        self.zero_points = {}
        self.calibration_mode = True
        self.current_group = None
        
    def set_timestep(self, step_index, total_steps):
        """Set timestep group based on current step"""
        progress = step_index / total_steps
        
        if progress < 0.33:
            self.current_group = "early"
        elif progress < 0.67:
            self.current_group = "mid"
        else:
            self.current_group = "late"
    
    def observe(self, layer_name, tensor):
        """Collect statistics during calibration"""
        if not self.calibration_mode or self.current_group is None:
            return
        
        self.timestep_stats[self.current_group][layer_name].append({
            'min': tensor.min().item(),
            'max': tensor.max().item()
        })
    
    def finalize_calibration(self):
        """Compute scales for each timestep group"""
        print("\n" + "="*70)
        print("FINALIZING TIMESTEP-AWARE CALIBRATION")
        print("="*70)
        
        for group in ["early", "mid", "late"]:
            print(f"\n{group.upper()} TIMESTEPS:")
            print("-"*70)
            
            for layer_name, stats_list in self.timestep_stats[group].items():
                if not stats_list:
                    print(f"  {layer_name}: No data collected!")
                    continue
                
                all_mins = [s['min'] for s in stats_list]
                all_maxs = [s['max'] for s in stats_list]
                
                qmin = np.percentile(all_mins, 1)
                qmax = np.percentile(all_maxs, 99)
                
                scale = (qmax - qmin) / (2**self.num_bits - 1)
                zero_point = -qmin / scale if scale > 0 else 0
                
                key = (group, layer_name)
                self.scales[key] = scale
                self.zero_points[key] = zero_point
                
                print(f"  {layer_name}: [{qmin:.4f}, {qmax:.4f}], scale={scale:.6f}, samples={len(stats_list)}")
        
        self.calibration_mode = False
        print("\n" + "="*70)
    
    def quantize(self, layer_name, tensor):
        """Apply quantization"""
        if self.calibration_mode:
            self.observe(layer_name, tensor)
            return tensor
        
        if self.current_group is None:
            return tensor
        
        key = (self.current_group, layer_name)
        if key not in self.scales:
            return tensor
        
        scale = self.scales[key]
        if scale == 0:
            return tensor
        
        zero_point = self.zero_points[key]
        quant_max = 2**self.num_bits - 1
        
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point),
            0, quant_max
        )
        dequantized = (quantized - zero_point) * scale
        
        return dequantized

class StaticQuantizer:
    """Standard static quantization"""
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scales = {}
        self.zero_points = {}
        self.stats = defaultdict(list)
        self.calibration_mode = True
    
    def observe(self, layer_name, tensor):
        if not self.calibration_mode:
            return
        
        self.stats[layer_name].append({
            'min': tensor.min().item(),
            'max': tensor.max().item()
        })
    
    def finalize_calibration(self):
        print("\n" + "="*70)
        print("FINALIZING STATIC CALIBRATION")
        print("="*70)
        
        for layer_name, stats_list in self.stats.items():
            all_mins = [s['min'] for s in stats_list]
            all_maxs = [s['max'] for s in stats_list]
            
            qmin = np.percentile(all_mins, 1)
            qmax = np.percentile(all_maxs, 99)
            
            scale = (qmax - qmin) / (2**self.num_bits - 1)
            zero_point = -qmin / scale if scale > 0 else 0
            
            self.scales[layer_name] = scale
            self.zero_points[layer_name] = zero_point
            
            print(f"{layer_name}: [{qmin:.4f}, {qmax:.4f}], scale={scale:.6f}, samples={len(stats_list)}")
        
        self.calibration_mode = False
        print("="*70)
    
    def quantize(self, layer_name, tensor):
        if self.calibration_mode:
            self.observe(layer_name, tensor)
            return tensor
        
        if layer_name not in self.scales or self.scales[layer_name] == 0:
            return tensor
        
        scale = self.scales[layer_name]
        zero_point = self.zero_points[layer_name]
        quant_max = 2**self.num_bits - 1
        
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point),
            0, quant_max
        )
        dequantized = (quantized - zero_point) * scale
        
        return dequantized

def wrap_layer_with_quantizer(layer, layer_name, quantizer):
    """Wrap a single layer with quantization"""
    original_forward = layer.forward
    
    def quantized_forward(x, *args, **kwargs):
        output = original_forward(x, *args, **kwargs)
        if isinstance(output, torch.Tensor):
            output = quantizer.quantize(layer_name, output)
        return output
    
    layer.forward = quantized_forward

def wrap_unet_for_timestep_tracking(unet, quantizer):
    """Wrap UNet forward to track timesteps"""
    original_forward = unet.forward
    
    def tracked_forward(sample, timestep, *args, **kwargs):
        # Reset counter at start of each image generation
        if not hasattr(tracked_forward, 'step_count'):
            tracked_forward.step_count = 0
        
        # Update timestep group
        quantizer.set_timestep(tracked_forward.step_count, 25)
        tracked_forward.step_count += 1
        
        # Reset after 25 steps
        if tracked_forward.step_count >= 25:
            tracked_forward.step_count = 0
        
        return original_forward(sample, timestep, *args, **kwargs)
    
    unet.forward = tracked_forward

def calibrate(quantizer, device, is_timestep_aware):
    """Run calibration"""
    print(f"\nCalibrating {'timestep-aware' if is_timestep_aware else 'static'} quantizer...")
    
    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        torch_dtype=torch.float32
    )
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=torch.float32
    )
    
    pipe_prior.to(device)
    pipe.to(device)
    
    # Wrap layers
    wrap_layer_with_quantizer(pipe.unet.conv_in, 'conv_in', quantizer)
    wrap_layer_with_quantizer(pipe.unet.conv_out, 'conv_out', quantizer)
    
    if is_timestep_aware:
        wrap_unet_for_timestep_tracking(pipe.unet, quantizer)
    
    # Calibration loop
    calibration_prompts = [
        "A sunset over mountains",
        "A cat sitting on chair", 
        "Modern city skyline",
        "Forest with trees",
        "Ocean waves",
        "Desert landscape",
        "Snowy mountain peak",
        "Tropical beach"
    ]
    
    for i, prompt in enumerate(calibration_prompts):
        print(f"  Prompt {i+1}/{len(calibration_prompts)}: {prompt[:30]}...")
        
        generator = torch.Generator(device=device).manual_seed(SEED + i)
        
        with torch.no_grad():
            image_embeds, negative_image_embeds = pipe_prior(
                prompt=prompt,
                num_inference_steps=10,
                generator=generator
            ).to_tuple()
            
            generator = torch.Generator(device=device).manual_seed(SEED + i)
            
            # Reset step counter before each generation
            if is_timestep_aware and hasattr(pipe.unet.forward, 'step_count'):
                pipe.unet.forward.step_count = 0
            
            _ = pipe(
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                height=512,
                width=512,
                num_inference_steps=25,
                generator=generator
            )
    
    quantizer.finalize_calibration()
    
    del pipe_prior, pipe
    torch.cuda.empty_cache()

def generate(quantizer, method_name, device, is_timestep_aware):
    """Generate image"""
    print(f"\nGenerating with {method_name}...")
    
    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        torch_dtype=torch.float32
    )
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=torch.float32
    )
    
    pipe_prior.to(device)
    pipe.to(device)
    
    if quantizer is not None:
        wrap_layer_with_quantizer(pipe.unet.conv_in, 'conv_in', quantizer)
        wrap_layer_with_quantizer(pipe.unet.conv_out, 'conv_out', quantizer)
        
        if is_timestep_aware:
            wrap_unet_for_timestep_tracking(pipe.unet, quantizer)
    
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    with torch.no_grad():
        image_embeds, negative_image_embeds = pipe_prior(
            prompt=PROMPT,
            num_inference_steps=10,
            generator=generator
        ).to_tuple()
        
        generator = torch.Generator(device=device).manual_seed(SEED)
        
        # Reset counter
        if quantizer is not None and is_timestep_aware and hasattr(pipe.unet.forward, 'step_count'):
            pipe.unet.forward.step_count = 0
        
        image = pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=512,
            width=512,
            num_inference_steps=25,
            generator=generator
        ).images[0]
    
    output_path = os.path.join(OUTPUT_DIR, f"{method_name}.png")
    image.save(output_path)
    print(f"Saved: {output_path}")
    
    del pipe_prior, pipe
    torch.cuda.empty_cache()
    
    return image

def calculate_psnr(img1, img2):
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    print("="*70)
    print("TIMESTEP-AWARE QUANTIZATION (IMPROVED)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    print("="*70)
    print("STEP 1: BASELINE")
    print("="*70)
    baseline = generate(None, "baseline", device, False)
    

    print("\n" + "="*70)
    print("STEP 2: STATIC QUANTIZATION")
    print("="*70)
    static_quant = StaticQuantizer(num_bits=8)
    calibrate(static_quant, device, False)
    static_img = generate(static_quant, "static_int8", device, False)

    print("\n" + "="*70)
    print("STEP 3: TIMESTEP-AWARE QUANTIZATION")
    print("="*70)
    ts_quant = ImprovedTimestepQuantizer(num_bits=8)
    calibrate(ts_quant, device, True)
    ts_img = generate(ts_quant, "timestep_aware_int8", device, True)
    

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    psnr_static = calculate_psnr(baseline, static_img)
    psnr_ts = calculate_psnr(baseline, ts_img)
    improvement = psnr_ts - psnr_static
    
    print(f"\nStatic INT8:           PSNR = {psnr_static:.2f} dB")
    print(f"Timestep-Aware INT8:   PSNR = {psnr_ts:.2f} dB")
    print(f"Improvement:           {improvement:+.2f} dB")
    
    if improvement > 0:
        print("\nTimestep-aware quantization is BETTER!")
    else:
        print("\nStatic quantization performed better in this case.")
        print("Possible reasons:")
        print("  - conv_in/conv_out may not benefit from timestep-aware approach")
        print("  - These layers see similar distributions across timesteps")
        print("  - Middle layers (resnets, attention) might benefit more")

    comparison = Image.new('RGB', (512 * 3, 512))
    comparison.paste(baseline, (0, 0))
    comparison.paste(static_img, (512, 0))
    comparison.paste(ts_img, (512 * 2, 0))
    comparison.save(os.path.join(OUTPUT_DIR, "comparison.png"))
    print(f"\nSaved: {OUTPUT_DIR}/comparison.png")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()