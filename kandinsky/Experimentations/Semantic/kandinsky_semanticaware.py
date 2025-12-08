import torch
import numpy as np
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from PIL import Image
import os
from collections import defaultdict

OUTPUT_DIR = "semantic_aware_quantization"
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

class SemanticAwareQuantizer:
    """
    NOVEL: Quantization that adapts based on prompt content complexity
    
    Key Innovation: Different prompts need different precision
    - Complex/detailed prompts → Higher precision (more bits)
    - Simple/abstract prompts → Lower precision (fewer bits)
    """
    def __init__(self):
        self.complexity_level = "medium"  
        self.scales = {}
        self.zero_points = {}
        self.stats = defaultdict(list)
        self.calibration_mode = True

        self.bit_allocation = {
            "high": 8,     
            "medium": 6,    
            "low": 4  
        }
    
    def analyze_prompt_complexity(self, prompt):
        """
        Analyze prompt to determine required precision
        
        Complexity indicators:
        - High: fur, texture, face, portrait, detailed, intricate, realistic
        - Medium: landscape, mountains, city, building
        - Low: abstract, simple, minimal, solid, geometric
        """
        prompt_lower = prompt.lower()
        
        # High complexity keywords
        high_complexity = [
            'fur', 'hair', 'face', 'portrait', 'person', 'human', 'animal',
            'detailed', 'intricate', 'realistic', 'photorealistic', 'texture',
            'fabric', 'cloth', 'skin', 'feather', 'scales', 'wrinkle'
        ]
        
        # Low complexity keywords
        low_complexity = [
            'abstract', 'simple', 'minimal', 'solid', 'geometric',
            'flat', 'basic', 'plain', 'clean', 'smooth'
        ]
        
        high_count = sum(1 for word in high_complexity if word in prompt_lower)
        low_count = sum(1 for word in low_complexity if word in prompt_lower)
        
        if high_count > 0:
            return "high"
        elif low_count > 0:
            return "low"
        else:
            return "medium"
    
    def set_complexity(self, prompt):
        """Set quantization complexity based on prompt"""
        self.complexity_level = self.analyze_prompt_complexity(prompt)
        print(f"  Prompt complexity: {self.complexity_level.upper()}")
        print(f"  Bit allocation: {self.bit_allocation[self.complexity_level]} bits")
    
    def observe(self, layer_name, tensor):
        """Collect statistics during calibration"""
        if not self.calibration_mode:
            return
        
        self.stats[layer_name].append({
            'min': tensor.min().item(),
            'max': tensor.max().item()
        })
    
    def finalize_calibration(self):
        """Compute scales"""
        print("\n" + "="*70)
        print("FINALIZING SEMANTIC-AWARE CALIBRATION")
        print("="*70)
        
        for layer_name, stats_list in self.stats.items():
            all_mins = [s['min'] for s in stats_list]
            all_maxs = [s['max'] for s in stats_list]
            
            qmin = np.percentile(all_mins, 1)
            qmax = np.percentile(all_maxs, 99)

            self.scales[layer_name] = {'qmin': qmin, 'qmax': qmax}
            
            print(f"{layer_name}: [{qmin:.4f}, {qmax:.4f}], samples={len(stats_list)}")
        
        self.calibration_mode = False
        print("="*70)
    
    def quantize(self, layer_name, tensor):
        """Apply semantic-aware quantization"""
        if self.calibration_mode:
            self.observe(layer_name, tensor)
            return tensor
        
        if layer_name not in self.scales:
            return tensor
        

        num_bits = self.bit_allocation[self.complexity_level]
        
        qmin = self.scales[layer_name]['qmin']
        qmax = self.scales[layer_name]['qmax']
        
        scale = (qmax - qmin) / (2**num_bits - 1)
        zero_point = -qmin / scale if scale > 0 else 0
        quant_max = 2**num_bits - 1
        
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point),
            0, quant_max
        )
        dequantized = (quantized - zero_point) * scale
        
        return dequantized

class SpatiallyAdaptiveQuantizer:
    """
    NOVEL: Quantize differently based on spatial importance
    
    Key Innovation: Not all pixels are equally important
    - High frequency regions (edges, details) → High precision
    - Low frequency regions (smooth areas) → Low precision
    """
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scales = {}
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
        print("FINALIZING SPATIALLY-ADAPTIVE CALIBRATION")
        print("="*70)
        
        for layer_name, stats_list in self.stats.items():
            all_mins = [s['min'] for s in stats_list]
            all_maxs = [s['max'] for s in stats_list]
            
            qmin = np.percentile(all_mins, 1)
            qmax = np.percentile(all_maxs, 99)
            
            scale = (qmax - qmin) / (2**self.num_bits - 1)
            zero_point = -qmin / scale if scale > 0 else 0
            
            self.scales[layer_name] = {'scale': scale, 'zero_point': zero_point}
            
            print(f"{layer_name}: [{qmin:.4f}, {qmax:.4f}], scale={scale:.6f}")
        
        self.calibration_mode = False
        print("="*70)
    
    def compute_spatial_importance(self, tensor):
        """
        Compute importance map based on local variance
        High variance = important (edges, details)
        Low variance = less important (smooth areas)
        """
        if len(tensor.shape) != 4: 
            return None
        

        kernel_size = 3
        padding = kernel_size // 2

        unfolded = torch.nn.functional.unfold(
            tensor, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        local_max = unfolded.max(dim=1, keepdim=True)[0]
        local_min = unfolded.min(dim=1, keepdim=True)[0]
        importance = local_max - local_min
        
        # Reshape back
        importance = torch.nn.functional.fold(
            importance,
            output_size=tensor.shape[2:],
            kernel_size=1
        )
        
        # Normalize to [0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        return importance
    
    def quantize(self, layer_name, tensor):
        """Apply spatially-adaptive quantization"""
        if self.calibration_mode:
            self.observe(layer_name, tensor)
            return tensor
        
        if layer_name not in self.scales:
            return tensor
        
        scale = self.scales[layer_name]['scale']
        zero_point = self.scales[layer_name]['zero_point']
        

        importance = self.compute_spatial_importance(tensor)
        
        if importance is not None:

            adaptive_scale = scale * (2.0 - importance) 
        else:
            adaptive_scale = scale
        
        quant_max = 2**self.num_bits - 1
        
        quantized = torch.clamp(
            torch.round(tensor / adaptive_scale + zero_point),
            0, quant_max
        )
        dequantized = (quantized - zero_point) * adaptive_scale
        
        return dequantized

class StaticQuantizer:
    """Baseline static quantization"""
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
            
            print(f"{layer_name}: [{qmin:.4f}, {qmax:.4f}], scale={scale:.6f}")
        
        self.calibration_mode = False
        print("="*70)
    
    def quantize(self, layer_name, tensor):
        if self.calibration_mode:
            self.observe(layer_name, tensor)
            return tensor
        
        if layer_name not in self.scales:
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

def wrap_layer(layer, layer_name, quantizer):
    original_forward = layer.forward
    
    def quantized_forward(x, *args, **kwargs):
        output = original_forward(x, *args, **kwargs)
        if isinstance(output, torch.Tensor):
            output = quantizer.quantize(layer_name, output)
        return output
    
    layer.forward = quantized_forward

def calibrate(quantizer, device):
    print("\nCalibrating quantizer...")
    
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
    
    wrap_layer(pipe.unet.conv_in, 'conv_in', quantizer)
    wrap_layer(pipe.unet.conv_out, 'conv_out', quantizer)
    
    calibration_prompts = [
        "A sunset over mountains",
        "A cat with detailed fur sitting on chair",
        "Abstract geometric shapes",
        "Photorealistic human portrait",
        "Simple minimalist design",
        "Intricate lace fabric texture",
        "Smooth gradient background",
        "Detailed forest scene"
    ]
    
    for i, prompt in enumerate(calibration_prompts):
        print(f"  Prompt {i+1}/8: {prompt[:40]}...")
        
        generator = torch.Generator(device=device).manual_seed(SEED + i)
        
        with torch.no_grad():
            image_embeds, negative_image_embeds = pipe_prior(
                prompt=prompt,
                num_inference_steps=10,
                generator=generator
            ).to_tuple()
            
            generator = torch.Generator(device=device).manual_seed(SEED + i)
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

def generate(quantizer, method_name, prompt, device, is_semantic=False):
    print(f"\nGenerating: {prompt}")
    print(f"Method: {method_name}")
    
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
        if is_semantic:
            quantizer.set_complexity(prompt)
        
        wrap_layer(pipe.unet.conv_in, 'conv_in', quantizer)
        wrap_layer(pipe.unet.conv_out, 'conv_out', quantizer)
    
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    with torch.no_grad():
        image_embeds, negative_image_embeds = pipe_prior(
            prompt=prompt,
            num_inference_steps=10,
            generator=generator
        ).to_tuple()
        
        generator = torch.Generator(device=device).manual_seed(SEED)
        image = pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=512,
            width=512,
            num_inference_steps=25,
            generator=generator
        ).images[0]
    
    safe_name = prompt[:30].replace(' ', '_').replace('/', '_')
    output_path = os.path.join(OUTPUT_DIR, f"{method_name}_{safe_name}.png")
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
    print("SEMANTIC-AWARE QUANTIZATION (NOVEL METHOD)")
    print("="*70)
    print("\nKey Innovation:")
    print("  Adapt quantization precision based on prompt complexity")
    print("  Complex prompts → More bits, Simple prompts → Fewer bits")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Test prompts with different complexity
    test_prompts = [
        ("A sunset over mountains", "medium"),
        ("Photorealistic portrait of a person with detailed facial features", "high"),
        ("Abstract geometric minimal art", "low")
    ]
    
    results = {}
    
    for prompt, expected_complexity in test_prompts:
        print("\n" + "="*70)
        print(f"TESTING: {prompt}")
        print(f"Expected complexity: {expected_complexity}")
        print("="*70)
        
        # Baseline
        print("\n[1/3] Baseline (no quantization)")
        baseline = generate(None, "baseline", prompt, device, False)
        results[(prompt, 'baseline')] = baseline
        
        # Static INT8
        print("\n[2/3] Static INT8")
        static_quant = StaticQuantizer(num_bits=8)
        calibrate(static_quant, device)
        static_img = generate(static_quant, "static", prompt, device, False)
        results[(prompt, 'static')] = static_img
        
        # Semantic-aware
        print("\n[3/3] Semantic-Aware")
        semantic_quant = SemanticAwareQuantizer()
        calibrate(semantic_quant, device)
        semantic_img = generate(semantic_quant, "semantic", prompt, device, True)
        results[(prompt, 'semantic')] = semantic_img
        
        # Compare
        psnr_static = calculate_psnr(baseline, static_img)
        psnr_semantic = calculate_psnr(baseline, semantic_img)
        
        print("\n" + "-"*70)
        print(f"RESULTS FOR: {prompt[:50]}")
        print("-"*70)
        print(f"Static INT8:      PSNR = {psnr_static:.2f} dB")
        print(f"Semantic-Aware:   PSNR = {psnr_semantic:.2f} dB")
        print(f"Improvement:      {psnr_semantic - psnr_static:+.2f} dB")
        print("-"*70)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nKey Finding:")
    print("  Semantic-aware quantization adapts to content complexity,")
    print("  potentially offering better quality-efficiency trade-off")
    print("  than one-size-fits-all static quantization.")

if __name__ == "__main__":
    main()