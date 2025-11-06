from __future__ import annotations
import os
import argparse

from PIL import Image
import torch

# we’ll reuse the metrics you already used
from math import log10
import lpips
from skimage.metrics import structural_similarity as ssim
import numpy as np


def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype("float32") / 255.0  # (H,W,3)
    arr = np.transpose(arr, (2, 0, 1))             # (3,H,W)
    return torch.from_numpy(arr).unsqueeze(0)      # (1,3,H,W)


def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
    t1 = pil_to_tensor(img1)
    t2 = pil_to_tensor(img2)
    mse = torch.mean((t1 - t2) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * log10(1.0 / mse)


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1).astype("float32") / 255.0
    arr2 = np.array(img2).astype("float32") / 255.0
    s = ssim(arr1, arr2, channel_axis=2, data_range=1.0)
    return float(s)


def compute_lpips(img1: Image.Image, img2: Image.Image) -> float:
    loss_fn = lpips.LPIPS(net="alex")
    t1 = pil_to_tensor(img1)
    t2 = pil_to_tensor(img2)
    with torch.no_grad():
        dist = loss_fn(t1, t2).item()
    return float(dist)


def load_flux_local(model_dir: str, device: str, dtype: torch.dtype):
    """
    Load FLUX.1-schnell from a *local* directory that you already downloaded
    with snapshot_download. We let diffusers decide placement, so we do NOT
    call pipe.to(...) later.
    """
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        use_safetensors=True,
        device_map="balanced",     # <-- this is the thing that was active
        low_cpu_mem_usage=True,
    )

    # try to enable some memory savers (safe to ignore if missing)
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pass

    return pipe


def maybe_quantize_cpu(pipe):
    """
    If we ever run this on CPU and torchao is installed, do a small
    weight-only quantization to have *something* to report.
    """
    if not (torch.device("cpu") == torch.device(pipe._execution_device)):
        # we’re probably on GPU / balanced — skip
        return pipe

    try:
        from torchao.quantization import quantize_, int8_weight_only
        target = getattr(pipe, "transformer", pipe)
        quantize_(target, int8_weight_only())
        print("[quant] applied cpu int8 weight-only quantization")
    except Exception as e:
        print(f"[quant] skipping quantization: {e}")

    return pipe


def generate(pipe, prompt: str, height: int, width: int, steps: int) -> Image.Image:
    out = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=0.0,
        max_sequence_length=128,
    )
    return out.images[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="local dir: /scratch/.../FLUX.1-schnell")
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument("--output_dir", required=True, help="where to save images + metrics.txt")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # decide device + dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[info] loading FLUX from {args.model_dir} on device={device} dtype={dtype}")

    pipe = load_flux_local(args.model_dir, device, dtype)

    # 1) baseline image
    print("[run] generating baseline image ...")
    img_base = generate(pipe, args.prompt, args.height, args.width, args.steps)
    base_path = os.path.join(args.output_dir, "flux_base.png")
    save_image(img_base, base_path)
    print(f"[ok] saved {base_path}")

    # 2) "quantized" image – here we just run again after maybe_quantize_cpu()
    print("[run] (optional) quantizing and generating second image ...")
    pipe = maybe_quantize_cpu(pipe)
    img_quant = generate(pipe, args.prompt, args.height, args.width, args.steps)
    quant_path = os.path.join(args.output_dir, "flux_quant.png")
    save_image(img_quant, quant_path)
    print(f"[ok] saved {quant_path}")

    # 3) metrics
    print("[metrics] computing PSNR / SSIM / LPIPS ...")
    psnr_v = compute_psnr(img_base, img_quant)
    ssim_v = compute_ssim(img_base, img_quant)
    lpips_v = compute_lpips(img_base, img_quant)

    metrics_txt = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write(f"prompt: {args.prompt}\n")
        f.write(f"base:   {base_path}\n")
        f.write(f"quant:  {quant_path}\n")
        f.write(f"PSNR:   {psnr_v:.4f}\n")
        f.write(f"SSIM:   {ssim_v:.4f}\n")
        f.write(f"LPIPS:  {lpips_v:.4f}\n")

    print("[done]")
    print(f"PSNR : {psnr_v:.4f}")
    print(f"SSIM : {ssim_v:.4f}")
    print(f"LPIPS: {lpips_v:.4f}")
    print(f"metrics saved to {metrics_txt}")


if __name__ == "__main__":
    main()
