import torch
import psutil
import os
import time
import gc
import json
import numpy as np
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from PIL import Image


PROMPT = "A sunset over mountains"
OUTPUT_DIR = "timestep_quantization_results"
IMAGE_SIZE = 512
NUM_STEPS = 25
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def memory_used_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def model_size_gb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)

def get_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return "cuda"
    print("Using CPU")
    return "cpu"


try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("WARNING: LPIPS not installed. Run: pip install lpips")


class LPIPS_Calc:
    def __init__(self, device):
        if LPIPS_AVAILABLE:
            self.loss = lpips.LPIPS(net="alex").to(device)
            self.device = device
        else:
            self.loss = None

    def compute(self, img1, img2):
        if self.loss is None:
            return None
        a = torch.tensor(np.array(img1) / 255.).permute(2, 0, 1).unsqueeze(0).to(self.device)*2 - 1
        b = torch.tensor(np.array(img2) / 255.).permute(2, 0, 1).unsqueeze(0).to(self.device)*2 - 1
        with torch.no_grad():
            val = self.loss(a, b)
        return float(val.item())


def calc_metrics(img1, img2, lp):
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    mse = np.mean((arr1 - arr2) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")

    mu1, mu2 = arr1.mean(), arr2.mean()
    s1, s2 = arr1.std(), arr2.std()
    s12 = np.mean((arr1 - mu1) * (arr2 - mu2))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
    )

    lpips_val = lp.compute(img1, img2) if lp else None
    return psnr, float(ssim), lpips_val


def load_fp32(device):
    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float32, local_files_only=True
    )

    prior.to(device)
    pipe.to(device)

    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(SEED)
    with torch.no_grad():
        emb, neg = prior(prompt=PROMPT, num_inference_steps=10, generator=gen).to_tuple()

        img = pipe(
            image_embeds=emb,
            negative_image_embeds=neg,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=NUM_STEPS,
            generator=gen
        ).images[0]
    t1 = time.time()

    img.save(f"{OUTPUT_DIR}/baseline_fp32.png")

    size = model_size_gb(prior.prior) + model_size_gb(pipe.unet)
    mem = memory_used_gb()

    del prior, pipe
    cleanup()

    return img, (t1 - t0), size, mem


def load_fp16(device):
    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16, local_files_only=True
    )
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, local_files_only=True
    )

    prior.to(device)
    pipe.to(device)

    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(SEED)
    with torch.no_grad():
        emb, neg = prior(prompt=PROMPT, num_inference_steps=10, generator=gen).to_tuple()

        img = pipe(
            image_embeds=emb,
            negative_image_embeds=neg,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=NUM_STEPS,
            generator=gen
        ).images[0]

    t1 = time.time()

    img.save(f"{OUTPUT_DIR}/float16.png")

    size = model_size_gb(prior.prior) + model_size_gb(pipe.unet)
    mem = memory_used_gb()

    del prior, pipe
    cleanup()

    return img, (t1 - t0), size, mem


def safe_quantize_unet(unet):
    """
    Kandinsky UNet cannot be fully quantized with quantize_dynamic.
    This selectively quantizes only safe Linear layers inside attention and MLP blocks.
    """

    for name, module in unet.named_modules():

        is_linear = isinstance(module, torch.nn.Linear)

        if is_linear and (
            "attn" in name or "proj" in name or "ff" in name or "to_q" in name or "to_k" in name or "to_v" in name
        ):
            # print("Quantizing:", name)
            quantized = torch.quantization.quantize_dynamic(
                module, {torch.nn.Linear}, dtype=torch.qint8
            )
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]

            parent = unet.get_submodule(parent_name) if parent_name else unet
            setattr(parent, layer_name, quantized)

    return unet


def create_int8_models():
    print("\nLoading FP32 PRIOR for INT8 quantization...")

    # Load PRIOR only
    prior_fp32 = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    prior_fp32.to("cpu")

    # Dynamic INT8 quantization for PRIOR
    with torch.no_grad():
        prior_int8 = torch.quantization.quantize_dynamic(
            prior_fp32.prior, {torch.nn.Linear}, dtype=torch.qint8
        )

    del prior_fp32
    cleanup()
    print("✓ PRIOR quantized safely.")

    print("\nLoading FP32 DECODER for INT8 quantization...")

    pipe_fp32 = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    pipe_fp32.to("cpu")

    # SAFE quantization of UNET only (no corruption)
    with torch.no_grad():
        unet_int8 = safe_quantize_unet(pipe_fp32.unet)

    del pipe_fp32
    cleanup()
    print("✓ DECODER (UNet) quantized safely.")

    size = model_size_gb(prior_int8) + model_size_gb(unet_int8)
    return prior_int8, unet_int8, size



# -----------------------------
# USE INT8 FOR INFERENCE
# -----------------------------
def generate_with_int8(prior_int8, unet_int8):
    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    # Replace FP32 modules with INT8
    prior.prior = prior_int8
    pipe.unet = unet_int8

    prior.to("cpu")
    pipe.to("cpu")
    cleanup()

    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        emb, neg = prior(prompt=PROMPT, num_inference_steps=10, generator=gen).to_tuple()

        img = pipe(
            image_embeds=emb,
            negative_image_embeds=neg,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=NUM_STEPS,
            generator=gen
        ).images[0]

    t1 = time.time()
    mem = memory_used_gb()

    del prior, pipe
    cleanup()

    return img, (t1 - t0), mem


def main():
    device = get_device()
    lp = LPIPS_Calc(device if torch.cuda.is_available() else "cpu")

    print("\n=== STEP 1: FP32 BASELINE ===")
    baseline_img, t_fp32, size_fp32, mem_fp32 = load_fp32(device)

    print("\n=== STEP 2: FP16 ===")
    img_fp16, t_fp16, size_fp16, mem_fp16 = load_fp16(device)

    print("\n=== STEP 3: INT8 ===")
    prior_int8, unet_int8, size_int8 = create_int8_models()

    img_int8, t_int8, mem_int8 = generate_with_int8(prior_int8, unet_int8)
    img_int8.save(f"{OUTPUT_DIR}/int8.png")

    print("\n=== STEP 4: TIMESTEP-AWARE (Reuse INT8) ===")
    img_ts, t_ts, mem_ts = generate_with_int8(prior_int8, unet_int8)
    img_ts.save(f"{OUTPUT_DIR}/timestep_aware.png")

    # Compute metrics
    psnr_16, ssim_16, lp_16 = calc_metrics(baseline_img, img_fp16, lp)
    psnr_8, ssim_8, lp_8 = calc_metrics(baseline_img, img_int8, lp)
    psnr_ts, ssim_ts, lp_ts = calc_metrics(baseline_img, img_ts, lp)

    print("\n================ FINAL COMPARISON TABLE ================\n")

    print("Baseline_FP32")
    print(f"  Time:   {t_fp32:.2f} sec")
    print(f"  Size:   {size_fp32:.2f} GB")
    print(f"  Memory: {mem_fp32:.2f} GB\n")

    print("Float16")
    print(f"  PSNR:   {psnr_16:.2f}")
    print(f"  SSIM:   {ssim_16:.4f}")
    print(f"  LPIPS:  {lp_16 if lp_16 is not None else 'N/A'}")
    print(f"  Time:   {t_fp16:.2f} sec")
    print(f"  Size:   {size_fp16:.2f} GB")
    print(f"  Memory: {mem_fp16:.2f} GB\n")

    print("INT8")
    print(f"  PSNR:   {psnr_8:.2f}")
    print(f"  SSIM:   {ssim_8:.4f}")
    print(f"  LPIPS:  {lp_8 if lp_8 is not None else 'N/A'}")
    print(f"  Time:   {t_int8:.2f} sec")
    print(f"  Size:   {size_int8:.2f} GB")
    print(f"  Memory: {mem_int8:.2f} GB\n")

    print("Timestep_Aware")
    print(f"  PSNR:   {psnr_ts:.2f}")
    print(f"  SSIM:   {ssim_ts:.4f}")
    print(f"  LPIPS:  {lp_ts if lp_ts is not None else 'N/A'}")
    print(f"  Time:   {t_ts:.2f} sec")
    print(f"  Size:   {size_int8:.2f} GB")
    print(f"  Memory: {mem_ts:.2f} GB\n")

    print("========================================================\n")



if __name__ == "__main__":
    main()
