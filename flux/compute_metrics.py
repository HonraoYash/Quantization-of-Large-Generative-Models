#!/usr/bin/env python3
"""
Compute simple image similarity metrics for our FLUX quantization experiments.

Usage:
  python compute_metrics.py --ref flux_out.png --test flux_out_quant.png

  or for folders:
  python compute_metrics.py --ref-dir outputs/baseline --test-dir outputs/quant
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, Optional

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


# ---------- basic helpers ----------

def pil_to_torch(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image -> torch tensor (1,3,H,W) in [0,1], float32.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)
    arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
    t = torch.from_numpy(arr).unsqueeze(0)  # (1,3,H,W)
    return t


def _resize_like(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != y.shape[-2:]:
        x = F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ---------- metrics ----------

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = _resize_like(x, y)
    mse = F.mse_loss(x, y)
    if mse.item() < eps:
        return 99.0
    return float(10.0 * torch.log10(1.0 / mse).item())


def ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Tiny SSIM: good enough to see degradation from quantization.
    """
    x = _resize_like(x, y)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 3x3 box filter
    kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0

    def _conv(z):
        outs = []
        for c in range(z.shape[1]):
            outs.append(F.conv2d(z[:, c:c+1], kernel, padding=1))
        return torch.cat(outs, dim=1)

    mu_x = _conv(x)
    mu_y = _conv(y)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = _conv(x ** 2) - mu_x2
    sigma_y2 = _conv(y ** 2) - mu_y2
    sigma_xy = _conv(x * y) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den
    return float(ssim_map.mean().item())


def lpips_score(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """
    Optional LPIPS. If lpips isn't installed, just return None.
    """
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    x = _resize_like(x, y)
    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0
    net = lpips.LPIPS(net="alex")
    with torch.no_grad():
        val = net(x, y)
    return float(val.item())


# ---------- core comparison ----------

def compare_pair(ref_path: str, test_path: str) -> Dict[str, float]:
    ref_img = Image.open(ref_path)
    test_img = Image.open(test_path)

    ref_t = pil_to_torch(ref_img)
    test_t = pil_to_torch(test_img)

    metrics: Dict[str, float] = {}
    metrics["psnr"] = psnr(ref_t, test_t)
    metrics["ssim"] = ssim(ref_t, test_t)
    lp = lpips_score(ref_t, test_t)
    if lp is not None:
        metrics["lpips"] = lp
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", help="reference image")
    parser.add_argument("--test", help="test image")
    parser.add_argument("--ref-dir", help="reference folder")
    parser.add_argument("--test-dir", help="test folder")
    args = parser.parse_args()

    # single image mode
    if args.ref and args.test:
        m = compare_pair(args.ref, args.test)
        print(f"Reference: {args.ref}")
        print(f"Test     : {args.test}")
        for k, v in m.items():
            print(f"{k.upper():6s}: {v:.4f}")
        return

    # folder mode
    if args.ref_dir and args.test_dir:
        ref_files = [f for f in os.listdir(args.ref_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        test_files = [f for f in os.listdir(args.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        common = sorted(set(ref_files) & set(test_files))
        if not common:
            raise SystemExit("No common image names in the two folders.")

        total: Dict[str, float] = {}
        count = 0
        for name in common:
            ref_path = os.path.join(args.ref_dir, name)
            test_path = os.path.join(args.test_dir, name)
            m = compare_pair(ref_path, test_path)
            print(f"[{name}] -> {m}")
            for k, v in m.items():
                total[k] = total.get(k, 0.0) + v
            count += 1

        print(f"\n=== AVERAGE OVER {count} IMAGES ===")
        for k, v in total.items():
            print(f"{k.upper():6s}: {v / count:.4f}")
        return

    raise SystemExit("Provide either --ref/--test or --ref-dir/--test-dir")


if __name__ == "__main__":
    main()
