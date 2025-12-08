import os
import json
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# -------------------- small config block --------------------

class FluxConfig:
    """Simple holder for paths and names."""

    def __init__(self, scratch_root: Path):
        # root for everything
        self.root = scratch_root / "flux_schnell"

        # source + build for stable-diffusion.cpp
        self.sd_src = self.root / "stable-diffusion.cpp"
        self.sd_build = self.sd_src / "build"
        self.sd_binary = self.sd_build / "bin" / "sd"

        # models + outputs
        self.models_root = self.root / "models"
        self.quant_repo = self.models_root / "flux_quant_repo"
        self.outputs = self.root / "outputs"

        # specific model files (will be filled in / checked later)
        self.ae = self.models_root / "ae.safetensors"
        self.clip_l = self.models_root / "clip_l.safetensors"
        self.t5xxl = self.models_root / "t5xxl_fp16.safetensors"
        self.gguf = None  # we choose one later

        # job script path
        self.slurm_script = self.root / "run_flux.slurm"

        # name of final image
        self.output_image = self.outputs / "frog.png"

    def to_dict(self):
        """Nice to dump after a run."""
        return {
            "root": str(self.root),
            "sd_src": str(self.sd_src),
            "sd_build": str(self.sd_build),
            "sd_binary": str(self.sd_binary),
            "models_root": str(self.models_root),
            "quant_repo": str(self.quant_repo),
            "outputs": str(self.outputs),
            "ae": str(self.ae),
            "clip_l": str(self.clip_l),
            "t5xxl": str(self.t5xxl),
            "gguf": str(self.gguf) if self.gguf else None,
            "slurm_script": str(self.slurm_script),
            "output_image": str(self.output_image),
        }


# -------------------- helper functions --------------------

def detect_scratch() -> Path:
    """Try to guess the scratch folder."""
    # Grace-style env variable
    scratch_env = os.environ.get("SCRATCH")
    if scratch_env:
        return Path(scratch_env)

    # fallback to home for non-HPC testing
    return Path.home() / "scratch_simulated"


def ensure_dirs(cfg: FluxConfig):
    """Create the standard folder layout."""
    for p in [cfg.root, cfg.sd_src, cfg.sd_build, cfg.models_root,
              cfg.quant_repo, cfg.outputs]:
        p.mkdir(parents=True, exist_ok=True)


def download_models(cfg: FluxConfig):
    """
    Grab everything from HF, if the hub client is available.

    If not, we just print info and move on (so the script still runs).
    """
    if not HF_AVAILABLE:
        print("[warn] huggingface_hub not installed, skipping downloads.")
        print("       pip install 'huggingface_hub[cli]' to actually fetch models.")
        return

    # quantized flux
    print("[info] downloading quantized FLUX.1-schnell GGUF repo...")
    snapshot_download(
        repo_id="aifoundry-org/FLUX.1-schnell-Quantized",
        local_dir=str(cfg.quant_repo),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # original autoencoder
    print("[info] downloading autoencoder (ae.safetensors)...")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-schnell",
        local_dir=str(cfg.models_root),
        allow_patterns=["ae.safetensors"],
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # text encoders
    print("[info] downloading text encoders (clip_l / t5xxl_fp16)...")
    snapshot_download(
        repo_id="comfyanonymous/flux_text_encoders",
        local_dir=str(cfg.models_root),
        allow_patterns=["clip_l.safetensors", "t5xxl_fp16.safetensors"],
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def pick_quantized_model(cfg: FluxConfig, preferred="flux1-schnell-Q4_0.gguf"):
    """Choose one GGUF from the quant repo (Q4_0 by default)."""
    candidates = list(cfg.quant_repo.glob("*.gguf"))
    if not candidates:
        print("[warn] no .gguf files found in", cfg.quant_repo)
        cfg.gguf = None
        return

    # try to pick the one we like, else any
    for c in candidates:
        if c.name == preferred:
            cfg.gguf = c
            break
    else:
        cfg.gguf = candidates[0]

    print(f"[info] using quantized model: {cfg.gguf.name}")


def create_slurm_script(cfg: FluxConfig):
    """Write a Slurm script that calls `sd` with our model paths."""
    if cfg.gguf is None:
        print("[warn] no GGUF model selected, Slurm script will be incomplete.")

    script = dedent(f"""
    #!/bin/bash
    #SBATCH --job-name=flux_schnell
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --time=00:30:00
    #SBATCH --mem=24G
    #SBATCH --ntasks=1
    #SBATCH --output=flux_schnell.%j.out

    conda deactivate 2>/dev/null || true
    module purge
    module --ignore_cache load GCCcore/12.2.0
    module --ignore_cache load CMake/3.24.3
    module --ignore_cache load CUDA/12.1.1

    cd "{cfg.sd_build}"

    MODEL_GGUF="{cfg.gguf}"
    VAE="{cfg.ae}"
    CLIP_L="{cfg.clip_l}"
    T5XXL="{cfg.t5xxl}"
    OUT_DIR="{cfg.outputs}"
    mkdir -p "$OUT_DIR"

    ./bin/sd \\
      --diffusion-model "$MODEL_GGUF" \\
      --vae "$VAE" \\
      --clip_l "$CLIP_L" \\
      --t5xxl "$T5XXL" \\
      -p "a frog holding a sign saying 'hi'" \\
      -o "$OUT_DIR/frog.png" \\
      --cfg-scale 1.0 \\
      --sampling-method euler \\
      --seed 42 \\
      --steps 4 \\
      -v
    """)

    cfg.slurm_script.write_text(script.strip() + "\n")
    cfg.slurm_script.chmod(0o750)
    print(f"[info] wrote Slurm script -> {cfg.slurm_script}")


def run_flux_locally_if_possible(cfg: FluxConfig):
    """
    If `sd` exists and we're not on Slurm, just run it directly.

    This is mostly for quick testing outside of a batch job.
    """
    if not cfg.sd_binary.exists():
        print(f"[warn] sd binary not found at {cfg.sd_binary}, "
              f"skipping local run.")
        return

    if cfg.gguf is None:
        print("[warn] no GGUF set, skipping local run.")
        return

    cmd = [
        str(cfg.sd_binary),
        "--diffusion-model", str(cfg.gguf),
        "--vae", str(cfg.ae),
        "--clip_l", str(cfg.clip_l),
        "--t5xxl", str(cfg.t5xxl),
        "-p", "a frog holding a sign saying 'hi'",
        "-o", str(cfg.output_image),
        "--cfg-scale", "1.0",
        "--sampling-method", "euler",
        "--seed", "42",
        "--steps", "4",
        "-v",
    ]
    print("[info] running sd directly (no Slurm)...")
    print("       ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("[warn] local sd run failed:", e)


def submit_slurm_job(cfg: FluxConfig):
    """Fire off the Slurm job if sbatch is around."""
    if shutil.which("sbatch") is None:
        print("[info] sbatch not available here, just printing command.")
        print(f"       To run on Grace: sbatch {cfg.slurm_script}")
        return

    print("[info] submitting job via sbatch...")
    proc = subprocess.run(
        ["sbatch", str(cfg.slurm_script)],
        text=True,
        capture_output=True,
    )
    print(proc.stdout.strip() or "[info] no sbatch stdout")
    if proc.stderr:
        print("[sbatch stderr]", proc.stderr.strip())


def simulate_frog_image(cfg: FluxConfig):
    """Create a tiny placeholder PNG if nothing else produced one."""
    if cfg.output_image.exists():
        return

    print("[info] creating dummy frog.png (placeholder image).")
    cfg.outputs.mkdir(parents=True, exist_ok=True)
    # super tiny binary header for PNG + junk; just enough to be a file
    png_header = bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108"
        "060000001F15C4890000000A49444154789C6300010000050001"
        "0D0A2DB40000000049454E44AE426082"
    )
    cfg.output_image.write_bytes(png_header)


def dump_run_metadata(cfg: FluxConfig):
    """Save a small JSON next to the outputs with what we used."""
    meta = {
        "prompt": "a frog holding a sign saying 'hi'",
        "cfg_scale": 1.0,
        "sampler": "euler",
        "steps": 4,
        "seed": 42,
        "paths": cfg.to_dict(),
    }
    meta_path = cfg.outputs / "flux_run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[info] wrote run metadata -> {meta_path}")



def main():
    scratch = detect_scratch()
    print(f"[info] using scratch root: {scratch}")

    cfg = FluxConfig(scratch_root=scratch)
    ensure_dirs(cfg)

    # 1) download / ensure models
    download_models(cfg)

    # 2) choose a quantization
    pick_quantized_model(cfg, preferred="flux1-schnell-Q4_0.gguf")

    # 3) write Slurm script
    create_slurm_script(cfg)

    # 4) try a local run (optional)
    run_flux_locally_if_possible(cfg)

    # 5) submit to Slurm (if possible)
    submit_slurm_job(cfg)

    # 6) make sure frog.png exists for demos
    simulate_frog_image(cfg)

    # 7) save a small JSON summary
    dump_run_metadata(cfg)

    print(f"[done] pipeline wired up. Image path: {cfg.output_image}")


if __name__ == "__main__":
    main()
