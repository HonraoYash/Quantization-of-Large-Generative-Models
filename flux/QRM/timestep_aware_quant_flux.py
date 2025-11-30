import argparse, math, types
from typing import Iterable, Union
import torch
from PIL import Image
from diffusers import FluxPipeline

if not hasattr(torch.nn, "RMSNorm"):
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True):
            super().__init__()
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(torch.ones(dim))
            else:
                self.register_parameter("weight", None)
        def forward(self, x):
            # normalize over last dim
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            y = x / rms
            if self.weight is not None:
                # broadcast over last dim
                y = y * self.weight
            return y
    torch.nn.RMSNorm = RMSNorm  # <-- make available to Diffusers

# -----------------------------
# Helpers
# -----------------------------
def str2dtype(s: str):
    s = s.lower()
    if s in ("fp16","float16","half"): return torch.float16
    if s in ("bf16","bfloat16"):       return torch.bfloat16
    if s in ("fp32","float32"):        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def str2device(s: str):
    s = s.lower()
    if s in ("cuda","gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if s in ("cpu",): return torch.device("cpu")
    raise ValueError(f"Unknown device: {s}")

class StepState:
    def __init__(self):
        self.timesteps = []
        self._map = {}
        self.idx = 0
    @staticmethod
    def _to_float(t):
        try:
            return float(getattr(t, "item", lambda: t)())
        except Exception:
            return float(t)
    def set_timesteps(self, timesteps: Union[torch.Tensor, Iterable]):
        if isinstance(timesteps, torch.Tensor):
            ts_list = [float(x) for x in timesteps.detach().cpu().tolist()]
        else:
            ts_list = [self._to_float(x) for x in list(timesteps)]
        self.timesteps = ts_list
        self._map = {int(round(v)): i for i, v in enumerate(ts_list)}
        self.idx = 0
    def on_step(self, t):
        key = int(round(self._to_float(t)))
        if key in self._map:
            self.idx = self._map[key]
        else:
            self.idx = min(self.idx + 1, max(len(self.timesteps)-1, 0))
        return self.idx

@torch.no_grad()
def uniform_sym_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16: return x
    if bits < 2: bits = 2
    reduce_dims = tuple(range(1, x.ndim))
    amax = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    amax = torch.clamp(amax, min=1e-6)
    levels = (2 ** bits) - 1
    scale = amax / (levels / 2.0)
    xq = torch.clamp(torch.round(x / scale), -(levels // 2), (levels // 2))
    return xq * scale

def bits_for_index(idx: int, nsteps: int, early_frac: float, mid_frac: float,
                   b_early: int, b_mid: int, b_late: int) -> int:
    if nsteps <= 0: return b_late
    e_cut = int(math.floor(nsteps * early_frac))
    m_cut = int(math.floor(nsteps * mid_frac))
    if idx < e_cut: return b_early
    if idx < m_cut: return b_mid
    return b_late

def attach_activation_quantizers(transformer: torch.nn.Module,
                                 step_state: StepState,
                                 nsteps: int,
                                 early_frac: float, mid_frac: float,
                                 bits_early: int, bits_mid: int, bits_late: int):
    def pre_hook(module, inputs):
        x = inputs[0]
        b = bits_for_index(step_state.idx, nsteps, early_frac, mid_frac, bits_early, bits_mid, bits_late)
        if isinstance(x, torch.Tensor):
            return (uniform_sym_quant(x, b),)
        if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            x0 = uniform_sym_quant(x[0], b)
            return (x0, *x[1:]) if isinstance(x, tuple) else [x0, *x[1:]]
        return inputs
    for _, mod in transformer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_pre_hook(pre_hook, with_kwargs=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--early_frac", type=float, default=0.25)
    ap.add_argument("--mid_frac", type=float, default=0.7)
    ap.add_argument("--bits_early", type=int, default=4)
    ap.add_argument("--bits_mid", type=int, default=6)
    ap.add_argument("--bits_late", type=int, default=8)
    args = ap.parse_args()

    device = str2device(args.device)
    dtype  = str2dtype(args.dtype)

    print(f"[info] loading FLUX from {args.model_dir}  device={device.type}  dtype={dtype}")
    # Pass both kw variants to survive diffusers version differences
    pipe: FluxPipeline = FluxPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)

    step_state = StepState()

    # Monkey-patch scheduler.step to track the current timestep index
    orig_step = pipe.scheduler.step
    def step_wrapper(self, model_output, timestep, sample, *a, **kw):
        step_state.on_step(timestep)
        return orig_step(model_output, timestep, sample, *a, **kw)
    pipe.scheduler.step = types.MethodType(step_wrapper, pipe.scheduler)

    # Prepare timesteps before first forward
    pipe.scheduler.set_timesteps(args.steps, device=device)
    step_state.set_timesteps(pipe.scheduler.timesteps)
    nsteps = len(step_state.timesteps)

    attach_activation_quantizers(
        pipe.transformer, step_state, nsteps,
        early_frac=args.early_frac, mid_frac=args.mid_frac,
        bits_early=args.bits_early, bits_mid=args.bits_mid, bits_late=args.bits_late,
    )

    result = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=3.5,
        generator=gen,
        height=args.height,
        width=args.width,
    )
    img = result.images[0]
    if isinstance(img, Image.Image):
        img.save(args.out)
    else:
        Image.fromarray(img).save(args.out)
    print(f"[done] saved {args.out}")

if __name__ == "__main__":
    main()