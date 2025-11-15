from __future__ import annotations
import os
import bentoml
from PIL.Image import Image

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SAMPLE_PROMPT = "A girl smiling"

@bentoml.service(
    name="bento-flux-timestep-distilled-service",
    image=bentoml.images.Image(python_version="3.11").requirements_file("requirements.txt"),
    traffic={"timeout": 300},
    envs=[{"name": "HUGGINGFACE_HUB_TOKEN"}],  # set this in your shell or .env
    # resources={"gpu": 1, "gpu_type": "nvidia-a100-80gb"},  # optional hint; ignored locally
)
class FluxTimestepDistilled:
    def __init__(self):
        self.pipe = None
        self.device = "cpu"

    @bentoml.on_startup
    def setup_pipeline(self) -> None:
        import torch
        from huggingface_hub import login
        from diffusers import FluxPipeline

        token = ""
        if not token:
            raise RuntimeError("HUGGINGFACE_HUB_TOKEN is not set.")

        # authenticate to Hugging Face (needed for gated models/rate limits)
        login(token='')

        # choose device + dtype
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        # load pipeline with token
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            token=token,
        )
        self.pipe.to(self.device)

    @bentoml.api
    def txt2img(self, prompt: str = SAMPLE_PROMPT) -> Image:
        image = self.pipe(
            prompt=prompt,
            guidance_scale=0.0,
            height=768,
            width=1360,
            num_inference_steps=4,
            max_sequence_length=256,
        ).images[0]
        return image
