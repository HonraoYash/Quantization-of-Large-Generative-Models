from __future__ import annotations
import bentoml
from PIL.Image import Image

# LOCAL path – we already downloaded the model here
LOCAL_MODEL_DIR = "/scratch/user/dishant.zaveri/hf_models/FLUX.1-schnell"
SAMPLE_PROMPT = "A girl smiling"


@bentoml.service(
    name="bento-flux-timestep-distilled-service",
    image=bentoml.images.Image(python_version="3.11").requirements_file("requirements.txt"),
    traffic={"timeout": 300},
)
class FluxTimestepDistilled:
    def __init__(self):
        self.pipe = None
        self.device = "cpu"
        self.dtype = None

    @bentoml.on_startup
    def setup_pipeline(self) -> None:
        import torch
        from diffusers import FluxPipeline

        # pick device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        # NO huggingface_hub.login here
        self.pipe = FluxPipeline.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=self.dtype,
            use_safetensors=True,
            device_map="balanced",
            low_cpu_mem_usage=True,
        )

        # try to reduce memory
        try:
            self.pipe.enable_attention_slicing("max")
        except Exception:
            pass
        try:
            self.pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

        try:
            self.pipe.to(self.device)
        except Exception:
            pass

    @bentoml.api
    def txt2img(
        self,
        prompt: str = SAMPLE_PROMPT,
        height: int = 768,
        width: int = 768,
        steps: int = 4,
        guidance: float = 0.0,
        max_seq_len: int = 128,
    ) -> Image:
        out = self.pipe(
            prompt=prompt,
            guidance_scale=guidance,
            height=height,
            width=width,
            num_inference_steps=steps,
            max_sequence_length=max_seq_len,
        )
        return out.images[0]
