from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def next_output_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    number = 1
    while (path := OUTPUT_DIR / f"output{number}.png").exists():
        number += 1
    return path

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    cache_dir=PROJECT_DIR / ".cache" / "huggingface" / "hub",
    dtype=torch.float32,
    use_safetensors=True,
)

# This replaces .to("cuda") because the GTX 1650 Ti has only 4 GB VRAM.
pipe.enable_sequential_cpu_offload()

image = pipe("A futuristic city at night").images[0]
output_path = next_output_path()
image.save(output_path)
print(f"Saved image: {output_path}")
