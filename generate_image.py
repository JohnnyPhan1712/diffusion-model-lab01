from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface" / "hub"
DEFAULT_PROMPT = (
    "Spider-Man sitting on a skyscraper edge, back to camera, "
    "New York skyline at sunset, cinematic wide shot"
)
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, extra limbs, duplicate, "
    "text, watermark, logo"
)


def next_output_path(output_dir: Path) -> Path:
    """Return the first unused outputN.png path without overwriting old images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while (candidate := output_dir / f"output{number}.png").exists():
        number += 1
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path. If omitted, use the next free outputs/outputN.png name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--precision",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Use float32 automatically on GTX 16-series GPUs to avoid FP16 NaN images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check the NVIDIA driver and install the CUDA-enabled "
            "PyTorch wheel from requirements.txt."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {args.prompt}")

    gpu_name = torch.cuda.get_device_name(0)
    if args.precision == "auto":
        dtype = torch.float32 if "GTX 16" in gpu_name.upper() else torch.float16
    else:
        dtype = getattr(torch, args.precision)
    print(f"Precision: {dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        dtype=dtype,
        use_safetensors=True,
    )

    # A GTX 1650 Ti has only 4 GB VRAM. Sequential CPU offload is slower, but
    # it has the lowest peak VRAM use and is the most reliable option here.
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    extrema = image.convert("RGB").getextrema()
    if all(channel_max <= 1 for _, channel_max in extrema):
        raise RuntimeError(
            "The generated image is black. Retry with --precision float32 and review "
            "any safety-checker message printed above."
        )

    output_path = args.output or next_output_path(Path(__file__).resolve().parent / "outputs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image: {output_path.resolve()}")


if __name__ == "__main__":
    main()
