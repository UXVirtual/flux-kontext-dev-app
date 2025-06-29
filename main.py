# main_flux_clean.py

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from PIL import Image
from torchao.quantization import int8_weight_only, quantize_

# --- 1. Constants and Configuration ---
# Use constants for values that don't change, improving readability and maintainability.
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
TRANSFORMER_SUBFOLDER = "transformer"
DTYPE = torch.bfloat16  # Data type for modern GPUs like the RTX 4090

def setup_logging():
    """Configures a structured logger instead of using print()."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

def parse_arguments() -> argparse.Namespace:
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FLUX.1 Image Editing with 8-bit Quantization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
        help="The URL of the input image to edit.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The scene is exactly the same, but add floral headwear with colorful flowers in her hair, and an elegant white lace collar around her neck",
        help="The text prompt describing the desired image edits.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="The directory where output images will be saved.",
    )
    return parser.parse_args()


def load_and_quantize_transformer(
    model_id: str, subfolder: str, dtype: torch.dtype
) -> FluxTransformer2DModel:
    """Loads the transformer model and applies 8-bit quantization in-place."""
    logging.info(f"Loading transformer from '{model_id}'...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder=subfolder, torch_dtype=dtype
    )
    logging.info("Applying 8-bit weight-only quantization to the transformer...")
    quantize_(transformer, int8_weight_only())
    logging.info("Transformer loaded and quantized successfully.")
    return transformer


def create_pipeline(
    model_id: str, transformer: FluxTransformer2DModel, dtype: torch.dtype
) -> FluxKontextPipeline:
    """Creates the full diffusers pipeline with the quantized transformer."""
    logging.info("Loading the full FluxKontextPipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        model_id, transformer=transformer, torch_dtype=dtype
    )
    logging.info("Moving pipeline to CUDA device...")
    pipe.to("cuda")
    logging.info("Pipeline ready.")
    return pipe


def prepare_inputs(image_url: str) -> Tuple[Image.Image, int, int]:
    """Loads the input image from a URL and returns it with its dimensions."""
    logging.info(f"Loading input image from: {image_url}")
    input_image = load_image(image_url).convert("RGB")
    width, height = input_image.size
    logging.info(f"Input image loaded with dimensions: {width}x{height}")
    return input_image, width, height


def run_inference(
    pipe: FluxKontextPipeline,
    prompt: str,
    image: Image.Image,
    height: int,
    width: int,
) -> Image.Image:
    """Runs the image generation process."""
    logging.info(f"Running inference with prompt: '{prompt}'...")
    generator = torch.manual_seed(42)
    output_image = pipe(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        guidance_scale=2.5,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
    logging.info("Inference complete.")
    return output_image


def save_output(output_dir: Path, prompt: str, image: Image.Image):
    """Saves the generated image to a file with a descriptive name."""
    output_dir.mkdir(exist_ok=True)
    # Create a safe filename from the prompt
    safe_prompt = re.sub(r"[^\w\s-]", "", prompt).strip()
    filename = "_".join(safe_prompt.split())[:50] + ".png"
    output_path = output_dir / filename
    
    image.save(output_path)
    logging.info(f"Successfully saved output image to: {output_path}")


def main():
    """Main function to orchestrate the image editing process."""
    setup_logging()
    args = parse_arguments()

    try:
        transformer = load_and_quantize_transformer(
            MODEL_ID, TRANSFORMER_SUBFOLDER, DTYPE
        )
        pipeline = create_pipeline(MODEL_ID, transformer, DTYPE)
        input_image, height, width = prepare_inputs(args.image_url)
        output_image = run_inference(
            pipeline, args.prompt, input_image, height, width
        )
        save_output(args.output_dir, args.prompt, output_image)
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()