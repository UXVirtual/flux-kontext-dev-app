import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from PIL import Image
from torchao.quantization import int8_weight_only, quantize_

# --- Constants and Configuration ---
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
TRANSFORMER_SUBFOLDER = "transformer"
DTYPE = torch.bfloat16
HUGGING_FACE_CACHE_DIR = Path("/root/.cache/huggingface/hub")
MODEL_CACHE_NAME = f"models--{MODEL_ID.replace('/', '--')}"

def setup_logging():
    """Configures a structured logger."""
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
        "--image_path", type=Path, required=True,
        help="The local file path of the input image to edit."
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The text prompt describing the desired image edits."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("output"),
        help="The directory where output images will be saved."
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Your Hugging Face token, only required for the first download."
    )
    return parser.parse_args()

def validate_image_path(image_path: Path):
    """Checks if the provided image path is a valid file."""
    logging.info(f"Validating input image path: {image_path}")
    if not image_path.is_file():
        logging.critical(f"Input Error: Image file not found at the specified path.")
        sys.exit(1)
    logging.info("Image path validation successful.")

def load_and_quantize_transformer(
    model_id: str, subfolder: str, dtype: torch.dtype, **kwargs
) -> FluxTransformer2DModel:
    """Loads the transformer model and applies 8-bit quantization in-place."""
    logging.info(f"Loading transformer from '{model_id}'...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder=subfolder, torch_dtype=dtype, **kwargs
    )
    logging.info("Applying 8-bit weight-only quantization to the transformer...")
    quantize_(transformer, int8_weight_only())
    logging.info("Transformer loaded and quantized successfully.")
    return transformer

def create_pipeline(
    model_id: str, transformer: FluxTransformer2DModel, dtype: torch.dtype, **kwargs
) -> FluxKontextPipeline:
    """Creates the full diffusers pipeline with the quantized transformer."""
    logging.info("Loading the full FluxKontextPipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        model_id, transformer=transformer, torch_dtype=dtype, **kwargs
    )
    logging.info("Moving pipeline to CUDA device...")
    pipe.to("cuda")
    logging.info("Pipeline ready.")
    return pipe

def load_image_from_path(image_path: Path) -> Tuple[Image.Image, int, int]:
    """Loads the input image from a local file path and returns it with its dimensions."""
    logging.info(f"Loading image from: {image_path}")
    input_image = Image.open(image_path).convert("RGB")
    width, height = input_image.size
    logging.info(f"Input image loaded with dimensions: {width}x{height}")
    return input_image, width, height

def run_inference(
    pipe: FluxKontextPipeline, prompt: str, image: Image.Image, height: int, width: int
) -> Image.Image:
    """Runs the image generation process."""
    logging.info(f"Running inference with prompt: '{prompt}'...")
    generator = torch.manual_seed(42)
    output_image = pipe(
        prompt=prompt, image=image, height=height, width=width,
        guidance_scale=2.5, num_inference_steps=50, generator=generator
    ).images[0]
    logging.info("Inference complete.")
    return output_image

def save_output(output_dir: Path, prompt: str, image: Image.Image):
    """Saves the generated image to a file with a descriptive name."""
    output_dir.mkdir(exist_ok=True)
    safe_prompt = re.sub(r"[^\w\s-]", "", prompt).strip()
    filename = "_".join(safe_prompt.split())[:50] + ".png"
    output_path = output_dir / filename
    image.save(output_path)
    logging.info(f"Successfully saved output image to: {output_path}")

def main():
    """Main function to orchestrate the image editing process."""
    setup_logging()
    args = parse_arguments()

    # --- 1. Validate Inputs First (Fail Fast) ---
    # This check happens before any heavy models are loaded into memory.
    validate_image_path(args.image_path)

    # --- 2. Smart Model Loading ---
    model_path_in_cache = HUGGING_FACE_CACHE_DIR / MODEL_CACHE_NAME
    loading_kwargs = {}
    if model_path_in_cache.exists():
        logging.info("Model cache found locally. Attempting to load files without internet.")
        loading_kwargs['local_files_only'] = True
    else:
        logging.info("Model cache not found. A Hugging Face token may be required for download.")
        if not args.hf_token:
            logging.warning("Warning: Model not found in cache and --hf_token not provided. Download may fail if model is gated.")
        else:
            loading_kwargs['token'] = args.hf_token

    try:
        # --- 3. Load Model and Run Pipeline ---
        transformer = load_and_quantize_transformer(
            MODEL_ID, TRANSFORMER_SUBFOLDER, DTYPE, **loading_kwargs
        )
        pipeline = create_pipeline(MODEL_ID, transformer, DTYPE, **loading_kwargs)
        input_image, height, width = load_image_from_path(args.image_path)
        output_image = run_inference(
            pipeline, args.prompt, input_image, height, width
        )
        save_output(args.output_dir, args.prompt, output_image)
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()