import argparse
import asyncio
import logging
import re
import select
import sys
import time
from datetime import datetime # Import the datetime module
from pathlib import Path
from typing import Tuple

import torch
import uvicorn
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from torchao.quantization import int8_weight_only, quantize_

# --- Constants and Configuration ---
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
TRANSFORMER_SUBFOLDER = "transformer"
DTYPE = torch.bfloat16

# --- Globals for shared state ---
pipeline: FluxKontextPipeline = None
last_activity_time = time.time()
inference_lock = asyncio.Lock()

# --- FastAPI App Setup ---
app = FastAPI(title="FLUX.1 Inference API", description="An API for running 8-bit quantized FLUX.1 image editing.")

class InferenceRequest(BaseModel):
    image_path: str
    prompt: str

def setup_logging():
    """Configures a structured logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the server."""
    parser = argparse.ArgumentParser(description="FLUX.1 Server with Interactive Console and API.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    parser.add_argument("--timeout", type=int, default=300, help="Inactivity timeout in seconds.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for first download.")
    return parser.parse_args()

def generate_filename_base_from_prompt(prompt: str) -> str:
    """Creates a safe filename base (without extension) from a prompt."""
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt).strip()
    # No longer adds the .png extension here
    return "_".join(safe_prompt.split())[:50]

# --- Core Logic Functions (unchanged but used by both interfaces) ---
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


def prepare_inputs(image_path: Path) -> Tuple[Image.Image, int, int]:
    """Loads the input image from a local file path and returns it with its dimensions."""
    if not image_path.is_file():
        logging.error(f"File not found: {image_path}. Please provide a valid file path.")
        return None, None, None

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
    """Saves the generated image with a timestamped and descriptive name."""
    output_dir.mkdir(exist_ok=True)

    # **CHANGE 1: Generate a timestamp string**
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the safe, prompt-based part of the filename
    prompt_base = generate_filename_base_from_prompt(prompt)
    
    # **CHANGE 2: Combine timestamp and prompt for the final filename**
    filename = f"{timestamp}_{prompt_base}.png"
    output_path = output_dir / filename
    
    image.save(output_path)
    logging.info(f"Successfully saved output image to: {output_path}")


async def process_inference_job(image_path_str: str, prompt_str: str):
    """A thread-safe function to handle a single inference request."""
    global last_activity_time
    # This lock ensures only one inference happens at a time.
    async with inference_lock:
        logging.info("Inference lock acquired, starting job.")
        try:
            input_image, height, width = prepare_inputs(Path(image_path_str))
            if input_image:
                output_image = run_inference(pipeline, prompt_str, input_image, height, width)
                save_output(Path("output"), prompt_str, output_image)
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}", exc_info=True)

    # **CHANGE 1: Update activity time AFTER the job is done and the lock is released.**
    last_activity_time = time.time()
    logging.info("Inference lock released. Inactivity timer has been reset.")


@app.post("/inference", status_code=202)
async def trigger_inference(request: InferenceRequest):
    """Triggers an asynchronous inference job."""
    global last_activity_time
    # Set the activity time when the request is received to keep the server alive.
    last_activity_time = time.time()
    logging.info(f"API request received for image '{request.image_path}'")
    asyncio.create_task(process_inference_job(request.image_path, request.prompt))
    return {"message": "Inference job accepted and is running in the background."}

# --- Concurrent Task Runners ---
async def run_api_server(host: str, port: int):
    """Runs the Uvicorn server and handles graceful shutdown on cancellation."""
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    # **CHANGE 1: Add try...except block to catch the expected cancellation error.**
    try:
        await server.serve()
    except asyncio.CancelledError:
        # This is expected on timeout. Log a clean message instead of a traceback.
        logging.info("API server task cancelled successfully.")


async def run_interactive_console():
    """Runs the interactive console loop and handles graceful shutdown on cancellation."""
    global last_activity_time
    loop = asyncio.get_event_loop()
    logging.info("Interactive console is ready. Format: /path/to/image.png \"prompt\"")
    
    # **CHANGE 2: Add try...except block around the main loop.**
    try:
        while True:
            ready = await loop.run_in_executor(None, select.select, [sys.stdin], [], [], 0.1)
            if ready[0]:
                last_activity_time = time.time()
                command = await loop.run_in_executor(None, sys.stdin.readline)
                command = command.strip()

                if command.lower() in ['quit', 'exit']:
                    logging.info("Exit command received from console. Shutting down.")
                    for task in asyncio.all_tasks():
                        task.cancel()
                    break
                try:
                    path_str, prompt_str = command.split('"')[:2]
                    await process_inference_job(path_str.strip(), prompt_str.strip())
                except ValueError:
                    logging.error("Invalid console input. Format: /path/to/your/image.png \"your prompt here\"")
    except asyncio.CancelledError:
        logging.info("Interactive console task cancelled successfully.")


async def timeout_monitor(timeout: int):
    """Monitors for inactivity and triggers a shutdown only when the model is idle."""
    global last_activity_time
    while True:
        # **CHANGE 2: Only check the timeout if the inference lock is NOT held.**
        if not inference_lock.locked():
            inactivity = time.time() - last_activity_time
            if inactivity > timeout:
                logging.warning(f"Inactivity timeout of {timeout} seconds reached. Shutting down all tasks.")
                for task in asyncio.all_tasks():
                    task.cancel()
                break
        await asyncio.sleep(5) # Check every 5 seconds


async def main():
    """Main function to initialize the model and run concurrent services."""
    global pipeline, last_activity_time
    setup_logging()
    args = parse_arguments()

    loading_kwargs = {'token': args.hf_token} if args.hf_token else {}
    
    try:
        logging.info("--- Initializing FLUX Pipeline (this may take a moment) ---")
        transformer = load_and_quantize_transformer(MODEL_ID, TRANSFORMER_SUBFOLDER, DTYPE, **loading_kwargs)
        pipeline = create_pipeline(MODEL_ID, transformer, DTYPE, **loading_kwargs)
        last_activity_time = time.time()
        logging.info("--- Initialization Complete. Ready for API and console input. ---")

        # Run API server, console, and timeout monitor concurrently
        await asyncio.gather(
            run_api_server(args.host, args.port),
            run_interactive_console(),
            timeout_monitor(args.timeout),
        )

    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.info("Shutdown signal received.")
    except Exception as e:
        logging.critical(f"A critical error occurred during initialization: {e}", exc_info=True)
    finally:
        logging.info("Unloading model from memory and freeing VRAM...")
        del pipeline
        del transformer
        torch.cuda.empty_cache()
        logging.info("Cleanup complete. Exiting.")


if __name__ == "__main__":
    # Ensure you copy the full definitions for all functions here.
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Script shutdown complete.")
