import argparse
import asyncio
import io
import logging
import re
import select
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import uvicorn
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from torchao.quantization import int8_weight_only, quantize_
from transformers import PreTrainedTokenizer

# --- Constants and Configuration ---
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
TRANSFORMER_SUBFOLDER = "transformer"
DTYPE = torch.bfloat16

# --- Globals for shared state ---
pipeline: FluxKontextPipeline = None
# We now also store the tokenizer globally to calculate input tokens
tokenizer: PreTrainedTokenizer = None
last_activity_time = time.time()
inference_lock = asyncio.Lock()

# --- FastAPI App Setup & New Response Models ---
app = FastAPI(title="FLUX.1-Like API", description="An API for image editing, mimicking the OpenAI specification.")

# --- Mount the static files directory ---
# This line tells FastAPI to serve any file in the 'output' directory
# when a request is made to a URL starting with '/output'.
app.mount("/output", StaticFiles(directory="output"), name="output")

# --- OpenAI-like Response Models ---
class ImageData(BaseModel):
    url: str

class TokenUsageDetails(BaseModel):
    text_tokens: int
    image_tokens: int = Field(0, description="Image token count is not applicable for this model.")

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int = Field(0, description="Output token count is not applicable for this model.")
    input_tokens_details: TokenUsageDetails

class ImageEditResponse(BaseModel):
    created: int
    data: List[ImageData]
    usage: TokenUsage

# --- Core Application Functions ---

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
    """Loads an image from a path and returns it with its dimensions."""
    if not image_path.is_file():
        logging.error(f"File not found: {image_path}. Please provide a valid file path.")
        return None, None, None
    logging.info(f"Loading image from: {image_path}")
    input_image = Image.open(image_path).convert("RGB")
    width, height = input_image.size
    logging.info(f"Input image loaded with dimensions: {width}x{height}")
    return input_image, width, height

def prepare_inputs_from_bytes(image_bytes: bytes) -> Tuple[Image.Image, int, int]:
    """Loads an image from raw bytes."""
    logging.info("Loading image from uploaded bytes.")
    # Use io.BytesIO to treat the byte string as a file
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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

def save_output(output_path: Path, image: Image.Image):
    """Saves the generated image to a pre-calculated path."""
    # Ensure the parent directory exists
    output_path.parent.mkdir(exist_ok=True)
    image.save(output_path)
    logging.info(f"Successfully saved output image to: {output_path}")

async def process_inference_job(image_path_str: str, prompt_str: str):
    """A thread-safe function to handle a single console inference request."""
    global last_activity_time
    async with inference_lock:
        logging.info("Inference lock acquired for console job.")
        try:
            # Pre-calculate the output path before inference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = re.sub(r"[^\w\s-]", "", prompt_str).strip()
            filename_base = "_".join(safe_prompt.split())[:50]
            filename = f"{timestamp}_{filename_base}.png"
            output_path = Path("output") / filename
            
            input_image, height, width = prepare_inputs(Path(image_path_str))
            if input_image:
                output_image = run_inference(pipeline, prompt_str, input_image, height, width)
                # --- THIS IS THE FIX ---
                # The 'prompt' argument is no longer needed here as the path is pre-calculated.
                save_output(output_path, output_image)
        except Exception as e:
            logging.error(f"An error occurred during console processing: {e}", exc_info=True)
    last_activity_time = time.time()
    logging.info("Inference lock released. Inactivity timer has been reset.")


async def process_inference_job_for_api(image_bytes: bytes, prompt_str: str, output_path: Path):
    """A thread-safe function that now saves the image to a pre-determined path."""
    global last_activity_time
    async with inference_lock:
        logging.info("Inference lock acquired for API job.")
        try:
            input_image, height, width = prepare_inputs_from_bytes(image_bytes)
            if input_image:
                output_image = run_inference(pipeline, prompt_str, input_image, height, width)
                # --- THIS IS THE FIX ---
                # The 'prompt' argument is no longer needed here.
                save_output(output_path, output_image)
        except Exception as e:
            logging.error(f"An error occurred during background API processing: {e}", exc_info=True)
    last_activity_time = time.time()
    logging.info("Inference lock released. Inactivity timer has been reset.")


# --- FastAPI Endpoint (Updated to return a valid relative URL) ---
@app.post("/v1/images/edits", response_model=ImageEditResponse)
async def create_image_edit(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="The image to edit."),
    prompt: str = Form(..., description="A text description of the desired edit."),
    n: int = Form(1, description="The number of images to generate (must be 1)."),
):
    if n > 1:
        raise HTTPException(status_code=400, detail="This implementation currently only supports generating 1 image (n=1).")
    if not tokenizer:
        raise HTTPException(status_code=503, detail="Server is not fully initialized, tokenizer not available.")

    # --- Pre-calculate Filename and URL ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = re.sub(r"[^\w\s-]", "", prompt).strip()
    filename_base = "_".join(safe_prompt.split())[:50]
    filename = f"{timestamp}_{filename_base}.png"
    
    # The output_path is still relative to the script's execution
    output_path = Path("output") / filename
    # The URL in the response is now a valid web path
    image_url = f"/output/{filename}"
    
    # --- 2. Calculate Token Usage ---
    # Calculate input text tokens using the globally loaded tokenizer
    input_text_tokens = len(tokenizer(prompt).input_ids)
    # Image and output tokens are not applicable, so we use placeholders
    usage = TokenUsage(
        total_tokens=input_text_tokens,
        input_tokens=input_text_tokens,
        input_tokens_details=TokenUsageDetails(text_tokens=input_text_tokens)
    )

    # --- Schedule Background Job ---
    image_bytes = await image.read()
    # Pass the full disk path to the background task for saving
    background_tasks.add_task(process_inference_job_for_api, image_bytes, prompt, output_path)

    # --- Return Immediate Response ---
    response = ImageEditResponse(
        created=int(time.time()),
        # Return the web-accessible URL
        data=[ImageData(url=image_url)],
        usage=usage
    )
    return response


# --- Concurrent Task Runners ---
async def run_api_server(host: str, port: int):
    """Runs the Uvicorn server and handles graceful shutdown."""
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except asyncio.CancelledError:
        logging.info("API server task cancelled successfully.")

async def run_interactive_console():
    """Runs the interactive console loop and handles graceful shutdown."""
    global last_activity_time
    loop = asyncio.get_event_loop()
    logging.info("Interactive console is ready. Format: /path/to/image.png \"prompt\"")
    try:
        while True:
            ready = await loop.run_in_executor(None, select.select, [sys.stdin], [], [], 0.1)
            if ready[0]:
                last_activity_time = time.time()
                command = await loop.run_in_executor(None, sys.stdin.readline)
                command = command.strip()
                if command.lower() in ['quit', 'exit']:
                    logging.info("Exit command received from console. Shutting down.")
                    for task in asyncio.all_tasks(): task.cancel()
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
        if not inference_lock.locked():
            inactivity = time.time() - last_activity_time
            if inactivity > timeout:
                logging.warning(f"Inactivity timeout of {timeout} seconds reached. Shutting down all tasks.")
                for task in asyncio.all_tasks(): task.cancel()
                break
        await asyncio.sleep(5)

# --- Main Application Entrypoint ---
async def main():
    """Main function to initialize the model and run concurrent services."""
    global pipeline, tokenizer, last_activity_time
    setup_logging()
    args = parse_arguments()
    loading_kwargs = {'token': args.hf_token} if args.hf_token else {}

    try:
        logging.info("--- Initializing FLUX Pipeline (this may take a moment) ---")
        transformer = load_and_quantize_transformer(
            MODEL_ID, TRANSFORMER_SUBFOLDER, DTYPE, **loading_kwargs
        )
        pipeline = create_pipeline(
            MODEL_ID, transformer, DTYPE, **loading_kwargs
        )

        # --- THIS IS THE FIX ---
        # The primary text tokenizer for FLUX is stored in the 'tokenizer_2' attribute.
        tokenizer = pipeline.tokenizer_2
        
        last_activity_time = time.time()
        logging.info("--- Initialization Complete. Ready for API and console input. ---")

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
        if 'pipeline' in globals() and pipeline: del pipeline
        if 'transformer' in globals() and transformer: del transformer
        torch.cuda.empty_cache()
        logging.info("Cleanup complete. Exiting.")

if __name__ == "__main__":
    # Ensure you copy the full definitions for all functions into your script.
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Script shutdown sequence initiated.")