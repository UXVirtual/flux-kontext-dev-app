# Dockerfile for FLUX.1 Quantized Inference (Simplified and Corrected)

# Use an official NVIDIA CUDA runtime image.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Standard Python environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git

# Install PyTorch for CUDA 12.1 first.
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the latest stable versions of the required libraries from PyPI.
# This is now the most reliable method as the necessary APIs are in the main releases.
# Note: The diffusers library is installed from a specific commit to ensure compatibility with Flux.1-Kontext-dev.
RUN pip3 install --no-cache-dir --upgrade \
    transformers \
    accelerate \
    torchao \
    sentencepiece \
    protobuf \
    'diffusers[torch]@git+https://github.com/huggingface/diffusers.git@e8e44a510c152fff17e3f1bba036d635776b5b9f'

# Copy the application code into the container.
COPY main.py .

# Set the default command to execute.
CMD ["python3", "main.py"]