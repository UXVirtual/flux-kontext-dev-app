# Dockerfile for FLUX.1 Quantized Inference

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Add fastapi and uvicorn to the dependencies
RUN pip3 install --no-cache-dir --upgrade \
    'diffusers[torch]@git+https://github.com/huggingface/diffusers.git@e8e44a510c152fff17e3f1bba036d635776b5b9f' \
    transformers \
    accelerate \
    torchao \
    sentencepiece \
    protobuf \
    fastapi \
    "uvicorn[standard]"

COPY main.py .

# Expose the port the API server will run on
EXPOSE 8000

# The CMD is now optional as the bat file specifies the command,
# but it's good practice to have a default.
CMD ["python3", "main.py"]