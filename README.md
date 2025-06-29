# Flux Kontext DEV Diffusers

This project containerizes `Flux.1-Kontext-dev` to run on a CUDA compatible host powered by an NVIDIA RTX 4090 series GPU. It provides an API to allow an input image to be transformed via a prompt and will return a coherant, modified output image.

This is intended for offline local usage and should not be deployed into a production or internet-facing environment!

Thanks to Andrew Zhu for his [Medium Article on running Flux.1 Kontext [dev] using Diffusers](https://xhinker.medium.com/run-flux-kontext-using-huggingface-diffusers-in-your-own-gpus-e9ea7cc51b8a).

## Windows Setup

### Step 1: Windows 11 Host Configuration

To run a GPU-accelerated Docker container, your Windows 11 machine must be set up correctly.

1. Install NVIDIA Drivers: Ensure you have the latest drivers for your NVIDIA GPU installed from the NVIDIA website.

2. Install WSL 2: Docker Desktop for Windows uses the Windows Subsystem for Linux (WSL) 2 as its backend for performance. If you don't have it, open PowerShell as an Administrator and run:

```
wsl --install
```

A system reboot is usually required after installation.

3. Install Docker Desktop: Download and install Docker Desktop for Windows. During setup, or in the settings (`Settings > General`), make sure the **Use the WSL 2 based engine** option is enabled. Docker Desktop should automatically detect your WSL 2 installation.

### Step 2: Build the New Docker Image

1. Open PowerShell or Command Prompt.

2. Navigate to your project directory (e.g. `C:\flux-kontext-app\`).

3. Open Powershell

```
docker build -t flux-kontext-app .
```

### Step 3: Run the Container with Hugging Face Authentication

The `FLUX.1-Kontext-dev` model is gated, meaning your container needs to authenticate with Hugging Face to download it. The best and most secure way to do this is by passing your Hugging Face token as an environment variable.

1. Sign in to Hugging Face in order to generate an access token.

2. Go to your Hugging Face profile: https://huggingface.co/settings/tokens

4. Generate a new token with only **Read access to contents of selected repos** permission. Ensure this can access the `black-forest-labs/FLUX.1-Kontext-dev` repo. Copy the token string (it will start with `hf_`).

5. Navigate to [https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev], read the licence agreement and click **Agree and access repository**.

## Usage

The included batch file will automatically download the model (~20GB disk space required) and start the docker container. This command securely passes your HuggingFace token to the container and maps your cache to persist the downloaded model.

For your safety, be sure to only use a token with read only access to the `black-forest-labs/FLUX.1-Kontext-dev` repo!

By default, a cat image hosted on Hugging Face will be edited using a text prompt. The input image and prompt
can be customized by adjusting the arguments passed into `main.py`.

1. Open Windows commandline to run:

```
.\run-flux-docker.bat hf_YOUR_TOKEN_GOES_HERE
```
Replace `hf_YOUR_TOKEN_HERE` with the actual token you copied.

## TODO

- Add OpenAI-compatible `images/createEdit` endpoint using FastAPI
- Allow specifying existing downloaded model weights on host PC
