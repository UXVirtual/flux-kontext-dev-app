# Flux Kontext DEV Diffusers

This project containerizes `Flux.1-Kontext-dev` to run on a CUDA compatible host powered by an NVIDIA RTX 4090 series GPU. It provides an [OpenAI compatible /images/edits API endpoint](https://platform.openai.com/docs/api-reference/images/createEdit) that takes an input image and prompt and will transform the image, returning a url to a coherant, modified output image. Currently this only supports a single image as an input.

This is intended for offline local usage and should not be deployed into a production or internet-facing environment!

Thanks to Andrew Zhu for his [Medium Article on running Flux.1 Kontext [dev] using Diffusers](https://xhinker.medium.com/run-flux-kontext-using-huggingface-diffusers-in-your-own-gpus-e9ea7cc51b8a).

Please read the `LICENSE` and `NOTICE.md` files in this repository so you understand your rights when using this software.

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

1. Start Docker Desktop.

2. Open Windows commandline to run:

```
.\run-flux-docker.bat --hf_token hf_YOUR_TOKEN_GOES_HERE --timeout 300
```
Replace `hf_YOUR_TOKEN_HERE` with the actual token you copied. You only need to do this once for the model to download. Subsequent runs don't require the token.

The timeout before the model auto-unloads from memory defaults to 5 minutes (300 seconds). This is adjustable depending on the use-case if this is used as part of a multi-step pipeline.

3. Make an HTTP request to the endpoint, with multi-part form data, including the input:

**Request**

```
curl -X POST 'http://localhost:8000/v1/images/edits' ^
  -H 'Content-Type: multipart/form-data' ^
  -H 'Accept: application/json' ^
  -F 'image=@"/C:/path/to/image/my_cat.jpg"' ^
  -F 'prompt="a cat wearing a top hat, studio lighting, detailed"'
  -F 'n="1"'
```

**Response**

The response will have a relative url to where the image is served by the API. The base URL should be [http://127.0.0.1:8000]. Alternately, you can find the image in the `output` subfolder in this project folder:

```
{
  "created": 1751719260,
  "data": [
    {
      "url": "output/20250705_114100_a_cat_wearing_a_top_hat_studio_lighting_deta.png"
    }
  ],
  "usage": {
      "total_tokens": 15,
      "input_tokens": 15,
      "output_tokens": 0,
      "input_tokens_details": {
          "text_tokens": 15,
          "image_tokens": 0
      }
  }
}`
```

4. Alternately, use the interactive commandline to trigger images to be processed by the model. Don't include spaces or encapsulate the file path in quotation marks. e.g:

```
/app/host_files/inputs/image.png "your prompt here"
```

### OpenAPI Specification

Once the server is started, open a browser window to [http://127.0.0.1:8000/docs] explore the API for this service. You can download the OpenAPI spec in JSON format from [http://127.0.0.1:8000/openapi.json].

## TODO

- Allow specifying existing downloaded model weights on host PC
