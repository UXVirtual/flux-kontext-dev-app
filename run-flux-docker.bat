@echo off
:: This batch script runs the FLUX AI Docker container.
:: It requires the Hugging Face Hub token as the first argument.

:: --- Argument Check ---
:: Check if the first argument (%1) is empty. If it is, jump to the error handler.
IF "%~1"=="" GOTO :handle_error

:: --- Set Variables ---
SET HF_TOKEN=%1
SET IMAGE_NAME=flux-kontext-app
SET SCRIPT_DIR=%~dp0
SET CACHE_DIR="%USERPROFILE%\.cache\huggingface"
SET OUTPUT_DIR="%SCRIPT_DIR%output"

echo [INFO] Starting Docker container for image: %IMAGE_NAME%
echo [INFO] Using cache directory: %CACHE_DIR%
echo [INFO] Using output directory: %OUTPUT_DIR%
echo.

:: --- Run Docker Container ---
docker run ^
  --rm ^
  -it ^
  --gpus all ^
  -e HUGGING_FACE_HUB_TOKEN=%HF_TOKEN% ^
  -v %CACHE_DIR%:/root/.cache/huggingface ^
  -v %OUTPUT_DIR%:/app/output ^
  %IMAGE_NAME%

echo.
echo [INFO] Docker container has finished.

:: Jump to the end of the file to skip the error handler section.
GOTO :eof


:: --- Error Handler Section ---
:handle_error
echo.
echo [ERROR] Hugging Face token not provided.
echo.
echo Usage:
echo   run-flux-docker.bat YOUR_HF_TOKEN
echo.
echo Please pass your Hugging Face token (starting with "hf_") as an argument.
echo.


:eof
:: Marks the end of the script.