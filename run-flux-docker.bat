@echo off
:: This batch script runs the FLUX AI Docker container in INTERACTIVE SERVER mode.
:: It correctly exposes the FastAPI port and a shared folder for input images.

:: --- Set Main Variables ---
SET "IMAGE_NAME=flux-kontext-app"
SET "SCRIPT_DIR=%~dp0"
SET "CACHE_DIR=%USERPROFILE%\.cache\huggingface"
SET "OUTPUT_DIR=%SCRIPT_DIR%output"
SET "MODEL_CACHE_PATH=%CACHE_DIR%\hub\models--black-forest-labs--FLUX.1-Kontext-dev"

:: --- Check if Model Exists and Token is Needed ---
:: This check is still useful for the very first run, looking for --hf_token.
IF NOT EXIST "%MODEL_CACHE_PATH%" (
    echo %* | find "--hf_token" >nul
    IF ERRORLEVEL 1 (
        echo.
        echo [ERROR] Model not downloaded. The --hf_token argument is required for the first run.
        echo.
        echo Usage for first run:
        echo   %~n0 --hf_token YOUR_TOKEN [--timeout 600]
        echo.
        GOTO :eof
    )
)

echo [INFO] Starting Docker container in interactive server mode...
echo [INFO] API will be available on http://localhost:8000
echo [INFO] Place input files in the '%SCRIPT_DIR%' folder.
echo.

:: --- Run Docker Container ---
:: - Forwards all optional arguments (%*) like --timeout to the Python script.
:: - Maps the port (-p) for the FastAPI server.
:: - Mounts the script directory (-v) to /app/host_files for easy file access.
docker run ^
  --rm ^
  -it ^
  --gpus all ^
  -p 8000:8000 ^
  -v "%CACHE_DIR%:/root/.cache/huggingface" ^
  -v "%OUTPUT_DIR%:/app/output" ^
  -v "%SCRIPT_DIR%:/app/host_files" ^
  %IMAGE_NAME% python3 main.py %*

echo.
echo [INFO] Docker container has finished.
GOTO :eof

:eof
