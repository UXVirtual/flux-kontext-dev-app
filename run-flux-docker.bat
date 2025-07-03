@echo off
setlocal enabledelayedexpansion

:: This batch script runs the FLUX AI Docker container.
:: It correctly handles all arguments and dynamically mounts the input image's directory.

:: --- Set Main Variables ---
SET "IMAGE_NAME=flux-kontext-app"
SET "SCRIPT_DIR=%~dp0"
SET "CACHE_DIR=%USERPROFILE%\.cache\huggingface"
SET "OUTPUT_DIR=%SCRIPT_DIR%output"
SET "MODEL_CACHE_PATH=%CACHE_DIR%\hub\models--black-forest-labs--FLUX.1-Kontext-dev"

:: --- Argument and Model Cache Check ---
IF NOT EXIST "%MODEL_CACHE_PATH%" (
    echo %* | find "--hf_token" >nul
    IF ERRORLEVEL 1 (
        echo.
        echo [ERROR] Model not downloaded. The --hf_token argument is required for the first run.
        GOTO :handle_error
    )
)

:: --- Parse Arguments and Find Image Path ---
SET "HOST_IMAGE_PATH="
SET "OTHER_ARGS="

set "arg_list=%*"
:arg_parse_loop
for /f "tokens=1,2,*" %%a in ("!arg_list!") do (
    if /i "%%a"=="--image_path" (
        SET "HOST_IMAGE_PATH=%%~b"
    ) else (
        SET "OTHER_ARGS=!OTHER_ARGS! %%a %%b"
    )
    set "arg_list=%%c"
    if defined arg_list goto :arg_parse_loop
)

IF NOT DEFINED HOST_IMAGE_PATH (
    echo [ERROR] --image_path argument not found or is missing a value.
    GOTO :handle_error
)

:: --- Dynamic Volume Mount Logic ---
FOR %%F IN ("%HOST_IMAGE_PATH%") DO SET "HOST_IMAGE_DIR=%%~dpF"
FOR %%F IN ("%HOST_IMAGE_PATH%") DO SET "IMAGE_FILENAME=%%~nxF"

IF NOT DEFINED HOST_IMAGE_DIR (
    echo [ERROR] Could not determine the directory for the input image. Please use a full path.
    GOTO :handle_error
)

SET "CONTAINER_INPUT_DIR=/app/input_image"
SET "CONTAINER_IMAGE_PATH_ARG=--image_path "%CONTAINER_INPUT_DIR%/%IMAGE_FILENAME%""

echo [INFO] Starting Docker container...
echo [INFO] Mapping host directory '!HOST_IMAGE_DIR!' to container directory '!CONTAINER_INPUT_DIR!'
echo.

:: --- Run Docker Container ---
docker run ^
  --rm ^
  -it ^
  --gpus all ^
  -v "!CACHE_DIR!:/root/.cache/huggingface" ^
  -v "!OUTPUT_DIR!:/app/output" ^
  -v "!HOST_IMAGE_DIR!:%CONTAINER_INPUT_DIR%" ^
  %IMAGE_NAME% python3 main.py %CONTAINER_IMAGE_PATH_ARG% !OTHER_ARGS!

echo.
echo [INFO] Docker container has finished.
GOTO :eof

:: --- Error Handler Section ---
:handle_error
echo.
echo Usage:
echo   %~n0 --image_path "C:\path\to\image.png" --prompt "your prompt here" [--hf_token YOUR_TOKEN]
echo.
GOTO :eof

:eof
