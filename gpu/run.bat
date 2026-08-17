@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo       DigitalOrganoid GPU Evolution Benchmark Runner
echo ================================================================
echo.
echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in system PATH.
    echo Please install Python 3.10+ and ensure "Add Python to PATH" is checked.
    echo.
    pause
    exit /b 1
)

echo Checking PyTorch availability...
python -c "import torch" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch is not installed in your Python environment.
    echo Please install PyTorch by running:
    echo     pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

echo Checking CUDA acceleration...
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] CUDA GPU detected! Running evolutionary sweep on GPU device...
    set DEVICE=cuda
) else (
    echo [WARNING] CUDA GPU not detected. Falling back to CPU execution...
    set DEVICE=cpu
)

echo.
echo Starting Evolutionary Benchmark with Flocke Mechanisms:
echo   - Innate CPG Prior (--cpg)
echo   - Cerebellar Predictive Model (--cerebellum)
echo   - Multi-Channel Neuromodulation (--multichannel-neuromod)
echo   - Bilateral Symmetric Initialization (--bilateral-symmetric)
echo.

python gpu_evolve.py --pop 64 --gens 60 --device %DEVICE% --cpg --cerebellum --multichannel-neuromod --bilateral-symmetric --out champion_flocke_gpu.npy

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================================
    echo [ERROR] GPU Evolution Benchmark encountered an error!
    echo Error Code: %ERRORLEVEL%
    echo Please review the traceback above for details.
    echo ================================================================
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================================
echo [SUCCESS] GPU Evolution Benchmark completed successfully!
echo Champion brain saved to champion_flocke_gpu.npy
echo Auto-closing in 3 seconds...
echo ================================================================
timeout /t 3 >nul
exit /b 0
