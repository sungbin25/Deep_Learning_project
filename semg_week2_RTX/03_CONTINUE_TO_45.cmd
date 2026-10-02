@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run 01_SETUP_GPU.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u -X utf8 run_gpu_pipeline.py --epochs 45 --batch-size 16 --threads 4
if errorlevel 1 (
  echo RUN FAILED. See output\week2\gpu_console.log
  pause
  exit /b 1
)
pause
