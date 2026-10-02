@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_gpu.ps1"
if errorlevel 1 (
  echo SETUP FAILED. Read the error above.
  pause
  exit /b 1
)
pause
