param([string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Assert-Exit([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" } }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'NVIDIA driver not found. Install/update the RTX driver, then try again.' }
& nvidia-smi
Assert-Exit 'NVIDIA driver check'
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    if ($PythonExe) {
        & $PythonExe -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
        Assert-Exit 'Python version check'
        & $PythonExe -m venv .venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } else {
        & python -c 'import sys; assert sys.version_info[:2] == (3,12), "Install Python 3.12 64-bit"'
        Assert-Exit 'Python version check'
        & python -m venv .venv
    }
    Assert-Exit 'Virtual environment creation'
}
$ProjectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $ProjectPython -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 environment required"'
Assert-Exit 'Virtual environment version check'
& $ProjectPython -m pip install --upgrade pip
Assert-Exit 'pip upgrade'
& $ProjectPython -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
Assert-Exit 'CUDA PyTorch installation'
& $ProjectPython -m pip install --no-cache-dir -r requirements-gpu.txt
Assert-Exit 'Analysis package installation'
& $ProjectPython -X utf8 check_gpu.py --require-cuda
Assert-Exit 'CUDA execution check'
Write-Host 'SETUP COMPLETE. Next: run 02_RUN_GPU.cmd' -ForegroundColor Green
