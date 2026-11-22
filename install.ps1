# One-command install for FairLens (Windows PowerShell 5.1 or PowerShell 7):
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Dev] [-TorchIndex URL]
#   -Dev             also install requirements-dev.txt and the Playwright Chromium browser (UI smoke tests)
#   -TorchIndex URL  reinstall torch + torchvision from this PyTorch wheel index after the requirements,
#                    e.g. a CUDA index from https://pytorch.org/get-started/locally/ (env: FAIRLENS_TORCH_INDEX)
# Creates .venv with Python 3.11, installs pinned requirements and the fairlens package.
param(
    [switch]$Dev,
    [string]$TorchIndex = $env:FAIRLENS_TORCH_INDEX
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

function Test-Py311 {
    param([string]$Exe, [string[]]$Prefix)
    try {
        & $Exe @Prefix -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$pyExe = $null
$pyPrefix = @()
if (Test-Py311 "py" @("-3.11")) {
    $pyExe = "py"
    $pyPrefix = @("-3.11")
} elseif (Test-Py311 "python" @()) {
    $pyExe = "python"
}
if (-not $pyExe) {
    throw "Python 3.11 not found. Install it from https://www.python.org/downloads/ (tick 'py launcher') and run this script again."
}

if (-not (Test-Path ".venv")) { Invoke-Checked $pyExe ($pyPrefix + @("-m", "venv", ".venv")) }
$venvPy = Join-Path (Join-Path (Join-Path $PSScriptRoot ".venv") "Scripts") "python.exe"
if (-not (Test-Py311 $venvPy @())) {
    throw "The existing .venv is not Python 3.11. Delete the .venv folder and run this script again."
}

Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked $venvPy @("-m", "pip", "install", "-e", ".", "--no-deps")

if ($TorchIndex) {
    $torchV = (& $venvPy -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])").Trim()
    $visionV = (& $venvPy -c "import importlib.metadata as m; print(m.version('torchvision').split('+')[0])").Trim()
    Write-Host "Reinstalling torch==$torchV and torchvision==$visionV from $TorchIndex"
    Invoke-Checked $venvPy @("-m", "pip", "install", "--force-reinstall", "--no-deps", "--index-url", $TorchIndex,
        "torch==$torchV", "torchvision==$visionV")
}

if ($Dev) {
    Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "requirements-dev.txt")
    Invoke-Checked $venvPy @("-m", "playwright", "install", "chromium")
}

Invoke-Checked $venvPy @("scripts\check_env.py", "--no-write") | Out-Null
Invoke-Checked $venvPy @("-c", "import fairlens.hardware as h; print('Compute device:', h.device_label())")
Write-Host ""
Write-Host "FairLens installed. Start the dashboard with:  .venv\Scripts\python.exe -m streamlit run app.py"
