# One-command install for FairLens (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Creates .venv with Python 3.11, installs pinned requirements and the fairlens package.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "py -3.11"
try { & py -3.11 --version | Out-Null } catch { $py = "python" }

if (-not (Test-Path ".venv")) { Invoke-Expression "$py -m venv .venv" }
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt
& $venvPy -m pip install -e . --no-deps
& $venvPy scripts\check_env.py | Out-Null
Write-Host "`nFairLens installed. Start the dashboard with:  .venv\Scripts\streamlit run app.py"
