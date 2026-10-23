# One-command install for FairLens (Windows PowerShell). Same steps as install.sh.
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Creates .venv with Python 3.11, installs pinned requirements and the fairlens package.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($argv -join ' ')" }
}

$pyExe = $null; $pyArgs = @()
if ($env:PYTHON) { $pyExe = $env:PYTHON }
else {
    try { & py -3.11 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArgs = @("-3.11") } } catch {}
    if (-not $pyExe) { $pyExe = "python" }
}
Invoke-Checked $pyExe ($pyArgs + @("-c", "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 required, found ' + sys.version"))

if (-not (Test-Path ".venv")) { Invoke-Checked $pyExe ($pyArgs + @("-m", "venv", ".venv")) }
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked $venvPy @("-m", "pip", "install", "-e", ".", "--no-deps")
Invoke-Checked $venvPy @("scripts\check_env.py") | Out-Null
Write-Host "`nFairLens installed. Start the dashboard with:  .venv\Scripts\streamlit run app.py"
