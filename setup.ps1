param(
    [switch]$Force,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host "Eiffel FL Security Lab - environment setup"
Write-Host "Repository: $Root"
Write-Host ""

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw @"
Python Launcher ('py') was not found.
Install Python 3.10, reopen the terminal, then run this script again.

Windows:
  winget install -e --id Python.Python.3.10
"@
}

& py -3.10 --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw @"
Python 3.10 is not installed or is not visible to the Python Launcher.

Install it with:
  winget install -e --id Python.Python.3.10

Then reopen the terminal and verify:
  py -0p
"@
}

if ($Force -and (Test-Path $Venv)) {
    Write-Host "Removing existing .venv because -Force was requested..."
    Remove-Item -Recurse -Force $Venv
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating .venv with Python 3.10..."
    & py -3.10 -m venv $Venv
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the virtual environment."
    }
} else {
    Write-Host "Using existing .venv."
}

Push-Location $Root
try {
    Write-Host "Upgrading pip/setuptools/wheel..."
    & $Python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }

    Write-Host "Installing Eiffel and runtime dependencies..."
    & $Python -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }

    if ($Dev) {
        Write-Host "Installing development/test dependencies..."
        & $Python -m pip install pytest
        if ($LASTEXITCODE -ne 0) { throw "Development dependency installation failed." }
    }

    Write-Host ""
    Write-Host "Running environment import check..."
    & $Python -c "import sys, tensorflow, flwr, hydra, h5py, eiffel; print('Python:', sys.version.split()[0]); print('TensorFlow:', tensorflow.__version__); print('Flower:', flwr.__version__); print('h5py:', h5py.__version__); print('Environment OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "Environment import check failed."
    }

    Write-Host ""
    Write-Host "Setup completed."
    Write-Host "Next:"
    Write-Host "  .\run.cmd -Doctor"
    Write-Host "  .\run.cmd synthetic_50k_quick_clean"
} finally {
    Pop-Location
}
