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

$Python310 = $null
try {
    $Python310 = (& py -3.10 -c "import sys; print(sys.executable)" 2>$null)
} catch {
    $Python310 = $null
}

if (-not $Python310) {
    Write-Host ""
    Write-Host "Python 3.10 was not found." -ForegroundColor Yellow
    Write-Host "Detected Python installations:"
    & py -0p
    Write-Host ""
    Write-Host "Install Python 3.10 with:" -ForegroundColor Yellow
    Write-Host "  winget install -e --id Python.Python.3.10"
    Write-Host ""
    Write-Host "Then close and reopen the terminal and run:"
    Write-Host "  .\setup.cmd"
    exit 1
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
    Write-Host "Upgrading pip/wheel and installing a Ray-compatible setuptools..."
    & $Python -m pip install --upgrade pip wheel
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
    & $Python -m pip install "setuptools==80.9.0"
    if ($LASTEXITCODE -ne 0) { throw "setuptools bootstrap failed." }

    Write-Host "Installing Eiffel and pinned runtime dependencies..."
    $Constraints = Join-Path $Root "constraints-py310.txt"
    & $Python -m pip install -c $Constraints -e .
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Dependency installation failed." -ForegroundColor Red
        Write-Host "Rebuild the environment after pulling the latest branch with:"
        Write-Host "  .\setup.cmd -Force"
        throw "Project installation failed."
    }

    if ($Dev) {
        Write-Host "Installing development/test dependencies..."
        & $Python -m pip install pytest
        if ($LASTEXITCODE -ne 0) { throw "Development dependency installation failed." }
    }

    Write-Host ""
    Write-Host "Running environment import check..."
    & $Python -c "import sys, importlib.metadata, pkg_resources, tensorflow, flwr, hydra, h5py, numpy, google.protobuf, eiffel; print('Python:', sys.version.split()[0]); print('setuptools:', importlib.metadata.version('setuptools')); print('TensorFlow:', tensorflow.__version__); print('Flower:', flwr.__version__); print('NumPy:', numpy.__version__); print('Protobuf:', google.protobuf.__version__); print('h5py:', h5py.__version__); assert tensorflow.__version__.startswith('2.10.'), tensorflow.__version__; assert flwr.__version__ == '1.5.0', flwr.__version__; print('pkg_resources: OK'); print('Environment OK')"
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
