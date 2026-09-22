param(
    [Parameter(Position=0)]
    [string]$Profile = "",

    [switch]$List,
    [switch]$DryRun,
    [switch]$Doctor,

    [ValidateSet("none", "synthetic50k")]
    [string]$Suite = "none",

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProfilesDir = Join-Path $Root "experiments\toml"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

function Get-Profiles {
    if (-not (Test-Path $ProfilesDir)) {
        throw "Experiment directory not found: $ProfilesDir"
    }
    return @(Get-ChildItem -Path $ProfilesDir -Filter "*.toml" -File | Sort-Object Name)
}

function Require-Environment {
    if (-not (Test-Path $Python)) {
        throw @"
The project virtual environment was not found.

Run the one-time setup first:
  .\setup.cmd

If Python 3.10 is not installed:
  winget install -e --id Python.Python.3.10
"@
    }
}

function Resolve-Profile([string]$Name) {
    if (Test-Path $Name -PathType Leaf) {
        return (Resolve-Path $Name).Path
    }

    $Candidate = $Name
    if (-not $Candidate.EndsWith(".toml")) {
        $Candidate = "$Candidate.toml"
    }

    $Full = Join-Path $ProfilesDir $Candidate
    if (Test-Path $Full -PathType Leaf) {
        return (Resolve-Path $Full).Path
    }

    throw "Unknown experiment profile '$Name'. Use '.\run.cmd -List' to see available profiles."
}

function Invoke-Profile([string]$Path, [switch]$ValidateOnly) {
    $Args = @("-m", "eiffel.toml_runner", $Path)
    if ($ValidateOnly) {
        $Args += "--dry-run"
    }
    if ($HydraOverrides) {
        $Args += $HydraOverrides
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Experiment: $(Split-Path $Path -Leaf)"
    Write-Host "============================================================"
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $(Split-Path $Path -Leaf)"
    }
}

$Profiles = Get-Profiles

if ($List) {
    Write-Host "Available TOML experiments:"
    foreach ($Item in $Profiles) {
        Write-Host "  $($Item.BaseName)"
    }
    exit 0
}

Require-Environment

Push-Location $Root
try {
    if ($Doctor) {
        Write-Host "Eiffel FL Security Lab - doctor"
        Write-Host ""
        $env:TF_CPP_MIN_LOG_LEVEL = "2"
        & $Python -m pip check
        if ($LASTEXITCODE -ne 0) { throw "pip check found missing or incompatible dependencies." }
        & $Python -c "import sys, importlib.metadata, pkg_resources, absl, tensorflow, flwr, hydra, h5py, numpy, google.protobuf, eiffel; assert sys.version_info[:2] == (3,10), sys.version; assert tensorflow.__version__.startswith('2.10.'), tensorflow.__version__; assert flwr.__version__ == '1.5.0', flwr.__version__; print('Python:', sys.version.split()[0]); print('setuptools:', importlib.metadata.version('setuptools')); print('absl-py:', importlib.metadata.version('absl-py')); print('TensorFlow:', tensorflow.__version__); print('Flower:', flwr.__version__); print('NumPy:', numpy.__version__); print('Protobuf:', google.protobuf.__version__); print('h5py:', h5py.__version__); print('pkg_resources: OK'); print('Imports: OK')"
        if ($LASTEXITCODE -ne 0) { throw "Environment import check failed." }

        $DoctorProfile = Resolve-Profile "synthetic_50k_quick_clean"
        Write-Host ""
        Write-Host "Checking TOML -> Hydra translation..."
        Invoke-Profile $DoctorProfile -ValidateOnly
        Write-Host ""
        Write-Host "Doctor completed successfully."
        exit 0
    }

    if ($Suite -eq "synthetic50k") {
        $Names = @(
            "synthetic_50k_clean",
            "synthetic_50k_label_flip",
            "synthetic_50k_sign_flip",
            "synthetic_50k_model_scaling",
            "synthetic_50k_gaussian_noise",
            "synthetic_50k_lie",
            "synthetic_50k_gradient_mimicry",
            "synthetic_50k_colluding_sign_flip"
        )
        foreach ($Name in $Names) {
            Invoke-Profile (Resolve-Profile $Name) -ValidateOnly:$DryRun
        }
        exit 0
    }

    if (-not $Profile) {
        Write-Host "Select an experiment:"
        Write-Host ""
        for ($Index = 0; $Index -lt $Profiles.Count; $Index++) {
            Write-Host ("[{0,2}] {1}" -f ($Index + 1), $Profiles[$Index].BaseName)
        }
        Write-Host ""
        $Selection = Read-Host "Experiment number"

        $Number = 0
        if (-not [int]::TryParse($Selection, [ref]$Number)) {
            throw "Invalid selection '$Selection'."
        }
        if ($Number -lt 1 -or $Number -gt $Profiles.Count) {
            throw "Selection must be between 1 and $($Profiles.Count)."
        }
        $ProfilePath = $Profiles[$Number - 1].FullName
    } else {
        $ProfilePath = Resolve-Profile $Profile
    }

    Invoke-Profile $ProfilePath -ValidateOnly:$DryRun
} finally {
    Pop-Location
}
