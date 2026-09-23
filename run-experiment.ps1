param(
    [Parameter(Position=0)]
    [string]$Profile = "",

    [switch]$List,
    [switch]$DryRun,
    [switch]$Doctor,
    [switch]$Attacks,
    [switch]$Analyze,
    [switch]$Tests,
    [switch]$Smoke,
    [switch]$Version,

    [string]$RunsRoot = "outputs",
    [string]$AnalysisOutput = "analysis-results",
    [string]$ValidateHdf5 = "",

    [ValidateSet("none", "synthetic50k", "multiclass")]
    [string]$Suite = "none",

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProfilesDir = Join-Path $Root "experiments\toml"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LauncherVersion = "2026-09-23-smoke-failfast-1"

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

function Show-AttackHelp {
    Write-Host "Eiffel FL Security Lab - attacks and TOML controls"
    Write-Host ""
    Write-Host "Clean"
    Write-Host "  No adversarial modification."
    Write-Host "  TOML: attack.mechanism = 'none'"
    Write-Host ""
    Write-Host "Label Flip (data poisoning, Eiffel baseline)"
    Write-Host "  Flips local binary Benign/Attack labels before training."
    Write-Host "  TOML: poison_rate, objective, target, malicious_fraction, schedule."
    Write-Host "  family_aware keeps binary training and reports each attack family separately."
    Write-Host "  True multiclass label flip is blocked until an explicit"
    Write-Host "  source_class -> destination_class mapping is configured."
    Write-Host ""
    Write-Host "Sign Flip (model poisoning)"
    Write-Host "  Submits Delta_m <- -strength * Delta_m."
    Write-Host "  TOML: strength, malicious_fraction, schedule."
    Write-Host ""
    Write-Host "Model Scaling (model poisoning)"
    Write-Host "  Amplifies the malicious model update."
    Write-Host "  TOML: strength or scale_factor, malicious_fraction, schedule."
    Write-Host ""
    Write-Host "Gaussian Noise (model poisoning)"
    Write-Host "  Adds deterministic seeded Gaussian perturbation to malicious updates."
    Write-Host "  TOML: noise_std, malicious_fraction, schedule."
    Write-Host ""
    Write-Host "LIE (model poisoning)"
    Write-Host "  Crafts malicious updates from the same-round benign mean and std."
    Write-Host "  TOML: lie_z, malicious_fraction, schedule."
    Write-Host ""
    Write-Host "Gradient Mimicry (model poisoning)"
    Write-Host "  Blends a malicious update with the benign same-round reference update."
    Write-Host "  TOML: mimicry_lambda, malicious_fraction, schedule."
    Write-Host ""
    Write-Host "Colluding Sign Flip (coordinated model poisoning)"
    Write-Host "  Malicious clients submit a common transformed malicious centroid."
    Write-Host "  TOML: strength, malicious_fraction, schedule."
    Write-Host ""
    Write-Host "Temporal schedules"
    Write-Host "  continuous | late | window | on_off | gradual"
    Write-Host "  TOML: start_round, end_round, on_rounds, off_rounds, ramp_rounds,"
    Write-Host "        gradual_start_strength, gradual_end_strength."
    Write-Host ""
    Write-Host "General experiment parameters"
    Write-Host "  experiment: seed, num_clients, rounds"
    Write-Host "  dataset: name, task=binary|family_aware|multiclass, num_classes"
    Write-Host "  partition: type=iid|dirichlet, dirichlet_alpha"
    Write-Host "  model: name and architecture-specific fields"
    Write-Host "  training: local_epochs, learning_rate, batch_size"
    Write-Host "  storage: enabled, path, compression, capture_inference, probe_size"
    Write-Host ""
    Write-Host "Hydra remains the backend. TOML is translated to Hydra overrides, so ad-hoc"
    Write-Host "Hydra overrides can still be appended to any .\run.cmd experiment command."
}

$Profiles = Get-Profiles

if ($Version) {
    Write-Host "Eiffel FL Security Lab launcher: $LauncherVersion"
    exit 0
}

if (-not $Profile -and $HydraOverrides) {
    $UnknownOptions = @($HydraOverrides | Where-Object { $_ -like "-*" })
    if ($UnknownOptions.Count -gt 0) {
        throw "Unknown launcher option(s): $($UnknownOptions -join ', '). Run '.\run.cmd -List' or update the repository."
    }
}

if ($List) {
    Write-Host "Available TOML experiments:"
    foreach ($Item in $Profiles) {
        Write-Host "  $($Item.BaseName)"
    }
    exit 0
}

if ($Attacks) {
    Show-AttackHelp
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

        foreach ($DoctorName in @(
            "synthetic_50k_quick_clean",
            "synthetic_50k_quick_sign_flip",
            "synthetic_50k_quick_multiclass_clean",
            "synthetic_50k_quick_multiclass_sign_flip"
        )) {
            Write-Host ""
            Write-Host "Checking TOML -> Hydra translation: $DoctorName"
            Invoke-Profile (Resolve-Profile $DoctorName) -ValidateOnly
        }
        Write-Host ""
        Write-Host "Doctor completed successfully."
        exit 0
    }

    if ($Smoke) {
        Write-Host "Running synthetic client integration smoke test..."
        & $Python -m pytest -q -s "eiffel\core\tests\synthetic_client_integration_test.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Synthetic client smoke test failed."
        }
        Write-Host "Synthetic client smoke test completed successfully."
        exit 0
    }

    if ($Tests) {
        Write-Host "Running focused FL-security tests..."
        $TestFiles = @(
            "eiffel\core\tests\model_attacks_test.py",
            "eiffel\core\tests\label_flip_attack_test.py",
            "eiffel\core\tests\multiclass_models_test.py",
            "eiffel\core\tests\round_store_test.py",
            "eiffel\core\tests\compare_metrics_test.py",
            "eiffel\core\tests\plot_callback_test.py",
            "eiffel\core\tests\toml_runner_test.py",
            "eiffel\core\tests\synthetic_stress_test.py",
            "eiffel\core\tests\synthetic_client_integration_test.py"
        )
        & $Python -m pytest @TestFiles
        if ($LASTEXITCODE -ne 0) {
            throw "Focused FL-security tests failed. If pytest is missing, run '.\setup.cmd -Dev'."
        }
        exit 0
    }

    if ($ValidateHdf5) {
        & $Python -m eiffel.analysis.validate_round_state $ValidateHdf5
        if ($LASTEXITCODE -ne 0) {
            throw "HDF5 validation failed: $ValidateHdf5"
        }
        exit 0
    }

    if ($Analyze) {
        $ResolvedRunsRoot = if ([System.IO.Path]::IsPathRooted($RunsRoot)) {
            $RunsRoot
        } else {
            Join-Path $Root $RunsRoot
        }
        $ResolvedAnalysisOutput = if ([System.IO.Path]::IsPathRooted($AnalysisOutput)) {
            $AnalysisOutput
        } else {
            Join-Path $Root $AnalysisOutput
        }
        & $Python -m eiffel.analysis.compare_metrics --runs-root $ResolvedRunsRoot --output-dir $ResolvedAnalysisOutput
        if ($LASTEXITCODE -ne 0) {
            throw "Metric analysis failed."
        }
        Write-Host ""
        Write-Host "Analysis written to: $ResolvedAnalysisOutput"
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

    if ($Suite -eq "multiclass") {
        $Names = @(
            "synthetic_50k_multiclass_clean",
            "synthetic_50k_multiclass_sign_flip",
            "synthetic_50k_multiclass_model_scaling",
            "synthetic_50k_multiclass_gaussian_noise",
            "synthetic_50k_multiclass_lie",
            "synthetic_50k_multiclass_gradient_mimicry",
            "synthetic_50k_multiclass_colluding_sign_flip"
        )
        foreach ($Name in $Names) {
            Invoke-Profile (Resolve-Profile $Name) -ValidateOnly:$DryRun
        }
        exit 0
    }

    if (-not $Profile) {
        Show-AttackHelp
        Write-Host ""
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
