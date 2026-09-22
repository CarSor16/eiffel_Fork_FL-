$ErrorActionPreference = "Stop"

$profiles = @(
    "synthetic_50k_clean.toml",
    "synthetic_50k_label_flip.toml",
    "synthetic_50k_sign_flip.toml",
    "synthetic_50k_model_scaling.toml",
    "synthetic_50k_gaussian_noise.toml",
    "synthetic_50k_lie.toml",
    "synthetic_50k_gradient_mimicry.toml",
    "synthetic_50k_colluding_sign_flip.toml"
)

foreach ($profile in $profiles) {
    Write-Host ""
    Write-Host "=== $profile ==="
    python -m eiffel.toml_runner "experiments/toml/$profile"
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $profile"
    }
}
