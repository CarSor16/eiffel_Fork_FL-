param(
    [Parameter(Position=0)]
    [string]$Run = "",
    [switch]$Show
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run .\setup.cmd first."
}

$Args = @("-m", "eiffel.analysis.plot_round_metrics")
if ($Run) {
    $Args += $Run
}
if ($Show) {
    $Args += "--show"
}

Push-Location $Root
try {
    & $Python @Args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
