param(
    [switch]$DryRun
)

$Args = @("-Suite", "synthetic50k")
if ($DryRun) {
    $Args += "-DryRun"
}

& "$PSScriptRoot\run-experiment.ps1" @Args
exit $LASTEXITCODE
