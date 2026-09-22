param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Profile,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraOverrides
)

$Args = @("-Profile", $Profile)
if ($DryRun) {
    $Args += "-DryRun"
}
if ($HydraOverrides) {
    $Args += $HydraOverrides
}

& "$PSScriptRoot\run-experiment.ps1" @Args
exit $LASTEXITCODE
