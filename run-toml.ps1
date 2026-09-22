param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Profile,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraOverrides
)

python -m eiffel.toml_runner $Profile @HydraOverrides
