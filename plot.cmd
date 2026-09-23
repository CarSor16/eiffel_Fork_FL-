@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0plot-metrics.ps1" %*
