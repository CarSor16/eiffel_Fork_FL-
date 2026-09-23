@echo off
setlocal

if /I "%~1"=="-Smoke" (
    if not exist "%~dp0.venv\Scripts\python.exe" (
        echo Virtual environment not found. Run .\setup.cmd -Dev first.
        exit /b 1
    )
    echo Running synthetic client integration smoke test...
    "%~dp0.venv\Scripts\python.exe" -m pytest -q -s "%~dp0eiffel\core\tests\synthetic_client_integration_test.py" "%~dp0eiffel\core\tests\synthetic_flower_integration_test.py"
    if errorlevel 1 exit /b 1
    echo Synthetic client smoke test completed successfully.
    exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-experiment.ps1" %*
