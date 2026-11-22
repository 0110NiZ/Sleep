@echo off
setlocal
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 "%~dp0serve.py"
) else (
    python "%~dp0serve.py"
)
if errorlevel 1 (
    echo Install Python 3.9 or newer if Python was not found.
    pause
)
