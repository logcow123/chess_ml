@echo off
rem Double-click to create/update .venv for training (runs setup_venv.ps1, log in setup_venv.log)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_venv.ps1"
echo.
pause
