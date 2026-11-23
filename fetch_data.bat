@echo off
rem Double-click to download + prepare the training data (runs fetch_data.ps1, log in fetch_data.log)
rem Extra options go through, e.g.:  fetch_data.bat -LichessShards 8   or   fetch_data.bat -SkipLichess
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fetch_data.ps1" %*
echo.
pause
