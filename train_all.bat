@echo off
rem Double-click to run overfit check -> Lichess pre-training -> fine-tuning (log in train.log)
rem Options pass through, e.g.:  train_all.bat -SkipOverfit   or   train_all.bat -SkipPretrain
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_all.ps1" %*
echo.
pause
