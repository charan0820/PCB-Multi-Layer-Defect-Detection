@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow START_TRAINING.md to install Python and dependencies.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\train_deeppcb.py %*
if errorlevel 1 (
  echo Training stopped. Read the error above. For CUDA out of memory, retry with --batch 2.
  pause
  exit /b 1
)
pause
