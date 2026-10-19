@echo off
setlocal
cd /d "%~dp0"
python -O main.py
if errorlevel 1 (
  echo.
  echo Emulator could not start.
  echo Python 3.10+ with Tkinter is required.
  echo.
  pause
)
endlocal
