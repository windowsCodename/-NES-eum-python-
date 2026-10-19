@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag a .nes file onto this BAT, or use:
  echo run_mario.bat "C:\path\game.nes"
  pause
  exit /b 0
)
python -O main.py "%~1"
if errorlevel 1 pause
endlocal
