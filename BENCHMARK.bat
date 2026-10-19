@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag a .nes file onto this BAT, or use:
  echo BENCHMARK.bat "C:\path\game.nes" 120
  pause
  exit /b 0
)
python bench_rom.py "%~1" %~2
pause
endlocal
