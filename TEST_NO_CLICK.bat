@echo off
setlocal
cd /d "%~dp0"
python tests\gui_no_click_smoke.py
if errorlevel 1 (
  echo.
  echo GUI refresh test failed. Check Python/Tkinter and nes_error.log.
  pause
  exit /b 1
)
echo.
echo GUI refresh test passed without clicking menus.
pause
endlocal
