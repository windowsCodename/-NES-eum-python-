@echo off
setlocal
cd /d "%~dp0"
python -m py_compile *.py tests\*.py
if errorlevel 1 goto fail
python tests\make_ppu_poll_stress.py
if errorlevel 1 goto fail
python tests\self_test.py
if errorlevel 1 goto fail
python tests\test_fast_extended_ops.py
if errorlevel 1 goto fail
python tests\test_fast_nrom_idle_jmp.py
if errorlevel 1 goto fail
python tests\test_sprite_cache_invalidation.py
if errorlevel 1 goto fail
python tests\test_core_nmi_controller.py
if errorlevel 1 goto fail
python tests\test_ppu_sync_quantum.py
if errorlevel 1 goto fail
python tests\test_fast_ppustatus.py
if errorlevel 1 goto fail
python tests\test_scanline_render.py
if errorlevel 1 goto fail
python tests\test_background_row_cache.py
if errorlevel 1 goto fail
python tests\test_dot_segment_rendering.py
if errorlevel 1 goto fail
python tests\test_sparse_sprite_equivalence.py
if errorlevel 1 goto fail
python tests\test_input_latency.py
if errorlevel 1 goto fail
python tests\test_core_regressions.py
if errorlevel 1 goto fail
python tests\test_ppu_poll_superinstruction.py
if errorlevel 1 goto fail
echo.
echo ALL CHECKS PASSED
pause
exit /b 0
:fail
echo.
echo CHECK FAILED
pause
exit /b 1
