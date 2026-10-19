Python NES Emulator — GUI Refresh Fixed

RUN
- Extract this ZIP to a NEW folder.
- Run run.bat, then press Ctrl+O to open a .nes ROM.
- Or drag a .nes file onto run_mario.bat.
- The emulation loop yields for 1 ms between slices and flushes Tk idle redraws; you should not have to click the File menu to make the image update.
- Run TEST_NO_CLICK.bat for a short visible test with a generated diagnostic ROM. It opens and animates automatically without mouse input.
- Python 3.10+ with Tkinter is required. No pygame/numpy/Pillow dependency.

CONTROLS
- Arrow keys: directions
- Z: A
- X: B
- Enter: Start
- Right Shift: Select
- F8: Key mapping
- F11: Fullscreen
- Alt+1..Alt+4: display scale
- F5: Reset; F6: Pause; F7: Audio

WHAT CHANGED IN THIS BUILD
- A checked CPU super-instruction recognizes exact Mapper-0/NROM polling loops `BIT $2002; BPL/BMI/BVC/BVS back`, with branch polarity checked. All other instruction sequences use the normal CPU decoder.
- A checked self-JMP batching path accelerates only a verified `JMP` to its own address and preserves CPU-cycle/frame state across the same sync boundaries.
- Sprite caches are not invalidated for unrelated NMI/nametable or mask changes; sprite decode/size changes and OAM/CHR updates still invalidate them.
- The full CPU/PPU state was compared between optimized and ordinary execution across 8 consecutive frames for BPL/BMI/BVC/BVS variants; register, cycle, timing, status, memory and frame data matched. A separate 12-frame self-JMP test also matched full state.
- PPU uses dot-segment rendering for mid-scanline scroll/mask/PPUCTRL changes, cached background rows, and sparse sprite-pixel composition. Regression tests cover scrolling, Sprite-0 timing, left-edge clipping, tile cache, OAM and simultaneous input.
- Default GUI instruction slice is 6000; set NES_INPUT_SLICE to a value from 800 through 12000 to trade emulation chunk size against keyboard/window response.
- Fixed the UI-refresh starvation symptom: continuing slices no longer form a chain of after(0) callbacks. A 1 ms positive delay lets Tk/window-manager paint events run, and draw_frame flushes idle redraw work so gameplay does not depend on clicking a menu.

MEASUREMENTS IN THIS ENVIRONMENT
- No-click GUI smoke test, opened automatically for 3 seconds with the included diagnostic ROM: ~60 drawn FPS; confirmed no mouse clicks were needed for continuous frame updates. This is a GUI/event-loop test, not a direct Super Mario Bros ROM test.
- Generated NROM self-JMP benchmark, 120 frames: 431.8 core FPS optimized vs 65.1 core FPS through the regular interpreter; exact state matched.
- Tk GUI on included generated ROMs: diagnostic ROM measured 60.0 drawn FPS and PPU polling stress measured 60.0 drawn FPS in the current run.
- These are measurements of included generated tests, not Super Mario Bros itself. The real Mario ROM was not supplied in this environment, so game-specific performance and compatibility have not been directly verified. Fixed 60 FPS cannot be guaranteed for every ROM/PC; this PPU is not transistor-level cycle-accurate.

TEST
- Run CHECK.bat to regenerate the synthetic PPU poll ROM and execute CPU, PPU, rendering, input and equivalence checks.
- tests/diagnostic.nes and tests/ppu_poll_stress.nes are generated test ROMs, not commercial game ROMs.
- `python bench_rom.py "path\game.nes" 240` measures core-only throughput; it excludes Tk rendering and is not the displayed in-game FPS.
