"""Core-only NES ROM benchmark; does not include Tk image rendering."""
import os
import sys
import time
from pathlib import Path

from nes import NESBus


def main():
    if len(sys.argv) < 2:
        print('Usage: python bench_rom.py "path\\game.nes" [frames]')
        return 2
    rom = Path(sys.argv[1]).expanduser().resolve()
    if not rom.is_file():
        print(f'ROM not found: {rom}')
        return 2
    frames = 120
    if len(sys.argv) > 2:
        try:
            frames = max(10, min(2000, int(sys.argv[2])))
        except ValueError:
            print('frames must be an integer from 10 to 2000')
            return 2
    nes = None
    try:
        nes = NESBus(str(rom), audio_enabled=False)
        # Warm up caches first; don't include the cold first frame in the timing.
        for _ in range(5):
            nes.begin_frame()
            if not nes.run_until_frame():
                raise RuntimeError('Frame did not complete within the instruction guard')
        samples = []
        for _ in range(frames):
            nes.begin_frame()
            start = time.perf_counter()
            completed = nes.run_until_frame()
            elapsed = time.perf_counter() - start
            if not completed:
                raise RuntimeError('Frame did not complete within the instruction guard')
            samples.append(elapsed)
        average = sum(samples) / len(samples)
        p95 = sorted(samples)[int(len(samples) * 0.95) - 1]
        print(f'ROM: {rom.name}')
        print(f'Frames measured: {len(samples)}')
        print(f'Average core frame time: {average * 1000:.3f} ms')
        print(f'Core-only FPS: {1.0 / average:.2f}')
        print(f'P95 core frame time: {p95 * 1000:.3f} ms')
        print('Note: excludes Tk/GUI presentation and audio; it is not the displayed in-game FPS.')
        return 0
    except Exception as exc:
        print(f'Benchmark failed: {exc}')
        return 1
    finally:
        if nes is not None:
            nes.close()


if __name__ == '__main__':
    raise SystemExit(main())
