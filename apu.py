"""Lightweight NES APU.

This is a practical audio implementation rather than transistor-level APU emulation.
It supports the two pulse channels, triangle, noise, $4015 status, and a Windows
sound backend using only Python's standard library. On non-Windows systems the APU
still runs internally but audio output is disabled.
"""

from __future__ import annotations

import math
import queue
import struct
import sys
import threading
import wave

CPU_HZ = 1_789_773
SAMPLE_RATE = 22_050
FRAME_SAMPLES = 735  # about 33 ms at 22.05 kHz


class AudioPlayer:
    def __init__(self, enabled=True):
        self.enabled = enabled and sys.platform.startswith("win")
        self.q = queue.Queue(maxsize=3)
        self.thread = None
        self.stop_event = threading.Event()
        if self.enabled:
            import winsound  # noqa: F401
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    @staticmethod
    def _wav(samples):
        raw = struct.pack("<" + "h" * len(samples), *samples)
        import io
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(raw)
        return bio.getvalue()

    def submit(self, samples):
        if not self.enabled or not samples:
            return
        try:
            self.q.put_nowait(self._wav(samples))
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(self._wav(samples))
            except queue.Full:
                pass

    def _worker(self):
        import winsound
        while not self.stop_event.is_set():
            try:
                data = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_SYNC)

    def close(self):
        self.stop_event.set()
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


class APU:
    DUTY = (
        (0, 1, 0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0, 0, 0),
        (0, 1, 1, 1, 1, 0, 0, 0),
        (1, 0, 0, 1, 1, 1, 1, 1),
    )

    def __init__(self, audio_enabled=True):
        self.reg = bytearray(0x18)
        self.enabled = [False] * 5
        self.sample_acc = 0
        self.pulse_phase = [0.0, 0.0]
        self.tri_phase = 0.0
        self.noise_lfsr = 1
        self.player = AudioPlayer(audio_enabled)
        self.samples = []

    def close(self):
        self.flush_audio()
        self.player.close()

    def write(self, addr, value):
        value &= 0xFF
        self.reg[addr - 0x4000] = value
        if addr == 0x4015:
            for i in range(5):
                self.enabled[i] = bool(value & (1 << i))

    def read_status(self):
        return sum((1 << i) for i, e in enumerate(self.enabled) if e)

    def _pulse(self, channel):
        base = 0x4000 if channel == 0 else 0x4004
        r0 = self.reg[base - 0x4000]
        lo = self.reg[base + 2 - 0x4000]
        hi = self.reg[base + 3 - 0x4000]
        timer = lo | ((hi & 7) << 8)
        volume = r0 & 0x0F
        duty = (r0 >> 6) & 3
        if not self.enabled[channel] or volume == 0 or timer < 2:
            return 0.0
        freq = CPU_HZ / (16.0 * (timer + 1.0))
        self.pulse_phase[channel] = (self.pulse_phase[channel] + freq / SAMPLE_RATE) % 1.0
        step = int(self.pulse_phase[channel] * 8) & 7
        return (1.0 if self.DUTY[duty][step] else -1.0) * (volume / 15.0)

    def _triangle(self):
        base = 0x4008
        lo = self.reg[base + 2 - 0x4000]
        hi = self.reg[base + 3 - 0x4000]
        timer = lo | ((hi & 7) << 8)
        if not self.enabled[2] or timer < 2:
            return 0.0
        freq = CPU_HZ / (32.0 * (timer + 1.0))
        self.tri_phase = (self.tri_phase + freq / SAMPLE_RATE) % 1.0
        p = self.tri_phase
        tri = 4.0 * p if p < 0.5 else 4.0 * (1.0 - p)
        return tri - 1.0

    def _noise(self):
        r = self.reg[0x0C]
        volume = r & 0x0F
        if not self.enabled[3] or volume == 0:
            return 0.0
        period_table = (4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068)
        period = period_table[self.reg[0x0E] & 0x0F]
        if period <= 0:
            return 0.0
        # Advance by a small deterministic amount each sample.
        steps = max(1, int(CPU_HZ / SAMPLE_RATE / period))
        for _ in range(steps):
            bit = ((self.noise_lfsr >> 0) ^ (self.noise_lfsr >> (6 if (self.reg[0x0E] & 0x80) else 1))) & 1
            self.noise_lfsr = (self.noise_lfsr >> 1) | (bit << 14)
        return (1.0 if (self.noise_lfsr & 1) == 0 else -1.0) * (volume / 15.0)

    def sample(self):
        p1 = self._pulse(0)
        p2 = self._pulse(1)
        tri = self._triangle()
        noise = self._noise()
        # Practical mix, normalized to keep clipping low.
        mixed = (p1 * 0.20) + (p2 * 0.20) + (tri * 0.20) + (noise * 0.20)
        return max(-1.0, min(1.0, mixed))

    def step_cpu(self, cycles=1):
        self.sample_acc += SAMPLE_RATE * cycles
        while self.sample_acc >= CPU_HZ:
            self.sample_acc -= CPU_HZ
            self.samples.append(int(self.sample() * 12_000))
            if len(self.samples) >= FRAME_SAMPLES:
                self.flush_audio()

    def flush_audio(self):
        if self.samples:
            self.player.submit(self.samples[:])
            self.samples.clear()
