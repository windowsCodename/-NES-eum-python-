"""NES system bus tying CPU, PPU, APU, controllers, DMA and cartridge together."""

from cartridge import Cartridge
from controller import Controller
from ppu2c02 import PPU2C02
from cpu6502 import CPU6502
from apu import APU


class NESBus:
    CPU_CYCLES_PER_FRAME = 29780

    def __init__(self, rom_path, audio_enabled=True):
        self.cart = Cartridge(rom_path)
        self.ppu = PPU2C02(self.cart)
        self.apu = APU(audio_enabled=audio_enabled)
        self.cpu_ram = bytearray(2048)
        self.controller1 = Controller()
        self.controller2 = Controller()
        self.nmi_pending = False
        self.dma_pending = False
        self.dma_page = 0
        self.dma_addr = 0
        self.total_cpu_cycles = 0
        self._apu_accum = 0
        self._pending_cpu_cycles = 0
        # Mapper 4 has scanline IRQs, so retain a fine sync there. Other
        # common mappers (especially Mapper 0/NROM used by Super Mario Bros.)
        # do not use mapper scanline IRQs. Their PPU register reads/writes flush
        # pending time synchronously, preserving $2002 sprite-0 polling and
        # mid-scanline register splits. Batch ordinary CPU instructions more
        # aggressively to avoid invoking the Python renderer once per ~48 dots.
        self._sync_quantum = 16 if self.cart.mapper == 4 else (128 if self.cart.mapper == 0 else 64)
        # Safe instruction batching for NROM's common idle JMP-self loop. It
        # preserves exact CPU cycles and flushes at the same sync quantum, which
        # is especially important for Super Mario Bros' frame-wait loop.
        self.fast_nrom_idle_jmp_enabled = True
        self.reset()

    def close(self):
        self.cart.save()
        self.apu.close()

    def reset(self):
        self.cpu_ram[:] = b"\x00" * len(self.cpu_ram)
        self.ppu.scanline = -1
        self.ppu.cycle = 0
        self.ppu.ppustatus = 0
        self.ppu.ppuctrl = 0
        self.ppu.ppumask = 0
        self.ppu.v = self.ppu.t = 0
        self.ppu.w = 0
        self.ppu.fine_x = 0
        self.ppu.frame_complete = False
        self.ppu._carry_ticks = 0
        self.ppu._render_queue.clear()
        self.ppu._opacity_state_key = [None] * 240
        self.ppu._bg_content_generation += 1
        self.ppu.frame_scroll_x = 0
        self.ppu.frame_scroll_y = 0
        self.ppu.frame_nt = 0
        self.ppu._frame_bg_table = 0
        self.ppu.sprite0_hit_x = None
        self.ppu.sprite0_hit_y = None
        self.ppu.nmi = False
        self.ppu.invalidate_sprite_cache()
        self.nmi_pending = False
        self.dma_pending = False
        self.total_cpu_cycles = 0
        self._apu_accum = 0
        self._pending_cpu_cycles = 0
        self.cpu = CPU6502(self)
        self.cpu.reset()

    def _flush_pending(self):
        cycles = self._pending_cpu_cycles
        if cycles <= 0:
            return
        self._pending_cpu_cycles = 0
        self.ppu.advance(cycles * 3)
        if self.ppu.nmi:
            self.ppu.nmi = False
            self.nmi_pending = True
        if self.apu.player.enabled:
            self._apu_accum += cycles
            if self._apu_accum >= 1024:
                self.apu.step_cpu(self._apu_accum)
                self._apu_accum = 0

    def sync_for_ppu_access(self):
        self._flush_pending()

    def cpu_read_ppu_status_fast(self):
        """Hot path for CPU reads of mirrored PPUSTATUS ($2002).

        Classic NROM games poll this register in tight loops for VBlank and
        Sprite-0 splits. Keep the same pending-cycle/NMI/APU semantics as
        _flush_pending, but avoid the additional sync and PPU-register dispatch
        calls for this very common read.
        """
        cycles = self._pending_cpu_cycles
        if cycles > 0:
            self._pending_cpu_cycles = 0
            ppu = self.ppu
            ppu.advance(cycles * 3)
            if ppu.nmi:
                ppu.nmi = False
                self.nmi_pending = True
            if self.apu.player.enabled:
                self._apu_accum += cycles
                if self._apu_accum >= 1024:
                    self.apu.step_cpu(self._apu_accum)
                    self._apu_accum = 0
        ppu = self.ppu
        value = (ppu.ppustatus & 0xE0) | (ppu.open_bus & 0x1F)
        ppu.ppustatus &= 0x7F
        ppu.w = 0
        ppu.open_bus = value
        return value

    def cpu_read(self, addr):
        addr &= 0xFFFF
        if addr < 0x2000:
            return self.cpu_ram[addr & 0x7FF]
        if addr < 0x4000:
            self._flush_pending()
            return self.ppu.cpu_read(addr & 7)
        if 0x4000 <= addr <= 0x4013:
            return 0
        if addr == 0x4014:
            return 0
        if addr == 0x4015:
            return self.apu.read_status()
        if addr == 0x4016:
            return self.controller1.read()
        if addr == 0x4017:
            return self.controller2.read()
        if 0x6000 <= addr <= 0xFFFF:
            value = self.cart.cpu_read(addr)
            return 0 if value is None else value
        return 0

    def cpu_write(self, addr, value):
        addr &= 0xFFFF
        value &= 0xFF
        if addr < 0x2000:
            self.cpu_ram[addr & 0x7FF] = value
        elif addr < 0x4000:
            self._flush_pending()
            self.ppu.cpu_write(addr & 7, value)
        elif addr == 0x4014:
            self._flush_pending()
            self.dma_pending = True
            self.dma_page = value
            self.dma_addr = 0
        elif addr == 0x4016:
            self.controller1.write(value)
            self.controller2.write(value)
        elif 0x4000 <= addr <= 0x4017:
            if addr != 0x4016:
                self.apu.write(addr, value)
        elif addr >= 0x6000:
            self.cart.cpu_write(addr, value)

    def controller_buttons(self, names, player=1):
        (self.controller1 if player == 1 else self.controller2).set_buttons(names)

    def _advance_cpu_cycles(self, cycles):
        cycles = int(cycles)
        if cycles <= 0:
            return
        self.total_cpu_cycles += cycles
        self._pending_cpu_cycles += cycles
        if (self._pending_cpu_cycles >= self._sync_quantum or
                (self.ppu.scanline >= 240 and self._near_vblank_edge())):
            self._flush_pending()

    def _near_vblank_edge(self):
        """Use a cheap guard to keep the asynchronous VBlank NMI timely.

        Mapper-0 batches more CPU work between PPU updates for speed. Only the
        short window at the VBlank boundary needs instruction-granular syncing;
        most of the frame exits from this function after two integer checks.
        PPU register accesses still flush exactly in cpu_read/cpu_write.
        """
        if self.cart.mapper == 4 or self._pending_cpu_cycles <= 0:
            return False
        ppu = self.ppu
        if ppu.scanline == 240:
            # VBlank starts on the next scanline. Enter the urgent window roughly
            # one normal batch before the edge. The pending CPU cycles are PPU
            # dots which have not yet been applied to ppu.cycle.
            # Cap asynchronous NMI phase error to a much smaller window than
            # the normal batch (16 CPU cycles), even when Mapper-0 batches 128.
            urgent_dots = 16 * 3
            return ppu.cycle + self._pending_cpu_cycles * 3 >= 341 - urgent_dots
        if ppu.scanline == 241 and ppu.cycle == 0:
            # We are at the first dot of the VBlank line; do not let the next
            # normal-sized batch run dozens of CPU instructions before latching NMI.
            return True
        return False

    def _do_dma(self):
        # OAM DMA blocks the CPU for roughly 513/514 cycles. The actual byte
        # transfer is performed immediately, then PPU/APU time is advanced in
        # bulk. This keeps gameplay timing close while avoiding 256 Python bus
        # cycles for every DMA.
        # A deferred NROM frame may still contain queued pixels using the old OAM.
        self.ppu.flush_render_queue()
        page = self.dma_page << 8
        base = self.ppu.oamaddr
        oam = self.ppu.oam
        for i in range(256):
            oam[(base + i) & 0xFF] = self.cpu_read(page | i)
        # OAM DMA changes the sprite table outside PPU register writes. Invalidate
        # the precomputed scanline lists or Mario's sprites may stay stale.
        self.ppu.invalidate_sprite_cache()
        self.dma_pending = False
        self.dma_addr = 0
        self._advance_cpu_cycles(513)

    def begin_frame(self):
        """Start the next frame and consume dots carried across the boundary."""
        self.ppu.frame_complete = False
        carry = self.ppu._carry_ticks
        self.ppu._carry_ticks = 0
        if carry:
            self.ppu.advance(carry)

    def _try_fast_nrom_idle_jmp(self, max_instructions):
        """Batch a verified `JMP $xxxx` whose target is its own opcode address.

        Super Mario Bros spends much of a frame in a one-instruction self-jump
        while the next NMI is pending. Re-entering the Python CPU dispatcher for
        every 3-cycle JMP wastes time. Since the loop touches no memory/registers,
        batch only up to the existing CPU/PPU sync quantum; the next iteration
        still handles NMI normally. Do not use it on VBlank scanlines.
        """
        if (not self.fast_nrom_idle_jmp_enabled or self.cart.mapper != 0 or
                max_instructions <= 0 or self.cpu.cycles or self.cpu.jammed or
                self.nmi_pending or self.ppu.scanline >= 240 or self.cpu.pc < 0x8000):
            return 0
        cpu = self.cpu
        pc = cpu.pc
        off = pc - 0x8000
        prg = cpu._prg
        if cpu._prg_mask_16k:
            i0 = off & 0x3FFF
            i1 = (off + 1) & 0x3FFF
            i2 = (off + 2) & 0x3FFF
        else:
            i0 = off & 0x7FFF if cpu._prg_len == 0x8000 else off % cpu._prg_len
            i1 = (i0 + 1) % cpu._prg_len
            i2 = (i0 + 2) % cpu._prg_len
        if prg[i0] != 0x4C:
            return 0
        target = prg[i1] | (prg[i2] << 8)
        if target != pc:
            return 0

        pending = self._pending_cpu_cycles
        quantum = self._sync_quantum
        # The regular interpreter runs whole JMP instructions until the exact
        # same pending-cycle threshold is reached or exceeded.
        to_sync = max(1, (quantum - pending + 2) // 3)
        count = min(int(max_instructions), to_sync)
        added = count * 3
        self.total_cpu_cycles += added
        self._pending_cpu_cycles += added
        if self._pending_cpu_cycles >= quantum:
            self._flush_pending()
        return count

    def run_frame_slice(self, max_instructions=1800):
        """Run a bounded slice of the current frame.

        The GUI calls this repeatedly instead of blocking until the whole frame
        finishes. This keeps keyboard input, window repainting and menus
        responsive even when a ROM is expensive or a mapper is imperfect.
        """
        instructions = 0
        cpu = self.cpu
        ppu = self.ppu
        cart = self.cart
        quantum = self._sync_quantum

        while not ppu.frame_complete and instructions < max_instructions:
            if self.dma_pending:
                self._do_dma()
                instructions += 1
                continue

            # Super-instruction for the exact NROM PPUSTATUS polling loop used
            # by classic games: this method advances all CPU/PPU timing itself.
            # It is tightly pattern-checked and falls back to the regular CPU
            # dispatcher for every other sequence.
            if cart.mapper == 0 and cpu.pc >= 0x8000:
                pc_off = cpu.pc - 0x8000
                prg = cpu._prg
                if cpu._prg_mask_16k:
                    probe_off = pc_off & 0x3FFF
                elif cpu._prg_len == 0x8000:
                    probe_off = pc_off & 0x7FFF
                else:
                    probe_off = pc_off % cpu._prg_len
                probe = prg[probe_off]
                remaining = max_instructions - instructions
                if probe == 0x4C:  # verified JMP-to-self idle loop
                    fast_used = self._try_fast_nrom_idle_jmp(remaining)
                    if fast_used:
                        instructions += fast_used
                        continue
                elif probe == 0x2C:  # BIT abs, potential $2002 polling loop
                    pair_budget = min(64, remaining // 2)
                    fast_used = cpu.try_fast_ppustatus_poll(pair_budget)
                    if fast_used:
                        instructions += fast_used
                        continue

            if cart.mapper == 4 and cart.mapper_impl.irq_pending and not cpu.get(cpu.FLAG_I):
                cart.clear_irq()
                cpu.irq()
                cycles = cpu.cycles
                cpu.cycles = 0
            else:
                cycles = cpu.step_instruction()

            self.total_cpu_cycles += cycles
            self._pending_cpu_cycles += cycles
            # The edge check only matters on VBlank scanlines. Calling a Python method
            # for every ordinary instruction was measurable overhead in long game loops.
            # PPU register accesses still flush synchronously, and the VBlank guard
            # remains active while the PPU is on scanlines 240/241.
            if self._pending_cpu_cycles >= quantum or (ppu.scanline >= 240 and self._near_vblank_edge()):
                self._flush_pending()
            instructions += 1

        if ppu.frame_complete:
            self._flush_pending()
            if self.apu.player.enabled and self._apu_accum:
                self.apu.step_cpu(self._apu_accum)
                self._apu_accum = 0
        return ppu.frame_complete

    def run_until_frame(self, max_instructions=50000):
        """Synchronous compatibility helper for tests/benchmarks."""
        self.begin_frame()
        remaining = int(max_instructions)
        while not self.ppu.frame_complete and remaining > 0:
            used = min(1800, remaining)
            self.run_frame_slice(used)
            remaining -= used
        if not self.ppu.frame_complete:
            self._flush_pending()
        return self.ppu.frame_complete

    def clock(self):
        """Compatibility single-CPU-cycle clock used by old diagnostics."""
        for _ in range(3):
            self.ppu.clock()
            if self.ppu.consume_nmi():
                self.nmi_pending = True
        if self.dma_pending:
            value = self.cpu_read((self.dma_page << 8) | self.dma_addr)
            self.ppu.oam[(self.ppu.oamaddr + self.dma_addr) & 0xFF] = value
            self.dma_addr += 1
            if self.dma_addr >= 256:
                self.dma_pending = False
                self.ppu.invalidate_sprite_cache()
        else:
            if (self.cart.mapper == 4 and self.cart.mapper_impl.irq_pending
                    and not self.cpu.get(self.cpu.FLAG_I)):
                self.cart.clear_irq()
                self.cpu.irq()
            self.cpu.clock()
        self.apu.step_cpu(1) if self.apu.player.enabled else None
        self.total_cpu_cycles += 1

    def frame_done(self):
        return self.ppu.frame_complete

    def take_frame(self):
        self.ppu.frame_complete = False
        return self.ppu.frame

