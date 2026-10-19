"""iNES cartridge loader and common NES mappers.

Implemented mappers: 0 (NROM), 1 (MMC1), 2 (UxROM), 3 (CNROM), 4 (MMC3).
The code intentionally stays dependency-free and uses plain Python bytearrays.
"""

from __future__ import annotations

import os


class Cartridge:
    """Load iNES 1.0 and NES 2.0 cartridge images.

    NES 2.0 keeps the same 16-byte magic/header layout, but uses bytes 8-15
    for mapper/submapper and extended ROM/RAM sizes.  We parse those fields
    here and keep the mapper layer independent from the header format.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 16 or data[:4] != b"NES\x1a":
            raise ValueError("فایل NES معتبر نیست؛ هدر iNES/NES 2.0 پیدا نشد.")

        self.header = data[:16]
        self.prg_banks = data[4]
        self.chr_banks = data[5]
        self.flags6 = data[6]
        self.flags7 = data[7]
        self.ines2 = (self.flags7 & 0x0C) == 0x08
        self.format_name = "NES 2.0" if self.ines2 else "iNES 1.0"

        # iNES mapper number: flags 6/7 high nibbles.
        # NES 2.0 extends it with the low nibble of byte 8 and adds submapper.
        self.submapper = 0
        if self.ines2:
            self.mapper = ((self.flags6 >> 4)
                           | (self.flags7 & 0xF0)
                           | ((data[8] & 0x0F) << 8))
            self.submapper = (data[8] >> 4) & 0x0F
        else:
            self.mapper = (self.flags6 >> 4) | (self.flags7 & 0xF0)

        self.mirroring = "vertical" if (self.flags6 & 1) else "horizontal"
        if self.flags6 & 8:
            self.mirroring = "four"
        self.battery = bool(self.flags6 & 2)
        self.trainer = bool(self.flags6 & 4)

        offset = 16 + (512 if self.trainer else 0)
        self.prg_size = self._rom_size(data[4], (data[9] & 0x0F) if self.ines2 else 0, 0x4000)
        self.chr_size = self._rom_size(data[5], (data[9] >> 4) if self.ines2 else 0, 0x2000)

        if self.prg_size <= 0:
            raise ValueError("ROM هیچ PRG-ROM ندارد یا هدر آن خراب است.")
        if len(data) < offset + self.prg_size + self.chr_size:
            raise ValueError(
                f"فایل NES ناقص است. اندازه اعلام‌شده: {self.prg_size + self.chr_size:,} بایت "
                f"(به‌علاوه هدر/تریلر)، اندازه فایل: {len(data):,} بایت."
            )

        self.prg_banks = max(1, self.prg_size // 0x4000)
        self.chr_banks = self.chr_size // 0x2000
        self.prg = bytearray(data[offset:offset + self.prg_size])
        offset += self.prg_size

        self.chr_is_ram = self.chr_size == 0
        self.chr_ram_size = self._ram_size(data[11] & 0x0F) if self.ines2 else 0
        self.chr_battery_ram_size = self._ram_size(data[11] >> 4) if self.ines2 else 0
        if self.chr_is_ram:
            # NES 2.0: if no CHR-ROM and no explicit CHR-RAM size is declared,
            # the default is 8 KiB CHR-RAM on ordinary cartridges.
            if self.chr_ram_size == 0 and self.chr_battery_ram_size == 0:
                self.chr_ram_size = 8192
            self.chr = bytearray(max(8192, self.chr_ram_size, self.chr_battery_ram_size))
        else:
            self.chr = bytearray(data[offset:offset + self.chr_size])
            offset += self.chr_size

        # PRG-RAM sizing.
        if self.ines2:
            work_ram = self._ram_size(data[10] & 0x0F)
            battery_ram = self._ram_size(data[10] >> 4)
            self.prg_ram_size = max(0, work_ram + battery_ram)
            # A battery flag with an unspecified size is conventionally 8 KiB.
            if self.battery and self.prg_ram_size == 0:
                self.prg_ram_size = 8192
        else:
            # iNES 1.0 byte 8: zero traditionally means 8 KiB on most emulators.
            self.prg_ram_size = (data[8] or 1) * 8192

        self.sram = bytearray(max(8192, self.prg_ram_size))
        self.save_path = os.path.splitext(self.path)[0] + ".sav"
        self.load_save()

        mapper_cls = {
            0: Mapper0,
            1: Mapper1,
            2: Mapper2,
            3: Mapper3,
            4: Mapper4,
            7: Mapper7,
            11: Mapper11,
            13: Mapper13,
            34: Mapper34,
            66: Mapper66,
            71: Mapper71,
        }.get(self.mapper)
        if mapper_cls is None:
            raise NotImplementedError(
                f"این ROM با {self.format_name} و Mapper {self.mapper} / Submapper {self.submapper} است. "
                "در نسخه فعلی Mapperهای 0، 1، 2، 3، 4، 7، 11، 13، 34، 66 و 71 پیاده‌سازی شده‌اند."
            )
        self.mapper_impl = mapper_cls(self)

    @staticmethod
    def _rom_size(low_byte: int, high_nibble: int, unit: int) -> int:
        """Decode an iNES or NES 2.0 ROM size field.

        NES 2.0 normally stores the size in units (16 KiB PRG / 8 KiB CHR).
        When the high nibble is 0xF it switches to exponent-multiplier form:
        size = 2^E * (2*M+1).
        """
        if high_nibble != 0x0F:
            return ((high_nibble << 8) | low_byte) * unit
        exponent = (low_byte >> 2) & 0x3F
        multiplier = (low_byte & 0x03) * 2 + 1
        return (1 << exponent) * multiplier

    @staticmethod
    def _ram_size(nibble: int) -> int:
        # NES 2.0 RAM sizes are 64 << n bytes, with 0 meaning absent.
        return 0 if nibble == 0 else (64 << nibble)

    @property
    def mapper_name(self) -> str:
        return self.mapper_impl.name

    @property
    def description(self) -> str:
        return f"{self.format_name} | Mapper {self.mapper} | Submapper {self.submapper} | {self.mapper_name}"

    def load_save(self):
        if not self.battery or not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "rb") as f:
                data = f.read()
            self.sram[:len(data[:len(self.sram)])] = data[:len(self.sram)]
        except OSError:
            pass

    def save(self):
        if not self.battery:
            return
        try:
            with open(self.save_path, "wb") as f:
                f.write(self.sram)
        except OSError:
            pass

    def cpu_read(self, addr: int):
        return self.mapper_impl.cpu_read(addr & 0xFFFF)

    def cpu_write(self, addr: int, value: int) -> bool:
        return self.mapper_impl.cpu_write(addr & 0xFFFF, value & 0xFF)

    def ppu_read(self, addr: int):
        return self.mapper_impl.ppu_read(addr & 0x1FFF)

    def ppu_write(self, addr: int, value: int) -> bool:
        return self.mapper_impl.ppu_write(addr & 0x1FFF, value & 0xFF)

    def scanline_tick(self):
        self.mapper_impl.scanline_tick()

    def consume_irq(self) -> bool:
        return self.mapper_impl.consume_irq()

    def clear_irq(self):
        self.mapper_impl.clear_irq()

    def mirror_nt(self, index: int) -> int:
        """Map logical nametable 0..3 to physical 1 KiB page 0..3."""
        mode = self.mapper_impl.mirroring
        if mode == "four":
            return index & 3
        if mode == "vertical":
            return index & 1
        if mode == "horizontal":
            return 0 if index < 2 else 1
        if mode == "single0":
            return 0
        if mode == "single1":
            return 1
        return index & 1


class BaseMapper:
    name = "Unknown"

    def __init__(self, cart: Cartridge):
        self.cart = cart
        self.mirroring = cart.mirroring
        self.irq_pending = False

    @property
    def prg(self):
        return self.cart.prg

    @property
    def chr(self):
        return self.cart.chr

    def cpu_read(self, addr):
        return 0

    def cpu_write(self, addr, value):
        return False

    def ppu_read(self, addr):
        return self.chr[addr % len(self.chr)]

    def ppu_write(self, addr, value):
        if self.cart.chr_is_ram:
            self.chr[addr % len(self.chr)] = value
            return True
        return False

    def scanline_tick(self):
        pass

    def consume_irq(self):
        if self.irq_pending:
            self.irq_pending = False
            return True
        return False

    def clear_irq(self):
        self.irq_pending = False

    def _sram_read(self, addr):
        if 0x6000 <= addr <= 0x7FFF:
            return self.cart.sram[(addr - 0x6000) % len(self.cart.sram)]
        return None

    def _sram_write(self, addr, value):
        if 0x6000 <= addr <= 0x7FFF:
            self.cart.sram[(addr - 0x6000) % len(self.cart.sram)] = value & 0xFF
            return True
        return False


class Mapper0(BaseMapper):
    name = "Mapper 0 / NROM"

    def cpu_read(self, addr):
        s = self._sram_read(addr)
        if s is not None:
            return s
        if addr >= 0x8000:
            if len(self.prg) == 0x4000:
                return self.prg[(addr - 0x8000) & 0x3FFF]
            return self.prg[(addr - 0x8000) % len(self.prg)]
        return None

    def cpu_write(self, addr, value):
        return self._sram_write(addr, value) or (addr >= 0x8000)


class Mapper1(BaseMapper):
    name = "Mapper 1 / MMC1"

    def __init__(self, cart):
        super().__init__(cart)
        self.shift = 0x10
        self.control = 0x0C
        self.chr_bank0 = 0
        self.chr_bank1 = 0
        self.prg_bank = 0
        self.mirroring = "horizontal"
        self._update_mirroring()

    def _update_mirroring(self):
        m = self.control & 3
        self.mirroring = {0: "single0", 1: "single1", 2: "vertical", 3: "horizontal"}[m]

    def _commit(self, addr, value):
        target = (addr >> 13) & 3
        if target == 0:
            self.control = value & 0x1F
            self._update_mirroring()
        elif target == 1:
            self.chr_bank0 = value & 0x1F
        elif target == 2:
            self.chr_bank1 = value & 0x1F
        else:
            self.prg_bank = value & 0x0F

    def cpu_read(self, addr):
        s = self._sram_read(addr)
        if s is not None:
            return s
        if addr < 0x8000:
            return None
        mode = (self.control >> 2) & 3
        banks = max(1, len(self.prg) // 0x4000)
        if mode in (0, 1):
            bank32 = (self.prg_bank & 0x0E) % banks
            idx = (bank32 * 0x4000) + (addr - 0x8000)
            return self.prg[idx % len(self.prg)]
        if mode == 2:
            if addr < 0xC000:
                return self.prg[(addr - 0x8000) % len(self.prg)]
            bank = self.prg_bank % banks
            return self.prg[(bank * 0x4000) + (addr - 0xC000)]
        bank = self.prg_bank % banks
        if addr < 0xC000:
            return self.prg[(bank * 0x4000) + (addr - 0x8000)]
        return self.prg[((banks - 1) * 0x4000) + (addr - 0xC000)]

    def cpu_write(self, addr, value):
        if self._sram_write(addr, value):
            return True
        if addr < 0x8000:
            return False
        if value & 0x80:
            self.shift = 0x10
            self.control |= 0x0C
            self._update_mirroring()
            return True
        complete = self.shift & 1
        self.shift >>= 1
        self.shift |= (value & 1) << 4
        if complete:
            self._commit(addr, self.shift & 0x1F)
            self.shift = 0x10
        return True

    def ppu_read(self, addr):
        if not self.cart.chr_banks and self.cart.chr_is_ram:
            return self.chr[addr]
        if (self.control & 0x10) == 0:
            bank = (self.chr_bank0 & 0x1E) * 0x1000
            return self.chr[(bank + addr) % len(self.chr)]
        if addr < 0x1000:
            bank = self.chr_bank0 * 0x1000
            return self.chr[(bank + addr) % len(self.chr)]
        bank = self.chr_bank1 * 0x1000
        return self.chr[(bank + (addr - 0x1000)) % len(self.chr)]

    def ppu_write(self, addr, value):
        if self.cart.chr_is_ram:
            self.chr[addr] = value & 0xFF
            return True
        return False


class Mapper2(BaseMapper):
    name = "Mapper 2 / UxROM"

    def __init__(self, cart):
        super().__init__(cart)
        self.bank = 0

    def cpu_read(self, addr):
        s = self._sram_read(addr)
        if s is not None:
            return s
        if 0x8000 <= addr < 0xC000:
            banks = len(self.prg) // 0x4000
            return self.prg[(self.bank % max(1, banks - 1)) * 0x4000 + (addr - 0x8000)]
        if addr >= 0xC000:
            bank = max(0, len(self.prg) // 0x4000 - 1)
            return self.prg[bank * 0x4000 + (addr - 0xC000)]
        return None

    def cpu_write(self, addr, value):
        if self._sram_write(addr, value):
            return True
        if addr >= 0x8000:
            self.bank = value & 0xFF
            return True
        return False


class Mapper3(BaseMapper):
    name = "Mapper 3 / CNROM"

    def __init__(self, cart):
        super().__init__(cart)
        self.chr_bank = 0

    def cpu_read(self, addr):
        s = self._sram_read(addr)
        if s is not None:
            return s
        if addr >= 0x8000:
            if len(self.prg) == 0x4000:
                return self.prg[(addr - 0x8000) & 0x3FFF]
            return self.prg[(addr - 0x8000) % len(self.prg)]
        return None

    def cpu_write(self, addr, value):
        if self._sram_write(addr, value):
            return True
        if addr >= 0x8000:
            self.chr_bank = value & 0xFF
            return True
        return False

    def ppu_read(self, addr):
        bank = self.chr_bank * 0x2000
        return self.chr[(bank + addr) % len(self.chr)]

    def ppu_write(self, addr, value):
        if self.cart.chr_is_ram:
            bank = self.chr_bank * 0x2000
            self.chr[(bank + addr) % len(self.chr)] = value & 0xFF
            return True
        return False


class Mapper7(BaseMapper):
    name = "Mapper 7 / AOROM"
    def __init__(self, cart):
        super().__init__(cart)
        self.bank = 0
        self.mirroring = "single0"
    def cpu_read(self, addr):
        s = self._sram_read(addr)
        if s is not None: return s
        if addr < 0x8000: return None
        banks = max(1, len(self.prg) // 0x8000)
        return self.prg[(self.bank % banks) * 0x8000 + (addr - 0x8000)]
    def cpu_write(self, addr, value):
        if self._sram_write(addr, value): return True
        if addr >= 0x8000:
            self.bank = value & 0x0F
            self.mirroring = "single1" if (value & 0x10) else "single0"
            return True
        return False


class Mapper11(BaseMapper):
    name = "Mapper 11 / Color Dreams"
    def __init__(self, cart):
        super().__init__(cart); self.bank = 0; self.chr_bank = 0
    def cpu_read(self, addr):
        s = self._sram_read(addr)
        if s is not None: return s
        if addr < 0x8000: return None
        banks = max(1, len(self.prg) // 0x8000)
        return self.prg[(self.bank % banks) * 0x8000 + (addr - 0x8000)]
    def cpu_write(self, addr, value):
        if self._sram_write(addr, value): return True
        if addr >= 0x8000:
            self.bank = value & 0x0F; self.chr_bank = (value >> 4) & 0x0F; return True
        return False
    def ppu_read(self, addr):
        return self.chr[((self.chr_bank * 0x2000) + addr) % len(self.chr)]


class Mapper13(BaseMapper):
    name = "Mapper 13 / CPROM"
    def __init__(self, cart):
        super().__init__(cart); self.chr_bank = 0
        if len(self.cart.chr) < 0x4000:
            self.cart.chr = bytearray(0x4000)
    def cpu_read(self, addr):
        s = self._sram_read(addr)
        if s is not None: return s
        if addr < 0x8000: return None
        bank = max(0, (len(self.prg)//0x8000) - 1)
        return self.prg[(bank * 0x8000) + (addr - 0x8000) % len(self.prg)]
    def cpu_write(self, addr, value):
        if self._sram_write(addr, value): return True
        if addr >= 0x8000: self.chr_bank = value & 3; return True
        return False
    def ppu_read(self, addr):
        if addr < 0x1000: return self.chr[addr]
        return self.chr[self.chr_bank * 0x1000 + (addr - 0x1000)]
    def ppu_write(self, addr, value):
        self.chr[addr] = value & 0xFF; return True


class Mapper34(BaseMapper):
    name = "Mapper 34 / BNROM"
    def __init__(self, cart):
        super().__init__(cart); self.bank = 0
    def cpu_read(self, addr):
        s=self._sram_read(addr)
        if s is not None: return s
        if addr < 0x8000: return None
        banks=max(1,len(self.prg)//0x8000)
        return self.prg[(self.bank%banks)*0x8000 + (addr-0x8000)]
    def cpu_write(self, addr, value):
        if self._sram_write(addr,value): return True
        if addr>=0x8000: self.bank=value & 0x1F; return True
        return False


class Mapper66(BaseMapper):
    name = "Mapper 66 / GxROM"
    def __init__(self, cart):
        super().__init__(cart); self.prg_bank=0; self.chr_bank=0
    def cpu_read(self, addr):
        s=self._sram_read(addr)
        if s is not None: return s
        if addr<0x8000: return None
        banks=max(1,len(self.prg)//0x8000)
        return self.prg[(self.prg_bank%banks)*0x8000 + (addr-0x8000)]
    def cpu_write(self, addr, value):
        if self._sram_write(addr,value): return True
        if addr>=0x8000:
            self.prg_bank=(value>>4)&3; self.chr_bank=value&3; return True
        return False
    def ppu_read(self,addr):
        return self.chr[(self.chr_bank*0x2000+addr)%len(self.chr)]


class Mapper71(BaseMapper):
    name = "Mapper 71 / Camerica"
    def __init__(self, cart):
        super().__init__(cart); self.bank=0
    def cpu_read(self,addr):
        s=self._sram_read(addr)
        if s is not None: return s
        if 0x8000<=addr<0xC000:
            banks=max(1,len(self.prg)//0x4000)
            return self.prg[(self.bank % max(1,banks-1))*0x4000 + (addr-0x8000)]
        if addr>=0xC000:
            bank=max(0,len(self.prg)//0x4000-1)
            return self.prg[bank*0x4000 + (addr-0xC000)]
        return None
    def cpu_write(self,addr,value):
        if self._sram_write(addr,value): return True
        if 0x9000<=addr<0xA000:
            self.mirroring='single1' if value&0x10 else 'single0'
        if addr>=0xC000 or 0x8000<=addr<0x9000:
            self.bank=value&0x0F; return True
        return False


class Mapper4(BaseMapper):
    name = "Mapper 4 / MMC3"

    def __init__(self, cart):
        super().__init__(cart)
        self.bank_select = 0
        self.regs = [0, 2, 4, 5, 6, 7, 0, 1]
        self.prg_bank_mode = 0
        self.chr_invert = 0
        self.mirroring = cart.mirroring if cart.mirroring != "four" else "vertical"
        self.prg_ram_enable = True
        self.prg_ram_write_protect = False
        self.irq_latch = 0
        self.irq_counter = 0
        self.irq_reload = False
        self.irq_enabled = False

    def _bank8(self, bank):
        count = max(1, len(self.prg) // 0x2000)
        return (bank % count) * 0x2000

    def cpu_read(self, addr):
        if 0x6000 <= addr <= 0x7FFF:
            if self.prg_ram_enable:
                return self.cart.sram[(addr - 0x6000) % len(self.cart.sram)]
            return 0
        if addr < 0x8000:
            return None
        last = max(0, len(self.prg) // 0x2000 - 1)
        second_last = max(0, last - 1)
        a = addr - 0x8000
        slot = a // 0x2000
        if self.prg_bank_mode == 0:
            bank = [self.regs[6], self.regs[7], second_last, last][slot]
        else:
            bank = [second_last, self.regs[7], self.regs[6], last][slot]
        return self.prg[self._bank8(bank) + (a & 0x1FFF)]

    def cpu_write(self, addr, value):
        value &= 0xFF
        if 0x6000 <= addr <= 0x7FFF:
            if self.prg_ram_enable and not self.prg_ram_write_protect:
                self.cart.sram[(addr - 0x6000) % len(self.cart.sram)] = value
                return True
            return True
        if addr < 0x8000:
            return False
        if 0x8000 <= addr <= 0x9FFF:
            if (addr & 1) == 0:
                self.bank_select = value & 7
                self.prg_bank_mode = (value >> 6) & 1
                self.chr_invert = (value >> 7) & 1
            else:
                self.regs[self.bank_select] = value
        elif 0xA000 <= addr <= 0xBFFF:
            if (addr & 1) == 0 and not (self.cart.flags6 & 8):
                self.mirroring = "horizontal" if (value & 1) else "vertical"
            elif (addr & 1) == 1:
                self.prg_ram_enable = not bool(value & 0x80)
                self.prg_ram_write_protect = bool(value & 0x40)
        elif 0xC000 <= addr <= 0xDFFF:
            if (addr & 1) == 0:
                self.irq_latch = value
            else:
                self.irq_reload = True
        elif addr >= 0xE000:
            if (addr & 1) == 0:
                self.irq_enabled = False
                self.irq_pending = False
            else:
                self.irq_enabled = True
        return True

    def _chr1(self, bank, addr):
        return self.chr[(bank * 0x400 + addr) % len(self.chr)]

    def ppu_read(self, addr):
        r = self.regs
        if self.chr_invert == 0:
            if addr < 0x800:
                bank = r[0] & 0xFE
                return self._chr1(bank + addr // 0x400, addr & 0x3FF)
            if addr < 0x1000:
                bank = r[1] & 0xFE
                return self._chr1(bank + (addr - 0x800) // 0x400, addr & 0x3FF)
            banks = [r[2], r[3], r[4], r[5]]
            return self._chr1(banks[(addr - 0x1000) // 0x400], addr & 0x3FF)
        if addr < 0x1000:
            banks = [r[2], r[3], r[4], r[5]]
            return self._chr1(banks[addr // 0x400], addr & 0x3FF)
        if addr < 0x1800:
            bank = r[0] & 0xFE
            return self._chr1(bank + (addr - 0x1000) // 0x400, addr & 0x3FF)
        bank = r[1] & 0xFE
        return self._chr1(bank * 2 + (addr - 0x1800) // 0x400, addr & 0x3FF)

    def ppu_write(self, addr, value):
        if self.cart.chr_is_ram:
            # CNROM/MMC3 CHR-RAM is uncommon; still expose direct writes to the currently mapped byte.
            if len(self.chr) >= 0x2000:
                # Fall back to the physical address without bank inversion for safety.
                self.chr[addr & (len(self.chr) - 1)] = value & 0xFF
            return True
        return False

    def scanline_tick(self):
        # Approximation of the MMC3 scanline IRQ counter. The simplified PPU calls
        # this once per visible scanline, which is enough for many games that use
        # status-bar IRQ splits even though it is not a transistor-accurate A12 model.
        if self.irq_counter == 0 or self.irq_reload:
            self.irq_counter = self.irq_latch
            self.irq_reload = False
        else:
            self.irq_counter = (self.irq_counter - 1) & 0xFF
        if self.irq_counter == 0 and self.irq_enabled:
            self.irq_pending = True
