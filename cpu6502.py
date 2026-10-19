class CPU6502:
    # NMOS 6502 JAM/KIL instructions. They halt the CPU core while PPU time continues.
    KIL_OPCODES = frozenset((0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x92, 0xB2, 0xD2, 0xF2))
    # A single membership check avoids paying the generic fast-path decode cost
    # on every unrelated instruction.
    NROM_EXTENDED_OPS = frozenset((
        0xA1,0xA4,0xA5,0xA6,0xA9,0xAD,0xB1,0xB5,0xB9,0xBD,
        0x81,0x85,0x8D,0x91,0x95,0x99,0x9D,
        0x01,0x05,0x09,0x0D,0x11,0x15,0x19,0x1D,
        0x21,0x25,0x29,0x2D,0x31,0x35,0x39,0x3D,
        0x41,0x45,0x49,0x4D,0x51,0x55,0x59,0x5D,
        0x61,0x65,0x69,0x6D,0x71,0x75,0x79,0x7D,
        0xC1,0xC5,0xC9,0xCD,0xD1,0xD5,0xD9,0xDD,
        0xE1,0xE5,0xE9,0xED,0xF1,0xF5,0xF9,0xFD,
    ))

    FLAG_C = 0x01; FLAG_Z = 0x02; FLAG_I = 0x04; FLAG_D = 0x08
    FLAG_B = 0x10; FLAG_U = 0x20; FLAG_V = 0x40; FLAG_N = 0x80

    def __init__(self, bus):
        self.bus = bus
        self.a = self.x = self.y = 0
        self.stkp = 0xFD
        self.status = self.FLAG_U | self.FLAG_I
        self.pc = 0
        self.cycles = 0
        self.jammed = False
        self._table = None
        self._extra = None
        self._build_hot_tables()

    def _build_hot_tables(self):
        d = self._build_dispatch()
        self._table = [None] * 256
        for op, entry in d.items():
            self._table[op] = entry
        d = self._build_extra()
        self._extra = [None] * 256
        for op, entry in d.items():
            self._extra[op] = entry
        self._ram = self.bus.cpu_ram
        self._ppu = self.bus.ppu
        self._apu = self.bus.apu
        self._c1 = self.bus.controller1
        self._c2 = self.bus.controller2
        self._cart = self.bus.cart
        self._mapper = self.bus.cart.mapper_impl
        self._mapper_id = self.bus.cart.mapper
        self._sram = self.bus.cart.sram
        self._prg = self.bus.cart.prg
        self._prg_len = len(self._prg)
        self._prg_mask_16k = self._prg_len == 0x4000
        self.fast_ppustatus_poll_enabled = True
        self.fast_extended_ops_enabled = True

    def try_fast_ppustatus_poll(self, max_pairs=64):
        """Run a verified NROM `BIT $2002 / BPL|BMI|BVC|BVS back` polling loop.

        This is a 6502 super-instruction, not a timing shortcut: each BIT and
        branch still contributes its exact CPU cycles through the normal bus
        synchronizer. It only avoids re-entering the full opcode dispatcher for
        the known five-byte loop. Return the number of emulated instructions
        consumed (0 means the current PC is not this exact pattern).
        """
        if (not self.fast_ppustatus_poll_enabled or max_pairs <= 0 or
                self.cycles or self.jammed or self.bus.nmi_pending or
                self._mapper_id != 0 or self.pc < 0x8000):
            return 0

        bus = self.bus
        ppu = bus.ppu
        prg = self._prg
        prg_len = self._prg_len
        mirror_16k = self._prg_mask_16k
        bit_pc = self.pc

        # Verify opcode/operand and that the supported conditional branch
        # returns to this exact BIT, including both branch polarities.
        off = bit_pc - 0x8000
        if mirror_16k:
            i0 = off & 0x3FFF
            i1 = (off + 1) & 0x3FFF
            i2 = (off + 2) & 0x3FFF
            i3 = (off + 3) & 0x3FFF
            i4 = (off + 4) & 0x3FFF
        else:
            i0 = off % prg_len
            i1 = (off + 1) % prg_len
            i2 = (off + 2) % prg_len
            i3 = (off + 3) % prg_len
            i4 = (off + 4) % prg_len
        branch_op = prg[i3]
        if (prg[i0] != 0x2C or prg[i1] != 0x02 or prg[i2] != 0x20 or
                branch_op not in (0x10, 0x30, 0x50, 0x70)):
            return 0
        # BPL/BMI use N (PPUSTATUS bit 7), BVC/BVS use V (bit 6).
        # Both polarities are supported; branch-exit polarity is tracked below.
        branch_flag = self.FLAG_N if branch_op in (0x10, 0x30) else self.FLAG_V
        # BPL/BVC leave their polling loops when the relevant bit is set;
        # BMI/BVS leave when it is clear.
        branch_exits_when_set = branch_op in (0x10, 0x50)
        relative = prg[i4] - 256 if prg[i4] & 0x80 else prg[i4]
        loop_target = ((bit_pc + 5 + relative) & 0xFFFF)
        if loop_target != bit_pc:
            return 0

        consumed = 0
        # Bound this super-instruction so a long wait cannot starve Tk input.
        for _ in range(min(int(max_pairs), 128)):
            # BIT fetches both operand bytes before reading $2002. Match that
            # visible PC state in case a pending NMI arrives at this boundary.
            self.pc = (bit_pc + 3) & 0xFFFF
            value = bus.cpu_read_ppu_status_fast()
            self.status = ((self.status & ~(self.FLAG_Z | self.FLAG_N | self.FLAG_V)) |
                           (self.FLAG_Z if (self.a & value) == 0 else 0) |
                           (value & 0xC0))
            bus._advance_cpu_cycles(4)
            consumed += 1
            if ppu.frame_complete or bus.nmi_pending:
                return consumed

            # Exit when the branch condition would be false.
            if bool(self.status & branch_flag) == branch_exits_when_set:
                self.pc = (bit_pc + 5) & 0xFFFF
                bus._advance_cpu_cycles(2)
                consumed += 1
                return consumed

            self.pc = bit_pc
            bus._advance_cpu_cycles(3)  # taken branch, same page
            consumed += 1
            if ppu.frame_complete or bus.nmi_pending:
                return consumed
        return consumed

    def reset(self):
        self.a = self.x = self.y = 0
        self.stkp = 0xFD
        self.status = self.FLAG_U | self.FLAG_I
        self.jammed = False
        self.pc = self.read16(0xFFFC)
        self.cycles = 8

    def irq(self):
        if not self.get(self.FLAG_I):
            self.push((self.pc >> 8) & 0xFF)
            self.push(self.pc & 0xFF)
            self.push((self.status & ~self.FLAG_B) | self.FLAG_U)
            self.set(self.FLAG_I, True)
            self.pc = self.read16(0xFFFE)
            self.cycles = 7

    def nmi(self):
        self.push((self.pc >> 8) & 0xFF)
        self.push(self.pc & 0xFF)
        self.push((self.status & ~self.FLAG_B) | self.FLAG_U)
        self.set(self.FLAG_I, True)
        self.pc = self.read16(0xFFFA)
        self.cycles = 8

    def get(self, f): return 1 if (self.status & f) else 0
    def set(self, f, v): self.status = (self.status | f) if v else (self.status & ~f)
    def read(self, a):
        a &= 0xFFFF
        if a < 0x2000:
            return self._ram[a & 0x07FF]
        if a < 0x4000:
            reg = a & 7
            if reg == 2:
                return self.bus.cpu_read_ppu_status_fast()
            # Other PPU register reads also need to observe current emulated time.
            self.bus.sync_for_ppu_access()
            return self._ppu.cpu_read(reg)
        if 0x4000 <= a <= 0x4013:
            return 0
        if a == 0x4014:
            return 0
        if a == 0x4015:
            return self._apu.read_status()
        if a == 0x4016:
            return self._c1.read()
        if a == 0x4017:
            return self._c2.read()
        if 0x6000 <= a <= 0x7FFF:
            return self._sram[(a - 0x6000) % len(self._sram)]
        if a >= 0x8000:
            if self._mapper_id == 0:
                off = a - 0x8000
                if self._prg_mask_16k:
                    off &= 0x3FFF
                else:
                    off %= self._prg_len
                return self._prg[off]
            return self._mapper.cpu_read(a) or 0
        return 0

    def write(self, a, v):
        # Writes are much less frequent than reads; retain the complete bus path.
        self.bus.cpu_write(a & 0xFFFF, v & 0xFF)

    def fetch(self):
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        if pc >= 0x8000:
            off = pc - 0x8000
            if self._mapper_id == 0:
                if self._prg_mask_16k:
                    off &= 0x3FFF
                else:
                    off %= self._prg_len
                return self._prg[off]
            return self._mapper.cpu_read(pc) or 0
        return self.read(pc)
    def read16(self, a): return self.read(a) | (self.read((a + 1) & 0xFFFF) << 8)
    def push(self, v): self.write(0x0100 + self.stkp, v); self.stkp = (self.stkp - 1) & 0xFF
    def pull(self): self.stkp = (self.stkp + 1) & 0xFF; return self.read(0x0100 + self.stkp)

    # Addressing modes return (address, crossed_page)
    def imm(self): return self.fetch(), False
    def zp0(self): return self.fetch() & 0xFF, False
    def zpx(self): return (self.fetch() + self.x) & 0xFF, False
    def zpy(self): return (self.fetch() + self.y) & 0xFF, False
    def abs_(self): return self.read16_pc(), False
    def read16_pc(self):
        lo = self.fetch(); hi = self.fetch(); return lo | (hi << 8)
    def abx(self):
        base = self.read16_pc(); a = (base + self.x) & 0xFFFF; return a, (base & 0xFF00) != (a & 0xFF00)
    def aby(self):
        base = self.read16_pc(); a = (base + self.y) & 0xFFFF; return a, (base & 0xFF00) != (a & 0xFF00)
    def ind(self):
        ptr = self.read16_pc(); lo = self.read(ptr); hi = self.read((ptr & 0xFF00) | ((ptr + 1) & 0xFF)); return lo | hi << 8, False
    def izx(self):
        zp = (self.fetch() + self.x) & 0xFF; return self.read(zp) | (self.read((zp + 1) & 0xFF) << 8), False
    def izy(self):
        zp = self.fetch(); base = self.read(zp) | (self.read((zp + 1) & 0xFF) << 8); a = (base + self.y) & 0xFFFF; return a, (base & 0xFF00) != (a & 0xFF00)

    def lda(self, v): self.a = v & 0xFF; self.setzn(self.a)
    def ldx(self, v): self.x = v & 0xFF; self.setzn(self.x)
    def ldy(self, v): self.y = v & 0xFF; self.setzn(self.y)
    def setzn(self, v): self.set(self.FLAG_Z, (v & 0xFF) == 0); self.set(self.FLAG_N, v & 0x80)

    def adc(self, v):
        v &= 0xFF; a = self.a; carry = self.get(self.FLAG_C); r = a + v + carry
        self.set(self.FLAG_C, r > 0xFF)
        rr = r & 0xFF
        self.set(self.FLAG_V, (~(a ^ v) & (a ^ rr) & 0x80) != 0)
        self.a = rr; self.setzn(self.a)

    def sbc(self, v):
        self.adc(v ^ 0xFF)

    def cmp(self, reg, v):
        r = (reg - (v & 0xFF)) & 0x1FF
        self.set(self.FLAG_C, reg >= (v & 0xFF)); self.setzn(r & 0xFF)

    def branch(self, cond):
        off = self.fetch(); off = off - 256 if off & 0x80 else off
        if cond:
            old = self.pc; self.pc = (self.pc + off) & 0xFFFF; self.cycles += 1
            if (old & 0xFF00) != (self.pc & 0xFF00): self.cycles += 1

    def un_anc_imm(self):
        self.a &= self.fetch()
        self.setzn(self.a)
        self.set(self.FLAG_C, bool(self.a & 0x80))

    def un_alr_imm(self):
        self.a &= self.fetch()
        self.a = self.lsr(self.a)
        self.setzn(self.a)

    def un_arr_imm(self):
        value = self.fetch()
        masked = self.a & value
        carry_in = self.get(self.FLAG_C)
        result = ((masked >> 1) | (carry_in << 7)) & 0xFF
        self.a = result
        self.setzn(result)
        self.set(self.FLAG_C, bool(result & 0x40))
        self.set(self.FLAG_V, bool(((result >> 6) ^ (result >> 5)) & 1))

    def un_xaa_imm(self):
        # XAA is unstable on NMOS 6502; the common NES approximation uses $EE
        # as the value present on the internal data bus.
        self.a = ((self.a | 0xEE) & self.x & self.fetch()) & 0xFF
        self.setzn(self.a)

    def un_lax_imm(self):
        # Immediate LAX/LXA is also unstable; use the widely used NES $EE model.
        value = ((self.a | 0xEE) & self.fetch()) & 0xFF
        self.a = self.x = value
        self.setzn(value)

    def un_las_abs_y(self):
        addr, _ = self.aby()
        value = self.read(addr) & self.stkp
        self.a = self.x = self.stkp = value
        self.setzn(value)

    def un_ahx_ind_y(self):
        addr, _ = self.izy()
        value = self.a & self.x & (((addr >> 8) + 1) & 0xFF)
        self.write(addr, value)

    def un_ahx_abs_y(self):
        addr, _ = self.aby()
        value = self.a & self.x & (((addr >> 8) + 1) & 0xFF)
        self.write(addr, value)

    def un_tas_abs_y(self):
        addr, _ = self.aby()
        self.stkp = self.a & self.x
        self.write(addr, self.stkp & (((addr >> 8) + 1) & 0xFF))

    def un_shy_abs_x(self):
        addr, _ = self.abx()
        self.write(addr, self.y & (((addr >> 8) + 1) & 0xFF))

    def un_shx_abs_y(self):
        addr, _ = self.aby()
        self.write(addr, self.x & (((addr >> 8) + 1) & 0xFF))

    def _build_dispatch(self):
        return {
            0x00:('BRK',self.brk,7),0x01:('ORA',lambda:self.logic(self.izx,self.ora),6),0x05:('ORA',lambda:self.logic(self.zp0,self.ora),3),
            0x06:('ASL',lambda:self.rmw(self.zp0,self.asl),5),0x08:('PHP',self.php,3),0x09:('ORA',lambda:self.logic(self.imm,self.ora),2),
            0x0A:('ASL',lambda:self.acc(self.asl),2),0x0D:('ORA',lambda:self.logic(self.abs_,self.ora),4),0x0E:('ASL',lambda:self.rmw(self.abs_,self.asl),6),
            0x10:('BPL',lambda:self.branch(not self.get(self.FLAG_N)),2),0x11:('ORA',lambda:self.logic(self.izy,self.ora),5),0x15:('ORA',lambda:self.logic(self.zpx,self.ora),4),
            0x16:('ASL',lambda:self.rmw(self.zpx,self.asl),6),0x18:('CLC',lambda:self.set(self.FLAG_C,False),2),0x19:('ORA',lambda:self.logic(self.aby,self.ora),4),
            0x1D:('ORA',lambda:self.logic(self.abx,self.ora),4),0x1E:('ASL',lambda:self.rmw(self.abx,self.asl),7),0x20:('JSR',self.jsr,6),0x21:('AND',lambda:self.logic(self.izx,self.and_),6),
            0x24:('BIT',lambda:self.bit(self.zp0),3),0x25:('AND',lambda:self.logic(self.zp0,self.and_),3),0x26:('ROL',lambda:self.rmw(self.zp0,self.rol),5),0x28:('PLP',self.plp,4),
            0x29:('AND',lambda:self.logic(self.imm,self.and_),2),0x2A:('ROL',lambda:self.acc(self.rol),2),0x2C:('BIT',lambda:self.bit(self.abs_),4),0x2D:('AND',lambda:self.logic(self.abs_,self.and_),4),
            0x2E:('ROL',lambda:self.rmw(self.abs_,self.rol),6),0x30:('BMI',lambda:self.branch(self.get(self.FLAG_N)==1),2),0x31:('AND',lambda:self.logic(self.izy,self.and_),5),0x35:('AND',lambda:self.logic(self.zpx,self.and_),4),
            0x36:('ROL',lambda:self.rmw(self.zpx,self.rol),6),0x38:('SEC',lambda:self.set(self.FLAG_C,True),2),0x39:('AND',lambda:self.logic(self.aby,self.and_),4),0x3D:('AND',lambda:self.logic(self.abx,self.and_),4),
            0x3E:('ROL',lambda:self.rmw(self.abx,self.rol),7),0x40:('RTI',self.rti,6),0x41:('EOR',lambda:self.logic(self.izx,self.eor),6),0x45:('EOR',lambda:self.logic(self.zp0,self.eor),3),
            0x46:('LSR',lambda:self.rmw(self.zp0,self.lsr),5),0x48:('PHA',self.pha,3),0x49:('EOR',lambda:self.logic(self.imm,self.eor),2),0x4A:('LSR',lambda:self.acc(self.lsr),2),
            0x4C:('JMP',self.jmp,3),0x4D:('EOR',lambda:self.logic(self.abs_,self.eor),4),0x4E:('LSR',lambda:self.rmw(self.abs_,self.lsr),6),0x50:('BVC',lambda:self.branch(not self.get(self.FLAG_V)),2),
            0x51:('EOR',lambda:self.logic(self.izy,self.eor),5),0x55:('EOR',lambda:self.logic(self.zpx,self.eor),4),0x56:('LSR',lambda:self.rmw(self.zpx,self.lsr),6),0x58:('CLI',lambda:self.set(self.FLAG_I,False),2),
            0x59:('EOR',lambda:self.logic(self.aby,self.eor),4),0x5D:('EOR',lambda:self.logic(self.abx,self.eor),4),0x5E:('LSR',lambda:self.rmw(self.abx,self.lsr),7),0x60:('RTS',self.rts,6),
            0x61:('ADC',lambda:self.logic(self.izx,self.adc),6),0x65:('ADC',lambda:self.logic(self.zp0,self.adc),3),0x66:('ROR',lambda:self.rmw(self.zp0,self.ror),5),0x68:('PLA',self.pla,4),
            0x69:('ADC',lambda:self.logic(self.imm,self.adc),2),0x6A:('ROR',lambda:self.acc(self.ror),2),0x6C:('JMP',self.ind_jmp,5),0x6D:('ADC',lambda:self.logic(self.abs_,self.adc),4),0x6E:('ROR',lambda:self.rmw(self.abs_,self.ror),6),
            0x70:('BVS',lambda:self.branch(self.get(self.FLAG_V)==1),2),0x71:('ADC',lambda:self.logic(self.izy,self.adc),5),0x75:('ADC',lambda:self.logic(self.zpx,self.adc),4),0x76:('ROR',lambda:self.rmw(self.zpx,self.ror),6),
            0x78:('SEI',lambda:self.set(self.FLAG_I,True),2),0x79:('ADC',lambda:self.logic(self.aby,self.adc),4),0x7D:('ADC',lambda:self.logic(self.abx,self.adc),4),0x7E:('ROR',lambda:self.rmw(self.abx,self.ror),7),
            0x81:('STA',lambda:self.store(self.izx,self.a),6),0x84:('STY',lambda:self.store(self.zp0,self.y),3),0x85:('STA',lambda:self.store(self.zp0,self.a),3),0x86:('STX',lambda:self.store(self.zp0,self.x),3),
            0x88:('DEY',lambda:self.dec_y(),2),0x8A:('TXA',lambda:self.lda(self.x),2),0x8C:('STY',lambda:self.store(self.abs_,self.y),4),0x8D:('STA',lambda:self.store(self.abs_,self.a),4),0x8E:('STX',lambda:self.store(self.abs_,self.x),4),
            0x90:('BCC',lambda:self.branch(not self.get(self.FLAG_C)),2),0x91:('STA',lambda:self.store(self.izy,self.a),6),0x94:('STY',lambda:self.store(self.zpx,self.y),4),0x95:('STA',lambda:self.store(self.zpx,self.a),4),0x96:('STX',lambda:self.store(self.zpy,self.x),4),
            0x98:('TYA',lambda:self.lda(self.y),2),0x99:('STA',lambda:self.store(self.aby,self.a),5),0x9A:('TXS',lambda:self.stxsp(),2),0x9D:('STA',lambda:self.store(self.abx,self.a),5),
            0xA0:('LDY',lambda:self.load(self.imm,self.ldy),2),0xA1:('LDA',lambda:self.load(self.izx,self.lda),6),0xA2:('LDX',lambda:self.load(self.imm,self.ldx),2),0xA4:('LDY',lambda:self.load(self.zp0,self.ldy),3),
            0xA5:('LDA',lambda:self.load(self.zp0,self.lda),3),0xA6:('LDX',lambda:self.load(self.zp0,self.ldx),3),0xA8:('TAY',lambda:self.ldy(self.a),2),0xA9:('LDA',lambda:self.load(self.imm,self.lda),2),
            0xAA:('TAX',lambda:self.ldx(self.a),2),0xAC:('LDY',lambda:self.load(self.abs_,self.ldy),4),0xAD:('LDA',lambda:self.load(self.abs_,self.lda),4),0xAE:('LDX',lambda:self.load(self.abs_,self.ldx),4),
            0xB0:('BCS',lambda:self.branch(self.get(self.FLAG_C)==1),2),0xB1:('LDA',lambda:self.load(self.izy,self.lda),5),0xB4:('LDY',lambda:self.load(self.zpx,self.ldy),4),0xB5:('LDA',lambda:self.load(self.zpx,self.lda),4),
            0xB6:('LDX',lambda:self.load(self.zpy,self.ldx),4),0xB8:('CLV',lambda:self.set(self.FLAG_V,False),2),0xB9:('LDA',lambda:self.load(self.aby,self.lda),4),0xBA:('TSX',lambda:self.ldx(self.stkp),2),0xBC:('LDY',lambda:self.load(self.abx,self.ldy),4),
            0xBD:('LDA',lambda:self.load(self.abx,self.lda),4),0xBE:('LDX',lambda:self.load(self.aby,self.ldx),4),0xC0:('CPY',lambda:self.logic(self.imm,lambda v:self.cmp(self.y,v)),2),0xC1:('CMP',lambda:self.logic(self.izx,lambda v:self.cmp(self.a,v)),6),
            0xC4:('CPY',lambda:self.logic(self.zp0,lambda v:self.cmp(self.y,v)),3),0xC5:('CMP',lambda:self.logic(self.zp0,lambda v:self.cmp(self.a,v)),3),0xC6:('DEC',lambda:self.rmw(self.zp0,self.dec),5),0xC8:('INY',lambda:self.iny(),2),
            0xC9:('CMP',lambda:self.logic(self.imm,lambda v:self.cmp(self.a,v)),2),0xCA:('DEX',lambda:self.dex(),2),0xCC:('CPY',lambda:self.logic(self.abs_,lambda v:self.cmp(self.y,v)),4),0xCD:('CMP',lambda:self.logic(self.abs_,lambda v:self.cmp(self.a,v)),4),
            0xCE:('DEC',lambda:self.rmw(self.abs_,self.dec),6),0xD0:('BNE',lambda:self.branch(not self.get(self.FLAG_Z)),2),0xD1:('CMP',lambda:self.logic(self.izy,lambda v:self.cmp(self.a,v)),5),0xD5:('CMP',lambda:self.logic(self.zpx,lambda v:self.cmp(self.a,v)),4),
            0xD6:('DEC',lambda:self.rmw(self.zpx,self.dec),6),0xD8:('CLD',lambda:self.set(self.FLAG_D,False),2),0xD9:('CMP',lambda:self.logic(self.aby,lambda v:self.cmp(self.a,v)),4),0xDD:('CMP',lambda:self.logic(self.abx,lambda v:self.cmp(self.a,v)),4),
            0xDE:('DEC',lambda:self.rmw(self.abx,self.dec),7),0xE0:('CPX',lambda:self.logic(self.imm,lambda v:self.cmp(self.x,v)),2),0xE1:('SBC',lambda:self.logic(self.izx,self.sbc),6),0xE4:('CPX',lambda:self.logic(self.zp0,lambda v:self.cmp(self.x,v)),3),
            0xE5:('SBC',lambda:self.logic(self.zp0,self.sbc),3),0xE6:('INC',lambda:self.rmw(self.zp0,self.inc),5),0xE8:('INX',lambda:self.inx(),2),0xE9:('SBC',lambda:self.logic(self.imm,self.sbc),2),
            0xEA:('NOP',lambda:None,2),0xEC:('CPX',lambda:self.logic(self.abs_,lambda v:self.cmp(self.x,v)),4),0xED:('SBC',lambda:self.logic(self.abs_,self.sbc),4),0xEE:('INC',lambda:self.rmw(self.abs_,self.inc),6),
            0xF0:('BEQ',lambda:self.branch(self.get(self.FLAG_Z)==1),2),0xF1:('SBC',lambda:self.logic(self.izy,self.sbc),5),0xF5:('SBC',lambda:self.logic(self.zpx,self.sbc),4),0xF6:('INC',lambda:self.rmw(self.zpx,self.inc),6),
            0xF8:('SED',lambda:self.set(self.FLAG_D,True),2),0xF9:('SBC',lambda:self.logic(self.aby,self.sbc),4),0xFD:('SBC',lambda:self.logic(self.abx,self.sbc),4),0xFE:('INC',lambda:self.rmw(self.abx,self.inc),7),
            # Common implied instructions
            0x18:('CLC',lambda:self.set(self.FLAG_C,False),2),0x38:('SEC',lambda:self.set(self.FLAG_C,True),2),0x8B:('XAA',self.un_xaa_imm,2),
        }

    def _build_extra(self):
        return {
            # NOPs with operands / implied NOPs.
            0x1A:(lambda:None,2), 0x3A:(lambda:None,2), 0x5A:(lambda:None,2),
            0x7A:(lambda:None,2), 0xDA:(lambda:None,2), 0xFA:(lambda:None,2),
            0x80:(self.nop_imm,2), 0x82:(self.nop_imm,2), 0x89:(self.nop_imm,2),
            0xC2:(self.nop_imm,2), 0xE2:(self.nop_imm,2),
            # Remaining common undocumented immediate opcodes.
            0x0B:(self.un_anc_imm,2), 0x2B:(self.un_anc_imm,2),
            0x4B:(self.un_alr_imm,2), 0x6B:(self.un_arr_imm,2),
            0xAB:(self.un_lax_imm,2), 0xBB:(self.un_las_abs_y,4),
            0x93:(self.un_ahx_ind_y,6), 0x9B:(self.un_tas_abs_y,5),
            0x9C:(self.un_shy_abs_x,5), 0x9E:(self.un_shx_abs_y,5),
            0x9F:(self.un_ahx_abs_y,5),
            0x04:(lambda:self.nop_zp(self.zp0),3), 0x44:(lambda:self.nop_zp(self.zp0),3), 0x64:(lambda:self.nop_zp(self.zp0),3),
            0x14:(lambda:self.nop_zp(self.zpx),4), 0x34:(lambda:self.nop_zp(self.zpx),4), 0x54:(lambda:self.nop_zp(self.zpx),4),
            0x74:(lambda:self.nop_zp(self.zpx),4), 0xD4:(lambda:self.nop_zp(self.zpx),4), 0xF4:(lambda:self.nop_zp(self.zpx),4),
            0x0C:(lambda:self.nop_abs(self.abs_),4),
            0x1C:(lambda:self.nop_abs(self.abx),4), 0x3C:(lambda:self.nop_abs(self.abx),4),
            0x5C:(lambda:self.nop_abs(self.abx),4), 0x7C:(lambda:self.nop_abs(self.abx),4),
            0xDC:(lambda:self.nop_abs(self.abx),4), 0xFC:(lambda:self.nop_abs(self.abx),4),
            # LAX/SAX.
            0xA7:(lambda:self.un_lax(self.zp0),3), 0xB7:(lambda:self.un_lax(self.zpy),4),
            0xAF:(lambda:self.un_lax(self.abs_),4), 0xBF:(lambda:self.un_lax(self.aby),4),
            0xA3:(lambda:self.un_lax(self.izx),6), 0xB3:(lambda:self.un_lax(self.izy),5),
            0x87:(lambda:self.un_sax(self.zp0),3), 0x97:(lambda:self.un_sax(self.zpy),4),
            0x8F:(lambda:self.un_sax(self.abs_),4), 0x83:(lambda:self.un_sax(self.izx),6),
            # RMW-combine family.
            0x07:(lambda:self.un_slo(self.zp0),5), 0x17:(lambda:self.un_slo(self.zpx),6), 0x0F:(lambda:self.un_slo(self.abs_),6),
            0x1F:(lambda:self.un_slo(self.abx),7), 0x1B:(lambda:self.un_slo(self.aby),7), 0x03:(lambda:self.un_slo(self.izx),8), 0x13:(lambda:self.un_slo(self.izy),8),
            0x47:(lambda:self.un_sre(self.zp0),5), 0x57:(lambda:self.un_sre(self.zpx),6), 0x4F:(lambda:self.un_sre(self.abs_),6),
            0x5F:(lambda:self.un_sre(self.abx),7), 0x5B:(lambda:self.un_sre(self.aby),7), 0x43:(lambda:self.un_sre(self.izx),8), 0x53:(lambda:self.un_sre(self.izy),8),
            0x27:(lambda:self.un_rla(self.zp0),5), 0x37:(lambda:self.un_rla(self.zpx),6), 0x2F:(lambda:self.un_rla(self.abs_),6),
            0x3F:(lambda:self.un_rla(self.abx),7), 0x3B:(lambda:self.un_rla(self.aby),7), 0x23:(lambda:self.un_rla(self.izx),8), 0x33:(lambda:self.un_rla(self.izy),8),
            0x67:(lambda:self.un_rra(self.zp0),5), 0x77:(lambda:self.un_rra(self.zpx),6), 0x6F:(lambda:self.un_rra(self.abs_),6),
            0x7F:(lambda:self.un_rra(self.abx),7), 0x7B:(lambda:self.un_rra(self.aby),7), 0x63:(lambda:self.un_rra(self.izx),8), 0x73:(lambda:self.un_rra(self.izy),8),
            0xC7:(lambda:self.un_dcp(self.zp0),5), 0xD7:(lambda:self.un_dcp(self.zpx),6), 0xCF:(lambda:self.un_dcp(self.abs_),6),
            0xDF:(lambda:self.un_dcp(self.abx),7), 0xDB:(lambda:self.un_dcp(self.aby),7), 0xC3:(lambda:self.un_dcp(self.izx),8), 0xD3:(lambda:self.un_dcp(self.izy),8),
            0xE7:(lambda:self.un_isc(self.zp0),5), 0xF7:(lambda:self.un_isc(self.zpx),6), 0xEF:(lambda:self.un_isc(self.abs_),6),
            0xFF:(lambda:self.un_isc(self.abx),7), 0xFB:(lambda:self.un_isc(self.aby),7), 0xE3:(lambda:self.un_isc(self.izx),8), 0xF3:(lambda:self.un_isc(self.izy),8),
            0xEB:(lambda:self.logic(self.imm,self.sbc),2),
            0xCB:(self.un_axs_imm,2),
        }

    def _ensure_tables(self):
        # Kept for compatibility; tables are built once during initialization.
        return

    def step_instruction(self):
        """Execute one complete 6502 instruction.

        Mapper-0 gets a hot path for common instructions. The fallback still
        uses the full opcode table, so less common opcodes and undocumented
        instructions retain the existing implementation.
        """
        # A real KIL opcode halts only the CPU; the PPU continues to advance.
        if self.jammed:
            return 2
        c = self.cycles
        if c:
            self.cycles = 0
            return c

        if self.bus.nmi_pending:
            self.bus.nmi_pending = False
            self.nmi()
            c = self.cycles
            self.cycles = 0
            return c

        # Fast fetch. Super Mario Bros and most simple NROM games live here.
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        ram = self._ram
        prg = self._prg
        fast_fetch = self._fast_fetch
        read_mem = self.read
        write_mem = self.write
        pull_stack = self.pull
        push_stack = self.push
        if pc >= 0x8000 and self._mapper_id == 0:
            off = pc - 0x8000
            if self._prg_mask_16k:
                op = prg[off & 0x3FFF]
            else:
                op = prg[off % self._prg_len]
        elif pc < 0x2000:
            op = ram[pc & 0x07FF]
        else:
            op = read_mem(pc)

        if op in self.KIL_OPCODES:
            self.jammed = True
            return 2

        # Extended NROM hot path. Keep common loads, stores, comparisons and
        # immediate ALU operations ahead of the larger dispatch chain. The
        # fallback table remains the reference implementation for other ops.
        if (self.fast_extended_ops_enabled and self._mapper_id == 0
                and op in self.NROM_EXTENDED_OPS):
            mode = op & 0x1F
            family = op & 0xE0
            # The following modes are NROM-only fast paths; bus-visible effective
            # accesses still use read_mem/write_mem, so PPU/controller side effects
            # and instruction cycle counts are retained.
            if op == 0xA4:  # LDY zp
                self.y = ram[fast_fetch()]; v = self.y
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 3
            if op == 0xA6:  # LDX zp
                self.x = ram[fast_fetch()]; v = self.x
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 3

            if family == 0xA0:  # LDA addressing family
                if mode == 0x09:
                    value = fast_fetch(); cycles = 2
                elif mode == 0x05:
                    value = ram[fast_fetch()]; cycles = 3
                elif mode == 0x15:
                    value = ram[(fast_fetch() + self.x) & 0xFF]; cycles = 4
                elif mode == 0x0D:
                    lo = fast_fetch(); hi = fast_fetch(); value = read_mem(lo | (hi << 8)); cycles = 4
                elif mode == 0x1D or mode == 0x19:
                    lo = fast_fetch(); hi = fast_fetch(); base = lo | (hi << 8)
                    addr = (base + (self.x if mode == 0x1D else self.y)) & 0xFFFF
                    value = read_mem(addr); cycles = 4 + int((base & 0xFF00) != (addr & 0xFF00))
                elif mode == 0x01:
                    zp = (fast_fetch() + self.x) & 0xFF
                    addr = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                    value = read_mem(addr); cycles = 6
                else:  # (zp),Y
                    zp = fast_fetch(); base = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                    addr = (base + self.y) & 0xFFFF
                    value = read_mem(addr); cycles = 5 + int((base & 0xFF00) != (addr & 0xFF00))
                self.a = value & 0xFF; v = self.a
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return cycles

            if family == 0x80:  # STA addressing family
                if mode == 0x05:
                    addr = fast_fetch(); write_mem(addr, self.a); return 3
                if mode == 0x15:
                    addr = (fast_fetch() + self.x) & 0xFF; write_mem(addr, self.a); return 4
                if mode == 0x0D:
                    lo = fast_fetch(); hi = fast_fetch(); write_mem(lo | (hi << 8), self.a); return 4
                if mode == 0x1D or mode == 0x19:
                    lo = fast_fetch(); hi = fast_fetch(); base = lo | (hi << 8)
                    addr = (base + (self.x if mode == 0x1D else self.y)) & 0xFFFF
                    write_mem(addr, self.a); return 5
                if mode == 0x01:
                    zp = (fast_fetch() + self.x) & 0xFF
                    addr = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                    write_mem(addr, self.a); return 6
                zp = fast_fetch(); base = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                write_mem((base + self.y) & 0xFFFF, self.a); return 6

            # ORA/AND/EOR/ADC/CMP/SBC. The mode low bits encode the addressing
            # variant, keeping this hot decode far shorter than nested lambdas.
            if mode == 0x09:
                value = fast_fetch(); cycles = 2
            elif mode == 0x05:
                value = ram[fast_fetch()]; cycles = 3
            elif mode == 0x15:
                value = ram[(fast_fetch() + self.x) & 0xFF]; cycles = 4
            elif mode == 0x0D:
                lo = fast_fetch(); hi = fast_fetch(); value = read_mem(lo | (hi << 8)); cycles = 4
            elif mode == 0x1D or mode == 0x19:
                lo = fast_fetch(); hi = fast_fetch(); base = lo | (hi << 8)
                addr = (base + (self.x if mode == 0x1D else self.y)) & 0xFFFF
                value = read_mem(addr); cycles = 4 + int((base & 0xFF00) != (addr & 0xFF00))
            elif mode == 0x01:
                zp = (fast_fetch() + self.x) & 0xFF
                addr = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                value = read_mem(addr); cycles = 6
            else:  # (zp),Y
                zp = fast_fetch(); base = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                addr = (base + self.y) & 0xFFFF
                value = read_mem(addr); cycles = 5 + int((base & 0xFF00) != (addr & 0xFF00))
            value &= 0xFF
            if family == 0x00:
                result = (self.a | value) & 0xFF
                self.a = result
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if result == 0 else 0) | (self.FLAG_N if result & 0x80 else 0)
            elif family == 0x20:
                result = self.a & value
                self.a = result
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if result == 0 else 0) | (self.FLAG_N if result & 0x80 else 0)
            elif family == 0x40:
                result = (self.a ^ value) & 0xFF
                self.a = result
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if result == 0 else 0) | (self.FLAG_N if result & 0x80 else 0)
            elif family == 0x60 or family == 0xE0:
                a = self.a
                carry = 1 if (self.status & self.FLAG_C) else 0
                operand = value ^ 0xFF if family == 0xE0 else value
                total = a + operand + carry
                result = total & 0xFF
                flags = self.FLAG_C if total > 0xFF else 0
                if (~(a ^ operand) & (a ^ result) & 0x80): flags |= self.FLAG_V
                if result == 0: flags |= self.FLAG_Z
                if result & 0x80: flags |= self.FLAG_N
                self.a = result
                self.status = (self.status & ~(self.FLAG_C | self.FLAG_Z | self.FLAG_N | self.FLAG_V)) | flags
            else:  # CMP
                diff = (self.a - value) & 0xFF
                flags = self.FLAG_C if self.a >= value else 0
                if diff == 0: flags |= self.FLAG_Z
                if diff & 0x80: flags |= self.FLAG_N
                self.status = (self.status & ~(self.FLAG_C | self.FLAG_Z | self.FLAG_N)) | flags
            return cycles

        if self.fast_extended_ops_enabled:
            if op == 0xA9:  # LDA #imm (reference-compatible fallback on other mappers)
                self.a = fast_fetch(); v = self.a
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 2
            if op == 0xAD:  # LDA abs
                lo = fast_fetch(); hi = fast_fetch(); self.a = read_mem(lo | (hi << 8)); v = self.a
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 4
            if op == 0xBD or op == 0xB9:  # LDA abs,X / abs,Y
                lo = fast_fetch(); hi = fast_fetch(); base = lo | (hi << 8)
                addr = (base + (self.x if op == 0xBD else self.y)) & 0xFFFF
                self.a = read_mem(addr); v = self.a
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 5 if (base & 0xFF00) != (addr & 0xFF00) else 4
            if op == 0xB1:  # LDA (zp),Y
                zp = fast_fetch(); base = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                addr = (base + self.y) & 0xFFFF
                self.a = read_mem(addr); v = self.a
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 6 if (base & 0xFF00) != (addr & 0xFF00) else 5
            if op == 0x8D:  # STA abs
                lo = fast_fetch(); hi = fast_fetch(); write_mem(lo | (hi << 8), self.a); return 4
            if op == 0x9D or op == 0x99:  # STA abs,X / abs,Y
                lo = fast_fetch(); hi = fast_fetch(); base = lo | (hi << 8)
                addr = (base + (self.x if op == 0x9D else self.y)) & 0xFFFF
                write_mem(addr, self.a); return 5
            if op == 0x91:  # STA (zp),Y
                zp = fast_fetch(); base = ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
                write_mem((base + self.y) & 0xFFFF, self.a); return 6
            if op == 0xA6:  # LDX zp
                self.x = ram[fast_fetch()]; v = self.x
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 3
            if op == 0xA4:  # LDY zp
                self.y = ram[fast_fetch()]; v = self.y
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 3
            if op in (0x09, 0x29, 0x49):  # ORA/AND/EOR #imm
                value = fast_fetch()
                if op == 0x09: self.a |= value
                elif op == 0x29: self.a &= value
                else: self.a ^= value
                self.a &= 0xFF; v = self.a
                self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N)) | (self.FLAG_Z if v == 0 else 0) | (self.FLAG_N if v & 0x80 else 0)
                return 2
            if op == 0x69 or op == 0xE9:  # ADC/SBC #imm (NES binary arithmetic)
                value = fast_fetch()
                a = self.a
                carry = 1 if (self.status & self.FLAG_C) else 0
                if op == 0xE9: value ^= 0xFF
                total = a + value + carry
                result = total & 0xFF
                flags = (self.FLAG_C if total > 0xFF else 0)
                if (~(a ^ value) & (a ^ result) & 0x80): flags |= self.FLAG_V
                if result == 0: flags |= self.FLAG_Z
                if result & 0x80: flags |= self.FLAG_N
                self.a = result
                self.status = (self.status & ~(self.FLAG_C | self.FLAG_Z | self.FLAG_N | self.FLAG_V)) | flags
                return 2
            if op == 0xC9 or op == 0xC5 or op == 0xCD:  # CMP #imm / zp / abs
                if op == 0xC9:
                    value = fast_fetch()
                elif op == 0xC5:
                    value = ram[fast_fetch()]
                else:
                    lo = fast_fetch(); hi = fast_fetch(); value = read_mem(lo | (hi << 8))
                reg = self.a; diff = (reg - value) & 0xFF
                flags = self.FLAG_C if reg >= value else 0
                if diff == 0: flags |= self.FLAG_Z
                if diff & 0x80: flags |= self.FLAG_N
                self.status = (self.status & ~(self.FLAG_C | self.FLAG_Z | self.FLAG_N)) | flags
                return 2 if op == 0xC9 else (3 if op == 0xC5 else 4)

        # Common 6502 instructions, inlined to avoid lambda + addressing-mode
        # + bound-method call overhead in the hot path.
        # Extremely common NROM opcodes are checked first. This matters on
        # Python because even a correct long if-chain costs time when a title
        # executes thousands of instructions per frame.
        if op == 0x2C:  # BIT abs; particularly hot in SMB's PPUSTATUS polling loop
            lo = fast_fetch(); hi = fast_fetch(); value = read_mem(lo | (hi << 8))
            self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N | self.FLAG_V)) | (self.FLAG_Z if (self.a & value) == 0 else 0) | (value & 0xC0)
            return 4
        if op == 0x24:  # BIT zp
            value = ram[fast_fetch()]
            self.status = (self.status & ~(self.FLAG_Z | self.FLAG_N | self.FLAG_V)) | (self.FLAG_Z if (self.a & value) == 0 else 0) | (value & 0xC0)
            return 3
        # All eight conditional branches are hot in NES game loops. Keep them
        # inline; the table fallback otherwise pays lambda + get() + branch().
        if op == 0x10:  # BPL
            off=fast_fetch()
            if not (self.status & self.FLAG_N):
                off=off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x30:  # BMI
            off=fast_fetch()
            if self.status & self.FLAG_N:
                off=off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x50:  # BVC
            off=fast_fetch()
            if not (self.status & self.FLAG_V):
                off=off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x70:  # BVS
            off=fast_fetch()
            if self.status & self.FLAG_V:
                off=off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x90:  # BCC
            off=fast_fetch()
            if not (self.status & self.FLAG_C):
                off=off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0xB0:  # BCS
            off=fast_fetch()
            if self.status & self.FLAG_C:
                off=off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0xEA:  # NOP
            return 2
        if op == 0xA9:  # LDA #imm
            self.a = fast_fetch(); v=self.a; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x85:  # STA zp
            write_mem(fast_fetch(), self.a); return 3
        if op == 0x8D:  # STA abs
            lo=fast_fetch(); hi=fast_fetch(); write_mem(lo|(hi<<8), self.a); return 4
        if op == 0xA5:  # LDA zp
            self.a = ram[fast_fetch()]; v=self.a; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 3
        if op == 0xD0:  # BNE
            off=fast_fetch()
            if not (self.status & self.FLAG_Z):
                off = off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0xF0:  # BEQ
            off=fast_fetch()
            if self.status & self.FLAG_Z:
                off = off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x20:  # JSR abs
            lo = fast_fetch(); hi = fast_fetch(); ret = (self.pc - 1) & 0xFFFF
            push_stack(ret >> 8); push_stack(ret & 0xFF); self.pc = lo | (hi << 8); return 6
        if op == 0x60:  # RTS
            self.pc = pull_stack() | (pull_stack() << 8); self.pc = (self.pc + 1) & 0xFFFF; return 6
        if op == 0x4C:  # JMP abs
            lo = fast_fetch(); hi = fast_fetch(); self.pc = lo | (hi << 8); return 3
        if op == 0xA2:  # LDX #imm
            self.x = fast_fetch(); v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xA0:  # LDY #imm
            self.y = fast_fetch(); v=self.y; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xE8:  # INX
            self.x=(self.x+1)&0xFF; v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xC8:  # INY
            self.y=(self.y+1)&0xFF; v=self.y; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xCA:  # DEX
            self.x=(self.x-1)&0xFF; v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x88:  # DEY
            self.y=(self.y-1)&0xFF; v=self.y; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2

        if op == 0x4C:  # JMP abs
            lo = fast_fetch(); hi = fast_fetch(); self.pc = lo | (hi << 8); return 3
        if op == 0x20:  # JSR abs
            lo = fast_fetch(); hi = fast_fetch(); ret = (self.pc - 1) & 0xFFFF
            push_stack(ret >> 8); push_stack(ret & 0xFF); self.pc = lo | (hi << 8); return 6
        if op == 0x60:  # RTS
            self.pc = pull_stack() | (pull_stack() << 8); self.pc = (self.pc + 1) & 0xFFFF; return 6
        if op == 0xA9:  # LDA #imm
            self.a = fast_fetch(); v=self.a; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xA2:  # LDX #imm
            self.x = fast_fetch(); v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xA0:  # LDY #imm
            self.y = fast_fetch(); v=self.y; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x8D:  # STA abs
            lo=fast_fetch(); hi=fast_fetch(); write_mem(lo|(hi<<8), self.a); return 4
        if op == 0x8E:  # STX abs
            lo=fast_fetch(); hi=fast_fetch(); write_mem(lo|(hi<<8), self.x); return 4
        if op == 0x8C:  # STY abs
            lo=fast_fetch(); hi=fast_fetch(); write_mem(lo|(hi<<8), self.y); return 4
        if op == 0x85:  # STA zp
            write_mem(fast_fetch(), self.a); return 3
        if op == 0x86:  # STX zp
            write_mem(fast_fetch(), self.x); return 3
        if op == 0x84:  # STY zp
            write_mem(fast_fetch(), self.y); return 3
        if op == 0xE8:  # INX
            self.x=(self.x+1)&0xFF; v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xC8:  # INY
            self.y=(self.y+1)&0xFF; v=self.y; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xCA:  # DEX
            self.x=(self.x-1)&0xFF; v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x88:  # DEY
            self.y=(self.y-1)&0xFF; v=self.y; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x18: self.status &= ~self.FLAG_C; return 2
        if op == 0x38: self.status |= self.FLAG_C; return 2
        if op == 0x58: self.status &= ~self.FLAG_I; return 2
        if op == 0x78: self.status |= self.FLAG_I; return 2
        if op == 0xEA: return 2
        if op == 0xD0: # BNE
            off=fast_fetch();
            if not (self.status & self.FLAG_Z):
                off = off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0xF0: # BEQ
            off=fast_fetch();
            if self.status & self.FLAG_Z:
                off = off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x10: # BPL
            off=fast_fetch();
            if not (self.status & self.FLAG_N):
                off = off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x30: # BMI
            off=fast_fetch();
            if self.status & self.FLAG_N:
                off = off-256 if off&0x80 else off; old=self.pc; self.pc=(self.pc+off)&0xFFFF; return 4 if (old&0xFF00)!=(self.pc&0xFF00) else 3
            return 2
        if op == 0x40:  # RTI
            self.status=(pull_stack()|self.FLAG_U)&~self.FLAG_B; self.pc=pull_stack()|(pull_stack()<<8); return 6
        if op == 0x48: push_stack(self.a); return 3
        if op == 0x68: self.a=pull_stack(); v=self.a; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 4
        if op == 0x08: push_stack(self.status|self.FLAG_B|self.FLAG_U); return 3
        if op == 0x28: self.status=(pull_stack()|self.FLAG_U)&~self.FLAG_B; return 4
        if op == 0xAA: self.x=self.a; v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x8A: self.a=self.x; v=self.a; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0xA8: self.y=self.a; v=self.y; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x98: self.a=self.y; v=self.a; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2
        if op == 0x9A: self.stkp=self.x; return 2
        if op == 0xBA: self.x=self.stkp; v=self.x; self.status=(self.status & ~(self.FLAG_Z|self.FLAG_N)) | (self.FLAG_Z if v==0 else 0) | (self.FLAG_N if v&0x80 else 0); return 2

        # Fallback to the complete, already-built dispatch table.
        entry = self._table[op]
        if entry is not None:
            _, fn, base = entry
            self.cycles = base - 1
            fn()
            used = self.cycles + 1
            self.cycles = 0
            return used
        entry = self._extra[op]
        if entry is not None:
            fn, base = entry
            self.cycles = base - 1
            fn()
            used = self.cycles + 1
            self.cycles = 0
            return used
        return 2

    def _fast_fetch(self):
        pc = self.pc
        self.pc = (pc + 1) & 0xFFFF
        if pc >= 0x8000 and self._mapper_id == 0:
            off = pc - 0x8000
            if self._prg_mask_16k:
                return self._prg[off & 0x3FFF]
            return self._prg[off % self._prg_len]
        if pc < 0x2000:
            return self._ram[pc & 0x07FF]
        return self.read(pc)

    def clock(self):
        """Cycle-compatible API retained for the test suite/diagnostics."""
        if self.cycles:
            self.cycles -= 1
            return
        self.step_instruction()

    def nop_imm(self):
        self.fetch()
    def nop_zp(self, mode):
        mode()
    def nop_abs(self, mode):
        mode()
    def un_lax(self, mode):
        a, cross = mode(); v = self.read(a); self.a = self.x = v; self.setzn(v); self.cycles += 1 if cross else 0
    def un_sax(self, mode):
        a, _ = mode(); self.write(a, self.a & self.x)
    def un_slo(self, mode):
        a, _ = mode(); v = self.asl(self.read(a)); self.write(a,v); self.ora(v)
    def un_sre(self, mode):
        a, _ = mode(); v = self.lsr(self.read(a)); self.write(a,v); self.eor(v)
    def un_rla(self, mode):
        a, _ = mode(); v = self.rol(self.read(a)); self.write(a,v); self.and_(v)
    def un_rra(self, mode):
        a, _ = mode(); v = self.ror(self.read(a)); self.write(a,v); self.adc(v)
    def un_dcp(self, mode):
        a, _ = mode(); v = self.dec(self.read(a)); self.write(a,v); self.cmp(self.a,v)
    def un_isc(self, mode):
        a, _ = mode(); v = self.inc(self.read(a)); self.write(a,v); self.sbc(v)
    def un_axs_imm(self):
        v=self.fetch(); r=(self.a & self.x) - v; self.set(self.FLAG_C, r >= 0); self.x = r & 0xFF; self.setzn(self.x)

    def logic(self, mode, op):
        a, cross = mode()
        # Immediate addressing returns the operand itself, not a memory address.
        v = a if getattr(mode, "__name__", "") == "imm" else self.read(a)
        op(v)
        if cross:
            self.cycles += 1
    def load(self, mode, op):
        a, cross = mode()
        # Immediate addressing returns the operand itself, not a memory address.
        v = a if getattr(mode, "__name__", "") == "imm" else self.read(a)
        op(v)
        if cross:
            self.cycles += 1
    def store(self, mode, value):
        a, _ = mode(); self.write(a, value)
    def rmw(self, mode, op):
        a, _ = mode(); self.write(a, op(self.read(a)))
    def acc(self, op): self.a = op(self.a)

    def ora(self,v): self.a |= v; self.setzn(self.a)
    def and_(self,v): self.a &= v; self.setzn(self.a)
    def eor(self,v): self.a ^= v; self.setzn(self.a)
    def asl(self,v): self.set(self.FLAG_C,v&0x80); r=(v<<1)&0xFF; self.setzn(r); return r
    def lsr(self,v): self.set(self.FLAG_C,v&1); r=(v>>1)&0xFF; self.setzn(r); return r
    def rol(self,v): c=self.get(self.FLAG_C); self.set(self.FLAG_C,v&0x80); r=((v<<1)&0xFF)|c; self.setzn(r); return r
    def ror(self,v): c=self.get(self.FLAG_C); self.set(self.FLAG_C,v&1); r=(v>>1)|(c<<7); self.setzn(r); return r
    def inc(self,v): r=(v+1)&0xFF; self.setzn(r); return r
    def dec(self,v): r=(v-1)&0xFF; self.setzn(r); return r
    def inx(self): self.x=(self.x+1)&0xFF; self.setzn(self.x)
    def iny(self): self.y=(self.y+1)&0xFF; self.setzn(self.y)
    def dex(self): self.x=(self.x-1)&0xFF; self.setzn(self.x)
    def dec_y(self): self.y=(self.y-1)&0xFF; self.setzn(self.y)
    def stxsp(self): self.stkp=self.x

    def bit(self,mode):
        a,_=mode(); v=self.read(a); self.set(self.FLAG_Z,(self.a&v)==0); self.set(self.FLAG_N,v&0x80); self.set(self.FLAG_V,v&0x40)
    def php(self): self.push(self.status|self.FLAG_B|self.FLAG_U)
    def plp(self): self.status=(self.pull()|self.FLAG_U)&~self.FLAG_B
    def pha(self): self.push(self.a)
    def pla(self): self.a=self.pull(); self.setzn(self.a)
    def jsr(self):
        lo = self.fetch(); hi = self.fetch(); target = lo | (hi << 8)
        ret=(self.pc-1)&0xFFFF; self.push(ret>>8); self.push(ret&0xFF); self.pc=target
    def rts(self): self.pc=(self.pull() | (self.pull()<<8)); self.pc=(self.pc+1)&0xFFFF
    def rti(self): self.status=(self.pull()|self.FLAG_U)&~self.FLAG_B; self.pc=self.pull() | (self.pull()<<8)
    def jmp(self):
        lo = self.fetch(); hi = self.fetch(); self.pc = lo | (hi << 8)
    def ind_jmp(self): self.pc=self.ind()[0]
    def brk(self):
        self.pc=(self.pc+1)&0xFFFF; self.push(self.pc>>8); self.push(self.pc&0xFF); self.push(self.status|self.FLAG_B|self.FLAG_U); self.set(self.FLAG_I,True); self.pc=self.read16(0xFFFE)
