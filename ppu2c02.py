"""Practical PPU 2C02 implementation for a dependency-free NES emulator.

The renderer is scanline based (not a transistor-accurate pixel pipeline), but it
implements the register behavior needed by common games: VRAM buffering, scroll,
nametable mirroring, palettes, background tiles, sprites, sprite priority,
sprite-0 hit, vblank/NMI and OAM DMA.
"""


class PPU2C02:
    def __init__(self, cart):
        self.cart = cart
        self.name_tables = bytearray(4096)
        self.palette = bytearray(32)
        self.oam = bytearray(256)
        self.frame = [0] * (256 * 240)
        self.scanline = -1
        self.cycle = 0
        self.frame_complete = False
        # PPU dots which overshoot a frame boundary during batched CPU sync are
        # carried into the next frame instead of rendering into the old frame.
        self._carry_ticks = 0
        self.nmi = False
        self.ppuctrl = 0
        self.ppumask = 0
        self.ppustatus = 0
        self.oamaddr = 0
        self.v = 0
        self.t = 0
        self.fine_x = 0
        self.w = 0
        self.data_buffer = 0
        self.bg_opaque = bytearray(256 * 240)
        self.frame_scroll_x = 0
        self.frame_scroll_y = 0
        self.frame_nt = 0
        self.sprite0_hit_x = None
        self.sprite0_hit_y = None
        self._sprite_line_cache = [()] * 240
        self._sprite_cache_dirty = True
        self._sprite_cache_generation = 0
        # Per-scanline resolved sprite pixels. Store palette indices, not RGB,
        # so palette writes during a line still affect pixels not yet emitted.
        self._sprite_pixel_cache = [None] * 240
        self._sprite_overflow_lines = bytearray(240)
        self._bg_row_cache = [None] * (2 * 256 * 8 * 4)
        # Cache decoded full 256-pixel background scanlines by physical nametable,
        # tile row, fine Y and pattern-table select. This removes 32 tile/attribute
        # lookups from every visible scanline while preserving per-scanline scroll.
        self._scanline_bg_cache = {}
        # Cache two adjacent physical nametable rows as one 512-pixel world row.
        # PPUSTATUS polling can advance only a handful of dots per call; this
        # avoids recalculating vertical scroll and mirroring on every tiny slice.
        self._horizontal_bg_cache = {}
        self._active_bg_row_cache_key = None
        self._active_bg_row_data = None
        self._background_dirty = True
        self._palette_generation = 0
        self._pattern_generation = 0
        self.open_bus = 0
        # Persistent CHR pattern cache.  For fixed-CHR mappers this avoids
        # re-decoding the same 8x8 tile row on every scanline.
        self._pattern_row_cache = [None] * 0x2000
        self._persistent_chr_cache = (not self.cart.chr_is_ram and self.cart.mapper in (0, 2, 7, 11, 13, 34, 71))
        # NROM can safely defer final RGB compositing because its CHR bank cannot
        # change. Timing-critical background opacity and sprite-0 hit are still
        # updated at each PPU slice; unchanged adjacent draw slices are merged.
        # Every rendering-state / VRAM / OAM write flushes the queue before mutation.
        self._deferred_render_enabled = (self.cart.mapper == 0 and self._persistent_chr_cache)
        self._render_queue = []
        # Dot-accurate BG opacity is needed for Sprite-0 polling, but copying a
        # few opacity bytes for every tiny $2002 read is expensive. Cache the
        # rendering state already filled through the end of each scanline.
        self._opacity_state_key = [None] * 240
        self._bg_content_generation = 0
        # Cache state was initialized above; keep a single owner for its arrays.
        self._background_dirty = True
        self._background_ready = False
        self._background_signature = None
        # Kept as lightweight state for dormant legacy helper methods; large
        # 512x480 pixel arrays are no longer allocated in the stable renderer.
        self._world_ready = False
        self._world_needs_full = True
        self._world_dirty_tiles = set()
        # The stable core uses incremental scanline rendering. The older full
        # 512x480 NROM world cache caused large synchronous rebuilds and wasted
        # memory; retain the flag for legacy code paths but keep it disabled.
        self._fast_nrom = False

        self.colors = [
            0x545454, 0x001E74, 0x081090, 0x300088, 0x440064, 0x5C0030, 0x540400, 0x3C1800,
            0x202A00, 0x083A00, 0x004200, 0x00402C, 0x00323C, 0x000000, 0x000000, 0x000000,
            0x989698, 0x084CC4, 0x3032EC, 0x5C1EE4, 0x8814B0, 0xA01464, 0x982220, 0x783800,
            0x545000, 0x286400, 0x0C6C34, 0x0C6058, 0x08505C, 0x000000, 0x000000, 0x000000,
            0xECEEEC, 0x4C9AEC, 0x787CEC, 0xB062E8, 0xD454C8, 0xE04C78, 0xD86834, 0xB07C24,
            0x888A24, 0x50A838, 0x20B05C, 0x20A090, 0x208890, 0x3C3C3C, 0x000000, 0x000000,
            0xECEEEC, 0xA8CCEC, 0xBCBCEC, 0xD0B2EC, 0xE8A8E4, 0xECA8B0, 0xECA890, 0xE0BC74,
            0xCCD078, 0xB0D080, 0x90D0A0, 0x90CCC0, 0x90C8CC, 0xA0A2A0, 0x000000, 0x000000,
        ]

    def cpu_write(self, reg, value):
        value &= 0xFF
        self.open_bus = value
        r = reg & 7
        if self._deferred_render_enabled and r in (0, 1, 3, 4, 5, 6, 7):
            self.flush_render_queue()
        if r == 0:
            old_ctrl = self.ppuctrl
            old_nmi = bool(old_ctrl & 0x80)
            old_bg_table = old_ctrl & 0x10
            old_sprite_decode = old_ctrl & 0x28  # sprite table + sprite size
            self.ppuctrl = value
            # Only bits 3 and 5 affect sprite decode/geometry. SMB changes NMI
            # and nametable bits during frame splits; invalidating every sprite
            # row for those writes caused needless per-frame recomputation.
            if old_sprite_decode != (value & 0x28):
                self.invalidate_sprite_cache()
            if old_bg_table != (value & 0x10):
                self._background_dirty = True
                self._world_needs_full = True
                self._world_dirty_tiles.clear()
            self.t = (self.t & 0xF3FF) | ((value & 3) << 10)
            self._update_scroll_snapshot()
            self._invalidate_opacity_for_current_line()
            if not old_nmi and (value & 0x80) and (self.ppustatus & 0x80):
                self.nmi = True
        elif r == 1:
            old_mask = self.ppumask
            self.ppumask = value
            # Mask bits only affect visibility/clipping; the sprite line cache
            # reads mask live during composition, so OAM/CHR caches remain valid.
            if (old_mask ^ value) & 0x08:
                self._background_dirty = True
            if (old_mask ^ value) & 0x0A:
                self._invalidate_opacity_for_current_line()
        elif r == 3:
            self.oamaddr = value
        elif r == 4:
            self.oam[self.oamaddr] = value
            self.oamaddr = (self.oamaddr + 1) & 0xFF
            self.invalidate_sprite_cache()
        elif r == 5:
            if self.w == 0:
                self.fine_x = value & 7
                old = self.t
                self.t = (self.t & 0xFFE0) | (value >> 3)
                self.w = 1
            else:
                self.t = (self.t & 0x8C1F) | ((value & 7) << 12) | ((value & 0xF8) << 2)
                self.w = 0
            # This scanline renderer cannot reproduce every dot-level mid-line
            # change, but sampling current scroll registers per scanline lets
            # games such as SMB perform their sprite-0 HUD split instead of
            # freezing on a permanent scroll offset.
            self._update_scroll_snapshot()
            self._invalidate_opacity_for_current_line()
        elif r == 6:
            if self.w == 0:
                self.t = (self.t & 0x00FF) | ((value & 0x3F) << 8)
                self.w = 1
            else:
                self.t = (self.t & 0xFF00) | value
                self.v = self.t
                self.w = 0
                self._update_scroll_snapshot()
                self._invalidate_opacity_for_current_line()
        elif r == 7:
            self.mem_write(self.v, value)
            self._increment_v()

    def cpu_read(self, reg, readonly=False):
        r = reg & 7
        if r == 2:
            value = (self.ppustatus & 0xE0) | (self.open_bus & 0x1F)
            if not readonly:
                self.ppustatus &= 0x7F
                self.w = 0
            self.open_bus = value
            return value
        if r == 4:
            return self.oam[self.oamaddr]
        if r == 7:
            addr = self.v & 0x3FFF
            if addr < 0x3F00:
                value = self.data_buffer
                self.data_buffer = self.mem_read(addr)
            else:
                value = self.mem_read(addr)
                self.data_buffer = self.mem_read((addr - 0x1000) & 0x3FFF)
            if not readonly:
                self._increment_v()
            self.open_bus = value
            return value
        return self.open_bus

    def _increment_v(self):
        self.v = (self.v + (32 if (self.ppuctrl & 4) else 1)) & 0x7FFF

    def _nt_index(self, nt):
        return self.cart.mirror_nt(nt)

    def mem_read(self, addr):
        addr &= 0x3FFF
        if addr < 0x2000:
            return self.cart.ppu_read(addr) or 0
        if addr < 0x3F00:
            if self._deferred_render_enabled:
                self.flush_render_queue()
            a = (addr - 0x2000) & 0x0FFF
            nt = a // 0x400
            off = a & 0x3FF
            return self.name_tables[self._nt_index(nt) * 0x400 + off]
        if self._deferred_render_enabled:
            self.flush_render_queue()
        a = (addr - 0x3F00) & 0x1F
        if a in (0x10, 0x14, 0x18, 0x1C):
            a -= 0x10
        return self.palette[a] & 0x3F

    def mem_write(self, addr, value):
        addr &= 0x3FFF
        value &= 0xFF
        if addr < 0x2000:
            if self._deferred_render_enabled and self.cart.chr_is_ram:
                self.flush_render_queue()
            changed = self.cart.ppu_write(addr, value)
            if changed:
                # CHR-RAM contents changed. Drop decoded pattern rows and make
                # any prefilled BG-opacity remainder stale.
                self._bg_content_generation += 1
                self._invalidate_opacity_for_current_line()
                self._pattern_row_cache = [None] * 0x2000
                self._pattern_generation += 1
                self._bg_row_cache = [None] * (2 * 256 * 8 * 4)
                self._scanline_bg_cache.clear()
                self._horizontal_bg_cache.clear()
                self._active_bg_row_cache_key = None
                self._active_bg_row_data = None
                self._background_dirty = True
                self.invalidate_sprite_cache()
                if self._fast_nrom:
                    self._world_needs_full = True
                    self._world_dirty_tiles.clear()
                else:
                    self._world_ready = False
            return
        if addr < 0x3F00:
            if self._deferred_render_enabled:
                # Render already-emitted pixels with the old nametable state.
                self.flush_render_queue()
                self._bg_content_generation += 1
                self._invalidate_opacity_for_current_line()
            a = (addr - 0x2000) & 0x0FFF
            nt = a // 0x400
            off = a & 0x3FF
            physical = self._nt_index(nt)
            self.name_tables[physical * 0x400 + off] = value
            self._invalidate_scanline_bg_cache(physical, off)
            if self._fast_nrom:
                self._mark_nrom_dirty(nt, off)
            else:
                self._background_dirty = True
            return
        if self._deferred_render_enabled:
            self.flush_render_queue()
        a = (addr - 0x3F00) & 0x1F
        if a in (0x10, 0x14, 0x18, 0x1C):
            a -= 0x10
        self.palette[a] = value & 0x3F
        self._palette_generation += 1
        self._bg_row_cache = [None] * (2 * 256 * 8 * 4)
        self._scanline_bg_cache.clear()
        self._horizontal_bg_cache.clear()
        self._active_bg_row_cache_key = None
        self._active_bg_row_data = None
        if self._fast_nrom:
            self._world_needs_full = True
            self._world_dirty_tiles.clear()
        else:
            self._background_ready = False

    def _mark_nrom_dirty(self, nt, off):
        if self._world_needs_full:
            return
        phys = self._nt_index(nt)
        logical = [i for i in range(4) if self._nt_index(i) == phys]
        if off < 0x3C0:
            tx = off & 31
            ty = (off >> 5) % 30
            for ln in logical:
                self._world_dirty_tiles.add(((ln & 1) * 32 + tx,
                                             ((ln >> 1) * 30) + ty))
        else:
            k = (off - 0x3C0) & 0x3F
            ax = k & 7
            ay = k >> 3
            x0 = ax * 4
            y0 = ay * 4
            for ln in logical:
                wx0 = (ln & 1) * 32 + x0
                wy0 = (ln >> 1) * 30 + y0
                for ty in range(wy0, wy0 + 4):
                    for tx in range(wx0, wx0 + 4):
                        self._world_dirty_tiles.add((tx, ty))

    def _render_nrom_tile(self, wx_tile, wy_tile):
        world = self._world
        opaque = self._world_opaque
        ntmem = self.name_tables
        mirror = self.cart.mirror_nt
        colors = self.colors
        pal = self.palette
        table = 0x1000 if (self.ppuctrl & 0x10) else 0
        nt_x = 1 if wx_tile >= 32 else 0
        nt_y = 1 if wy_tile >= 30 else 0
        tx = wx_tile & 31
        ty = wy_tile % 30
        nt = nt_x | (nt_y << 1)
        base = mirror(nt) * 0x400
        tile = ntmem[base + ty * 32 + tx]
        attr = ntmem[base + 0x3C0 + ((ty >> 2) * 8) + (tx >> 2)]
        quadrant = (((ty & 2) != 0) << 1) | ((tx & 2) != 0)
        pset = (attr >> (quadrant * 2)) & 3
        pbase = pset << 2
        bg0 = colors[pal[0] & 0x3F]
        world_x = wx_tile * 8
        world_y = wy_tile * 8
        for fy in range(8):
            brow = self._pattern_row(tile, fy, table)
            rgb = [bg0] * 8
            op = bytearray(8)
            for j, cv in enumerate(brow):
                if cv:
                    rgb[j] = colors[pal[(pbase + cv) & 0x1F] & 0x3F]
                    op[j] = 1
            pos = (world_y + fy) * 512 + world_x
            world[pos:pos + 8] = rgb
            opaque[pos:pos + 8] = op

    def _pattern_row(self, tile, fine_y, table):
        key = (table + (tile & 0xFF) * 16 + fine_y) & 0x1FFF
        cached = self._pattern_row_cache[key]
        if cached is not None:
            return cached
        lo = self.cart.ppu_read(key) or 0
        hi = self.cart.ppu_read((key + 8) & 0x1FFF) or 0
        row = bytes((
            (((hi >> 7) & 1) << 1) | ((lo >> 7) & 1),
            (((hi >> 6) & 1) << 1) | ((lo >> 6) & 1),
            (((hi >> 5) & 1) << 1) | ((lo >> 5) & 1),
            (((hi >> 4) & 1) << 1) | ((lo >> 4) & 1),
            (((hi >> 3) & 1) << 1) | ((lo >> 3) & 1),
            (((hi >> 2) & 1) << 1) | ((lo >> 2) & 1),
            (((hi >> 1) & 1) << 1) | ((lo >> 1) & 1),
            (((hi >> 0) & 1) << 1) | ((lo >> 0) & 1),
        ))
        if self._persistent_chr_cache:
            self._pattern_row_cache[key] = row
        return row

    def _pattern_row_addr(self, addr):
        key = addr & 0x1FF8
        cached = self._pattern_row_cache[key]
        if cached is not None:
            return cached
        lo = self.cart.ppu_read(key) or 0
        hi = self.cart.ppu_read((key + 8) & 0x1FFF) or 0
        row = bytes(((((hi >> b) & 1) << 1) | ((lo >> b) & 1)) for b in range(7, -1, -1))
        self._pattern_row_cache[key] = row
        return row

    def _bg_rgb_row(self, tile, fine_y, table, pset):
        key = (((table >> 12) & 1) * 256 + (tile & 0xFF)) * 32 + ((fine_y & 7) * 4) + (pset & 3)
        cached = self._bg_row_cache[key]
        if cached is not None:
            return cached
        brow = self._pattern_row(tile, fine_y, table)
        pal0 = self.colors[self.palette[0] & 0x3F]
        base = (pset & 3) << 2
        pal = self.palette
        colors = self.colors
        rgb = [pal0] * 8
        opaque = bytearray(8)
        for i in range(8):
            cv = brow[i]
            if cv:
                rgb[i] = colors[pal[base + cv] & 0x3F]
                opaque[i] = 1
        result = (tuple(rgb), bytes(opaque))
        self._bg_row_cache[key] = result
        return result

    def _bg_pixel(self, world_x, world_y):
        # Logical world is a 2x2 nametable area.
        nt_x = (world_x // 256) & 1
        nt_y = (world_y // 240) & 1
        nt = nt_x | (nt_y << 1)
        x = world_x & 255
        y = world_y % 240
        base = self._nt_index(nt) * 0x400
        tile = self.name_tables[base + (y // 8) * 32 + (x // 8)]
        fine_y = y & 7
        fine_x = x & 7
        table = 0x1000 if (self.ppuctrl & 0x10) else 0x0000
        p = tile * 16 + table
        key = p + fine_y
        pair = self._pattern_row_cache.get(key)
        if pair is None:
            lo = self.cart.ppu_read(key & 0x1FFF) or 0
            hi = self.cart.ppu_read((key + 8) & 0x1FFF) or 0
            pair = (lo, hi)
            self._pattern_row_cache[key] = pair
        else:
            lo, hi = pair
        bit = 7 - fine_x
        c = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1)
        if c == 0:
            return 0, False
        ax = (x // 32) & 7
        ay = (y // 32) & 7
        attr = self.name_tables[base + 0x3C0 + ay * 8 + ax]
        quadrant = (((y & 0x10) != 0) << 1) | ((x & 0x10) != 0)
        pal = (attr >> (quadrant * 2)) & 3
        return (pal << 2) | c, True

    def _palette_index(self, idx, sprite):
        if (idx & 3) == 0:
            return self.palette[0] & 0x3F
        p = (0x10 if sprite else 0) + idx
        return self.palette[p & 0x1F] & 0x3F

    def _rgb(self, palette_index):
        return self.colors[palette_index & 0x3F]

    @staticmethod
    def _sprite_row(tile, row, size16, ctrl):
        if size16:
            table = (tile & 1) * 0x1000
            tile &= 0xFE
            if row >= 8:
                tile += 1
                row -= 8
            return table + tile * 16 + row
        table = 0x1000 if (ctrl & 0x08) else 0
        return table + tile * 16 + row

    def _invalidate_opacity_for_current_line(self):
        """Force only the unrendered tail of the active visible row to refresh."""
        sy = self.scanline
        if 0 <= sy < 240:
            self._opacity_state_key[sy] = None

    def invalidate_sprite_cache(self):
        """Invalidate OAM-to-scanline and decoded per-line sprite pixels."""
        self._sprite_cache_dirty = True
        self._sprite_cache_generation += 1

    def _rebuild_sprite_line_cache(self):
        """Build visible-sprite lists in OAM order with one pass over OAM.

        This replaces 64 OAM checks on every scanline (15,360 checks per frame)
        with at most 64*8 range inserts after OAM DMA or a size-register change.
        """
        lines = [[] for _ in range(240)]
        overflow = bytearray(240)
        sprite_h = 16 if (self.ppuctrl & 0x20) else 8
        oam = self.oam
        for i in range(64):
            y = oam[i * 4] + 1  # raw Y=255 means top=256, fully off-screen
            if y >= 240 or y + sprite_h <= 0:
                continue
            for sy in range(max(0, y), min(240, y + sprite_h)):
                if len(lines[sy]) < 8:
                    lines[sy].append(i)
                else:
                    overflow[sy] = 1
        self._sprite_line_cache = [tuple(line) for line in lines]
        self._sprite_overflow_lines = overflow
        self._sprite_cache_dirty = False

    def _render_sprites_on_scanline(self, sy):
        """Compatibility helper: render the whole visible sprite scanline."""
        self._render_sprites_segment(sy, 0, 256)

    def _render_sprites_segment_reference(self, sy, x0, x1):
        """Reference renderer retained for tests and dynamic-CHR mapper fallback.

        Pixels already emitted are never repainted after a $2000/$2001/$2005/$2006
        write, which is important for split-scroll effects used by classic games.
        """
        x0 = max(0, int(x0))
        x1 = min(256, int(x1))
        if x1 <= x0 or not (self.ppumask & 0x10):
            return
        if self._sprite_cache_dirty:
            self._rebuild_sprite_line_cache()
        if self._sprite_overflow_lines[sy]:
            self.ppustatus |= 0x20

        visible = self._sprite_line_cache[sy]
        size16 = bool(self.ppuctrl & 0x20)
        sprite_h = 16 if size16 else 8
        oam = self.oam
        frame = self.frame
        bg = self.bg_opaque
        colors = self.colors
        pal = self.palette
        ctrl = self.ppuctrl
        mask = self.ppumask
        row_start = sy * 256

        # OAM index 0 has highest priority, so paint it last.
        for i in reversed(visible):
            y = oam[i * 4] + 1
            tile = oam[i * 4 + 1]
            attr = oam[i * 4 + 2]
            sx = oam[i * 4 + 3]
            row = sy - y
            if attr & 0x80:
                row = sprite_h - 1 - row
            addr = self._sprite_row(tile, row, size16, ctrl)
            lo = self.cart.ppu_read(addr) or 0
            hi = self.cart.ppu_read((addr + 8) & 0x1FFF) or 0
            left = max(x0, sx)
            right = min(x1, sx + 8)
            if right <= left:
                continue
            for px in range(left, right):
                if px < 8 and not (mask & 0x04):
                    continue
                col = px - sx
                bit = col if (attr & 0x40) else (7 - col)
                c = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1)
                if c == 0:
                    continue
                pos = row_start + px
                bg_opaque = bool(bg[pos])

                # Sprite-0 hit is a dot-timed event, not a result delayed until
                # the end of the scanline. It is independent of sprite priority.
                if (i == 0 and bg_opaque and (mask & 0x18) == 0x18
                        and px < 255 and (px >= 8 or (mask & 0x06) == 0x06)
                        and not (self.ppustatus & 0x40)):
                    self.sprite0_hit_x = px
                    self.sprite0_hit_y = sy
                    self.ppustatus |= 0x40

                if not (attr & 0x20) or not bg_opaque:
                    palette_index = 0x10 + ((attr & 3) * 4) + c
                    frame[pos] = colors[pal[palette_index & 0x1F] & 0x3F]

    def _build_sprite_pixel_line(self, sy):
        """Resolve sprite pixels for one scanline once instead of per CPU sync.

        On fixed-CHR mappers, games such as SMB can poll $2002 many times while
        the PPU advances only a few dots. Re-walking OAM/pattern bytes on every
        tiny segment was the dominant renderer cost. Cache the winning OAM pixel
        and its behind-BG bit; palette lookup remains live at draw time.
        """
        indices = bytearray(256)  # 0 = transparent; sprite palette indices are 0x11..0x1f
        behind = bytearray(256)
        sprite0 = bytearray(256)
        visible = self._sprite_line_cache[sy]
        size16 = bool(self.ppuctrl & 0x20)
        sprite_h = 16 if size16 else 8
        oam = self.oam
        ppu_read = self.cart.ppu_read
        ctrl = self.ppuctrl

        # Iterate back-to-front, overwriting only with opaque pixels, so lower
        # OAM indices win overlap exactly as on the NES. Priority versus BG is
        # decided after the winning sprite pixel is chosen.
        for i in reversed(visible):
            base = i * 4
            y = oam[base] + 1
            tile = oam[base + 1]
            attr = oam[base + 2]
            sx = oam[base + 3]
            row = sy - y
            if attr & 0x80:
                row = sprite_h - 1 - row
            addr = self._sprite_row(tile, row, size16, ctrl)
            lo = ppu_read(addr) or 0
            hi = ppu_read((addr + 8) & 0x1FFF) or 0
            left = max(0, sx)
            right = min(256, sx + 8)
            palette_base = 0x10 + ((attr & 3) * 4)
            is_back = 1 if (attr & 0x20) else 0
            for px in range(left, right):
                col = px - sx
                bit = col if (attr & 0x40) else (7 - col)
                color = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1)
                if color == 0:
                    continue
                indices[px] = palette_base + color
                behind[px] = is_back
                if i == 0:
                    sprite0[px] = 1
        return bytes(indices), bytes(behind), bytes(sprite0)

    def _render_sprites_segment(self, sy, x0, x1):
        """Fast per-dot compositor using a cached sprite layer for this scanline."""
        x0 = max(0, int(x0))
        x1 = min(256, int(x1))
        mask = self.ppumask
        if x1 <= x0 or not (mask & 0x10):
            return
        if self._sprite_cache_dirty:
            self._rebuild_sprite_line_cache()
        if self._sprite_overflow_lines[sy]:
            self.ppustatus |= 0x20

        # Bank-switching CHR mappers use the reference path because their pattern
        # data can change without an OAM update. Fixed-CHR mappers, notably NROM,
        # can safely reuse the decoded line for all tiny PPU dot slices.
        if not self._persistent_chr_cache:
            self._render_sprites_segment_reference(sy, x0, x1)
            return

        cached = self._get_sprite_pixel_cache(sy)
        _, indices, behind, sprite0, opaque_positions = cached[:5]

        row_start = sy * 256
        frame = self.frame
        bg = self.bg_opaque
        palette = self.palette
        colors = self.colors
        hit_allowed = bool((mask & 0x18) == 0x18 and not (self.ppustatus & 0x40))
        left_clip = not (mask & 0x04)
        bg_left_visible = bool(mask & 0x02)

        # Visit only opaque sprite pixels (max ~64 on a normal visible row), not
        # all 256 screen pixels. The sparse list is built once per OAM/pattern
        # generation and is reused across frames.
        for px in opaque_positions:
            if px < x0:
                continue
            if px >= x1:
                break
            if px < 8 and left_clip:
                continue
            pos = row_start + px
            bg_opaque = bool(bg[pos])
            if (hit_allowed and sprite0[px] and bg_opaque and px < 255
                    and (px >= 8 or bg_left_visible)):
                self.sprite0_hit_x = px
                self.sprite0_hit_y = sy
                self.ppustatus |= 0x40
                hit_allowed = False
            palette_index = indices[px]
            if not behind[px] or not bg_opaque:
                frame[pos] = colors[palette[palette_index & 0x1F] & 0x3F]

    def _invalidate_scanline_bg_cache(self, physical_nt, off):
        """Invalidate cached decoded rows after a nametable/attribute write."""
        # A scroll update may touch many neighboring tiles. Clearing the modest
        # combined-row cache is faster than maintaining a reverse index here.
        self._horizontal_bg_cache.clear()
        self._active_bg_row_cache_key = None
        self._active_bg_row_data = None
        if off < 0x3C0:
            tile_rows = (off >> 5,)
        else:
            attr_row = (off - 0x3C0) >> 3
            first = attr_row * 4
            tile_rows = range(first, min(first + 4, 30))
        for tile_y in tile_rows:
            for fine_y in range(8):
                self._scanline_bg_cache.pop((physical_nt, tile_y, fine_y, 0), None)
                self._scanline_bg_cache.pop((physical_nt, tile_y, fine_y, 1), None)

    def _get_cached_background_row(self, physical_base, tile_y, fine_y, table_bit):
        """Return a 256-pixel RGB row and its opacity mask, rebuilding on demand."""
        physical_nt = physical_base >> 10
        key = (physical_nt, tile_y, fine_y, table_bit)
        cached = self._scanline_bg_cache.get(key)
        if cached is not None:
            return cached

        ntmem = self.name_tables
        pal = self.palette
        colors = self.colors
        tile_base = physical_base + tile_y * 32
        attr_base = physical_base + 0x3C0 + ((tile_y >> 2) * 8)
        table = table_bit * 0x1000
        bg0 = colors[pal[0] & 0x3F]
        rgb = [bg0] * 256
        opaque = bytearray(256)
        bgcache = self._bg_row_cache
        quadrant_y = 2 if (tile_y & 2) else 0
        for tx in range(32):
            tile = ntmem[tile_base + tx]
            attr = ntmem[attr_base + (tx >> 2)]
            quadrant = quadrant_y | (1 if (tx & 2) else 0)
            pset = (attr >> (quadrant * 2)) & 3
            ckey = (((table_bit * 256) + tile) * 32) + (fine_y * 4) + pset
            tile_cached = bgcache[ckey]
            if tile_cached is None:
                brow = self._pattern_row(tile, fine_y, table)
                pbase = pset << 2
                tile_rgb = [bg0] * 8
                tile_opaque = bytearray(8)
                for j in range(8):
                    cv = brow[j]
                    if cv:
                        tile_rgb[j] = colors[pal[(pbase + cv) & 0x1F] & 0x3F]
                        tile_opaque[j] = 1
                tile_cached = (tuple(tile_rgb), bytes(tile_opaque))
                bgcache[ckey] = tile_cached
            tile_rgb, tile_opaque = tile_cached
            start = tx * 8
            rgb[start:start + 8] = tile_rgb
            opaque[start:start + 8] = tile_opaque

        cached = (tuple(rgb), bytes(opaque))
        self._scanline_bg_cache[key] = cached
        return cached

    def _get_horizontal_background_row(self, nt_y, tile_y, fine_y, table_bit):
        """Return RGB/opacity for both side-by-side nametables as one world row."""
        mirror = self.cart.mirror_nt
        left_nt = mirror(nt_y << 1)
        right_nt = mirror(1 | (nt_y << 1))
        key = (left_nt, right_nt, tile_y, fine_y, table_bit)
        cached = self._horizontal_bg_cache.get(key)
        if cached is not None:
            return cached
        left_rgb, left_opaque = self._get_cached_background_row(
            left_nt * 0x400, tile_y, fine_y, table_bit
        )
        right_rgb, right_opaque = self._get_cached_background_row(
            right_nt * 0x400, tile_y, fine_y, table_bit
        )
        cached = (left_rgb + right_rgb, left_opaque + right_opaque)
        self._horizontal_bg_cache[key] = cached
        return cached

    def _render_background_scanline(self, sy):
        """Compatibility helper: render a full visible background scanline."""
        self._render_background_segment(sy, 0, 256)

    def _background_source_for_segment(self, sy, x0):
        table_bit = 1 if (self.ppuctrl & 0x10) else 0
        state = (sy, self.frame_nt, self.frame_scroll_y, table_bit)
        if state != self._active_bg_row_cache_key:
            world_y = (((self.frame_nt >> 1) & 1) * 240 + self.frame_scroll_y + sy) % 480
            nt_y = 1 if world_y >= 240 else 0
            local_y = world_y - (240 if nt_y else 0)
            self._active_bg_row_data = self._get_horizontal_background_row(
                nt_y, local_y >> 3, local_y & 7, table_bit
            )
            self._active_bg_row_cache_key = state
        rgb_row, opacity_row = self._active_bg_row_data
        world_x = ((((self.frame_nt & 1) * 256) + self.frame_scroll_x + x0) & 0x1FF)
        return rgb_row, opacity_row, world_x

    def _copy_background_opacity(self, sy, x0, x1, opacity_row, world_x):
        row_start = sy * 256
        dst = row_start + x0
        count = x1 - x0
        end_x = world_x + count
        bg = self.bg_opaque
        if end_x <= 512:
            bg[dst:dst + count] = opacity_row[world_x:end_x]
        else:
            first = 512 - world_x
            bg[dst:dst + first] = opacity_row[world_x:]
            rest = count - first
            bg[dst + first:dst + count] = opacity_row[:rest]
        if x0 < 8 and not (self.ppumask & 0x02):
            clip_end = min(8, x1)
            if clip_end > x0:
                bg[row_start + x0:row_start + clip_end] = bytes(clip_end - x0)

    def _render_background_opacity_segment(self, sy, x0, x1):
        """Maintain timely BG opacity without redrawing it for every CPU poll.

        Busy loops around PPUSTATUS can call this thousands of times per frame,
        often advancing only a few dots. Since the current state's opacity is
        known for the remaining pixels, fill the row tail once per state. A
        scroll/mask/CHR/nametable change produces a new key and only repaints
        pixels at or after the current dot. Sprite-0 is still tested only in
        the current [x0, x1) interval, so this look-ahead never raises status
        early.
        """
        x0 = max(0, int(x0)); x1 = min(256, int(x1))
        if x1 <= x0:
            return
        if not self._deferred_render_enabled:
            if not (self.ppumask & 0x08):
                self.bg_opaque[sy * 256 + x0:sy * 256 + x1] = bytes(x1 - x0)
                return
            _, opacity_row, world_x = self._background_source_for_segment(sy, x0)
            self._copy_background_opacity(sy, x0, x1, opacity_row, world_x)
            return

        if self._opacity_state_key[sy] is not None:
            return
        # None means either this line has just started or a rendering-state / VRAM
        # write invalidated it. Fill only from the current dot to the right edge.
        row_start = sy * 256
        if not (self.ppumask & 0x08):
            self.bg_opaque[row_start + x0:row_start + 256] = bytes(256 - x0)
            self._opacity_state_key[sy] = True
            return
        _, opacity_row, world_x = self._background_source_for_segment(sy, x0)
        self._copy_background_opacity(sy, x0, 256, opacity_row, world_x)
        self._opacity_state_key[sy] = True

    def _get_sprite_pixel_cache(self, sy):
        """Return the resolved sprite row and its sparse opaque-pixel index."""
        cached = self._sprite_pixel_cache[sy]
        generation = self._sprite_cache_generation
        if cached is None or cached[0] != generation or len(cached) < 7:
            decoded = self._build_sprite_pixel_line(sy)
            sprite0 = decoded[2]
            positions = tuple(px for px, color in enumerate(decoded[0]) if color)
            cached = (generation, *decoded, positions,
                      sprite0.find(b"\x01"), sprite0.rfind(b"\x01"))
            self._sprite_pixel_cache[sy] = cached
        return cached

    def _check_sprite0_hit_segment(self, sy, x0, x1):
        """Latch sprite-0 hit at emulated time without compositing RGB yet."""
        mask = self.ppumask
        if ((mask & 0x18) != 0x18 or (self.ppustatus & 0x40)
                or x1 <= x0 or not self._persistent_chr_cache):
            return
        if self._sprite_cache_dirty:
            self._rebuild_sprite_line_cache()
        visible = self._sprite_line_cache[sy]
        # OAM is examined in priority order. If sprite 0 is not the first
        # visible entry on this line, it cannot produce a Sprite-0 hit here.
        if not visible or visible[0] != 0:
            return
        cached = self._get_sprite_pixel_cache(sy)
        sprite0 = cached[3]
        first_sprite0, last_sprite0 = cached[5], cached[6]
        if first_sprite0 < 0 or last_sprite0 < x0 or first_sprite0 >= x1:
            return
        start = max(x0, 0 if (mask & 0x06) == 0x06 else 8, first_sprite0)
        end = min(x1, 255, last_sprite0 + 1)  # dot 255 never sets Sprite-0 Hit
        while start < end:
            px = sprite0.find(b"\x01", start, end)
            if px < 0:
                return
            if self.bg_opaque[sy * 256 + px]:
                self.sprite0_hit_x = px
                self.sprite0_hit_y = sy
                self.ppustatus |= 0x40
                return
            start = px + 1

    def _queue_render_segment(self, sy, x0, x1):
        """Queue RGB composition and merge contiguous unchanged-state segments."""
        if self._render_queue:
            last = self._render_queue[-1]
            if last[0] == sy and last[2] == x0:
                last[2] = x1
                return
        self._render_queue.append([sy, x0, x1])

    def flush_render_queue(self):
        """Compose queued RGB spans before a rendering-state mutation or display."""
        if not self._render_queue:
            return
        queued = self._render_queue
        self._render_queue = []
        for sy, x0, x1 in queued:
            self._render_background_segment(sy, x0, x1)
            self._render_sprites_segment(sy, x0, x1)

    def _render_background_segment(self, sy, x0, x1):
        """Render RGB and BG opacity for pixels [x0, x1)."""
        x0 = max(0, int(x0)); x1 = min(256, int(x1))
        if x1 <= x0:
            return
        mask = self.ppumask
        frame = self.frame
        bg = self.bg_opaque
        palette = self.palette
        colors = self.colors
        row_start = sy * 256
        count = x1 - x0
        if not (mask & 0x08):
            color = colors[palette[0] & 0x3F]
            frame[row_start + x0:row_start + x1] = [color] * count
            bg[row_start + x0:row_start + x1] = bytes(count)
            return
        rgb_row, opacity_row, world_x = self._background_source_for_segment(sy, x0)
        dst = row_start + x0
        end_x = world_x + count
        if end_x <= 512:
            frame[dst:dst + count] = rgb_row[world_x:end_x]
            bg[dst:dst + count] = opacity_row[world_x:end_x]
        else:
            first = 512 - world_x
            frame[dst:dst + first] = rgb_row[world_x:]
            bg[dst:dst + first] = opacity_row[world_x:]
            rest = count - first
            frame[dst + first:dst + count] = rgb_row[:rest]
            bg[dst + first:dst + count] = opacity_row[:rest]
        if x0 < 8 and not (mask & 0x02):
            clip_end = min(8, x1)
            if clip_end > x0:
                color = colors[palette[0] & 0x3F]
                start = row_start + x0
                stop = row_start + clip_end
                frame[start:stop] = [color] * (stop - start)
                bg[start:stop] = bytes(stop - start)

    def _render_background_frame(self):
        if self._fast_nrom:
            self._render_nrom_world()
            self._copy_nrom_viewport()
            return
        # Fallback renderer for other mappers.
        if not self._background_dirty and self._background_ready:
            return
        for sy in range(240):
            self._render_background_scanline(sy)
        self._background_dirty = False
        self._background_ready = True

    def _update_scroll_snapshot(self):
        self.frame_scroll_x = ((self.t & 0x1F) << 3) | self.fine_x
        self.frame_scroll_y = (((self.t >> 5) & 0x1F) << 3) | ((self.t >> 12) & 7)
        self.frame_nt = (self.t >> 10) & 3
        self.frame_scroll_x &= 0x1FF
        self.frame_scroll_y %= 480
        self._frame_bg_table = 1 if (self.ppuctrl & 0x10) else 0

    def _capture_scroll(self):
        self.flush_render_queue()
        self._update_scroll_snapshot()
        self.sprite0_hit_x = None
        self.sprite0_hit_y = None

    def advance(self, ticks):
        """Advance PPU time in bulk while preserving scanline events.

        Rendering happens once per visible scanline, while VBlank/NMI and the
        pre-render register effects occur at their documented cycle positions.
        This removes hundreds of thousands of tiny Python method calls per frame.
        """
        ticks = int(ticks)
        if ticks <= 0:
            return

        if self.frame_complete:
            self._carry_ticks += ticks
            return
        remaining = ticks + self._carry_ticks
        self._carry_ticks = 0

        # Mapper-0 games often execute BIT $2002 loops, advancing the PPU by a
        # handful of dots on each status read. Avoid the general multi-line loop
        # when this call stays inside the current scanline. Event boundaries,
        # visible pixels, sprite-0 timing, VBlank/NMI, and pre-render effects are
        # still handled at their exact crossing points.
        if self._deferred_render_enabled and remaining < (341 - self.cycle):
            if 0 <= self.scanline < 240 and self.cycle == 0:
                self._opacity_state_key[self.scanline] = None
                if self._sprite_cache_dirty and (self.ppumask & 0x18):
                    self._rebuild_sprite_line_cache()
                if (self.ppumask & 0x18) and self._sprite_overflow_lines[self.scanline]:
                    self.ppustatus |= 0x20
                self.cart.scanline_tick()
            old_cycle = self.cycle
            new_cycle = old_cycle + remaining
            if 0 <= self.scanline < 240 and old_cycle < 256:
                x0 = old_cycle
                x1 = min(256, new_cycle)
                if x1 > x0:
                    if self._opacity_state_key[self.scanline] is None:
                        self._render_background_opacity_segment(self.scanline, x0, x1)
                    if not (self.ppustatus & 0x40):
                        if self._sprite_cache_dirty:
                            self._rebuild_sprite_line_cache()
                        line_sprites = self._sprite_line_cache[self.scanline]
                        if line_sprites and line_sprites[0] == 0:
                            self._check_sprite0_hit_segment(self.scanline, x0, x1)
                    self._queue_render_segment(self.scanline, x0, x1)
            if self.scanline == 241 and old_cycle < 1 <= new_cycle:
                self.ppustatus |= 0x80
                if self.ppuctrl & 0x80:
                    self.nmi = True
            if self.scanline == -1 and old_cycle < 1 <= new_cycle:
                self.ppustatus &= ~0xE0
                self._capture_scroll()
            self.cycle = new_cycle
            return

        while remaining > 0:
            if 0 <= self.scanline < 240 and self.cycle == 0:
                self._opacity_state_key[self.scanline] = None
                # Sprite evaluation / mapper scanline events are latched at the
                # beginning of the visible line. Pixels themselves are emitted
                # incrementally below so mid-scanline register writes take effect.
                if self.scanline == 0 and not self._persistent_chr_cache:
                    self._pattern_row_cache = [None] * 0x2000
                if self._sprite_cache_dirty and (self.ppumask & 0x18):
                    self._rebuild_sprite_line_cache()
                if (self.ppumask & 0x18) and self._sprite_overflow_lines[self.scanline]:
                    self.ppustatus |= 0x20
                self.cart.scanline_tick()

            old_cycle = self.cycle
            step = min(remaining, 341 - self.cycle)
            new_cycle = self.cycle + step

            # NES visible pixels map to dots 1..256. Render exactly the dots
            # covered by this advancement before any later CPU register write.
            if 0 <= self.scanline < 240 and old_cycle < 256:
                x0 = max(0, old_cycle)
                x1 = min(256, new_cycle)
                if x1 > x0:
                    if self._deferred_render_enabled:
                        # Preserve status-bit timing immediately, but merge final
                        # RGB drawing across repeated $2002 polling slices.
                        if self._opacity_state_key[self.scanline] is None:
                            self._render_background_opacity_segment(self.scanline, x0, x1)
                        if not (self.ppustatus & 0x40):
                            if self._sprite_cache_dirty:
                                self._rebuild_sprite_line_cache()
                            line_sprites = self._sprite_line_cache[self.scanline]
                            if line_sprites and line_sprites[0] == 0:
                                self._check_sprite0_hit_segment(self.scanline, x0, x1)
                        self._queue_render_segment(self.scanline, x0, x1)
                    else:
                        self._render_background_segment(self.scanline, x0, x1)
                        self._render_sprites_segment(self.scanline, x0, x1)

            # VBlank begins at scanline 241, cycle 1.
            if self.scanline == 241 and old_cycle < 1 <= new_cycle:
                self.ppustatus |= 0x80
                if self.ppuctrl & 0x80:
                    self.nmi = True

            # Pre-render line clears status and captures the scroll registers.
            if self.scanline == -1 and old_cycle < 1 <= new_cycle:
                self.ppustatus &= ~0xE0
                self._capture_scroll()

            self.cycle = new_cycle
            remaining -= step

            if self.cycle >= 341:
                self.cycle = 0
                self.scanline += 1
                if self.scanline >= 261:
                    self.scanline = -1
                    self.frame_complete = True
                    self.flush_render_queue()
                    # Do not let this batch redraw the beginning of the next
                    # frame into the one which is about to be presented.
                    if remaining > 0:
                        self._carry_ticks += remaining
                    break

    def clock(self):
        # Compatibility path for diagnostics that clock one PPU cycle at a time.
        self.advance(1)

    def consume_nmi(self):
        if self.nmi:
            self.nmi = False
            return True
        return False
