from __future__ import annotations

import base64
import json
import os
import sys
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

from nes import NESBus

W, H = 256, 240
SCALE = int(os.getenv("NES_SCALE", "1"))
TARGET_FPS = 60.0
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
# Keep GUI callbacks short so key press/release events are handled promptly.
# Shorter bounded slices let Tk dispatch pending paint/input/window messages
# while a frame is still being emulated. This is adjustable for very weak PCs.
INPUT_SLICE_INSTRUCTIONS = max(800, min(12000, int(os.getenv("NES_INPUT_SLICE", "6000"))))
# A zero-delay callback chain can starve Tk idle redraws on Windows/Tk 8.6.
# Yield at least 1 ms between slices so expose/paint events are serviced even
# when the user does not click a menu or another widget.
GUI_SLICE_DELAY_MS = 1
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls.json")

DEFAULT_P1 = {
    "A": "z", "B": "x", "SELECT": "shift_r", "START": "return",
    "UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right",
}
DEFAULT_P2 = {
    "A": "c", "B": "v", "SELECT": "b", "START": "n",
    "UP": "w", "DOWN": "s", "LEFT": "a", "RIGHT": "d",
}

DISPLAY_NAMES = {
    "up": "↑ Up", "down": "↓ Down", "left": "← Left", "right": "→ Right",
    "return": "Enter", "shift_r": "Right Shift", "shift_l": "Left Shift",
    "space": "Space", "control_l": "Left Ctrl", "control_r": "Right Ctrl",
    "alt_l": "Left Alt", "alt_r": "Right Alt", "escape": "Esc", "tab": "Tab",
}
BUTTON_ORDER = ["A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]


def load_controls():
    data = {"P1": DEFAULT_P1.copy(), "P2": DEFAULT_P2.copy()}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for player in ("P1", "P2"):
            if isinstance(saved.get(player), dict):
                for button in BUTTON_ORDER:
                    key = saved[player].get(button)
                    if isinstance(key, str) and key:
                        data[player][button] = key
    except (OSError, ValueError, TypeError):
        pass
    return data


def save_controls(data):
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_FILE)


def pretty_key(keysym: str) -> str:
    return DISPLAY_NAMES.get(keysym, keysym.upper() if len(keysym) == 1 else keysym)


# RGB byte cache for the fixed NES palette.
NES_PALETTE_RGB = [
    0x545454, 0x001E74, 0x081090, 0x300088, 0x440064, 0x5C0030, 0x540400, 0x3C1800,
    0x202A00, 0x083A00, 0x004200, 0x00402C, 0x00323C, 0x000000, 0x000000, 0x000000,
    0x989698, 0x084CC4, 0x3032EC, 0x5C1EE4, 0x8814B0, 0xA01464, 0x982220, 0x783800,
    0x545000, 0x286400, 0x0C6C34, 0x0C6058, 0x08505C, 0x000000, 0x000000, 0x000000,
    0xECEEEC, 0x4C9AEC, 0x787CEC, 0xB062E8, 0xD454C8, 0xE04C78, 0xD86834, 0xB07C24,
    0x888A24, 0x50A838, 0x20B05C, 0x20A090, 0x208890, 0x3C3C3C, 0x000000, 0x000000,
    0xECEEEC, 0xA8CCEC, 0xBCBCEC, 0xD0B2EC, 0xE8A8E4, 0xECA8B0, 0xECA890, 0xE0BC74,
    0xCCD078, 0xB0D080, 0x90D0A0, 0x90CCC0, 0x90C8CC, 0xA0A2A0, 0x000000, 0x000000,
]


class NESApp:
    ppm_cache = {c: bytes(((c >> 16) & 255, (c >> 8) & 255, c & 255)) for c in NES_PALETTE_RGB}

    def __init__(self, root):
        self.root = root
        self.root.title("Python NES Emulator — NES Compatibility Core")
        self.root.resizable(False, False)
        self.scale = max(1, min(4, SCALE if SCALE > 0 else 2))
        self.canvas = tk.Canvas(root, width=W * self.scale, height=H * self.scale, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.configure(takefocus=True)
        self.native_photo = tk.PhotoImage(width=W, height=H)
        self.photo = self.native_photo if self.scale == 1 else tk.PhotoImage(width=W * self.scale, height=H * self.scale)
        self.image_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.image = self.photo

        self.info = tk.StringVar(value="Ctrl+O = Open ROM | F5 = Reset | F6 = Pause | F7 = Audio | F8 = Controls | Mapper 0 compatibility mode")
        tk.Label(root, textvariable=self.info, anchor="w").pack(fill="x")
        self.input_info = tk.StringVar(value="P1: ↑↓←→  Z=A  X=B  Enter=Start  Right Shift=Select")
        tk.Label(root, textvariable=self.input_info, anchor="w", fg="#444").pack(fill="x")

        self.nes = None
        self.running = False
        self.paused = False
        self.keys = set()
        self.frames = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0
        self.next_frame_time = time.perf_counter()
        self.frame_started = False
        self.audio_enabled = False
        self._after_id = None
        controls = load_controls()
        self.p1 = controls["P1"]
        self.p2 = controls["P2"]
        self.rebuild_reverse_maps()

        # bind_all captures keys even when a menu, label or canvas has focus.
        # This fixes the classic Tk issue where arrows/Enter appear to work only
        # intermittently or seem to be mapped onto another button.
        root.bind_all("<KeyPress>", self.key_down, add="+")
        root.bind_all("<KeyRelease>", self.key_up, add="+")
        root.bind_all("<FocusOut>", self.clear_keys_on_focus_loss, add="+")
        root.bind("<Control-o>", lambda e: self.open_rom())
        root.bind("<F1>", lambda e: self.open_rom())
        root.bind("<F5>", lambda e: self.reset())
        root.bind("<F6>", lambda e: self.toggle_pause())
        root.bind("<F7>", lambda e: self.toggle_audio())
        root.bind("<F8>", lambda e: self.open_controls())
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        for key, scale in (("1",1),("2",2),("3",3),("4",4)):
            root.bind(f"<Alt-KeyPress-{key}>", lambda e, s=scale: self.set_scale(s))
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.make_menu()

    def rebuild_reverse_maps(self):
        self.reverse_p1 = {key: button for button, key in self.p1.items()}
        self.reverse_p2 = {key: button for button, key in self.p2.items()}

    def make_menu(self):
        menu = tk.Menu(self.root)
        filem = tk.Menu(menu, tearoff=0)
        filem.add_command(label="Open .NES...", command=self.open_rom, accelerator="Ctrl+O")
        filem.add_command(label="Reset", command=self.reset, accelerator="F5")
        filem.add_separator()
        filem.add_command(label="Exit", command=self.close)
        menu.add_cascade(label="File", menu=filem)

        emu = tk.Menu(menu, tearoff=0)
        emu.add_command(label="Pause / Resume", command=self.toggle_pause, accelerator="F6")
        emu.add_command(label="Audio on/off", command=self.toggle_audio, accelerator="F7")
        emu.add_command(label="Controls / Key Mapping", command=self.open_controls, accelerator="F8")
        emu.add_command(label="Fullscreen", command=self.toggle_fullscreen, accelerator="F11")
        menu.add_cascade(label="Emulator", menu=emu)
        display = tk.Menu(menu, tearoff=0)
        for scale, label in ((1,"1× (256×240)"),(2,"2× (512×480)"),(3,"3× (768×720)"),(4,"4× (1024×960)")):
            display.add_command(label=label, command=lambda s=scale: self.set_scale(s))
        display.add_command(label="Fullscreen", command=self.toggle_fullscreen, accelerator="F11")
        menu.add_cascade(label="Resolution / Scale", menu=display)
        self.root.config(menu=menu)

    def apply_scale(self, scale):
        self.scale = max(1, min(4, int(scale)))
        self.canvas.config(width=W * self.scale, height=H * self.scale)
        self.photo = self.native_photo if self.scale == 1 else tk.PhotoImage(width=W * self.scale, height=H * self.scale)
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        self.canvas.image = self.photo
        self.root.update_idletasks()

    def set_scale(self, scale):
        self.apply_scale(scale)
        self.info.set(self.info.get().split(" | Scale=")[0] + f" | Scale={self.scale}× ({W*self.scale}×{H*self.scale})")

    def toggle_fullscreen(self):
        state = not bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", state)
        if state:
            self.root.update_idletasks()
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            fit = min(sw // W, sh // H)
            self.apply_scale(max(1, min(4, fit)))

    def controls_summary(self):
        return (
            f"P1: A={pretty_key(self.p1['A'])} B={pretty_key(self.p1['B'])} "
            f"Start={pretty_key(self.p1['START'])} Select={pretty_key(self.p1['SELECT'])} | "
            f"P2: A={pretty_key(self.p2['A'])} B={pretty_key(self.p2['B'])} "
            f"Start={pretty_key(self.p2['START'])} Select={pretty_key(self.p2['SELECT'])}"
        )

    def open_controls(self):
        win = tk.Toplevel(self.root)
        win.title("NES Controls / Key Mapping")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text="برای تغییر هر دکمه روی Change کلیک کن، سپس کلید دلخواه را بزن.",
                 font=("Segoe UI", 10)).grid(row=0, column=0, columnspan=4, padx=12, pady=(12, 8))

        vars_by_player = {"P1": {}, "P2": {}}
        waiting = {"player": None, "button": None, "button_widget": None}

        for col, player in enumerate(("P1", "P2")):
            frame = tk.LabelFrame(win, text="Player 1" if player == "P1" else "Player 2", padx=10, pady=8)
            frame.grid(row=1, column=col, padx=10, pady=8, sticky="nsew")
            for row, button in enumerate(BUTTON_ORDER):
                tk.Label(frame, text=button, width=9, anchor="w").grid(row=row, column=0, padx=(0, 6), pady=3)
                var = tk.StringVar(value=pretty_key(getattr(self, player.lower())[button]))
                vars_by_player[player][button] = var
                tk.Label(frame, textvariable=var, width=13, relief="sunken", anchor="w").grid(row=row, column=1, pady=3)

                def change_key(p=player, b=button):
                    waiting["player"] = p
                    waiting["button"] = b
                    if waiting["button_widget"]:
                        waiting["button_widget"].config(text="PRESS A KEY...", state="disabled")
                    messagebox.showinfo("Key Mapping", f"کلید جدید برای {p} / {b} را بزن.", parent=win)
                    win.focus_force()

                btn = tk.Button(frame, text="Change", width=9, command=change_key)
                btn.grid(row=row, column=2, padx=(6, 0), pady=3)
                if button == BUTTON_ORDER[-1]:
                    pass

        def controls_key(event):
            if waiting["player"] is None:
                return
            key = event.keysym.lower()
            p, b = waiting["player"], waiting["button"]
            target = self.p1 if p == "P1" else self.p2
            other_player = self.p2 if p == "P1" else self.p1
            used = {k: btn for btn, k in target.items() if btn != b}
            if key in used:
                messagebox.showwarning("Key Mapping", f"این کلید قبلاً برای {p} / {used[key]} استفاده شده است.", parent=win)
                return
            other_used = {k: btn for btn, k in other_player.items()}
            if key in other_used:
                answer = messagebox.askyesno(
                    "Key Mapping",
                    f"کلید {pretty_key(key)} در {('P2' if p == 'P1' else 'P1')} / {other_used[key]} هم استفاده شده.\n\nادامه بده؟",
                    parent=win,
                )
                if not answer:
                    return
            target[b] = key
            vars_by_player[p][b].set(pretty_key(key))
            self.rebuild_reverse_maps()
            waiting["player"] = None
            waiting["button"] = None

        def save_and_close():
            try:
                save_controls({"P1": self.p1, "P2": self.p2})
                self.info.set(self.info.get().split(" | P1:")[0] + " | " + self.controls_summary())
                win.destroy()
            except OSError as e:
                messagebox.showerror("NES Emulator", f"ذخیره کنترل‌ها ناموفق بود:\n{e}", parent=win)

        def reset_defaults():
            self.p1.clear(); self.p1.update(DEFAULT_P1)
            self.p2.clear(); self.p2.update(DEFAULT_P2)
            self.rebuild_reverse_maps()
            for p, source in (("P1", self.p1), ("P2", self.p2)):
                for b in BUTTON_ORDER:
                    vars_by_player[p][b].set(pretty_key(source[b]))

        win.bind("<KeyPress>", controls_key)
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        bottom = tk.Frame(win)
        bottom.grid(row=2, column=0, columnspan=2, pady=(2, 12))
        tk.Button(bottom, text="Restore Defaults", command=reset_defaults).pack(side="left", padx=5)
        tk.Button(bottom, text="Save", width=12, command=save_and_close).pack(side="left", padx=5)
        tk.Button(bottom, text="Cancel", width=12, command=win.destroy).pack(side="left", padx=5)

        win.focus_force()

    def create_core(self, path):
        # Compatibility-first core: use the complete CPU/PPU implementation.
        # This is especially important for Super Mario Bros., which depends on
        # normal NES PPU register timing and sprite-0/VBlank behavior.
        return NESBus(path, audio_enabled=self.audio_enabled)

    def using_native(self):
        return False

    def open_rom(self, initial_path=None):
        path = initial_path or filedialog.askopenfilename(
            title="انتخاب بازی NES",
            filetypes=[("NES ROM", "*.nes"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            if self.nes:
                self.nes.close()
            self.nes = self.create_core(path)
            self.running = True
            self.paused = False
            self.keys.clear()
            self.next_frame_time = time.perf_counter()
            self.frame_started = False
            self.root.focus_force()
            self.canvas.focus_set()
            self.info.set(
                f"{os.path.basename(path)} | {self.nes.cart.description} | {self.controls_summary()} | "
                "F5 Reset | F6 Pause | F7 Audio | F8 Controls"
            )
            self.schedule()
        except Exception as e:
            self.running = False
            self._show_error(e, f"باز کردن ROM:\n{path}")

    def reset(self):
        if self.nes:
            self.nes.reset()
            self.paused = False
            self.frame_started = False
            self.next_frame_time = time.perf_counter()

    def toggle_pause(self):
        if self.nes:
            self.paused = not self.paused
            self.info.set(self.info.get().split(" | PAUSED")[0] + (" | PAUSED" if self.paused else ""))

    def toggle_audio(self):
        self.audio_enabled = not self.audio_enabled
        if self.nes:
            path = self.nes.cart.path
            self.nes.close()
            self.nes = self.create_core(path)
        state = "ON" if self.audio_enabled else "OFF"
        self.info.set(self.info.get().split(" | Audio=")[0] + f" | Audio={state}")

    def _normalize_key(self, event):
        key = event.keysym.lower()
        # Tk uses Return/Shift_L/Shift_R; keep a few aliases so keyboard
        # layouts and synthetic key events do not break the mapping.
        aliases = {
            "return": "return", "enter": "return",
            "shift_l": "shift_l", "shift_r": "shift_r",
            "kp_enter": "return",
            "left": "left", "right": "right",
            "up": "up", "down": "down",
            "kp_left": "left", "kp_right": "right",
            "kp_up": "up", "kp_down": "down",
            "control_l": "control_l", "control_r": "control_r",
        }
        return aliases.get(key, key)

    def key_down(self, event):
        key = self._normalize_key(event)
        changed = key not in self.keys
        self.keys.add(key)
        # Push input to the emulated pads inside the input event itself instead
        # of waiting for the next emulation slice. This removes an avoidable
        # queue of up to one slice between a physical key event and controller
        # state becoming visible to the CPU.
        if changed and self.nes is not None:
            self.update_controllers()
        # Returning "break" prevents widgets/menus from consuming game keys.
        return "break"

    def key_up(self, event):
        key = self._normalize_key(event)
        was_down = key in self.keys
        self.keys.discard(key)
        if was_down and self.nes is not None:
            self.update_controllers()
        return "break"

    def clear_keys_on_focus_loss(self, event):
        if self.keys:
            self.keys.clear()
            if self.nes is not None:
                self.update_controllers()

    def update_controllers(self):
        if not self.nes:
            return
        p1 = [button for key, button in self.reverse_p1.items() if key in self.keys]
        p2 = [button for key, button in self.reverse_p2.items() if key in self.keys]
        self.nes.controller_buttons(p1, 1)
        self.nes.controller_buttons(p2, 2)

    @staticmethod
    def ppm_data(frame):
        # bytearray.extend is substantially faster than per-pixel slice writes.
        cache = NESApp.ppm_cache
        raw = b"".join(map(cache.__getitem__, frame))
        return b"P6\n256 240\n255\n" + raw

    def draw_frame(self):
        if not self.nes:
            return
        ppm = self.ppm_data(self.nes.ppu.frame)
        # Decode the 256x240 RGB frame once. Then use Tk's C-level PhotoImage
        # copy/zoom into a persistent destination image. Recreating a new zoomed
        # PhotoImage on every frame caused memory churn and visible stutter on
        # Tk 8.6 (common in Windows Python installations).
        if self.scale == 1:
            # At native size, use one persistent PhotoImage directly. Avoid an
            # unnecessary second Tcl image copy each frame on slower PCs.
            self.native_photo.configure(data=ppm, format="PPM")
        else:
            self.native_photo.configure(data=ppm, format="PPM")
            self.photo.tk.call(
                str(self.photo), "copy", str(self.native_photo),
                "-compositingrule", "set", "-zoom", self.scale, self.scale
            )
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        # Flush queued image/canvas redraw work now. This is idle-only (does not
        # recursively dispatch keyboard or timer callbacks) and avoids the
        # symptom where the game appears to advance only after clicking File.
        self.root.update_idletasks()

    def schedule(self):
        if self._after_id is None:
            self._after_id = self.root.after(0, self.loop)

    def loop(self):
        self._after_id = None
        if not self.running or not self.nes:
            return
        try:
            if self.paused:
                self.update_controllers()
                self._after_id = self.root.after(16, self.loop)
                return

            # Input is sampled on every tiny emulation slice, not once per full
            # frame. This makes arrow/Z/X/Enter response effectively immediate.
            self.update_controllers()

            if not self.frame_started:
                self.nes.begin_frame()
                self.frame_started = True

            # Bound GUI callback work more tightly. Controller state is also
            # pushed directly from key events, but shorter slices prevent a
            # long Python callback from delaying key-up/keyup processing and
            # window events on slower PCs. Tuneable via NES_INPUT_SLICE.
            completed = self.nes.run_frame_slice(max_instructions=INPUT_SLICE_INSTRUCTIONS)

            if completed:
                self.draw_frame()
                self.frames += 1
                self.frame_started = False

                stamp = time.perf_counter()
                if stamp - self.last_fps_time >= 0.5:
                    self.fps = self.frames / (stamp - self.last_fps_time)
                    self.frames = 0
                    self.last_fps_time = stamp
                    base = self.info.get().split(' | FPS=')[0]
                    self.info.set(base + f' | FPS={self.fps:.1f}')

                now = time.perf_counter()
                # Advance a fixed frame deadline. The old scheduler added a full
                # 16.7 ms *after* emulation had finished, effectively throttling
                # a 60 Hz loop to ~30 FPS whenever rendering took noticeable time.
                self.next_frame_time += TARGET_FRAME_TIME
                if self.next_frame_time < now:
                    # Avoid accumulating lag if a frame ran long. The next frame
                    # starts as soon as Tk can process it instead of waiting twice.
                    self.next_frame_time = now
                # Never chain a zero-delay frame callback: if we fell behind,
                # give Tk and the window manager one millisecond to paint/input.
                delay = max(1, int((self.next_frame_time - now) * 1000))
                self._after_id = self.root.after(delay, self.loop)
            else:
                # Continue this frame after a tiny positive delay. Repeated
                # after(0) callbacks can out-prioritize Tk idle redraw work, making
                # the visible game update only when a click opens a nested menu.
                self._after_id = self.root.after(GUI_SLICE_DELAY_MS, self.loop)
        except Exception as e:
            self.running = False
            self._show_error(e, 'اجرای ROM')

    def _show_error(self, exc, title):
        try:
            with open(os.path.join(os.path.dirname(__file__), "nes_error.log"), "a", encoding="utf-8") as f:
                f.write("\n=== " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                traceback.print_exc(file=f)
        except OSError:
            pass
        messagebox.showerror("NES Emulator", f"{title}\n\n{exc}\n\nجزئیات در nes_error.log ذخیره شد.\n\nمی‌توانی بازی دیگری را با Ctrl+O باز کنی.")

    def close(self):
        self.running = False
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        if self.nes:
            try:
                self.nes.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    try:
        root = tk.Tk()
    except Exception as e:
        print("خطا در اجرای Tkinter:", e)
        print("مطمئن شوید Python با Tcl/Tk نصب شده است.")
        return 2
    app = NESApp(root)
    if len(sys.argv) > 1:
        rom = os.path.abspath(sys.argv[1])
        if os.path.isfile(rom):
            root.after(100, lambda: app.open_rom(rom))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
