"""'CUBE is working' overlay: theme-matched CUBE logo + determinate progress bar + live status text + elapsed time.

Same pattern as the Behaviour Analyser's Group Predictor loading overlay (logo by theme, "underway - please wait",
progress bar, status line); worker threads never touch Tk - they post (fraction, text) to a queue that the UI thread
drains (see ``BusyOverlay.update`` callers in cube_guided_tab.py). The elapsed-time counter keeps ticking on its own,
so even a step that reports no progress for a minute visibly isn't frozen.
"""
from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

_HERE = Path(__file__).parent


def is_dark(theme: dict) -> bool:
    """Dark theme if the panel colour is dark (works for any theme dict with a hex 'panel' / 'bg')."""
    for key in ("panel", "bg"):
        c = str(theme.get(key, "")).lstrip("#")
        if len(c) == 6:
            try:
                r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
                return (0.299 * r + 0.587 * g + 0.114 * b) < 128
            except ValueError:
                continue
    return False


def logo_path(theme: dict) -> Path:
    return _HERE / ("CUBE_logo dark theme.png" if is_dark(theme) else "CUBE_logo.png")


class BusyOverlay(tk.Frame):
    def __init__(self, master, theme: dict):
        super().__init__(master, bg=theme["panel"])
        self.T = theme
        self._t0 = None
        self._after = None
        self._photo = None
        box = tk.Frame(self, bg=theme["panel"])
        box.place(relx=0.5, rely=0.45, anchor="center")
        self.logo = tk.Label(box, bg=theme["panel"], bd=0, highlightthickness=0)
        self.logo.pack(pady=(0, 26))
        self._load_logo()
        self.title = tk.Label(box, text="", bg=theme["panel"], fg=theme["text"], font=("Segoe UI", 13, "bold"))
        self.title.pack(pady=(0, 4))
        tk.Label(box, text="CUBE is working - this window will update when it is done. Please wait.",
                 bg=theme["panel"], fg=theme["text_dim"], font=("Segoe UI", 9)).pack(pady=(0, 16))
        self.bar = ttk.Progressbar(box, mode="determinate", length=440, maximum=1000)
        self.bar.pack(pady=(0, 8))
        self.status = tk.Label(box, text="", bg=theme["panel"], fg=theme["text_dim"], font=("Segoe UI", 10),
                               wraplength=460, justify="center")
        self.status.pack()
        self.elapsed = tk.Label(box, text="", bg=theme["panel"], fg=theme["text_dim"], font=("Segoe UI", 9))
        self.elapsed.pack(pady=(6, 0))
        self.fraction = 0.0

    def _load_logo(self):
        try:
            from PIL import Image, ImageTk
            p = logo_path(self.T)
            if p.is_file():
                im = Image.open(str(p))
                im.load()
                im.thumbnail((380, 170), Image.LANCZOS)
                # master= the owning toplevel: the annotator is a second Tk root inside CUBE, an image made in the
                # default root would be invisible here (same lesson as the Analyser's loading overlay)
                self._photo = ImageTk.PhotoImage(im.copy(), master=self.winfo_toplevel())
                self.logo.config(image=self._photo)
                return
        except Exception:                                           # noqa: BLE001
            pass
        self.logo.config(text="CUBE", font=("Segoe UI", 44, "bold"), fg=self.T["text_dim"])

    # ------------------------------------------------------------------ API
    def begin(self, title: str, text: str = ""):
        self.title.config(text=title)
        self.update_progress(0.0, text)
        self._t0 = time.time()
        self._tick()

    def update_progress(self, fraction, text: str | None = None):
        """fraction in [0, 1] (None leaves the bar where it is); text replaces the status line."""
        if fraction is not None:
            self.fraction = min(1.0, max(0.0, float(fraction)))
            try:
                self.bar["value"] = self.fraction * 1000
            except tk.TclError:
                return
        if text is not None:
            self.status.config(text=text)

    def end(self):
        if self._after is not None:
            try:
                self.after_cancel(self._after)
            except Exception:                                       # noqa: BLE001
                pass
            self._after = None
        self._t0 = None

    def _tick(self):
        if self._t0 is None:
            return
        s = int(time.time() - self._t0)
        try:
            self.elapsed.config(text=f"elapsed {s // 60}:{s % 60:02d}")
            self._after = self.after(500, self._tick)
        except tk.TclError:
            self._after = None
