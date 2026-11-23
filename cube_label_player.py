"""cube_label_player.py - seekable video player with a two-row label timeline.

Part of CUBE guided labelling (see GUIDED_LABELLING_CONTRACT.md section 4).

* ScrubLogic / SegmentModel are Tk-free (importable and testable headless).
* LabelPlayer(tk.Frame) is the widget: decode thread + queue + after() polling,
  seek via CAP_PROP_POS_FRAMES only on explicit seek (latest target wins).
  Tk is never touched from the worker thread.
"""
from __future__ import annotations

import copy
import queue
import threading
import time
from pathlib import Path

import numpy as np

try:  # GUI parts are optional so the logic classes import anywhere
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None

# --------------------------------------------------------------------------
# Pure logic
# --------------------------------------------------------------------------


class ScrubLogic:
    """Frame<->time conversion, clamping, wheel math and seek coalescing."""

    def __init__(self, n_frames: int, fps: float):
        self.n_frames = max(1, int(n_frames))
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._pending = None

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def clamp(self, frame) -> int:
        return int(min(max(int(round(frame)), 0), self.n_frames - 1))

    def frame_to_time(self, frame) -> float:
        return self.clamp(frame) / self.fps

    def time_to_frame(self, t: float) -> int:
        return self.clamp(round(t * self.fps))

    def step(self, frame, frames: int = 0, seconds: float = 0.0) -> int:
        return self.clamp(frame + frames + int(round(seconds * self.fps)))

    def wheel(self, frame, notches: int, shift: bool = False, ctrl: bool = False):
        """notches > 0 = scroll down/forward. Returns ("seek", frame) or
        ("zoom", view_length_multiplier). Ctrl wins over Shift."""
        if ctrl:
            return ("zoom", 1.25 ** notches)  # forward notch widens, back zooms in
        if shift:
            return ("seek", self.step(frame, seconds=float(notches)))
        return ("seek", self.step(frame, frames=int(notches)))

    # coalescing: only the newest requested frame is ever decoded
    def request(self, frame) -> int:
        self._pending = self.clamp(frame)
        return self._pending

    def take(self):
        p, self._pending = self._pending, None
        return p


class SegmentModel:
    """Behaviour segments in seconds with undo/redo. Later edit wins on overlap."""

    def __init__(self, fps: float, duration_s: float | None = None):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.duration_s = duration_s
        self._segs: list[dict] = []
        self._undo: list[list[dict]] = []
        self._redo: list[list[dict]] = []

    @property
    def eps(self) -> float:
        return 0.25 / self.fps

    def snap(self, t: float) -> float:
        t = round(float(t) * self.fps) / self.fps
        if self.duration_s is not None:
            t = min(t, self.duration_s)
        return max(0.0, t)

    # ---- history ----
    def _push(self):
        self._undo.append(copy.deepcopy(self._segs))
        self._redo.clear()
        del self._undo[:-200]

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self._segs))
        self._segs = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self._segs))
        self._segs = self._redo.pop()
        return True

    # ---- core ----
    def _normalise(self, segs):
        segs = sorted(segs, key=lambda s: (s["start_s"], s["end_s"]))
        out = []
        for s in segs:
            if out and out[-1]["behaviour"] == s["behaviour"] and s["start_s"] - out[-1]["end_s"] <= self.eps:
                out[-1]["end_s"] = max(out[-1]["end_s"], s["end_s"])
            else:
                out.append(dict(s))
        return out

    def _insert(self, segs, s, e, b):
        eps = self.eps
        res = []
        for g in segs:
            if g["end_s"] <= s + eps or g["start_s"] >= e - eps:
                res.append(dict(g))
                continue
            if g["start_s"] < s - eps:
                res.append({**g, "end_s": s})
            if g["end_s"] > e + eps:
                res.append({**g, "start_s": e})
        res.append({"start_s": s, "end_s": e, "behaviour": b})
        return self._normalise(res)

    def _bounds(self, start, end):
        s, e = self.snap(min(start, end)), self.snap(max(start, end))
        return (s, e) if e - s >= 1.0 / self.fps - 1e-9 else None

    def add(self, start, end, behaviour):
        """Returns index of the new segment or None if shorter than a frame."""
        b = self._bounds(start, end)
        if b is None:
            return None
        self._push()
        self._segs = self._insert(self._segs, b[0], b[1], behaviour)
        return self.at((b[0] + b[1]) / 2)

    def resize(self, idx, start, end):
        b = self._bounds(start, end)
        if b is None or not (0 <= idx < len(self._segs)):
            return None
        self._push()
        beh = self._segs[idx]["behaviour"]
        rest = self._segs[:idx] + self._segs[idx + 1:]
        self._segs = self._insert(rest, b[0], b[1], beh)
        return self.at((b[0] + b[1]) / 2)

    def delete(self, idx) -> bool:
        if not (0 <= idx < len(self._segs)):
            return False
        self._push()
        del self._segs[idx]
        return True

    def relabel(self, idx, behaviour) -> bool:
        if not (0 <= idx < len(self._segs)):
            return False
        self._push()
        self._segs[idx]["behaviour"] = behaviour
        self._segs = self._normalise(self._segs)
        return True

    def at(self, t):
        for i, s in enumerate(self._segs):
            if s["start_s"] - 1e-9 <= t < s["end_s"] - 1e-9:
                return i
        return None

    def __len__(self):
        return len(self._segs)

    def to_list(self) -> list[dict]:
        return [dict(s) for s in self._segs]

    def from_list(self, segs, record_undo: bool = False):
        clean = []
        for s in segs or []:
            try:
                a, b = float(s["start_s"]), float(s["end_s"])
                if b < a:
                    a, b = b, a
                if b - a > 0:
                    clean.append({"start_s": a, "end_s": b, "behaviour": str(s["behaviour"])})
            except (KeyError, TypeError, ValueError):
                continue
        if record_undo:
            self._push()
        else:
            self._undo.clear()
            self._redo.clear()
        self._segs = self._normalise(clean)


# --------------------------------------------------------------------------
# Widget
# --------------------------------------------------------------------------

_FALLBACK = ["#1565c0", "#2e7d32", "#e65100", "#c62828", "#6a1b9a", "#00838f",
             "#ad1457", "#558b2f", "#4527a0", "#ef6c00", "#00695c", "#6d4c41"]
_ROW1 = (4, 22)
_ROW2 = (28, 56)
_TL_H = 62


def _cluster_colour(cid, colors):
    if colors and cid in colors:
        return colors[cid]
    if colors and str(cid) in colors:
        return colors[str(cid)]
    if cid is None or (isinstance(cid, (int, np.integer)) and cid < 0):
        return "#b0b6c0"
    try:
        return _FALLBACK[int(cid) % len(_FALLBACK)]
    except Exception:
        return "#b0b6c0"


class LabelPlayer(tk.Frame if tk else object):
    EDGE_PX = 5

    def __init__(self, master, theme: dict, tile_size=(640, 400), compact: bool = False, **kw):
        super().__init__(master, bg=theme["panel"], **kw)
        self.compact = compact          # compact: playback only (no segment-editing rows), used by the question card
        self.T = theme
        self.tile_w, self.tile_h = tile_size
        self.on_segment_changed = None
        self.on_seek = None
        self.cap = None
        self.fps = 30.0
        self.n_frames = 0
        self.scrub = ScrubLogic(1, 30.0)
        self.model = SegmentModel(30.0)
        self.cluster_labels = None
        self.cluster_colors = None
        self.behaviours: list[dict] = []
        self.cur = 0
        self._want = None
        self._sel = None
        self._mark = None
        self._drag = None
        self._view0 = 0.0
        self._view_len = 1.0
        self._photo = None
        self._poll_id = None
        self._thread = None
        self._cond = threading.Condition()
        self._target = None
        self._playing = False
        self._speed = 1.0
        self._quit = False
        self._out: queue.Queue = queue.Queue(maxsize=3)
        self._loaded = False
        self.loop_range = None
        self.play_fps = 30.0           # real playback rate (the VIDEO's own fps); self.fps = analysis/label time base
        self.fps_note = ""
        self.readonly_segments: list = []
        self._build()

    # ------------------------------------------------------------- build
    def _btn(self, parent, text, cmd, tip_color=None):
        T = self.T
        return tk.Button(parent, text=text, command=cmd, takefocus=0, bg=tip_color or T["btn"],
                         fg=T["text"] if not tip_color else "#ffffff", activebackground=T["btn_hover"],
                         relief="flat", padx=8, pady=4, cursor="hand2", font=("Segoe UI", 9))

    def _build(self):
        T = self.T
        # Elastic video area: the container requests only a small size and takes whatever room the
        # controls below leave over; frames are rescaled to it (see _on_video_resize / _fit), so the
        # controls are never pushed out of view at small window sizes or high DPI scaling.
        self.vbox = tk.Frame(self, bg="#000000", width=200, height=120)
        self.vbox.pack(fill="both", expand=True, padx=4, pady=(4, 2))
        self.vbox.pack_propagate(False)
        self.video = tk.Label(self.vbox, bg="#000000", fg="#ffffff", width=1, height=1,
                              text="No video loaded", font=("Segoe UI", 11), wraplength=360)
        self.video.pack(expand=True)
        self._resize_id = None
        self.vbox.bind("<Configure>", self._on_video_resize)

        self.info = tk.Label(self, text="", bg=T["panel"], fg=T["text_dim"], anchor="w", font=("Segoe UI", 9))
        self.info.pack(fill="x", padx=6)

        self.tl = tk.Canvas(self, height=_TL_H, bg=T["panel2"], highlightthickness=1,
                            highlightbackground=T["border"], takefocus=1)
        self.tl.pack(fill="x", padx=4, pady=2)

        r1 = tk.Frame(self, bg=T["panel"])
        r1.pack(fill="x", padx=4, pady=1)
        r1b = tk.Frame(self, bg=T["panel"])      # nudge buttons on their own row so no row overflows
        r1b.pack(fill="x", padx=4, pady=1)
        self.b_play = self._btn(r1, "Play  [Space]", self.toggle_play, T["accent"])
        self.b_play.pack(side="left", padx=2)
        for txt, fr, sec in (("-1 s  [Shift+Left]", 0, -1), ("-1 frame  [Left]", -1, 0),
                             ("+1 frame  [Right]", 1, 0), ("+1 s  [Shift+Right]", 0, 1)):
            self._btn(r1b, txt, lambda f=fr, s=sec: self._nudge(f, s)).pack(side="left", padx=2)
        tk.Label(r1, text="Speed", bg=T["panel"], fg=T["text_dim"]).pack(side="left", padx=(10, 2))
        self.speed_var = tk.StringVar(master=self, value="1x")
        sp = ttk.Combobox(r1, textvariable=self.speed_var, width=6, state="readonly",
                          values=["0.25x", "0.5x", "1x", "1.5x", "2x"])
        sp.pack(side="left")
        sp.bind("<<ComboboxSelected>>", lambda e: self._set_speed())

        r2 = tk.Frame(self, bg=T["panel"])
        r2.pack(fill="x", padx=4, pady=1)
        self._btn(r2, "Start behaviour here  [S]", self.mark_start, T["accent2"]).pack(side="left", padx=2)
        self._btn(r2, "End behaviour here  [E]", self.mark_end, T["accent3"]).pack(side="left", padx=2)
        # Behaviour buttons (one coloured, titled button per behaviour; keys 1-9). After Start -> End the new
        # segment is unassigned and selected: click a button to assign it. Click any segment on the bottom
        # timeline row, then a button, to change its behaviour afterwards.
        self.beh_hint = tk.Label(self, text="", anchor="w", bg=T["panel"], fg=T["text_dim"], font=("Segoe UI", 9))
        self.beh_hint.pack(fill="x", padx=6)
        self.beh_row = tk.Frame(self, bg=T["panel"])
        self.beh_row.pack(fill="x", padx=4, pady=1)
        self._beh_buttons: dict = {}
        self._beh_key = None
        r3 = tk.Frame(self, bg=T["panel"])
        r3.pack(fill="x", padx=4, pady=(1, 4))
        self._btn(r3, "Delete  [Del]", self.delete_selected).pack(side="left", padx=2)
        self._btn(r3, "Undo  [Ctrl+Z]", self.undo).pack(side="left", padx=2)
        self._btn(r3, "Redo  [Ctrl+Y]", self.redo).pack(side="left", padx=2)
        if self.compact:
            for _r in (r1b, r2, r3, self.beh_hint, self.beh_row, self.tl, self.info):
                _r.pack_forget()

        # keys: only widgets that belong to the player, never Entry widgets
        for w in (self, self.video, self.tl, self.info):
            w.bind("<Key>", self._on_key)
            w.bind("<Button-1>", lambda e: self.focus_set() if e.widget is not self.tl else self.tl.focus_set(), add="+")
        for w in (self.video, self.tl):
            w.bind("<MouseWheel>", self._on_wheel)
            w.bind("<Button-4>", lambda e: self._on_wheel(e, -1))
            w.bind("<Button-5>", lambda e: self._on_wheel(e, 1))
        self.tl.bind("<Configure>", lambda e: self._redraw())
        self.tl.bind("<Motion>", self._on_motion)
        self.tl.bind("<Leave>", lambda e: self.tl.delete("tip"))
        self.tl.bind("<ButtonPress-1>", self._on_press)
        self.tl.bind("<B1-Motion>", self._on_drag)
        self.tl.bind("<ButtonRelease-1>", self._on_release)
        self.tl.bind("<Button-3>", self._on_right)
        self.bind("<Destroy>", lambda e: self.close() if e.widget is self else None)

    # ------------------------------------------------------------- public
    def load(self, video, fps: float, cluster_labels=None, cluster_colors=None, play_fps=None) -> bool:
        self.close()
        self._quit = False
        self._loaded = False
        path = Path(str(video)) if video else None
        try:
            import cv2
            if path is None or not path.is_file():
                raise FileNotFoundError("missing")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise IOError("open")
            ok, _ = cap.read()
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if not ok or n <= 0:
                cap.release()
                raise IOError("unreadable")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            self.cap = None
            self.video.config(image="", text="Video not found or can't be opened.\n"
                              "Check that the file still exists and is a valid video.")
            self._photo = None
            self.info.config(text="")
            self._redraw()
            return False
        self.cap = cap
        if not fps or fps <= 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        native = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.play_fps = float(play_fps) if play_fps else (native if native > 1.0 else float(fps))
        self.fps_note = ""
        if native > 1.0 and abs(native - float(fps)) / float(fps) > 0.01:
            self.fps_note = (f"Video is {native:.2f} fps but this run's frame labels use {float(fps):.2f} fps: it plays "
                             f"at the video's real speed; label times follow the analysis frame rate.")
        self.fps, self.n_frames = float(fps), n
        self.scrub = ScrubLogic(n, self.fps)
        keep = self.model.to_list()
        self.model = SegmentModel(self.fps, self.scrub.duration_s)
        self.model.from_list(keep)
        self.cluster_labels = None if cluster_labels is None else np.asarray(cluster_labels)
        self.cluster_colors = cluster_colors
        self.cur, self._want, self._sel, self._mark = 0, None, None, None
        self._view0, self._view_len = 0.0, self.scrub.duration_s
        self._loaded = True
        self._playing = False
        self._thread = threading.Thread(target=self._worker, args=(cap,), daemon=True)
        self._thread.start()
        self._schedule_poll()
        self.seek(0)
        return True

    def get_segments(self):
        return self.model.to_list()

    def set_segments(self, segs):
        self.model.from_list(segs)
        self._sel = None
        self._redraw()

    def set_readonly_segments(self, segs):
        """Segments drawn as a thin read-only strip (answers from the question card)."""
        self.readonly_segments = list(segs or [])
        self._redraw()

    def set_behaviours(self, behaviours):
        self.behaviours = list(behaviours or [])
        for w in self.beh_row.winfo_children():
            w.destroy()
        self._beh_buttons = {}
        self._beh_key = None
        per_row = 4
        for i, b in enumerate(self.behaviours):
            col = b.get("color", "#888888")
            label = b["name"] + (f"  [{i + 1}]" if i < 9 else "")
            btn = tk.Button(self.beh_row, text=label, command=lambda n=b["name"]: self.assign_behaviour(n),
                            takefocus=0, bg=col, fg="#ffffff", activebackground=col, activeforeground="#ffffff",
                            relief="flat", bd=3, padx=8, pady=4, cursor="hand2", font=("Segoe UI", 9, "bold"))
            btn.grid(row=i // per_row, column=i % per_row, padx=2, pady=2, sticky="ew")
            self.beh_row.columnconfigure(i % per_row, weight=1, uniform="beh")
            self._beh_buttons[b["name"]] = btn
        if not self.behaviours:
            tk.Label(self.beh_row, text="No behaviours yet - add one in the Behaviours panel.", bg=self.T["panel"],
                     fg=self.T["text_dim"]).grid(row=0, column=0, sticky="w")
        self._redraw()

    def seek(self, frame):
        if not self._loaded:
            return
        f = self.scrub.request(frame)
        self.scrub.take()
        self._want = f
        self.cur = f
        with self._cond:
            self._target = f
            self._cond.notify()
        self._draw_playhead()

    def play(self):
        if not self._loaded:
            return
        if self.cur >= self.n_frames - 1:
            self.seek(0)
        with self._cond:
            self._playing = True
            self._cond.notify()
        self.b_play.config(text="Pause  [Space]")

    def pause(self):
        with self._cond:
            self._playing = False
        self.b_play.config(text="Play  [Space]")

    def play_range(self, start_frame: int, end_frame: int):
        # Loop [start_frame, end_frame] (used by the query card's candidate/exemplar clips).
        if not self._loaded:
            return
        a = self.scrub.clamp(start_frame)
        b = max(a + 1, self.scrub.clamp(end_frame))
        self.loop_range = (a, b)
        self.seek(a)
        self.play()

    def clear_range(self):
        self.loop_range = None

    def toggle_play(self):
        self.pause() if self._playing else self.play()

    def close(self):
        with self._cond:
            self._quit = True
            self._playing = False
            self._cond.notify_all()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        stuck = t is not None and t.is_alive() and t is not threading.current_thread()
        self._thread = None
        if self.cap is not None:
            if not stuck:                # never release a capture another thread may still be reading
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
        if getattr(self, "_resize_id", None) is not None:
            try:
                self.after_cancel(self._resize_id)
            except Exception:
                pass
            self._resize_id = None
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None
        self._loaded = False

    # ------------------------------------------------------------- decode thread
    def _worker(self, cap):
        import cv2
        pos = 0  # index of the next frame cap.read() will return
        due = time.perf_counter()
        while True:
            with self._cond:
                while not self._quit and self._target is None and not self._playing:
                    self._cond.wait(0.2)
                if self._quit:
                    return
                tgt, self._target = self._target, None
                playing, speed = self._playing, self._speed
            if tgt is not None:
                if tgt != pos:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                    pos = tgt
                due = time.perf_counter()
            elif playing:
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, 0.05))
                    if due - time.perf_counter() > 0:
                        with self._cond:  # re-check commands while waiting
                            if self._target is not None or self._quit or not self._playing:
                                continue
                        continue
            else:
                continue
            ok, frame = cap.read()
            idx = pos
            if not ok:
                pos = max(0, pos)
                self._put(("end", idx, None))
                with self._cond:
                    self._playing = False
                continue
            pos += 1
            due = max(due, time.perf_counter() - 0.1) + 1.0 / (self.play_fps * speed)
            self._put(("frame", idx, self._fit(frame)))

    def _on_video_resize(self, e):
        w, h = max(160, e.width - 8), max(90, e.height - 8)
        if abs(w - self.tile_w) < 4 and abs(h - self.tile_h) < 4:
            return
        self.tile_w, self.tile_h = w, h
        if self._resize_id is not None:
            try:
                self.after_cancel(self._resize_id)
            except Exception:
                pass
        # debounce, then redraw the current frame at the new size (a playing video refits on its own)
        self._resize_id = self.after(120, self._refit_current)

    def _refit_current(self):
        self._resize_id = None
        if self._loaded and not self._playing:
            self.seek(self.cur)

    def _fit(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        s = min(self.tile_w / w, self.tile_h / h)
        if abs(s - 1) > 0.01:
            bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _put(self, item):
        while True:
            try:
                self._out.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._out.get_nowait()
                except queue.Empty:
                    pass

    # ------------------------------------------------------------- UI poll (main thread)
    def _schedule_poll(self):
        self._poll_id = self.after(15, self._poll)

    def _poll(self):
        self._poll_id = None
        last = None
        try:
            while True:
                last = self._out.get_nowait()
        except queue.Empty:
            pass
        if last is not None:
            kind, idx, rgb = last
            if kind == "end":
                self.pause()
            else:
                self._show(idx, rgb)
        if self._loaded:
            self._schedule_poll()

    def _show(self, idx, rgb):
        if self._want is not None and idx != self._want and not self._playing:
            return  # stale frame from before the latest seek
        if idx == self._want:
            self._want = None
        try:
            from PIL import Image, ImageTk
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb), master=self.video)
            self.video.config(image=self._photo, text="", width=rgb.shape[1], height=rgb.shape[0])
        except Exception:
            return
        if self._want is None:
            changed = idx != self.cur
            self.cur = idx
            self._draw_playhead()
            if changed and self.on_seek:
                self.on_seek(idx, idx / self.fps)
            lr = self.loop_range
            if lr is not None and self._playing and idx >= lr[1]:
                self.seek(lr[0])

    # ------------------------------------------------------------- actions
    def _nudge(self, frames, seconds):
        self.pause()
        self.seek(self.scrub.step(self.cur, frames=frames, seconds=seconds))
        if self.on_seek:
            self.on_seek(self.cur, self.cur / self.fps)

    def _set_speed(self):
        try:
            v = float(self.speed_var.get().rstrip("x"))
        except ValueError:
            v = 1.0
        with self._cond:
            self._speed = v

    def unassigned_count(self) -> int:
        return sum(1 for s in self.model.to_list() if not s["behaviour"])

    def assign_behaviour(self, name) -> bool:
        """Give the SELECTED segment (a fresh Start->End one, or any segment clicked on the bottom row) this
        behaviour. Works for the first assignment and for changing it later."""
        if not self._loaded or self._sel is None or not (0 <= self._sel < len(self.model)):
            self.beh_hint.config(text="First mark Start [S] and End [E], or click a segment on the bottom "
                                      "timeline row, then pick its behaviour.")
            return False
        seg = self.model.to_list()[self._sel]
        mid = (seg["start_s"] + seg["end_s"]) / 2
        if seg["behaviour"] != name:
            self.model.relabel(self._sel, name)
            self._sel = self.model.at(mid)
            self._changed()
        self.beh_hint.config(text=f"Set to {name}. Click another segment to change it, or keep scrubbing.",
                             fg=self.T["text_dim"], font=("Segoe UI", 9))
        return True

    def _refresh_beh_buttons(self):
        """Emphasise the button of the selected segment's behaviour; hint when a segment needs one."""
        segs = self.model.to_list()
        cur = segs[self._sel]["behaviour"] if (self._sel is not None and 0 <= self._sel < len(segs)) else None
        key = (cur, self._sel is not None)
        if key == self._beh_key:
            return
        self._beh_key = key
        for n, b in self._beh_buttons.items():
            b.config(relief="sunken" if n == cur else "flat")
        if cur == "":
            self.beh_hint.config(text="Which behaviour is this segment?  Click a behaviour below (or press its number).",
                                 fg=self.T["accent3"], font=("Segoe UI", 9, "bold"))
        elif cur:
            self.beh_hint.config(text=f"Selected segment: {cur}.  Click another behaviour to change it.",
                                 fg=self.T["text_dim"], font=("Segoe UI", 9))

    def mark_start(self):
        if not self._loaded:
            return
        self._mark = self.cur / self.fps
        self._redraw()

    def mark_end(self):
        if not self._loaded:
            return
        if self._mark is None:
            self.info.config(text="Press Start behaviour here [S] first, then End [E].")
            return
        i = self.model.add(self._mark, self.cur / self.fps, "")      # behaviour is chosen next, by button
        self._mark = None
        self._sel = i
        self._beh_key = None
        self._changed()

    def delete_selected(self):
        if self._sel is not None and self.model.delete(self._sel):
            self._sel = None
            self._changed()

    def undo(self):
        if self.model.undo():
            self._sel = None
            self._changed()

    def redo(self):
        if self.model.redo():
            self._sel = None
            self._changed()

    def _changed(self):
        self._redraw()
        if self.on_segment_changed:
            self.on_segment_changed(self.model.to_list())

    def _beh_color(self, name):
        for b in self.behaviours:
            if b.get("name") == name:
                return b.get("color", "#888888")
        return "#9aa0a8"

    # ------------------------------------------------------------- keyboard / wheel
    def _on_key(self, e):
        w = self.focus_get()
        if w is not None and w.winfo_class() in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"):
            return None
        ctrl = bool(e.state & 0x4)
        shift = bool(e.state & 0x1)
        k = e.keysym
        if self.compact:                 # question-card clip players: playback only; S/Y/N/X belong to the card
            if k == "space":
                self.toggle_play()
                return "break"
            return None
        if len(k) == 1 and k in "123456789" and not ctrl:
            i = int(k) - 1
            if i < len(self.behaviours):
                self.assign_behaviour(self.behaviours[i]["name"])
                return "break"
            return None
        if ctrl and k.lower() == "z":
            self.undo()
        elif ctrl and k.lower() == "y":
            self.redo()
        elif k in ("Left", "Right"):
            sgn = -1 if k == "Left" else 1
            self._nudge(0 if shift else sgn, sgn if shift else 0)
        elif k == "space":
            self.toggle_play()
        elif k.lower() == "s" and not ctrl:
            self.mark_start()
        elif k.lower() == "e" and not ctrl:
            self.mark_end()
        elif k == "Delete":
            self.delete_selected()
        else:
            return None
        return "break"

    def _on_wheel(self, e, notches=None):
        if not self._loaded:
            return
        if notches is None:
            notches = -1 if e.delta > 0 else 1  # wheel down = forward
        kind, val = self.scrub.wheel(self.cur, notches, shift=bool(e.state & 0x1), ctrl=bool(e.state & 0x4))
        if kind == "zoom":
            self._zoom(val, self._x_to_t(e.x) if e.widget is self.tl else self.cur / self.fps)
        else:
            self.pause()
            self.seek(val)  # seek() coalesces: worker only decodes the newest target
            if self.on_seek:
                self.on_seek(self.cur, self.cur / self.fps)

    # ------------------------------------------------------------- timeline
    def _x_to_t(self, x):
        W = max(1, self.tl.winfo_width())
        return self._view0 + (x / W) * self._view_len

    def _t_to_x(self, t):
        W = max(1, self.tl.winfo_width())
        return (t - self._view0) / self._view_len * W

    def _zoom(self, mult, centre_t):
        dur = self.scrub.duration_s
        new = min(max(self._view_len * mult, 10.0 / self.fps), dur)
        frac = (centre_t - self._view0) / self._view_len if self._view_len else 0.5
        self._view0 = min(max(centre_t - frac * new, 0.0), max(0.0, dur - new))
        self._view_len = new
        self._redraw()

    def _redraw(self):
        c = self.tl
        c.delete("all")
        if not self._loaded:
            return
        W = max(1, c.winfo_width())
        t0, t1 = self._view0, self._view0 + self._view_len
        # row 1: cluster runs
        cl = self.cluster_labels
        if cl is not None and len(cl):
            f0 = max(0, int(t0 * self.fps))
            f1 = min(len(cl), int(t1 * self.fps) + 2)
            i = f0
            while i < f1:
                j = i
                while j + 1 < f1 and cl[j + 1] == cl[i]:
                    j += 1
                v = cl[i]
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    pass
                x0, x1 = self._t_to_x(i / self.fps), self._t_to_x((j + 1) / self.fps)
                c.create_rectangle(x0, _ROW1[0], max(x1, x0 + 1), _ROW1[1],
                                   fill=_cluster_colour(v, self.cluster_colors), outline="")
                i = j + 1
        else:
            c.create_rectangle(0, _ROW1[0], W, _ROW1[1], fill=self.T["btn"], outline="")
        c.create_text(4, _ROW1[0] + 9, anchor="w", text="CUBE", fill="#ffffff", font=("Segoe UI", 7, "bold"))
        # row 2: user segments
        c.create_rectangle(0, _ROW2[0], W, _ROW2[1], fill=self.T["panel"], outline=self.T["border"])
        for i, s in enumerate(self.model.to_list()):
            x0, x1 = self._t_to_x(s["start_s"]), self._t_to_x(s["end_s"])
            if x1 < 0 or x0 > W:
                continue
            if s["behaviour"]:
                c.create_rectangle(x0, _ROW2[0] + 1, x1, _ROW2[1] - 1, fill=self._beh_color(s["behaviour"]),
                                   outline="#000000" if i == self._sel else "", width=2 if i == self._sel else 1)
                # a number badge is always drawn (colour alone fails for colour-blind users and narrow segments);
                # the full name is added when there is room
                num = next((str(k + 1) for k, b in enumerate(self.behaviours[:9]) if b.get("name") == s["behaviour"]),
                           s["behaviour"][:1].upper())
                label = f"{num} {s['behaviour']}" if x1 - x0 > 70 else (num if x1 - x0 > 9 else "")
                if label:
                    c.create_text(x0 + 4, (_ROW2[0] + _ROW2[1]) / 2, anchor="w", text=label,
                                  fill="#ffffff", font=("Segoe UI", 8, "bold"))
            else:                                     # marked but not yet assigned: loud, asks for a behaviour
                c.create_rectangle(x0, _ROW2[0] + 1, x1, _ROW2[1] - 1, fill="#ffe0b2", stipple="gray50",
                                   outline=self.T["accent3"], dash=(5, 2), width=3)
                c.create_text(x0 + 4, (_ROW2[0] + _ROW2[1]) / 2, anchor="w",
                              text="? pick a behaviour" if x1 - x0 > 110 else "?", fill=self.T["accent3"],
                              font=("Segoe UI", 9, "bold"))
        for s in self.readonly_segments:              # question-card answers: thin read-only strip under row 2
            x0, x1 = self._t_to_x(s["start_s"]), self._t_to_x(s["end_s"])
            if x1 < 0 or x0 > W:
                continue
            c.create_rectangle(x0, _ROW2[1] - 7, x1, _ROW2[1] - 1, fill=self._beh_color(s["behaviour"]),
                               outline="#ffffff", stipple="gray50" if not s["behaviour"] else "")
        if self._mark is not None:
            x = self._t_to_x(self._mark)
            c.create_line(x, _ROW2[0] - 2, x, _ROW2[1] + 2, fill=self.T["accent2"], dash=(3, 2), width=2)
        self._draw_playhead()
        self._refresh_beh_buttons()

    def _draw_playhead(self):
        c = self.tl
        c.delete("ph")
        if not self._loaded:
            return
        t = self.cur / self.fps
        if not (self._view0 <= t <= self._view0 + self._view_len) and self._playing:
            self._view0 = min(max(t - 0.1 * self._view_len, 0.0), max(0.0, self.scrub.duration_s - self._view_len))
            self._redraw()
            return
        x = self._t_to_x(t)
        c.create_line(x, 0, x, _TL_H, fill=self.T["accent4"], width=1, tags="ph")
        self.info.config(text=f"{t:7.2f} s   frame {self.cur} / {self.n_frames - 1}")

    def _hit(self, x, y):
        """Returns (index, 'l'|'r'|'in') for a row-2 segment under the cursor, or None."""
        t = self._x_to_t(x)
        for i, s in enumerate(self.model.to_list()):
            xs, xe = self._t_to_x(s["start_s"]), self._t_to_x(s["end_s"])
            if abs(x - xs) <= self.EDGE_PX:
                return i, "l"
            if abs(x - xe) <= self.EDGE_PX:
                return i, "r"
            if s["start_s"] <= t < s["end_s"]:
                return i, "in"
        return None

    def _on_motion(self, e):
        if not self._loaded:
            return
        c = self.tl
        c.delete("tip")
        t = self._x_to_t(e.x)
        f = self.scrub.time_to_frame(t)
        txt = f"{t:.2f} s"
        if self.cluster_labels is not None and 0 <= f < len(self.cluster_labels):
            txt += f"   cluster {self.cluster_labels[f]}"
        hit = self._hit(e.x, e.y) if e.y >= _ROW2[0] else None
        if hit:
            txt += f"   {self.model.to_list()[hit[0]]['behaviour']}"
            c.config(cursor="sb_h_double_arrow" if hit[1] in "lr" else "hand2")
        else:
            c.config(cursor="")
        x = min(e.x + 10, max(0, c.winfo_width() - 8 * len(txt) - 10))
        tid = c.create_text(x + 3, _TL_H - 8, anchor="w", text=txt, fill=self.T["text"],
                            font=("Segoe UI", 8), tags="tip")
        bb = c.bbox(tid)
        if bb:
            r = c.create_rectangle(bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1, fill=self.T["tag_new"],
                                   outline=self.T["border"], tags="tip")
            c.tag_lower(r, tid)

    def _on_press(self, e):
        if not self._loaded:
            return
        self.tl.focus_set()
        if e.y < _ROW2[0]:  # row 1 / ruler: jump
            self.pause()
            self.seek(self.scrub.time_to_frame(self._x_to_t(e.x)))
            if self.on_seek:
                self.on_seek(self.cur, self.cur / self.fps)
            self._drag = ("scrub",)
            return
        hit = self._hit(e.x, e.y)
        if hit and hit[1] in ("l", "r"):
            self._sel = hit[0]
            s = self.model.to_list()[hit[0]]
            self._drag = ("edge", hit[0], hit[1], s)
        elif hit:
            self._sel = hit[0]
            self._drag = None
            self.pause()
            self.seek(self.scrub.time_to_frame(self._x_to_t(e.x)))
        else:
            self._sel = None
            self._drag = ("new", self._x_to_t(e.x), self._x_to_t(e.x))
        self._redraw()

    def _on_drag(self, e):
        d = self._drag
        if not d or not self._loaded:
            return
        t = min(max(self._x_to_t(e.x), 0.0), self.scrub.duration_s)
        if d[0] == "scrub":
            self.seek(self.scrub.time_to_frame(t))
        elif d[0] == "new":
            self._drag = ("new", d[1], t)
            self._redraw()
            self.tl.create_rectangle(self._t_to_x(d[1]), _ROW2[0] + 1, self._t_to_x(t), _ROW2[1] - 1,
                                     outline=self.T["accent"], dash=(3, 2), width=2, tags="ph")
        elif d[0] == "edge":
            self._drag = d[:3] + (d[3], t)
            self.tl.delete("ghost")
            s = d[3]
            a, b = (t, s["end_s"]) if d[2] == "l" else (s["start_s"], t)
            self.tl.create_rectangle(self._t_to_x(min(a, b)), _ROW2[0] + 1, self._t_to_x(max(a, b)), _ROW2[1] - 1,
                                     outline=self.T["accent"], dash=(3, 2), width=2, tags="ghost")

    def _on_release(self, e):
        d, self._drag = self._drag, None
        self.tl.delete("ghost")
        if not d or not self._loaded:
            return
        if d[0] == "new":
            if abs(d[2] - d[1]) * self.fps >= 1:
                self._sel = self.model.add(d[1], d[2], "")       # then pick its behaviour with a button
                self._beh_key = None
                self._changed()
            else:  # plain click: jump
                self.pause()
                self.seek(self.scrub.time_to_frame(d[1]))
        elif d[0] == "edge" and len(d) == 5:
            s, t = d[3], d[4]
            a, b2 = (t, s["end_s"]) if d[2] == "l" else (s["start_s"], t)
            new = self.model.resize(d[1], a, b2)
            if new is not None:
                self._sel = new
                self._changed()
        self._redraw()

    def _on_right(self, e):
        hit = self._hit(e.x, e.y) if (self._loaded and e.y >= _ROW2[0]) else None
        if not hit:
            return
        idx = hit[0]
        self._sel = idx
        self._redraw()
        m = tk.Menu(self, tearoff=0)
        sub = tk.Menu(m, tearoff=0)
        for b in self.behaviours:
            sub.add_command(label=b["name"], command=lambda n=b["name"]: (self.model.relabel(idx, n), self._changed()))
        m.add_cascade(label="Relabel as", menu=sub)
        m.add_command(label="Delete segment", command=lambda: (self.model.delete(idx), setattr(self, "_sel", None), self._changed()))
        m.tk_popup(e.x_root, e.y_root)


