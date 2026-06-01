# -*- coding: utf-8 -*-
"""
Created on Fri May 15 03:18:44 2026

@author: param
"""

#!/usr/bin/env python3
"""
BSOID Behavioural Cluster Annotator  v2
=========================================
Requirements:
    pip install pillow opencv-python-headless

Run:
    python bsoid_annotator.py

Features
--------
  Light theme, landscape layout (1440 x 880 px)
  Embedded video player - MP4s play directly in the GUI (no external app needed)
  If a GIF exists it is shown; if not the first MP4 is played instead
  Multiple examples play simultaneously in a tiled grid
  All media can be rotated 0 / 90 / 180 / 270  - applied to every tile
  Colour-coded cluster list with status badges
  Behaviour-group manager: create / rename / recolour / delete
  Keyboard shortcuts:   -> navigate  Space replay  N new-group  I ignore  1-9 assign
  Session save/load (JSON) and TSV export

Robustness
----------
  ImageTk.PhotoImage references are kept on the widget to prevent GC
  Video threads are daemon threads - killed automatically on close
  All file I/O wrapped in try/except with user-facing messages
  GIF frame timing clamped to [30, 200] ms
"""

import os, re, csv, json, time, threading, queue, platform, subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
from collections import defaultdict, deque

#   PIL  
try:
    from PIL import Image, ImageTk, ImageSequence, ImageOps
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("!  Pillow not found. Install with:  pip install pillow")

#   OpenCV  
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("!  OpenCV not found. Install with:  pip install opencv-python-headless")

#  
#  LIGHT COLOUR PALETTE
#  
C = {
    "bg":          "#f0f2f5",   # page background
    "panel":       "#ffffff",   # panel / card white
    "panel2":      "#f7f8fa",   # slightly off-white
    "border":      "#d0d5de",   # subtle borders
    "header":      "#1a2340",   # dark navy header bar
    "header_text": "#ffffff",
    "accent":      "#1565c0",   # royal blue (primary)
    "accent2":     "#2e7d32",   # green  (assigned)
    "accent3":     "#e65100",   # deep orange (pending)
    "accent4":     "#c62828",   # deep red (ignored)
    "accent5":     "#6a1b9a",   # purple (selected/current)
    "text":        "#1a1a2e",   # near-black
    "text_dim":    "#5f6b7c",   # muted
    "btn":         "#e3e8f0",
    "btn_hover":   "#cdd5e0",
    "sel_row":     "#e8f0fe",   # selected cluster row tint
    "tag_new":     "#e3f2fd",
}

STATUS_COLOUR = {
    "pending":  C["accent3"],
    "assigned": C["accent2"],
    "ignored":  C["accent4"],
}
STATUS_BADGE = {"pending": " ", "assigned": "v", "ignored": "X"}

GROUP_COLOURS = [
    "#1565c0","#2e7d32","#e65100","#6a1b9a","#00695c",
    "#ad1457","#0277bd","#558b2f","#f9a825","#4527a0",
    "#00838f","#c62828","#37474f","#4e342e","#1b5e20",
]

ROTATION_OPTIONS = [0, 90, 180, 270]

#  
#  DATA MODEL
#  

class ClusterInfo:
    def __init__(self, group_id):
        self.group_id        = group_id
        self.gif_path        = None
        self.mp4_paths       = []          # sorted list
        self.status          = "pending"   # pending | assigned | ignored
        self.behaviour_group = None        # str name

    @property
    def example_count(self):
        return len(self.mp4_paths)

    @property
    def best_preview(self):
        """GIF if available, else first MP4."""
        return self.gif_path or (self.mp4_paths[0] if self.mp4_paths else None)


class BehaviourGroup:
    _counter = 0
    def __init__(self, name, colour=None):
        BehaviourGroup._counter += 1
        self.uid         = BehaviourGroup._counter
        self.name        = name
        self.colour      = colour or GROUP_COLOURS[(self.uid-1) % len(GROUP_COLOURS)]
        self.cluster_ids = []


SPEED_OPTIONS = [0.25, 0.5, 1.0]
ZOOM_OPTIONS  = [1.0, 1.5, 2.0, 3.0]


class AppState:
    def __init__(self):
        self.folder_path      = None
        self.clusters         = {}      # int -> ClusterInfo
        self.behaviour_groups = {}      # uid -> BehaviourGroup
        self.cluster_order    = []      # list[int]
        self.current_index    = 0
        self.undo_stack       = deque(maxlen=60)
        self.session_file     = None
        self.rotation         = 0      # degrees: 0, 90, 180, 270
        self.speed            = 1.0    # playback speed: 0.25, 0.5, 1.0
        self.zoom             = 1.0    # tile zoom factor: 1.0, 1.5, 2.0, 3.0

    @property
    def current_cluster(self):
        if not self.cluster_order:
            return None
        return self.clusters.get(self.cluster_order[self.current_index])

    def _bg_by_name(self, name):
        for bg in self.behaviour_groups.values():
            if bg.name == name:
                return bg
        return None

    def assign(self, group_id, behaviour_uid):
        cl = self.clusters[group_id]
        self.undo_stack.append((group_id, cl.status, cl.behaviour_group))
        if cl.behaviour_group:
            old = self._bg_by_name(cl.behaviour_group)
            if old and group_id in old.cluster_ids:
                old.cluster_ids.remove(group_id)
        bg = self.behaviour_groups[behaviour_uid]
        cl.behaviour_group = bg.name
        cl.status = "assigned"
        if group_id not in bg.cluster_ids:
            bg.cluster_ids.append(group_id)

    def ignore(self, group_id):
        cl = self.clusters[group_id]
        self.undo_stack.append((group_id, cl.status, cl.behaviour_group))
        if cl.behaviour_group:
            old = self._bg_by_name(cl.behaviour_group)
            if old and group_id in old.cluster_ids:
                old.cluster_ids.remove(group_id)
        cl.status = "ignored"
        cl.behaviour_group = None

    def undo(self):
        if not self.undo_stack:
            return None
        group_id, old_status, old_bg_name = self.undo_stack.pop()
        cl = self.clusters[group_id]
        if cl.behaviour_group:
            cur = self._bg_by_name(cl.behaviour_group)
            if cur and group_id in cur.cluster_ids:
                cur.cluster_ids.remove(group_id)
        cl.status = old_status
        cl.behaviour_group = old_bg_name
        if old_bg_name:
            old = self._bg_by_name(old_bg_name)
            if old and group_id not in old.cluster_ids:
                old.cluster_ids.append(group_id)
        return group_id

    def add_behaviour_group(self, name):
        colour = GROUP_COLOURS[len(self.behaviour_groups) % len(GROUP_COLOURS)]
        bg = BehaviourGroup(name, colour)
        self.behaviour_groups[bg.uid] = bg
        return bg

    def rename_behaviour_group(self, uid, new_name):
        bg = self.behaviour_groups[uid]
        old = bg.name
        bg.name = new_name
        for cl in self.clusters.values():
            if cl.behaviour_group == old:
                cl.behaviour_group = new_name

    def delete_behaviour_group(self, uid):
        bg = self.behaviour_groups.pop(uid, None)
        if bg:
            for gid in bg.cluster_ids:
                cl = self.clusters.get(gid)
                if cl:
                    cl.status = "pending"
                    cl.behaviour_group = None

    def stats(self):
        total    = len(self.clusters)
        assigned = sum(1 for c in self.clusters.values() if c.status == "assigned")
        ignored  = sum(1 for c in self.clusters.values() if c.status == "ignored")
        return total, assigned, ignored, total - assigned - ignored


#  
#  FILE DISCOVERY
#  

def discover_clusters(folder: Path):
    # Legacy flat naming: group_N_example_M.mp4
    pat_old = re.compile(r"group_(\d+)_example_(\d+)\.(mp4|gif)$", re.I)
    # New per-cluster subfolder naming: cluster_N[_animal]_example_M.mp4
    pat_new = re.compile(r"cluster_(\d+)(?:_[^/\\]*)?_example_(\d+)\.(mp4|gif)$", re.I)
    raw = defaultdict(lambda: {"gifs": [], "mp4s": []})

    try:
        for entry in sorted(folder.rglob("*")):
            if not entry.is_file():
                continue
            m = pat_new.match(entry.name) or pat_old.match(entry.name)
            if m:
                gid = int(m.group(1))
                if m.group(3).lower() == "gif":
                    raw[gid]["gifs"].append(entry)
                else:
                    raw[gid]["mp4s"].append(entry)
    except PermissionError:
        pass

    clusters = {}
    for gid, files in sorted(raw.items()):
        cl = ClusterInfo(gid)
        cl.gif_path  = files["gifs"][0] if files["gifs"] else None
        cl.mp4_paths = sorted(files["mp4s"])
        clusters[gid] = cl
    return clusters


#  
#  SINGLE VIDEO TILE  - plays one GIF or MP4 in a tk.Label, looping forever
#  

class VideoTile(tk.Label):
    """
    A tk.Label that plays a single media file (GIF via PIL, MP4 via OpenCV).
    Keeps a hard reference to every PhotoImage to prevent garbage collection.
    Thread-safe: the decode thread puts frames into a queue; the Tk after-loop
    pulls them on the main thread.
    """

    TILE_W = 300
    TILE_H = 240

    def __init__(self, master, path: Path, rotation: int = 0, label: str = "",
                 speed: float = 1.0, **kw):
        super().__init__(master, **kw)
        self.configure(bg=C["panel"], anchor="center",
                       relief="flat", bd=0,
                       text="Loading...", font=("Segoe UI", 9),
                       fg=C["text_dim"])
        self._path     = path
        self._rotation = rotation  # 0 / 90 / 180 / 270
        self._label    = label
        self._speed    = max(0.1, float(speed))
        self._stop_evt = threading.Event()
        self._frames   = []        # list of ImageTk.PhotoImage - keeps refs alive
        self._q        = queue.Queue(maxsize=16)  # larger buffer = smoother playback
        self._after_id = None
        self._photo    = None      # current displayed photo (hard ref)
        self._running  = False

        if path is None:
            self.configure(text="No media file", image="")
            return
        if not path.exists():
            self.configure(text=f"File not found:\n{path.name}", image="")
            return

        ext = path.suffix.lower()
        if ext == ".gif" and PIL_OK:
            self._load_gif()
        elif ext == ".mp4" and CV2_OK:
            self._start_mp4_thread()
        elif ext == ".mp4" and PIL_OK:
            # OpenCV unavailable: show placeholder + offer to open externally
            self.configure(text=f"[MP4 - click to open]\n{path.name}")
            self.bind("<Button-1>", lambda e: open_external(path))
        else:
            self.configure(text=f"Cannot play:\n{path.name}\n(install Pillow + OpenCV)")

    #   GIF  

    def _load_gif(self):
        try:
            img = Image.open(str(self._path))
            frames, delays = [], []
            for frame in ImageSequence.Iterator(img):
                f = self._transform(frame.convert("RGBA"))
                f.thumbnail((self.TILE_W, self.TILE_H), Image.LANCZOS)
                padded = Image.new("RGBA", (self.TILE_W, self.TILE_H), (247,248,250,255))
                x = (self.TILE_W - f.width)  // 2
                y = (self.TILE_H - f.height) // 2
                padded.paste(f, (x, y), f)
                frames.append(padded)
                raw_delay = max(30, min(200, int(frame.info.get("duration", 80))))
                # Apply speed: slower speed = longer delay per frame
                delays.append(int(raw_delay / self._speed))
            if not frames:
                raise ValueError("Empty GIF")
            # pre-convert all to PhotoImage (keeps refs in self._frames)
            self._frames = [ImageTk.PhotoImage(f, master=self) for f in frames]
            self._delays = delays
            self._gi     = 0
            self.configure(text="")
            self._running = True
            self._animate_gif()
        except Exception as e:
            self.configure(text=f"GIF error:\n{e}", image="")

    def _animate_gif(self):
        if not self._running or self._stop_evt.is_set():
            return
        if not self._frames:
            return
        try:
            if not self.winfo_exists():
                return
            ph = self._frames[self._gi]
            self._photo = ph                         # hard ref - prevents GC
            self.configure(image=ph, text="")
        except tk.TclError:
            return
        delay = self._delays[self._gi]
        self._gi = (self._gi + 1) % len(self._frames)
        self._after_id = self.after(delay, self._animate_gif)

    #   MP4  

    def _start_mp4_thread(self):
        self._running = True
        # Compute poll interval matching the video frame rate (capped 20-100ms)
        cap_tmp = cv2.VideoCapture(str(self._path))
        raw_fps = cap_tmp.get(cv2.CAP_PROP_FPS) if cap_tmp.isOpened() else 25.0
        cap_tmp.release()
        raw_fps = raw_fps if raw_fps > 0 else 25.0
        # At slower speeds, poll less frequently so we don't waste cycles
        effective_fps = raw_fps * self._speed
        self._poll_delay_ms = max(20, min(100, int(1000.0 / effective_fps)))
        t = threading.Thread(target=self._decode_mp4, daemon=True)
        t.start()
        self._poll_mp4()

    def _decode_mp4(self):
        """Run in background thread: decode frames and push to queue."""
        path_str = str(self._path)
        while not self._stop_evt.is_set():
            cap = cv2.VideoCapture(path_str)
            if not cap.isOpened():
                self._q.put(None)          # signal error
                return
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25
            # speed < 1 means slower: frame_delay gets longer
            delay = max(0.01, 1.0 / fps / self._speed)
            t0    = time.monotonic()
            frame_no = 0
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break   # end of file -> restart loop

                # colour convert and resize (BILINEAR is much faster than LANCZOS,
                # giving smoother frame timing with multiple simultaneous tiles)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img   = Image.fromarray(frame_rgb)
                pil_img   = self._transform(pil_img)
                pil_img.thumbnail((self.TILE_W, self.TILE_H), Image.BILINEAR)
                padded    = Image.new("RGB", (self.TILE_W, self.TILE_H), (247,248,250))
                x = (self.TILE_W - pil_img.width)  // 2
                y = (self.TILE_H - pil_img.height) // 2
                padded.paste(pil_img, (x, y))

                # throttle to target frame rate; skip sleep if we're already behind
                frame_no += 1
                next_t = t0 + frame_no * delay
                sleep  = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                # If we're more than 2 frames behind, drop this frame to catch up
                elif sleep < -delay * 2:
                    continue

                # push (non-blocking; drop frame if queue is full to avoid stalling)
                try:
                    self._q.put_nowait(padded)
                except queue.Full:
                    pass   # display lag is better than decode stall
            cap.release()
            # small pause before restart
            if not self._stop_evt.is_set():
                time.sleep(0.1)

    def _poll_mp4(self):
        """Called on main thread via after(); pulls frames from queue."""
        if self._stop_evt.is_set():
            return
        try:
            if not self.winfo_exists():
                return
            item = self._q.get_nowait()
            if item is None:
                self.configure(text=f"Cannot open:\n{self._path.name}", image="")
                return
            ph = ImageTk.PhotoImage(item, master=self)
            self._photo = ph              # hard ref - prevents GC
            self.configure(image=ph, text="")
        except queue.Empty:
            pass
        except tk.TclError:
            return
        self._after_id = self.after(self._poll_delay_ms, self._poll_mp4)

    #   shared  

    def _transform(self, img: "Image.Image") -> "Image.Image":
        """Apply rotation to a PIL image."""
        if self._rotation == 0:
            return img
        # PIL rotate is CCW; BSOID videos typically need CW correction
        angle_map = {90: Image.ROTATE_270,   # 90  CW
                     180: Image.ROTATE_180,
                     270: Image.ROTATE_90}   # 270  CW = 90  CCW
        return img.transpose(angle_map[self._rotation])

    def destroy(self):
        self._running = False
        self._stop_evt.set()
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = None
        self._photo    = None
        self._frames   = []
        super().destroy()

    def stop(self):
        self._running = False
        self._stop_evt.set()
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        self._frames = []
        self._photo  = None


#  
#  MULTI-TILE PLAYER PANEL  - holds N VideoTile widgets in a grid
#  

class MultiPlayer(tk.Frame):
    """
    Displays all examples of a cluster simultaneously in a tiled layout.
    Columns: 1 tile -> 1 col; 2 4 tiles -> 2 cols; 5+ -> 3 cols.
    """

    BASE_TILE_W = 320
    BASE_TILE_H = 240
    MAX_TILE_W  = 960
    MAX_TILE_H  = 720

    def __init__(self, master, **kw):
        super().__init__(master, bg=C["panel2"], **kw)
        self._tiles: list[VideoTile] = []

    def load(self, paths: list[Path], rotation: int = 0,
             speed: float = 1.0, zoom: float = 1.0):
        """Stop old tiles, create new ones for given paths."""
        self._stop_all()
        for w in self.winfo_children():
            w.destroy()
        self._tiles = []

        n = len(paths)
        if n == 0:
            tk.Label(self, text="No media files for this cluster.",
                     font=("Segoe UI", 11), bg=C["panel2"],
                     fg=C["text_dim"]).pack(expand=True)
            return

        cols = 1 if n == 1 else (2 if n <= 4 else 3)
        # scale tile size: zoom applied on top of column-based sizing
        base_w = min(self.MAX_TILE_W, max(160, 640 // cols))
        tile_w = min(self.MAX_TILE_W, int(base_w * zoom))
        tile_h = min(self.MAX_TILE_H, int(tile_w * 0.75))
        VideoTile.TILE_W = tile_w
        VideoTile.TILE_H = tile_h

        for i, path in enumerate(paths):
            row, col = divmod(i, cols)
            lbl = f"Example {i}" if n > 1 else ""
            tile = VideoTile(self, path=path, rotation=rotation,
                             speed=speed, label=lbl,
                             width=tile_w, height=tile_h)
            tile.grid(row=row*2, column=col, padx=6, pady=(6,0), sticky="nsew")
            if lbl:
                tk.Label(self, text=lbl,
                         font=("Segoe UI", 8), bg=C["panel2"],
                         fg=C["text_dim"]).grid(row=row*2+1, column=col,
                                                padx=6, pady=(0,4))
            self._tiles.append(tile)

        for c in range(cols):
            self.columnconfigure(c, weight=1)

    def stop_all(self):
        self._stop_all()

    def _stop_all(self):
        for t in self._tiles:
            try:
                t.stop()
            except Exception:
                pass
        self._tiles = []


#  
#  HELPERS
#  

def open_external(path: Path):
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            os.startfile(str(path))
        elif sys_name == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        messagebox.showerror("Cannot open file", str(e))


#  
#  COLOUR PICKER DIALOG
#  

class ColourPickerDialog(tk.Toplevel):
    def __init__(self, parent, bg_obj: BehaviourGroup, callback=None):
        super().__init__(parent)
        self.title(f"Colour for '{bg_obj.name}'")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.grab_set()
        self._bg_obj = bg_obj
        self._cb     = callback
        tk.Label(self, text="Select a colour:",
                 font=("Segoe UI", 10), bg=C["bg"],
                 fg=C["text"]).pack(pady=10, padx=20)
        grid = tk.Frame(self, bg=C["bg"])
        grid.pack(padx=20, pady=4)
        for i, col in enumerate(GROUP_COLOURS):
            r, c = divmod(i, 5)
            sel  = (col == bg_obj.colour)
            tk.Button(grid, bg=col, width=4, height=2,
                      relief="sunken" if sel else "raised",
                      bd=3 if sel else 1,
                      command=lambda c2=col: self._pick(c2),
                      cursor="hand2").grid(row=r, column=c, padx=3, pady=3)
        tk.Button(self, text="Cancel", command=self.destroy,
                  bg=C["btn"], fg=C["text"], font=("Segoe UI", 9),
                  relief="flat", bd=0, padx=12, pady=4,
                  cursor="hand2").pack(pady=10)

    def _pick(self, colour):
        self._bg_obj.colour = colour
        if self._cb:
            self._cb()
        self.destroy()


#  
#  MAIN APPLICATION
#  

class BSoidAnnotator(tk.Tk):

    def __init__(self, auto_open=True):
        super().__init__()
        self.title("CUBE Behavioural Cluster Annotator")
        self.configure(bg=C["bg"])
        self.geometry("1440x880")
        self.minsize(1200, 740)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.sd  = AppState()
        self._cl_row_widgets: dict[int, tk.Frame] = {}

        self._build_ui()
        self._bind_keys()
        if auto_open:
            self.after(200, self._open_folder)

    def _on_close(self):
        """Stop all video threads, exit the mainloop, then destroy the window."""
        try:
            self._player.stop_all()
        except Exception:
            pass
        # quit() exits any nested mainloop started by the launcher so that
        # the parent app's _running flag is cleared correctly.
        try:
            self.quit()
        except Exception:
            pass
        self.destroy()

    #  
    #  BUILD UI
    #  

    def _build_ui(self):
        #   top header bar  
        top = tk.Frame(self, bg=C["header"], height=54)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(top, text="   CUBE Cluster Annotator",
                 font=("Segoe UI", 15, "bold"),
                 bg=C["header"], fg=C["header_text"]
                 ).pack(side="left", padx=18, pady=10)

        self.lbl_stats = tk.Label(top, text="",
                                   font=("Segoe UI", 9),
                                   bg=C["header"], fg="#b0bec5")
        self.lbl_stats.pack(side="left", padx=20)

        for txt, cmd, col in [
            ("   Export TSV",   self._export_tsv,   "#ffe082"),
            ("Save:  Save Session", self._save_session,  "#a5d6a7"),
            ("Open:  Open Folder",  self._open_folder,   "#90caf9"),
        ]:
            tk.Button(top, text=txt, command=cmd,
                      bg=C["header"], fg=col,
                      font=("Segoe UI", 10, "bold"),
                      relief="flat", bd=0, padx=12, pady=6,
                      activebackground="#263355",
                      activeforeground=col,
                      cursor="hand2").pack(side="right", padx=5, pady=8)

        #   three-column body  
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=6, pady=6)

        self._build_left(body)
        self._build_centre(body)
        self._build_right(body)

        #   status bar  
        bot = tk.Frame(self, bg=C["border"], height=1)
        bot.pack(fill="x")
        sb  = tk.Frame(self, bg=C["panel"], height=28)
        sb.pack(fill="x")
        sb.pack_propagate(False)
        self.lbl_status = tk.Label(sb,
            text="Open a CUBE output folder to begin.",
            font=("Segoe UI", 9), bg=C["panel"], fg=C["text_dim"])
        self.lbl_status.pack(side="left", padx=14)
        tk.Label(sb,
            text=" -> navigate  |  Space replay  |  N new group  |  I ignore  |  1 9 assign  |  Ctrl+S save  |  Ctrl+Z undo",
            font=("Segoe UI", 8), bg=C["panel"], fg=C["text_dim"]
            ).pack(side="right", padx=14)

    #   LEFT PANEL  

    def _build_left(self, parent):
        lf = tk.Frame(parent, bg=C["panel"], width=228,
                      relief="flat", bd=0,
                      highlightbackground=C["border"],
                      highlightthickness=1)
        lf.pack(side="left", fill="y", padx=(0,6))
        lf.pack_propagate(False)

        # header
        hdr = tk.Frame(lf, bg=C["accent"], height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="CLUSTERS",
                 font=("Segoe UI", 10, "bold"),
                 bg=C["accent"], fg="white").pack(side="left", padx=10, pady=6)

        # filter row
        flt = tk.Frame(lf, bg=C["panel2"])
        flt.pack(fill="x")
        self.filter_var = tk.StringVar(value="all")
        for val, txt, col in [
            ("all",      "All",     C["text"]),
            ("pending",  "Pending", C["accent3"]),
            ("assigned", "Done",    C["accent2"]),
            ("ignored",  "Ignored", C["accent4"]),
        ]:
            tk.Radiobutton(flt, text=txt, variable=self.filter_var, value=val,
                           command=self._refresh_cluster_list,
                           bg=C["panel2"], fg=col, selectcolor=C["panel"],
                           activebackground=C["panel2"],
                           font=("Segoe UI", 8), relief="flat", bd=0,
                           cursor="hand2").pack(side="left", padx=2, pady=4)

        tk.Frame(lf, bg=C["border"], height=1).pack(fill="x")

        # scrollable list
        sf = tk.Frame(lf, bg=C["panel"])
        sf.pack(fill="both", expand=True)
        self._cl_canvas = tk.Canvas(sf, bg=C["panel"], highlightthickness=0)
        sb = tk.Scrollbar(sf, orient="vertical", command=self._cl_canvas.yview)
        self._cl_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._cl_canvas.pack(side="left", fill="both", expand=True)
        self._cl_frame = tk.Frame(self._cl_canvas, bg=C["panel"])
        self._cl_win   = self._cl_canvas.create_window(
            (0,0), window=self._cl_frame, anchor="nw")
        self._cl_frame.bind("<Configure>",
            lambda e: self._cl_canvas.configure(
                scrollregion=self._cl_canvas.bbox("all")))
        self._cl_canvas.bind("<Configure>",
            lambda e: self._cl_canvas.itemconfig(self._cl_win, width=e.width))
        self._bind_mousewheel(self._cl_canvas)

    #   CENTRE PANEL  

    def _build_centre(self, parent):
        cf = tk.Frame(parent, bg=C["bg"])
        cf.pack(side="left", fill="both", expand=True)

        # title row
        title_row = tk.Frame(cf, bg=C["bg"])
        title_row.pack(fill="x", pady=(4,2))

        self.lbl_title = tk.Label(title_row, text="Select a cluster ->",
                                   font=("Segoe UI", 16, "bold"),
                                   bg=C["bg"], fg=C["accent5"])
        self.lbl_title.pack(side="left", padx=12)

        self.lbl_cur_status = tk.Label(title_row, text="",
                                        font=("Segoe UI", 11, "bold"),
                                        bg=C["bg"])
        self.lbl_cur_status.pack(side="left", padx=8)

        self.lbl_sub = tk.Label(cf, text="",
                                 font=("Segoe UI", 9),
                                 bg=C["bg"], fg=C["text_dim"])
        self.lbl_sub.pack(anchor="w", padx=12, pady=(0,4))

        # rotation controls
        rot_row = tk.Frame(cf, bg=C["bg"])
        rot_row.pack(fill="x", padx=12, pady=(0,2))
        tk.Label(rot_row, text="  Rotation:",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        self._rot_var = tk.IntVar(value=0)
        for deg in ROTATION_OPTIONS:
            tk.Radiobutton(rot_row, text=f"{deg} ",
                           variable=self._rot_var, value=deg,
                           command=self._apply_rotation,
                           bg=C["bg"], fg=C["accent"],
                           selectcolor=C["panel"],
                           activebackground=C["bg"],
                           font=("Segoe UI", 9, "bold"),
                           relief="flat", bd=0,
                           cursor="hand2").pack(side="left", padx=6)
        tk.Label(rot_row, text="(applied to all media)",
                 font=("Segoe UI", 8), bg=C["bg"],
                 fg=C["text_dim"]).pack(side="left", padx=6)

        # speed controls
        spd_row = tk.Frame(cf, bg=C["bg"])
        spd_row.pack(fill="x", padx=12, pady=(0,2))
        tk.Label(spd_row, text="  Speed:",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        self._speed_var = tk.DoubleVar(value=1.0)
        for spd in SPEED_OPTIONS:
            tk.Radiobutton(spd_row, text=f"{spd:g}x",
                           variable=self._speed_var, value=spd,
                           command=self._apply_speed,
                           bg=C["bg"], fg=C["accent3"],
                           selectcolor=C["panel"],
                           activebackground=C["bg"],
                           font=("Segoe UI", 9, "bold"),
                           relief="flat", bd=0,
                           cursor="hand2").pack(side="left", padx=6)
        tk.Label(spd_row, text="(0.25x/0.5x reduce jitter on slow machines)",
                 font=("Segoe UI", 8), bg=C["bg"],
                 fg=C["text_dim"]).pack(side="left", padx=6)

        # zoom controls
        zoom_row = tk.Frame(cf, bg=C["bg"])
        zoom_row.pack(fill="x", padx=12, pady=(0,4))
        tk.Label(zoom_row, text="  Zoom:",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        self._zoom_var = tk.DoubleVar(value=1.0)
        for z in ZOOM_OPTIONS:
            tk.Radiobutton(zoom_row, text=f"{z:g}x",
                           variable=self._zoom_var, value=z,
                           command=self._apply_zoom,
                           bg=C["bg"], fg=C["accent2"],
                           selectcolor=C["panel"],
                           activebackground=C["bg"],
                           font=("Segoe UI", 9, "bold"),
                           relief="flat", bd=0,
                           cursor="hand2").pack(side="left", padx=6)
        tk.Label(zoom_row, text="(scroll to pan zoomed view)",
                 font=("Segoe UI", 8), bg=C["bg"],
                 fg=C["text_dim"]).pack(side="left", padx=6)

        #   video player area  
        player_border = tk.Frame(cf, bg=C["border"], bd=1)
        player_border.pack(fill="both", expand=True, padx=8, pady=4)

        # scrollable player frame (for many examples)
        self._player_outer = tk.Canvas(player_border, bg=C["panel2"],
                                       highlightthickness=0)
        player_sb = tk.Scrollbar(player_border, orient="vertical",
                                  command=self._player_outer.yview)
        self._player_outer.configure(yscrollcommand=player_sb.set)
        player_sb.pack(side="right", fill="y")
        self._player_outer.pack(side="left", fill="both", expand=True)
        self._bind_mousewheel(self._player_outer)

        self._player = MultiPlayer(self._player_outer)
        self._player_win = self._player_outer.create_window(
            (0,0), window=self._player, anchor="nw")
        self._player.bind("<Configure>",
            lambda e: self._player_outer.configure(
                scrollregion=self._player_outer.bbox("all")))
        self._player_outer.bind("<Configure>",
            lambda e: self._player_outer.itemconfig(
                self._player_win, width=e.width))

        #   controls below player  
        ctrl = tk.Frame(cf, bg=C["bg"])
        ctrl.pack(fill="x", padx=8, pady=4)

        # assign buttons area
        assign_card = tk.Frame(cf, bg=C["panel"],
                               highlightbackground=C["border"],
                               highlightthickness=1)
        assign_card.pack(fill="x", padx=8, pady=4)

        top_ac = tk.Frame(assign_card, bg=C["panel"])
        top_ac.pack(fill="x", padx=8, pady=(6,0))
        tk.Label(top_ac, text="ASSIGN TO BEHAVIOUR GROUP",
                 font=("Segoe UI", 8, "bold"),
                 bg=C["panel"], fg=C["text_dim"]).pack(side="left")

        self.assign_btn_frame = tk.Frame(assign_card, bg=C["panel"])
        self.assign_btn_frame.pack(fill="x", padx=8, pady=4)

        qa = tk.Frame(assign_card, bg=C["panel"])
        qa.pack(pady=(0,8))

        for txt, cmd, bg2, fg2 in [
            ("   New Group (N)",  self._new_group_dialog, C["tag_new"], C["accent"]),
            ("   Ignore (I)",     self._ignore_current,   "#fdecea",   C["accent4"]),
            ("   Undo (Ctrl+Z)",  self._undo,              C["btn"],    C["text_dim"]),
        ]:
            tk.Button(qa, text=txt, command=cmd,
                      bg=bg2, fg=fg2,
                      font=("Segoe UI", 9, "bold"),
                      relief="flat", bd=0, padx=12, pady=5,
                      activebackground=C["btn_hover"], cursor="hand2"
                      ).pack(side="left", padx=5)

        # navigation
        nav = tk.Frame(cf, bg=C["bg"])
        nav.pack(pady=6)
        self._nbtn(nav, "   Previous", self._prev_cluster
                   ).pack(side="left", padx=8)
        self.lbl_nav = tk.Label(nav, text="",
                                 font=("Segoe UI", 10),
                                 bg=C["bg"], fg=C["text_dim"])
        self.lbl_nav.pack(side="left", padx=12)
        tk.Button(nav, text="Next   ",
                  command=self._next_cluster,
                  bg=C["accent"], fg="white",
                  font=("Segoe UI", 11, "bold"),
                  relief="flat", bd=0, padx=18, pady=8,
                  activebackground="#0d47a1", cursor="hand2"
                  ).pack(side="left", padx=8)

    #   RIGHT PANEL  

    def _build_right(self, parent):
        rf = tk.Frame(parent, bg=C["panel"], width=298,
                      highlightbackground=C["border"], highlightthickness=1)
        rf.pack(side="right", fill="y", padx=(6,0))
        rf.pack_propagate(False)

        hdr = tk.Frame(rf, bg=C["accent2"], height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="BEHAVIOUR GROUPS",
                 font=("Segoe UI", 10, "bold"),
                 bg=C["accent2"], fg="white").pack(side="left", padx=10)
        tk.Button(hdr, text="+ New",
                  command=self._new_group_dialog,
                  bg="#1b5e20", fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", bd=0, padx=10, pady=4,
                  activebackground="#2e7d32", cursor="hand2"
                  ).pack(side="right", padx=6, pady=4)

        sf = tk.Frame(rf, bg=C["panel"])
        sf.pack(fill="both", expand=True)
        self._bg_canvas = tk.Canvas(sf, bg=C["panel"], highlightthickness=0)
        bg_sb = tk.Scrollbar(sf, orient="vertical", command=self._bg_canvas.yview)
        self._bg_canvas.configure(yscrollcommand=bg_sb.set)
        bg_sb.pack(side="right", fill="y")
        self._bg_canvas.pack(side="left", fill="both", expand=True)
        self._bg_frame = tk.Frame(self._bg_canvas, bg=C["panel"])
        self._bg_win   = self._bg_canvas.create_window(
            (0,0), window=self._bg_frame, anchor="nw")
        self._bg_frame.bind("<Configure>",
            lambda e: self._bg_canvas.configure(
                scrollregion=self._bg_canvas.bbox("all")))
        self._bg_canvas.bind("<Configure>",
            lambda e: self._bg_canvas.itemconfig(self._bg_win, width=e.width))
        self._bind_mousewheel(self._bg_canvas)

    #   MOUSEWHEEL HELPER  

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>",
            lambda e: widget.yview_scroll(-1*(e.delta//120), "units"))
        widget.bind("<Button-4>",
            lambda e: widget.yview_scroll(-1, "units"))
        widget.bind("<Button-5>",
            lambda e: widget.yview_scroll(1, "units"))

    #   BUTTON HELPER  

    def _nbtn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         bg=C["btn"], fg=C["text"],
                         font=("Segoe UI", 10, "bold"),
                         relief="flat", bd=0, padx=14, pady=8,
                         activebackground=C["btn_hover"], cursor="hand2")

    #  
    #  KEY BINDINGS
    #  

    def _bind_keys(self):
        self.bind("<Right>",     lambda e: self._next_cluster())
        self.bind("<Left>",      lambda e: self._prev_cluster())
        self.bind("<space>",     lambda e: self._replay())
        self.bind("n",           lambda e: self._new_group_dialog())
        self.bind("N",           lambda e: self._new_group_dialog())
        self.bind("i",           lambda e: self._ignore_current())
        self.bind("I",           lambda e: self._ignore_current())
        self.bind("<Control-s>", lambda e: self._save_session())
        self.bind("<Control-z>", lambda e: self._undo())
        for k in range(1, 10):
            self.bind(str(k), lambda e, n=k: self._assign_by_number(n))

    #  
    #  FOLDER LOADING
    #  

    
    def _toggle_theme(self):
        global C
        if C["bg"] == "#f0f2f5":
            # Switch to Dark
            C["bg"] = "#09090f"
            C["panel"] = "#111120"
            C["panel2"] = "#16162a"
            C["text"] = "#eaeaea"
            C["text_dim"] = "#7788aa"
        else:
            # Switch to Light
            C["bg"] = "#f0f2f5"
            C["panel"] = "#ffffff"
            C["panel2"] = "#f8f9fa"
            C["text"] = "#222222"
            C["text_dim"] = "#666666"
        import tkinter.messagebox
        tkinter.messagebox.showinfo("Theme Changed", "Please restart the Video Explorer to apply the new theme.")

    def _open_folder(self):

        folder = filedialog.askdirectory(
            title="Select CUBE output folder (contains group_N_example_M files)")
        if not folder:
            return
        path = Path(folder)
        clusters = discover_clusters(path)
        if not clusters:
            # User may have selected BSOID_Project_Ready while clips live in the
            # sibling cube_results/ folder — search the parent folder too.
            parent_clusters = discover_clusters(path.parent)
            if parent_clusters:
                clusters = parent_clusters
                path = path.parent
        if not clusters:
            messagebox.showerror("No clusters found",
                "No example clip files found.\n\n"
                "Run Step 3 (CUBE Clustering) to generate them, then\n"
                "select the output folder (cube_results) or its parent.")
            return

        self._player.stop_all()

        sd = self.sd
        sd.folder_path      = path
        sd.clusters         = clusters
        sd.cluster_order    = sorted(clusters.keys())
        sd.current_index    = 0
        sd.behaviour_groups = {}
        BehaviourGroup._counter = 0
        sd.undo_stack.clear()
        sd.rotation = 0
        sd.speed    = 1.0
        sd.zoom     = 1.0
        self._rot_var.set(0)
        self._speed_var.set(1.0)
        self._zoom_var.set(1.0)

        self.title(f"CUBE Annotator     {path.name}")
        self._status(f"Loaded {len(clusters)} clusters from: {path}")
        self._refresh_cluster_list()
        self._refresh_group_panel()
        self._refresh_assign_buttons()
        self._load_current()

    #  
    #  CLUSTER LIST
    #  

    def _refresh_cluster_list(self):
        for w in self._cl_row_widgets.values():
            w.destroy()
        self._cl_row_widgets.clear()
        filt = self.filter_var.get()
        for gid in self.sd.cluster_order:
            cl = self.sd.clusters[gid]
            if filt == "all" or cl.status == filt:
                self._make_cluster_row(gid, cl)
        self._update_stats()

    def _make_cluster_row(self, gid: int, cl: ClusterInfo):
        is_cur = (bool(self.sd.cluster_order) and
                  self.sd.cluster_order[self.sd.current_index] == gid)
        bg  = C["sel_row"] if is_cur else C["panel"]
        sc  = STATUS_COLOUR[cl.status]

        row = tk.Frame(self._cl_frame, bg=bg, cursor="hand2")
        row.pack(fill="x", padx=2, pady=1)

        # colour stripe
        tk.Frame(row, bg=sc, width=4).pack(side="left", fill="y")

        inner = tk.Frame(row, bg=bg)
        inner.pack(side="left", fill="x", expand=True, padx=6, pady=4)

        top_r = tk.Frame(inner, bg=bg)
        top_r.pack(fill="x")
        tk.Label(top_r,
                 text=f"Group {gid}",
                 font=("Segoe UI", 9, "bold"),
                 bg=bg, fg=C["accent5"] if is_cur else C["text"]
                 ).pack(side="left")
        tk.Label(top_r,
                 text=STATUS_BADGE[cl.status],
                 font=("Segoe UI", 9, "bold"),
                 bg=bg, fg=sc
                 ).pack(side="right", padx=4)

        if cl.behaviour_group:
            tk.Label(inner,
                     text=cl.behaviour_group,
                     font=("Segoe UI", 8),
                     bg=bg,
                     fg=C["accent2"] if cl.status == "assigned" else C["text_dim"]
                     ).pack(anchor="w")

        n = cl.example_count
        tk.Label(inner,
                 text=f"{n} example{'s' if n != 1 else ''}",
                 font=("Segoe UI", 8),
                 bg=bg, fg=C["text_dim"]
                 ).pack(anchor="w")

        for widget in (row, inner, top_r):
            widget.bind("<Button-1>", lambda e, g=gid: self._jump_to(g))

        self._cl_row_widgets[gid] = row

    def _jump_to(self, gid: int):
        if gid in self.sd.cluster_order:
            self.sd.current_index = self.sd.cluster_order.index(gid)
            self._load_current()

    #  
    #  CENTRE PANEL UPDATES
    #  

    def _load_current(self):
        cl = self.sd.current_cluster
        if cl is None:
            return

        self.lbl_title.configure(text=f"Group {cl.group_id}")
        self.lbl_sub.configure(
            text=f"{cl.example_count} example{'s' if cl.example_count != 1 else ''}  "
                 f"|  GIF: {'v' if cl.gif_path else 'X'}  "
                 f"|  MP4s: {len(cl.mp4_paths)}")

        sc = STATUS_COLOUR[cl.status]
        if cl.status == "assigned":
            stxt = f"v  {cl.behaviour_group}"
        elif cl.status == "ignored":
            stxt = "X  Ignored"
        else:
            stxt = "   Pending"
        self.lbl_cur_status.configure(text=stxt, fg=sc, bg=C["bg"])

        total = len(self.sd.cluster_order)
        self.lbl_nav.configure(text=f"{self.sd.current_index + 1} / {total}")

        # Build media path list:
        # Always use MP4s as primary; if none exist fall back to GIF
        if cl.mp4_paths:
            media_paths = list(cl.mp4_paths)
        elif cl.gif_path:
            media_paths = [cl.gif_path]
        else:
            media_paths = []

        self._player.load(media_paths, rotation=self.sd.rotation,
                          speed=self.sd.speed, zoom=self.sd.zoom)

        self._refresh_cluster_list()
        self._scroll_to_current()
        self._update_stats()

    def _apply_rotation(self):
        self.sd.rotation = self._rot_var.get()
        self._load_current()

    def _apply_speed(self):
        self.sd.speed = self._speed_var.get()
        self._load_current()

    def _apply_zoom(self):
        self.sd.zoom = self._zoom_var.get()
        self._load_current()

    def _replay(self):
        self._load_current()

    def _scroll_to_current(self):
        cl = self.sd.current_cluster
        if not cl:
            return
        w = self._cl_row_widgets.get(cl.group_id)
        if not w:
            return
        self._cl_frame.update_idletasks()
        h  = max(self._cl_frame.winfo_height(), 1)
        y  = w.winfo_y()
        ch = self._cl_canvas.winfo_height()
        self._cl_canvas.yview_moveto(max(0.0, y/h - ch/h/2))

    #  
    #  NAVIGATION
    #  

    def _next_cluster(self):
        sd = self.sd
        if not sd.cluster_order:
            return
        sd.current_index = (sd.current_index + 1) % len(sd.cluster_order)
        self._load_current()

    def _prev_cluster(self):
        sd = self.sd
        if not sd.cluster_order:
            return
        sd.current_index = (sd.current_index - 1) % len(sd.cluster_order)
        self._load_current()

    #  
    #  ASSIGNMENT
    #  

    def _ignore_current(self):
        cl = self.sd.current_cluster
        if not cl:
            return
        self.sd.ignore(cl.group_id)
        self._status(f"Group {cl.group_id} -> ignored.")
        self._after_action()
        self._next_cluster()

    def _assign_to(self, uid: int):
        cl = self.sd.current_cluster
        if not cl:
            return
        self.sd.assign(cl.group_id, uid)
        bg = self.sd.behaviour_groups[uid]
        self._status(f"Group {cl.group_id}  ->  '{bg.name}'")
        self._after_action()
        self._next_cluster()

    def _assign_by_number(self, n: int):
        uids = sorted(self.sd.behaviour_groups.keys())
        if n <= len(uids):
            self._assign_to(uids[n-1])

    def _undo(self):
        gid = self.sd.undo()
        if gid is None:
            self._status("Nothing to undo.")
            return
        self._status(f"Undone: Group {gid} reverted.")
        self._after_action()
        self._load_current()

    def _after_action(self):
        self._refresh_group_panel()
        self._refresh_assign_buttons()
        self._update_stats()
        self._load_current()

    #  
    #  GROUP MANAGEMENT
    #  

    def _new_group_dialog(self, prefill=""):
        name = simpledialog.askstring(
            "New Behaviour Group",
            "Enter a name for the new behaviour group:",
            initialvalue=prefill, parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        if any(b.name == name for b in self.sd.behaviour_groups.values()):
            messagebox.showwarning("Duplicate name",
                f"A group named '{name}' already exists.")
            return
        bg = self.sd.add_behaviour_group(name)
        self._status(f"Created group: '{name}'")
        self._refresh_group_panel()
        self._refresh_assign_buttons()
        cl = self.sd.current_cluster
        if cl and messagebox.askyesno("Assign?",
                f"Assign Group {cl.group_id} to '{name}' now?"):
            self.sd.assign(cl.group_id, bg.uid)
            self._after_action()
            self._next_cluster()

    def _refresh_group_panel(self):
        for w in self._bg_frame.winfo_children():
            w.destroy()
        if not self.sd.behaviour_groups:
            tk.Label(self._bg_frame,
                     text="No groups yet.\nPress N to create one.",
                     font=("Segoe UI", 9), bg=C["panel"],
                     fg=C["text_dim"], justify="center"
                     ).pack(pady=20)
            return
        for uid, bg in sorted(self.sd.behaviour_groups.items(),
                               key=lambda kv: kv[1].name):
            self._make_group_card(bg)

    def _make_group_card(self, bg: BehaviourGroup):
        card = tk.Frame(self._bg_frame, bg=C["panel"],
                        highlightbackground=C["border"],
                        highlightthickness=1)
        card.pack(fill="x", padx=6, pady=4)

        # top colour stripe
        tk.Frame(card, bg=bg.colour, height=4).pack(fill="x")

        body = tk.Frame(card, bg=C["panel"])
        body.pack(fill="x", padx=8, pady=6)

        uids = sorted(self.sd.behaviour_groups.keys())
        idx  = uids.index(bg.uid) if bg.uid in uids else -1
        sc   = str(idx+1) if 0 <= idx < 9 else ""

        name_row = tk.Frame(body, bg=C["panel"])
        name_row.pack(fill="x")
        if sc:
            tk.Label(name_row, text=f"[{sc}]",
                     font=("Segoe UI", 8, "bold"),
                     bg=C["panel"], fg=bg.colour).pack(side="left", padx=(0,4))
        tk.Label(name_row, text=bg.name,
                 font=("Segoe UI", 10, "bold"),
                 bg=C["panel"], fg=C["text"]).pack(side="left")

        n     = len(bg.cluster_ids)
        ids_s = (", ".join(f"G{g}" for g in sorted(bg.cluster_ids)[:8])
                 + ("..." if n > 8 else "")) if n else "-"
        tk.Label(body,
                 text=f"{n} cluster{'s' if n != 1 else ''}: {ids_s}",
                 font=("Segoe UI", 8), bg=C["panel"], fg=C["text_dim"],
                 wraplength=240, justify="left").pack(anchor="w")

        btn_row = tk.Frame(body, bg=C["panel"])
        btn_row.pack(anchor="w", pady=(4,0))
        for txt, cmd, fg2, bg2 in [
            ("  Rename",  lambda u=bg.uid: self._rename_group(u),  C["text_dim"], C["btn"]),
            ("  Colour", lambda u=bg.uid: self._change_colour(u),  C["text_dim"], C["btn"]),
            ("  Delete",  lambda u=bg.uid: self._delete_group(u),  C["accent4"],  "#fdecea"),
        ]:
            tk.Button(btn_row, text=txt, command=cmd,
                      bg=bg2, fg=fg2, font=("Segoe UI", 8),
                      relief="flat", bd=0, padx=6, pady=2,
                      activebackground=C["btn_hover"], cursor="hand2"
                      ).pack(side="left", padx=2)

    def _rename_group(self, uid: int):
        bg = self.sd.behaviour_groups.get(uid)
        if not bg:
            return
        new = simpledialog.askstring("Rename", f"Rename '{bg.name}' to:",
                                     initialvalue=bg.name, parent=self)
        if not new or not new.strip() or new.strip() == bg.name:
            return
        new = new.strip()
        if any(b.name == new for b in self.sd.behaviour_groups.values()
               if b.uid != uid):
            messagebox.showwarning("Duplicate", f"'{new}' already exists.")
            return
        self.sd.rename_behaviour_group(uid, new)
        self._status(f"Renamed to '{new}'")
        self._after_action()

    def _change_colour(self, uid: int):
        bg = self.sd.behaviour_groups.get(uid)
        if bg:
            ColourPickerDialog(self, bg, callback=self._after_action)

    def _delete_group(self, uid: int):
        bg = self.sd.behaviour_groups.get(uid)
        if not bg:
            return
        if not messagebox.askyesno("Delete Group",
                f"Delete '{bg.name}'?\n"
                f"({len(bg.cluster_ids)} clusters will return to pending)"):
            return
        self.sd.delete_behaviour_group(uid)
        self._status(f"Deleted '{bg.name}'")
        self._after_action()

    def _refresh_assign_buttons(self):
        for w in self.assign_btn_frame.winfo_children():
            w.destroy()
        if not self.sd.behaviour_groups:
            tk.Label(self.assign_btn_frame,
                     text="Create a behaviour group to start classifying.",
                     font=("Segoe UI", 9), bg=C["panel"],
                     fg=C["text_dim"]).pack(pady=6)
            return
        wrap = tk.Frame(self.assign_btn_frame, bg=C["panel"])
        wrap.pack(fill="x")
        for i, (uid, bg) in enumerate(
                sorted(self.sd.behaviour_groups.items(),
                       key=lambda kv: kv[1].name)):
            sc  = str(i+1) if i < 9 else ""
            lbl = f"[{sc}] {bg.name}" if sc else bg.name
            tk.Button(wrap, text=lbl,
                      command=lambda u=uid: self._assign_to(u),
                      bg=C["panel"], fg=bg.colour,
                      font=("Segoe UI", 9, "bold"),
                      relief="solid", bd=1,
                      highlightbackground=bg.colour, highlightthickness=1,
                      padx=10, pady=4,
                      activebackground=C["btn_hover"],
                      activeforeground=bg.colour,
                      cursor="hand2"
                      ).pack(side="left", padx=3, pady=4)

    #  
    #  EXPORT / SESSION
    #  

    def _export_tsv(self):
        sd = self.sd
        if not sd.clusters:
            messagebox.showwarning("Nothing to export", "Load a folder first.")
            return
        folder = filedialog.askdirectory(title="Choose export folder", parent=self)
        if not folder:
            return
        out = Path(folder)

        # 1. Full mapping
        try:
            with open(out/"cluster_behaviour_mapping.tsv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["bsoid_group_id","behaviour_group","status",
                             "example_count","gif_path","mp4_paths"])
                for gid in sorted(sd.clusters):
                    cl = sd.clusters[gid]
                    w.writerow([gid, cl.behaviour_group or "", cl.status,
                                cl.example_count,
                                str(cl.gif_path) if cl.gif_path else "",
                                ";".join(str(p) for p in cl.mp4_paths)])
        except OSError as e:
            messagebox.showerror("Export error", str(e))
            return

        # 2. Per-group TSVs
        gfiles = []
        for uid, bg in sorted(sd.behaviour_groups.items(),
                               key=lambda kv: kv[1].name):
            safe = re.sub(r"[^\w\-]", "_", bg.name)
            p    = out / f"behaviour_{safe}.tsv"
            try:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["bsoid_group_id","example_count",
                                 "gif_path","mp4_paths"])
                    for gid in sorted(bg.cluster_ids):
                        cl = sd.clusters.get(gid)
                        if cl:
                            w.writerow([gid, cl.example_count,
                                        str(cl.gif_path) if cl.gif_path else "",
                                        ";".join(str(p2) for p2 in cl.mp4_paths)])
                gfiles.append(p.name)
            except OSError as e:
                messagebox.showerror("Export error", str(e))

        # 3. Ignored
        try:
            with open(out/"ignored_clusters.tsv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["bsoid_group_id","example_count"])
                for gid in sorted(sd.clusters):
                    cl = sd.clusters[gid]
                    if cl.status == "ignored":
                        w.writerow([gid, cl.example_count])
        except OSError as e:
            messagebox.showerror("Export error", str(e))

        total, assigned, ignored, pending = sd.stats()
        messagebox.showinfo("Export complete",
            f"Saved to: {folder}\n\n"
            f"Files:\n  cluster_behaviour_mapping.tsv\n"
            f"  ignored_clusters.tsv\n"
            + "".join(f"  {f}\n" for f in gfiles) +
            f"\nTotal: {total}  |  Assigned: {assigned}  |  "
            f"Ignored: {ignored}  |  Pending: {pending}")
        self._status(f"Exported {len(gfiles)+2} TSV files to {folder}")

    def _save_session(self):
        sd = self.sd
        if not sd.clusters:
            messagebox.showwarning("Nothing to save", "Load a folder first.")
            return
        if not sd.session_file:
            p = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON session","*.json"),("All","*.*")],
                title="Save annotation session", parent=self)
            if not p:
                return
            sd.session_file = Path(p)
        data = {
            "folder":     str(sd.folder_path),
            "bg_counter": BehaviourGroup._counter,
            "rotation":   sd.rotation,
            "speed":      sd.speed,
            "zoom":       sd.zoom,
            "clusters":   {str(gid): {"status": cl.status,
                                       "behaviour_group": cl.behaviour_group}
                           for gid, cl in sd.clusters.items()},
            "behaviour_groups": {str(uid): {"name":        bg.name,
                                             "colour":      bg.colour,
                                             "cluster_ids": bg.cluster_ids}
                                 for uid, bg in sd.behaviour_groups.items()},
        }
        try:
            with open(sd.session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._status(f"Session saved: {sd.session_file.name}")
        except OSError as e:
            messagebox.showerror("Save error", str(e))

    def load_session(self, path: Path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        folder   = Path(data["folder"])
        clusters = discover_clusters(folder)
        if not clusters:
            messagebox.showerror("Error",
                f"Cannot reload clusters from:\n{folder}")
            return
        self._player.stop_all()
        sd = self.sd
        sd.folder_path = folder
        sd.clusters    = clusters
        sd.cluster_order = sorted(clusters.keys())
        sd.behaviour_groups = {}
        BehaviourGroup._counter = data.get("bg_counter", 0)
        for uid_s, bgd in data.get("behaviour_groups", {}).items():
            bg = BehaviourGroup.__new__(BehaviourGroup)
            bg.uid = int(uid_s); bg.name = bgd["name"]
            bg.colour = bgd["colour"]; bg.cluster_ids = bgd["cluster_ids"]
            sd.behaviour_groups[bg.uid] = bg
        for gid_s, cld in data.get("clusters", {}).items():
            gid = int(gid_s)
            if gid in sd.clusters:
                sd.clusters[gid].status          = cld["status"]
                sd.clusters[gid].behaviour_group = cld["behaviour_group"]
        sd.session_file  = path
        sd.current_index = 0
        sd.rotation      = data.get("rotation", 0)
        sd.speed         = data.get("speed",    1.0)
        sd.zoom          = data.get("zoom",     1.0)
        self._rot_var.set(sd.rotation)
        self._speed_var.set(sd.speed)
        self._zoom_var.set(sd.zoom)
        self.title(f"CUBE Annotator     {folder.name}")
        self._refresh_cluster_list()
        self._refresh_group_panel()
        self._refresh_assign_buttons()
        self._load_current()
        self._status(f"Session loaded: {path.name}")

    #  
    #  HELPERS
    #  

    def _status(self, msg: str):
        self.lbl_status.configure(text=msg)

    def _update_stats(self):
        if not self.sd.clusters:
            self.lbl_stats.configure(text="")
            return
        total, assigned, ignored, pending = self.sd.stats()
        pct = int(100*(assigned+ignored)/max(total,1))
        self.lbl_stats.configure(
            text=f"  {total} clusters  -  "
                 f"v{assigned} assigned  -  "
                 f"X{ignored} ignored  -  "
                 f" {pending} pending  -  {pct}% done")


#  
#  ENTRY POINT
#  

def main():
    missing = []
    if not PIL_OK:
        missing.append("pillow")
    if not CV2_OK:
        missing.append("opencv-python-headless")
    if missing:
        print(f"!  Missing packages: {', '.join(missing)}")
        print(f"   Install with:  pip install {' '.join(missing)}\n")

    app = BSoidAnnotator()

    # Menu bar
    menubar = tk.Menu(app)
    sm = tk.Menu(menubar, tearoff=0,
                 bg=C["panel"], fg=C["text"],
                 activebackground=C["accent"], activeforeground="white")
    sm.add_command(label="Open Folder...",   command=app._open_folder)
    sm.add_command(label="Save Session",   command=app._save_session)
    sm.add_command(label="Load Session...",
        command=lambda: _load_session_dlg(app))
    sm.add_separator()
    sm.add_command(label="Export TSVs...",   command=app._export_tsv)
    sm.add_separator()
    sm.add_command(label="Quit",           command=app._on_close)
    menubar.add_cascade(label="Session", menu=sm)

    hm = tk.Menu(menubar, tearoff=0,
                 bg=C["panel"], fg=C["text"],
                 activebackground=C["accent"], activeforeground="white")
    hm.add_command(label="Keyboard shortcuts", command=lambda: _show_help(app))
    menubar.add_cascade(label="Help", menu=hm)
    app.configure(menu=menubar)

    app.mainloop()


def _load_session_dlg(app: BSoidAnnotator):
    p = filedialog.askopenfilename(
        filetypes=[("JSON session","*.json"),("All","*.*")],
        title="Load annotation session", parent=app)
    if p:
        app.load_session(Path(p))


def _show_help(parent):
    win = tk.Toplevel(parent)
    win.title("Keyboard Shortcuts")
    win.configure(bg=C["bg"])
    win.resizable(False, False)
    rows = [
        ("   /  ->",  "Previous / next cluster"),
        ("Space",     "Reload / replay current cluster"),
        ("N",         "Create a new behaviour group"),
        ("I",         "Ignore current cluster"),
        ("1   9",    "Assign to behaviour group 1 9"),
        ("Ctrl+S",   "Save session (JSON)"),
        ("Ctrl+Z",   "Undo last assignment"),
    ]
    tk.Label(win, text="Keyboard Shortcuts",
             font=("Segoe UI", 12, "bold"),
             bg=C["bg"], fg=C["accent"]).pack(pady=(14,8))
    for key, desc in rows:
        r = tk.Frame(win, bg=C["bg"])
        r.pack(fill="x", padx=24, pady=3)
        tk.Label(r, text=key,
                 font=("Consolas", 10, "bold"),
                 bg=C["btn"], fg=C["accent3"],
                 width=12, anchor="e", padx=6
                 ).pack(side="left")
        tk.Label(r, text=desc,
                 font=("Segoe UI", 10),
                 bg=C["bg"], fg=C["text"]
                 ).pack(side="left", padx=10)
    tk.Button(win, text="Close", command=win.destroy,
              bg=C["btn"], fg=C["text"],
              font=("Segoe UI", 9),
              relief="flat", bd=0, padx=12, pady=4,
              cursor="hand2").pack(pady=14)


if __name__ == "__main__":
    main()