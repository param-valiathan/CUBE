# -*- coding: utf-8 -*-
"""
CUBE: Comprehensive Unsupervised Behavioral Explorer
====================================================
v3.0  —  DeepLabCut  ▸  CUBE engine  ▸  Annotator  ▸  Analyser

New in v3
---------
Smart Adapt mode (Scenario A): instead of running per-video Zoo adaptation
on every file (slow), CUBE now selects a single *representative* video
(closest to the dataset's median pixel brightness), adapts the SuperAnimal
Zoo model to that one video, extracts the fine-tuned weights, creates a
named DLC inference project (base-folder name), injects init_weights via
ruamel.yaml, and then runs high-throughput batch inference via
dlc.analyze_videos on all remaining videos with adaptive OOM recovery.

Enable via the "Smart Adapt (v3)" checkbox in DLC & Prep Settings.

Single-file launcher.  Place in the same folder as:
    cube_core.py           (required — V2 analysis engine)
    cube_analyser.py       (required for Step 5)
    cube_video_explorer.py (required for Step 4)

Step 1 — Run DLC inference       (DeepLabCut SuperAnimal)
Step 2 — CUBE pre-processing     (bodypart filtering, H5/CSV export)
Step 3 — CUBE clustering engine  (V2 features · UMAP · HDBSCAN · MLP)
Step 4 — Video annotation        (label clusters via example clips)
Step 5 — Behaviour analysis      (metrics, ethograms, statistics)

Sessions are saved as JSON after every step so analysis can resume after crash.

Requirements
------------
    pip install pillow opencv-python-headless scipy scikit-learn umap-learn customtkinter
    conda install -c conda-forge hdbscan
"""

#  " "  stdlib  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 
import importlib.util
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

#  " "  GUI  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

#  " "  local engine  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 
try:
    from cube_core import (
        PipelineLogger, BSoidEngine, run_bsoid_prep, filter_dlc_h5,
        cleanup_video_byproducts,
    )
    CORE_OK = True
except ImportError as _ce:
    CORE_OK = False
    _CORE_ERR = str(_ce)
    def cleanup_video_byproducts(*_a, **_kw): pass   # safe no-op if core missing
    def filter_dlc_h5(h5_path, *_a, out_path=None, **_kw):  # no-op; just copy
        import shutil as _sh
        dst = out_path or h5_path.with_name(h5_path.stem + "_filtered.h5")
        _sh.copy2(str(h5_path), str(dst))
        return dst

#  " "   optional companion scripts  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 
HERE = Path(__file__).resolve().parent

def _load_script(names: list):
    for name in names:
        p = HERE / name
        if p.is_file():
            try:
                spec = importlib.util.spec_from_file_location(
                    p.stem.replace("-","_"), p)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod, p
            except Exception:
                return None, p
    return None, None

_MOD_VIDEO,    _PATH_VIDEO    = _load_script(["cube_video_explorer.py",
                                               "BSOID_VIDEO_EXPLR.py"])
_MOD_ANALYSER, _PATH_ANALYSER = _load_script(
    ["cube_analyser.py"])

CTK_OK = importlib.util.find_spec("customtkinter") is not None

#  
#  COLOUR PALETTE  &  THEME
#  

C = dict(
    bg      = "#09090f",
    panel   = "#111120",
    card    = "#16162a",
    card2   = "#1e1e35",
    border  = "#2a2a4a",
    accent  = "#e94560",
    green   = "#4caf50",
    cyan    = "#00b4d8",
    yellow  = "#ffd60a",
    orange  = "#ff9800",
    purple  = "#9c27b0",
    red     = "#f44336",
    text    = "#eaeaea",
    subtext = "#7788aa",
    dim     = "#444466",
    log_bg  = "#07070d",
    log_fg  = "#00ff88",
)

try:
    with open(HERE / "theme.txt", "r", encoding="utf-8") as _f:
        if _f.read().strip() == "light":
            C["bg"] = "#f0f2f5"
            C["panel"] = "#ffffff"
            C["card"] = "#f8f9fa"
            C["card2"] = "#e9ecef"
            C["border"] = "#dee2e6"
            C["text"] = "#222222"
            C["subtext"] = "#666666"
            C["log_bg"] = "#ffffff"
            C["log_fg"] = "#333333"
except Exception:
    pass

# Per-step colours
STEP_META = [
    dict(num=1, key="dlc",        icon=" ", title="DLC Inference",
         subtitle="DeepLabCut SuperAnimal on raw videos",
         bg="#1a3a1a", accent="#4caf50"),
    dict(num=2, key="bsoid_prep", icon="", title="CUBE Pre-processing",
         subtitle="Filter bodyparts · export H5/CSV",
         bg="#1a1a3a", accent="#00b4d8"),
    dict(num=3, key="bsoid_run",  icon=" ",  title="CUBE Clustering",
         subtitle="V2 features · UMAP · HDBSCAN · MLP",
         bg="#2a1a4a", accent="#9c27b0"),
    dict(num=4, key="annotate",   icon=" ", title="Video Annotation",
         subtitle="Label clusters via example clips",
         bg="#4a2a1a", accent="#ff9800"),
    dict(num=5, key="analyse",    icon=" ", title="Behaviour Analysis",
         subtitle="Metrics, ethograms, statistics",
         bg="#1a1a4a", accent="#e94560"),
]

# DLC settings
RESOLUTION_PRESETS = {
    "Original":                          None,
    "Long-edge 1280  (720p equivalent)": 1280,
    "Long-edge 854   (480p equivalent)": 854,
    "Long-edge 640   (360p equivalent)": 640,
}
FILTER_OPTIONS = {
    "None (raw output)":                    [],
    "Median (spike removal)":               ["median"],
    "Gaussian smooth":                      ["gaussian"],
    "Butterworth low-pass":                 ["butterworth"],
    "Savitzky-Golay":                       ["savgol"],
    "Kalman smoother":                      ["kalman"],
    "Sequential  Median -> Gaussian":       ["median", "gaussian"],
    "Sequential  Median -> Butterworth":    ["median", "butterworth"],
    "Sequential  Median -> Savitzky-Golay": ["median", "savgol"],
}
COOLDOWN_OPTIONS = {"None":0,"5 s":5,"15 s":15,"30 s":30,"60 s":60}


#  
#  SESSION STATE  (JSON-serialisable)
#  

SESSION_EXT = ".pipeline_session.json"

class SessionState:
    DEFAULTS = dict(
        version         = "2.0",
        created         = "",
        last_saved      = "",
        step_status     = {},       # key  ' idle|running|done|error|skipped
        video_folders   = [],
        output_root     = "",
        fps             = 30,
        # DLC settings
        dlc_resolution  = "Long-edge 1280  (720p equivalent)",
        dlc_adapt       = True,
        dlc_epochs      = 15,
        auto_bsoid      = False,
        dlc_pseudo_thr  = 0.50,
        dlc_filter      = "Sequential  Median -> Gaussian",
        dlc_filtered_vid= True,
        dlc_delete_orig = False,
        dlc_cooldown    = "15 s",
        dlc_run_prep    = True,
        dlc_smart_adapt = False,
        # BSOID prep
        bsoid_min_conf  = 0.35,
        # BSOID engine
        engine_cfg      = {},
        # Experimental group assignments  {folder_path: group_name}
        video_groups    = {},
        # paths set by steps
        bsoid_ready_dirs= [],
        engine_out_dirs = [],
        mapping_file    = "",
        bout_lengths_paths = [],
        ntfy_topic      = "",
    )

    def __init__(self):
        self._d = dict(self.DEFAULTS)
        self._d["created"] = datetime.now().isoformat()
        self._d["step_status"] = {s["key"]: "idle" for s in STEP_META}
        self._path: Path | None = None

    def save(self, path: Path | None = None):
        path = path or self._path
        if path is None:
            return
        self._path = path
        self._d["last_saved"] = datetime.now().isoformat()
        try:
            path.write_text(json.dumps(self._d, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def load(cls, path: Path) -> "SessionState":
        obj = cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            obj._d.update(raw)
            obj._path = path
        except Exception:
            pass
        return obj

    def __getitem__(self, k):
        return self._d.get(k, self.DEFAULTS.get(k))
    def __setitem__(self, k, v):
        self._d[k] = v
    def get(self, k, default=None):
        return self._d.get(k, default)
    def is_done(self, key):
        return self._d["step_status"].get(key) == "done"
    def set_status(self, key, status):
        self._d["step_status"][key] = status


#  
#  LOG PANEL  (polls PipelineLogger queue)
#  

class LogPanel(tk.Frame):
    _COLOURS = {
        "INFO":    C["text"],
        "STEP":    "#ff88ff",
        "WARN":    C["yellow"],
        "ERROR":   C["red"],
        "SUCCESS": C["green"],
        "DEBUG":   C["dim"],
    }

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        self._logger = None

        tb = tk.Frame(self, bg=C["panel"])
        tb.pack(fill="x", padx=6, pady=(4, 2))
        tk.Label(tb, text="   Pipeline Log",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["panel"], fg=C["cyan"]).pack(side="left")
        tk.Button(tb, text="Clear",
                  font=("Segoe UI", 8), bg=C["card2"], fg=C["subtext"],
                  relief="flat", padx=6, cursor="hand2",
                  command=self.clear).pack(side="right", padx=2)
        tk.Button(tb, text="  Open log",
                  font=("Segoe UI", 8), bg=C["card2"], fg=C["subtext"],
                  relief="flat", padx=6, cursor="hand2",
                  command=self._open_log).pack(side="right", padx=2)

        tf = tk.Frame(self, bg=C["panel"])
        tf.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        sb = tk.Scrollbar(tf, bg=C["card"], troughcolor=C["panel"])
        sb.pack(side="right", fill="y")
        self._txt = tk.Text(tf, bg=C["log_bg"], fg=C["log_fg"],
                            font=("Consolas", 9), wrap="word",
                            state="disabled", relief="flat", bd=0,
                            insertbackground=C["text"],
                            yscrollcommand=sb.set)
        self._txt.pack(side="left", fill="both", expand=True)
        sb.config(command=self._txt.yview)
        for tag, colour in self._COLOURS.items():
            self._txt.tag_config(tag, foreground=colour)
        self._txt.tag_config("TS", foreground=C["dim"])

    def attach(self, logger: PipelineLogger):
        self._logger = logger
        self._poll()

    def _poll(self):
        if not self.winfo_exists():
            return
        if self._logger is None:
            self.after(200, self._poll)
            return
        try:
            while True:
                level, msg, ts = self._logger._q.get_nowait()
                self._append(level, msg, ts)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _append(self, level, msg, ts):
        self._txt.configure(state="normal")
        self._txt.insert("end", f"[{ts}] ", "TS")
        tag = level if level in self._COLOURS else "INFO"
        self._txt.insert("end", msg + "\n", tag)
        self._txt.see("end")
        self._txt.configure(state="disabled")

    def append_direct(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._append(level, msg, ts)

    def clear(self):
        self._txt.configure(state="normal")
        self._txt.delete("1.0", "end")
        self._txt.configure(state="disabled")

    def _open_log(self):
        if self._logger and self._logger.log_path.is_file():
            _open_path(self._logger.log_path)


#  
#  PROGRESS BAR  (dual: overall + per-step)
#  

class DualProgressBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        style = ttk.Style()
        style.configure("Overall.Horizontal.TProgressbar",
                        troughcolor=C["card"], background=C["cyan"],
                        thickness=10)
        style.configure("Step.Horizontal.TProgressbar",
                        troughcolor=C["card"], background=C["green"],
                        thickness=8)
        tk.Label(self, text="Overall", font=("Segoe UI", 8),
                 bg=C["panel"], fg=C["subtext"]).pack(anchor="w", padx=8)
        self._overall = ttk.Progressbar(
            self, style="Overall.Horizontal.TProgressbar",
            mode="determinate", maximum=len(STEP_META))
        self._overall.pack(fill="x", padx=8, pady=(0, 4))
        self._step_lbl = tk.Label(self, text="Step:  ",
                                  font=("Segoe UI", 8),
                                  bg=C["panel"], fg=C["subtext"])
        self._step_lbl.pack(anchor="w", padx=8)
        self._step = ttk.Progressbar(
            self, style="Step.Horizontal.TProgressbar",
            mode="determinate")
        self._step.pack(fill="x", padx=8, pady=(0, 2))
        self._eta_lbl = tk.Label(self, text="",
                                  font=("Segoe UI", 8),
                                  bg=C["panel"], fg=C["subtext"])
        self._eta_lbl.pack(anchor="e", padx=8, pady=(0, 4))
        self._t0 = 0.0

    def set_overall(self, done: int):
        self._overall["value"] = done

    def step_start(self, label: str, maximum: int = 100):
        self._step["mode"]    = "determinate"
        self._step["value"]   = 0
        self._step["maximum"] = max(1, maximum)
        self._step_lbl.configure(text=f"Step: {label}")
        self._eta_lbl.configure(text="")
        self._t0 = time.time()

    def step_tick(self, value: int, maximum: int | None = None):
        if maximum is not None:
            self._step["maximum"] = max(1, maximum)
        self._step["value"] = value
        elapsed = time.time() - self._t0
        if self._t0 and value > 0 and elapsed > 0:
            rate   = value / elapsed
            remain = (self._step["maximum"] - value) / max(rate, 1e-9)
            m, s   = divmod(int(remain), 60)
            self._eta_lbl.configure(text=f"ETA {m:02d}:{s:02d}")

    def step_indeterminate(self, label: str):
        self._step["mode"] = "indeterminate"
        self._step_lbl.configure(text=f"Step: {label}")
        self._step.start(12)

    def step_label(self, text: str):
        """Update the sub-step label text without resetting the progress bar."""
        self._step_lbl.configure(text=f"Step: {text}")

    def step_done(self):
        self._step.stop()
        self._step["mode"]  = "determinate"
        self._step["value"] = self._step["maximum"]
        self._eta_lbl.configure(text="")


#  
#  STEP CARD  WIDGET
#  

_BADGE_STATES = {
    "idle":    ("#555566", " -   Waiting"),
    "ready":   ("#00b4d8", " -   Ready"),
    "running": ("#ffd60a", "   Running "),
    "done":    ("#4caf50", "v  Complete"),
    "error":   ("#f44336", " -  Error"),
    "skipped": ("#888899", "   Skipped"),
}

class StepCard(tk.Frame):
    def __init__(self, parent, meta: dict, launch_cmd, **kw):
        bg = meta["bg"]
        super().__init__(parent, bg=bg, bd=0,
                         highlightbackground=meta["accent"],
                         highlightthickness=2, **kw)
        self._bg     = bg
        self._accent = meta["accent"]
        self._key    = meta["key"]

        # header
        hdr = tk.Frame(self, bg=bg)
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(hdr, text=f"{meta['icon']}  Step {meta['num']}",
                 font=("Segoe UI", 10, "bold"),
                 bg=bg, fg=meta["accent"]).pack(side="left")
        self._badge = tk.Label(hdr, text=" -   Waiting",
                               font=("Segoe UI", 8, "bold"),
                               fg="#555566", bg=bg, padx=8, pady=2)
        self._badge.pack(side="right")

        tk.Label(self, text=meta["title"],
                 font=("Segoe UI", 12, "bold"),
                 bg=bg, fg=C["text"],
                 wraplength=195, justify="left").pack(
            anchor="w", padx=10, pady=(0, 2))
        tk.Label(self, text=meta["subtitle"],
                 font=("Segoe UI", 8),
                 bg=bg, fg=C["subtext"],
                 wraplength=195, justify="left").pack(
            anchor="w", padx=10, pady=(0, 8))

        self._btn = tk.Button(
            self, text=f"   Step {meta['num']}",
            font=("Segoe UI", 9, "bold"),
            bg=meta["accent"], fg="white",
            activebackground=meta["accent"],
            relief="flat", padx=10, pady=6,
            cursor="hand2", command=launch_cmd)
        self._btn.pack(fill="x", padx=10, pady=(0, 10))

    def set_status(self, state: str):
        colour, text = _BADGE_STATES.get(state, _BADGE_STATES["idle"])
        self._badge.configure(text=text, fg=colour)

    def enable(self):
        self._btn.configure(state="normal")

    def disable(self):
        self._btn.configure(state="disabled")


#  
#  SETTINGS PANEL  (collapsible)
#  

class SettingsPanel(tk.Frame):
    _DLC_ROWS = [
        ("dlc_resolution",   "Resolution",          "combo",
         list(RESOLUTION_PRESETS.keys()),
         "Long-edge 1280  (720p equivalent)",
         "Resize before inference"),
        ("dlc_adapt",        "Video adapt",         "bool", None, True,
         "Fine-tune per video (better tracking)"),
        ("dlc_smart_adapt",  "Smart Adapt (v3)",    "bool", None, True,
         "Select 1 representative video → adapt once → reuse for all (Scenario A)"),
        ("dlc_epochs",       "Adapt epochs",        "int",  (4,30,2),  15,
         "15=recommended  4=fast"),
        ("dlc_pseudo_thr",   "Pseudo threshold",    "float",(0.2,0.8,0.05),0.50,
         "Higher = stricter pseudo-labels"),
        ("dlc_filter",       "Post-filter",         "combo",
         list(FILTER_OPTIONS.keys()),
         "Sequential  Median -> Gaussian",
         "Smoothing applied to H5 trajectories"),
        ("dlc_filtered_vid", "Filtered video",      "bool", None, True,
         "Create quad overlay from filtered H5"),
        ("dlc_delete_orig",  "Delete original",     "bool", None, False,
         "   Irreversible — deletes source video"),
        ("dlc_run_prep",     "Run CUBE prep",       "bool", None, True,
         "Auto-run Step 2 after DLC"),
        ("auto_bsoid",       "Auto-run Steps 2+3",  "bool", None, True,
         "Run pre-processing + clustering after DLC completes"),
        ("dlc_cooldown",     "Cooldown",            "combo",
         list(COOLDOWN_OPTIONS.keys()), "15 s",
         "GPU cooldown between videos"),
        ("fps",              "Recording FPS",       "int",  (1,500,1), 30,
         "Frames per second"),
        ("bsoid_min_conf",   "Min BP confidence",   "float",(0.1,0.9,0.05),0.35,
         "Bodyparts below this are excluded"),
        ("ntfy_topic",       "Notification Topic",  "str",  None, "",
         "ntfy.sh topic name for push alerts"),
    ]
    _ENGINE_ROWS = [
        ("body_normalise",       "Body normalisation",   "bool", None, True,
         "Divide distances by nose-to-tailbase length"),
        ("likelihood_thresh",    "Likelihood threshold", "float",(0.1,0.9,0.05),0.30,""),
        ("boxcar_win_sec",       "Boxcar smooth (s)",    "float",(0.0,0.5,0.01),0.07,""),
        ("train_frac",           "UMAP train fraction",  "float",(0.05,1.0,0.05),0.30,""),
        ("umap_n_neighbors",     "UMAP n_neighbors",     "int",  (5,200,5),   60,""),
        ("umap_n_components",    "UMAP n_components",    "int",  (2,8,1),      2,""),
        ("umap_min_dist",        "UMAP min_dist",        "float",(0.0,1.0,0.05),0.10,""),
        ("umap_random_state",    "UMAP random seed",     "int",  (0,9999,1),  42,""),
        ("hdbscan_metric",       "HDBSCAN metric",       "combo",
         ["euclidean","manhattan","cosine"],"euclidean",""),
        ("hdbscan_method",       "HDBSCAN method",       "combo",
         ["eom","leaf"],"eom","eom=larger, leaf=finer clusters"),
        ("mlp_hidden",           "MLP layers",           "str",  None, "100,50",""),
        ("mlp_max_iter",         "MLP max iter",         "int",  (100,5000,100),1000,""),
        ("cv_folds",             "CV folds",             "int",  (2,10,1),     5,""),
        ("min_epoch_dur_s",      "Min epoch dur (s)",    "float",(0.0,60.0,0.1),0.0,""),
        ("max_epoch_dur_s",      "Max epoch dur (s)",    "float",(1.0,600.0,1.0),999.0,""),
        ("output_fps",           "Output video FPS",     "int",  (1,60,1),    15,""),
        ("max_clips_per_cluster","Max clips/cluster",    "int",  (1,10,1),     3,""),
        ("save_plots",           "Save plots",           "bool", None, True,""),
        ("save_videos",          "Save videos",          "bool", None, True,""),
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        self._vars: dict = {}
        self._bodies: list = []
        self._expanded: list = []
        # Initialise all DLC vars with defaults — UI lives in DLCPrepSettingsWindow
        self._init_dlc_vars()

    def _init_dlc_vars(self):
        """Create tk.Var objects for every DLC row (no widgets — popup owns those)."""
        for key, _label, wtype, _opts, default, _tip in self._DLC_ROWS:
            if wtype == "bool":
                self._vars[key] = tk.BooleanVar(value=bool(default))
            elif wtype == "int":
                self._vars[key] = tk.IntVar(value=int(default))
            elif wtype == "float":
                self._vars[key] = tk.DoubleVar(value=float(default))
            else:
                self._vars[key] = tk.StringVar(value=str(default))

    def _build_section(self, title: str, rows: list):
        expanded = tk.BooleanVar(value=False)
        self._expanded.append(expanded)

        hdr = tk.Frame(self, bg=C["card2"],
                       highlightbackground=C["border"],
                       highlightthickness=1)
        hdr.pack(fill="x", pady=(4, 0))
        btn = tk.Button(hdr, text=f"   {title}",
                        font=("Segoe UI", 9, "bold"),
                        bg=C["card2"], fg=C["yellow"],
                        relief="flat", anchor="w", padx=12, pady=5,
                        cursor="hand2")
        btn.pack(fill="x")

        body = tk.Frame(self, bg=C["card"])
        self._bodies.append(body)

        def _toggle(b=body, e=expanded, bt=btn, t=title):
            if e.get():
                b.pack_forget()
                bt.configure(text=f"   {t}")
                e.set(False)
            else:
                b.pack(fill="x", padx=8, pady=4)
                bt.configure(text=f"   {t}")
                e.set(True)

        btn.configure(command=_toggle)

        for key, label, wtype, opts, default, tip in rows:
            row = tk.Frame(body, bg=C["card"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, width=24, anchor="w",
                     font=("Segoe UI", 9), bg=C["card"],
                     fg=C["text"]).pack(side="left")
            if wtype == "bool":
                v = tk.BooleanVar(value=default)
                tk.Checkbutton(row, variable=v, bg=C["card"],
                               fg=C["green"], selectcolor=C["card2"],
                               activebackground=C["card"]).pack(side="left")
            elif wtype == "combo":
                v = tk.StringVar(value=str(default))
                ttk.Combobox(row, textvariable=v, values=opts,
                             state="readonly", width=26,
                             font=("Segoe UI", 9)).pack(side="left")
            elif wtype == "int":
                lo, hi, step = opts
                v = tk.IntVar(value=int(default))
                tk.Spinbox(row, from_=lo, to=hi, increment=step,
                           textvariable=v, width=7,
                           bg=C["card2"], fg=C["text"],
                           buttonbackground=C["card2"],
                           font=("Segoe UI", 9)).pack(side="left")
            elif wtype == "float":
                lo, hi, step = opts
                v = tk.DoubleVar(value=float(default))
                tk.Spinbox(row, from_=lo, to=hi, increment=step,
                           format="%.2f", textvariable=v, width=7,
                           bg=C["card2"], fg=C["text"],
                           buttonbackground=C["card2"],
                           font=("Segoe UI", 9)).pack(side="left")
            else:
                v = tk.StringVar(value=str(default))
                tk.Entry(row, textvariable=v, width=12,
                         bg=C["card2"], fg=C["text"],
                         insertbackground=C["text"],
                         relief="flat").pack(side="left")
            if tip:
                tk.Label(row, text=tip, font=("Segoe UI", 7),
                         bg=C["card"], fg=C["dim"],
                         wraplength=260).pack(side="left", padx=4)
            self._vars[key] = v

    def get(self, key, default=None):
        v = self._vars.get(key)
        if v is None:
            return default
        try:
            return v.get()
        except Exception:
            return default

    def set_val(self, key, value):
        v = self._vars.get(key)
        if v is None:
            return
        try:
            v.set(value)
        except Exception:
            pass

    def get_engine_cfg(self) -> dict:
        return {}   # engine cfg is now stored in session["engine_cfg"] via AdvancedCUBEWindow

    def apply_session(self, session: SessionState):
        for key in self._vars:
            val = session[key] if key in session._d else None
            if val is not None:
                self.set_val(key, val)
        # engine cfg nested dict
        ec = session.get("engine_cfg", {})
        for k, v in ec.items():
            self.set_val(k, v)

    def export_to_session(self, session: SessionState):
        dlc_keys = [r[0] for r in self._DLC_ROWS]
        for k in dlc_keys:
            session[k] = self.get(k)
        ec = self.get_engine_cfg()
        if ec:  # Don't overwrite engine_cfg set by AdvancedCUBEWindow with {}
            session["engine_cfg"] = ec


#  
#  FOLDER LIST  WIDGET
#  

class FolderList(tk.Frame):
    def __init__(self, parent, on_change=None, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        self._on_change = on_change

        hdr = tk.Frame(self, bg=C["card"])
        hdr.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(hdr, text="   Video source folders",
                 font=("Segoe UI", 10, "bold"),
                 bg=C["card"], fg=C["yellow"]).pack(side="left")
        self._cnt = tk.Label(hdr, text="", font=("Segoe UI", 8),
                             bg=C["card"], fg=C["subtext"])
        self._cnt.pack(side="right")

        lf = tk.Frame(self, bg=C["card"])
        lf.pack(fill="both", expand=True, padx=8, pady=2)
        sb = tk.Scrollbar(lf, bg=C["card2"], troughcolor=C["card"])
        sb.pack(side="right", fill="y")
        self._lb = tk.Listbox(lf, bg=C["card2"], fg=C["text"],
                              font=("Consolas", 9),
                              selectbackground=C["cyan"],
                              selectforeground=C["bg"],
                              relief="flat", bd=0, height=5,
                              yscrollcommand=sb.set)
        self._lb.pack(side="left", fill="both", expand=True)
        sb.config(command=self._lb.yview)

        bf = tk.Frame(self, bg=C["card"])
        bf.pack(fill="x", padx=8, pady=(4, 8))
        for text, cmd, colour in [
            ("  Add",    self._add,    C["green"]),
            ("  Remove", self._remove, C["red"]),
        ]:
            tk.Button(bf, text=text, font=("Segoe UI", 9, "bold"),
                      bg=colour, fg="white", relief="flat",
                      padx=10, pady=4, cursor="hand2",
                      command=cmd).pack(side="left", padx=3)

    def _add(self):
        d = filedialog.askdirectory(title="Select folder containing videos")
        if d and d not in self._lb.get(0, "end"):
            self._lb.insert("end", d)
            self._refresh()
            if self._on_change:
                self._on_change()

    def _remove(self):
        for i in reversed(self._lb.curselection()):
            self._lb.delete(i)
        self._refresh()
        if self._on_change:
            self._on_change()

    def _refresh(self):
        n = self._lb.size()
        self._cnt.configure(text=f"{n} folder(s)")

    def get_folders(self) -> list:
        return list(self._lb.get(0, "end"))

    def set_folders(self, folders: list):
        self._lb.delete(0, "end")
        for f in folders:
            self._lb.insert("end", f)
        self._refresh()


#  
#  HELP WINDOW
#  

def show_help(parent):
    win = tk.Toplevel(parent)
    win.title("CUBE — User Guide")
    win.configure(bg=C["bg"])
    win.geometry("700x600")
    win.resizable(True, True)
    tk.Label(win, text="CUBE: Comprehensive Unsupervised Behavioral Explorer — User Guide",
             font=("Segoe UI", 13, "bold"),
             bg=C["bg"], fg=C["accent"]).pack(pady=(14, 4))

    SECTIONS = [
        ("  Workflow",
         "Add video folders → Step 1 (DLC) → Step 2 (Pre-processing) → "
         "Step 3 (Clustering) → Step 4 (Annotate clips) → Step 5 (Analyse).\n"
         "Each step saves progress automatically.  After a crash, load the "
         "session JSON and continue from the last completed step."),
        ("  Step 1 — DLC Inference",
         "Requires DeepLabCut installed in the active conda environment.\n"
         "Uses SuperAnimal quadruped model.  Video adapt fine-tunes per video "
         "(better tracking, slower).  Outputs H5 pose files and labeled videos "
         "in <video>_results/ subfolders.\n"
         "Pseudo-label folders are deleted automatically after each video to "
         "keep disk usage low and avoid Windows path-length errors."),
        ("  Step 2 — CUBE Pre-processing",
         "Reads *_filtered.h5 files.  Drops the 'individuals' level added by "
         "SuperAnimal.  Filters bodyparts to those meeting the confidence "
         "threshold across ALL sessions.  Exports BSOID_Project_Ready/ with "
         "h5/, csv/, videos/, output/ subdirectories.\n"
         "File names are automatically shortened to prevent MAX_PATH errors."),
        ("  Step 3 — CUBE Clustering (V2 Engine)",
         "Fully programmatic — no external app required.\n"
         "V2 features: fps-adaptive scales (100+200 ms at 30fps; 50+100+200 ms "
         "at 60fps+), optional body-size normalisation (nose-to-tailbase), "
         "smoothed velocity+acceleration, angular body-axis features.\n"
         "→ UMAP (n_components=3) → HDBSCAN (auto-sweep) → MLP classifier.\n"
         "Only required user input: Min / Max bout duration (seconds).\n"
         "Outputs: bout_lengths CSVs, frame labels, epoch stats, UMAP plot,\n"
         "ethograms, validation_dashboard.png, validation_report.json."),
        ("  Experimental Groups",
         "Assign a group name to each video folder using the 'Experimental "
         "Groups' panel.\nSelect a folder in the list, type a group name "
         "(e.g. 'Control' or 'Drug'), and click Apply.\n"
         "Groups are saved in the session and pre-populated in the Analyser "
         "when Step 5 is launched, enabling automatic split-group plots and "
         "Kruskal-Wallis statistics."),
        ("  Step 4 — Video Annotation",
         "Opens the Video Explorer.  Browse example clips per cluster.\n"
         "Assign clusters to named behaviour groups.  Export TSV mapping.\n"
         "Keyboard: arrows navigate | Space replay | N new group | "
         "I ignore | 1–9 assign"),
        ("  Step 5 — Behaviour Analysis",
         "Opens the CUBE Analyser (requires customtkinter).\n"
         "Group Editor → Full Analysis → Combined multi-animal → "
         "Unbiased Analytics with Kruskal-Wallis, volcano plot, reclustering."),
        ("  Sessions",
         "Sessions are auto-saved after every step to:\n"
         "  <output_root>/autosave.pipeline_session.json\n"
         "Load via the 'Load' button.  Step cards show ✓ Complete / ✗ Error."),
        ("  Troubleshooting",
         "cube_core.py not found: place in same folder as this script.\n"
         "DLC not found: activate your DLC conda environment first.\n"
         "H5 MultiIndex error: Steps 2 and 3 fix this automatically.\n"
         "UMAP/HDBSCAN missing: pip install umap-learn + "
         "conda install -c conda-forge hdbscan\n"
         "customtkinter missing: pip install customtkinter\n"
         "validation_report.json: check this file after Step 3 for quality gates."),
    ]

    canvas = tk.Canvas(win, bg=C["bg"], highlightthickness=0)
    sb     = tk.Scrollbar(win, command=canvas.yview, bg=C["card"])
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(fill="both", expand=True)
    inner  = tk.Frame(canvas, bg=C["bg"])
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    for heading, body in SECTIONS:
        tk.Label(inner, text=heading,
                 font=("Segoe UI", 11, "bold"),
                 bg=C["bg"], fg=C["yellow"], anchor="w").pack(
            fill="x", padx=20, pady=(12, 2))
        tk.Label(inner, text=body,
                 font=("Segoe UI", 9),
                 bg=C["bg"], fg=C["text"],
                 justify="left", anchor="w",
                 wraplength=620).pack(fill="x", padx=28, pady=(0, 4))

    tk.Button(win, text="Close", command=win.destroy,
              bg=C["accent"], fg="white",
              font=("Segoe UI", 10), relief="flat",
              padx=20, pady=6, cursor="hand2").pack(pady=12)


#  
#  OS HELPERS
#  

def _open_path(p: Path):
    try:
        if platform.system() == "Windows":
            os.startfile(str(p))
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception:
        pass


#
#  OUTPUT PATH HELPER  — always keeps files on the data drive, never C:
#

def _resolve_work_dir(session: "SessionState") -> Path:
    """
    Return the root directory for logs, workspace, and session autosave.

    Rule: output MUST live on the same drive as the video data — never on
    the system/home drive (C: on Windows).  Priority:
      1. session["output_root"] if set AND not on the system drive
      2. Drive-root\\CUBE_Pipeline  (derived from first video folder's drive)
      3. HERE / "CUBE_Pipeline"     (script directory — last resort)

    Using the drive root (e.g. D:\\CUBE_Pipeline) keeps the workspace
    sibling-level with data folders and avoids run_bsoid_prep scanning it.
    """
    import os as _os
    _sys_drive = Path.home().drive.upper()   # "C:" on most Windows installs

    raw = (session.get("output_root") or "").strip()
    if raw:
        p = Path(raw)
        # Accept only if it is NOT on the system (C:) drive
        if not _sys_drive or p.drive.upper() != _sys_drive:
            return p

    # Derive from first video folder — same drive as the data
    folders = session.get("video_folders", [])
    if folders:
        drive = Path(folders[0]).drive          # e.g. "D:"
        if drive:
            return Path(drive + _os.sep) / "CUBE_Pipeline"

    # Last resort — script directory (likely on D: for this project)
    return HERE / "CUBE_Pipeline"


def _resolve_ffmpeg() -> str:
    """Resolve ffmpeg executable in PATH or Conda Library/bin on Windows."""
    import shutil
    import platform
    import sys
    from pathlib import Path
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    # Windows Conda Env fallback
    if platform.system() == "Windows":
        py_dir = Path(sys.executable).parent
        conda_ffmpeg = py_dir / "Library" / "bin" / "ffmpeg.exe"
        if conda_ffmpeg.is_file():
            return str(conda_ffmpeg)
    return "ffmpeg"


def send_push_notification(session: SessionState, message: str,
                           title: str = "CUBE", logger=None):
    """Send an instant push notification via ntfy.sh."""
    topic = session.get("ntfy_topic", "").strip()
    if not topic:
        if logger:
            logger.warn("[Notify] ntfy_topic is empty — no notification sent. "
                        "Set it in DLC & Prep Settings.")
        return
    try:
        import urllib.request
        import urllib.error
        url = "https://ntfy.sh/" + topic
        headers = {
            "Title": title.replace("—", "-").encode("ascii", "replace").decode("ascii"),
            "Priority": "default"
        }
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if logger:
                logger.info(f"[Notify] Sent to topic '{topic}': {title}")
    except urllib.error.HTTPError as e:
        if logger:
            logger.warn(f"[Notify] ntfy.sh HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        if logger:
            logger.warn(f"[Notify] ntfy.sh unreachable: {e.reason}")
    except Exception as e:
        if logger:
            logger.warn(f"[Notify] Notification failed: {e}")


def _validated_dlc_model_name(model_name: str, session, logger, fallback: str = "hrnet_w32") -> str:
    """Return model_name if DLC knows it; otherwise fall back and warn."""
    try:
        import json as _json, os as _os
        import deeplabcut as _dlc_tmp
        _mzoo = _os.path.join(_os.path.dirname(_dlc_tmp.__file__),
                              "modelzoo", "models_to_framework.json")
        _avail = set(_json.load(open(_mzoo)).keys())
        if model_name not in _avail:
            msg = (f"Architecture '{model_name}' is not supported by the installed "
                   f"DeepLabCut version (supported: {', '.join(sorted(_avail))}). "
                   f"Falling back to '{fallback}'.")
            logger.warn(f"  [DLC] {msg}")
            send_push_notification(session, msg, title="CUBE — Unsupported Architecture", logger=logger)
            return fallback
    except Exception:
        pass
    return model_name


def _apply_dlc_monkeypatch(logger):
    """Dynamically unfreeze BatchNorm statistics in DeepLabCut PyTorch engine during adaptation training."""
    try:
        from deeplabcut.pose_estimation_pytorch.modelzoo.train_from_coco import COCOLoader
        original_update = COCOLoader.update_model_cfg
        
        def custom_update(self, updates):
            if "model.backbone.freeze_bn_stats" in updates:
                updates["model.backbone.freeze_bn_stats"] = False
                logger.info("  [MONKEYPATCH] Unfreezing pose model backbone BatchNorm stats.")
            if "detector.model.freeze_bn_stats" in updates:
                updates["detector.model.freeze_bn_stats"] = False
                logger.info("  [MONKEYPATCH] Unfreezing detector model backbone BatchNorm stats.")
            original_update(self, updates)
            
        COCOLoader.update_model_cfg = custom_update
        logger.info("  [MONKEYPATCH] Successfully wrapped COCOLoader to unfreeze BatchNorm statistics.")
    except Exception as e:
        logger.warn(f"  [MONKEYPATCH] Could not apply BatchNorm unfreeze: {e}")


#
#  STEP IMPLEMENTATIONS  (run in background threads)
#  

def _run_dlc_step(session: SessionState, settings: SettingsPanel,
                  logger: PipelineLogger, pb: DualProgressBar,
                  after_fn):
    """
    Run DeepLabCut SuperAnimal inference on all video folders.
    Mirrors the logic of BatchDLC_2_PreBSOID_Combined_Analyser.py.

    When 'Smart Adapt (v3)' is enabled in settings, dispatches to
    _run_dlc_smart_adapt_step (Scenario A: one representative video
    adapted, adapted weights reused for all videos via analyze_videos).
    """
    # ── Smart Adapt mode (Scenario A / v3) ────────────────────────────────────
    if bool(settings.get("dlc_smart_adapt", False)):
        _run_dlc_smart_adapt_step(session, settings, logger, pb, after_fn)
        return

    try:
        import deeplabcut as dlc
    except ImportError:
        raise ImportError(
            "DeepLabCut is not installed in this environment.\n"
            "Activate your DLC conda environment and relaunch.")

    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not found.  pip install opencv-python-headless")

    import gc

    folders = session["video_folders"]
    if not folders:
        raise ValueError("No video folders selected.")

    # collect videos
    VIDEO_EXTS = {".avi",".mp4",".mov",".mkv",".wmv"}
    video_entries = []
    for root_folder in folders:
        for sub, dirs, files in os.walk(root_folder):
            dirs[:] = [d for d in dirs if not d.endswith("_results")]
            if Path(sub).name.endswith("_results"):
                continue
            for fname in sorted(files):
                if fname.startswith("resized_"):
                    continue
                if Path(fname).suffix.lower() in VIDEO_EXTS:
                    video_entries.append((os.path.join(sub, fname), sub))

    if not video_entries:
        raise ValueError("No video files found in selected folders.")

    total          = len(video_entries)
    long_edge      = RESOLUTION_PRESETS.get(settings.get("dlc_resolution"))
    filter_key     = settings.get("dlc_filter", "Sequential  Median  ' Gaussian")
    filter_types   = FILTER_OPTIONS.get(filter_key, ["median","gaussian"])
    cooldown_secs  = COOLDOWN_OPTIONS.get(settings.get("dlc_cooldown","15 s"), 15)
    use_adapt      = bool(settings.get("dlc_adapt", True))
    n_epochs       = int(settings.get("dlc_epochs", 15))
    pseudo_thr     = float(settings.get("dlc_pseudo_thr", 0.5))
    create_filt_v  = bool(settings.get("dlc_filtered_vid", True))
    delete_orig    = bool(settings.get("dlc_delete_orig", False))
    run_prep       = bool(settings.get("dlc_run_prep", True))

    # ── Advanced DLC parameters (from AdvancedDLCWindow) ─────────────────────
    _adv = session.get("dlc_advanced_cfg", {})
    _use_custom     = bool(_adv.get("dlc_use_custom",    False))
    _sa_name        = str(_adv.get("dlc_superanimal_name", "superanimal_quadruped"))
    _model_name     = _validated_dlc_model_name(
                          str(_adv.get("dlc_architecture", "hrnet_w32")),
                          session, logger)
    _detector_name  = str(_adv.get("dlc_detector",
                          "fasterrcnn_mobilenet_v3_large_fpn"))
    _pcutoff        = float(_adv.get("dlc_pcutoff",        0.6))
    _bbox_thr       = float(_adv.get("dlc_bbox_threshold", 0.6))
    _max_ind        = int(_adv.get("dlc_max_individuals",  1))
    _det_epochs     = int(_adv.get("dlc_det_epochs",       n_epochs))
    _pose_epochs    = int(_adv.get("dlc_pose_epochs",      n_epochs))
    _transfer       = bool(_adv.get("dlc_transfer",        True))
    _custom_config  = str(_adv.get("dlc_custom_config",    ""))
    _scale_mode     = str(_adv.get("dlc_scale_mode",       "Auto"))
    _scale_min      = int(_adv.get("dlc_scale_min",        100))
    _scale_max      = int(_adv.get("dlc_scale_max",        600))
    _scale_step     = int(_adv.get("dlc_scale_step",       50))
    _inf_batch_ov   = int(_adv.get("dlc_inf_batch",        0))
    _det_batch_ov   = int(_adv.get("dlc_det_batch",        0))

    # GPU batch size (auto-detect unless user overrides); capped at 85% of free VRAM
    inf_batch = 8
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            usable_gb = free_gb * 0.85
            inf_batch = 32 if usable_gb >= 10 else (16 if usable_gb >= 5 else 8)
    except Exception:
        pass
    if _inf_batch_ov > 0:
        inf_batch = _inf_batch_ov
    det_batch = _det_batch_ov if _det_batch_ov > 0 else inf_batch

    pb.step_start("DLC inference", total)
    logger.step(f"DLC: {total} video(s) across {len(folders)} folder(s)")
    errors = []

    for idx, (video_path, subfolder) in enumerate(video_entries, 1):
        vname       = os.path.basename(video_path)
        base_noext  = os.path.splitext(vname)[0]
        dest_folder = os.path.join(subfolder, f"{base_noext}_results")
        os.makedirs(dest_folder, exist_ok=True)

        logger(f"[{idx}/{total}]  {vname}")

        # resize
        if long_edge:
            inf_path = os.path.join(dest_folder, f"resized_{base_noext}.mp4")
            if not os.path.exists(inf_path):
                _cap = cv2.VideoCapture(video_path)
                _ow_raw = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _oh_raw = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                _fps    = _cap.get(cv2.CAP_PROP_FPS) or 30
                try:
                    _rot = int(_cap.get(cv2.CAP_PROP_ORIENTATION_META))
                except Exception:
                    _rot = 0
                # Swap visual dimensions for 90°/270° rotated videos
                if _rot in (90, 270):
                    _ow, _oh = _oh_raw, _ow_raw
                else:
                    _ow, _oh = _ow_raw, _oh_raw
                _scale = min(long_edge / max(_ow, _oh), 1.0)
                _nw = int(_ow * _scale) & ~1
                _nh = int(_oh * _scale) & ~1
                if _nw == _ow and _nh == _oh and _rot == 0:
                    _cap.release()
                    shutil.copy2(video_path, inf_path)
                    logger(f"  Video already at/below target — copied to workspace")
                else:
                    logger(f"  Resizing {_ow}x{_oh} → {_nw}x{_nh} via cv2 (rotation={_rot}°)")
                    _fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    _writer = cv2.VideoWriter(inf_path, _fourcc, _fps, (_nw, _nh))
                    _rot_map = {
                        90:  cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }
                    while True:
                        _ret, _frame = _cap.read()
                        if not _ret:
                            break
                        if _rot in _rot_map:
                            _frame = cv2.rotate(_frame, _rot_map[_rot])
                        _writer.write(cv2.resize(_frame, (_nw, _nh),
                                                 interpolation=cv2.INTER_AREA))
                    _cap.release()
                    _writer.release()
                    logger(f"  Resized video saved (long edge {long_edge})")
                if delete_orig:
                    try: os.remove(video_path)
                    except Exception: pass
        else:
            inf_path = os.path.join(dest_folder, vname)
            if video_path != inf_path and not os.path.exists(inf_path):
                shutil.copy2(video_path, inf_path)

        # ── Scale list for SuperAnimal detector ──────────────────────────────
        if _scale_mode == "Manual":
            scale_list = list(range(_scale_min,
                                    _scale_max + _scale_step,
                                    _scale_step))
        else:
            try:
                cap3  = cv2.VideoCapture(inf_path)
                short = min(int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                cap3.release()
            except Exception:
                short = 720
            centre     = max(150, int(short * 0.35))
            scale_list = list(range(max(100, centre - 150),
                                    min(1200, centre + 200), 50))

        # Re-measure free VRAM before each video; DLC retains model weights
        # across calls so VRAM depletes steadily — the startup reading is stale.
        if _inf_batch_ov <= 0:
            try:
                import torch
                if torch.cuda.is_available():
                    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                    usable_gb = free_gb * 0.85
                    inf_batch = 32 if usable_gb >= 10 else (16 if usable_gb >= 5 else 8)
                    det_batch = _det_batch_ov if _det_batch_ov > 0 else inf_batch
                    logger(f"  VRAM: {free_gb:.1f} GB free → batch {inf_batch}")
            except Exception:
                inf_batch = 8
                det_batch = _det_batch_ov if _det_batch_ov > 0 else 8

        h5_before = set(Path(dest_folder).glob("*.h5"))
        try:
            if _use_custom:
                # ── Custom DLC project ────────────────────────────────────────
                if not _custom_config or not Path(_custom_config).is_file():
                    raise FileNotFoundError(
                        f"Custom DLC config not found: {_custom_config!r}\n"
                        "Set it in ⚙ Advanced DLC Parameters.")
                logger(f"  Custom DLC model: {Path(_custom_config).parent.name}")
                dlc.analyze_videos(
                    _custom_config,
                    [inf_path],
                    save_as_csv    = True,
                    destfolder     = dest_folder,
                    batchsize      = inf_batch,
                    robust_nframes = True,
                )
                if create_filt_v:
                    try:
                        dlc.create_labeled_video(
                            _custom_config, [inf_path],
                            destfolder=dest_folder)
                    except Exception:
                        pass
            else:
                # ── SuperAnimal (Zoo model) ───────────────────────────────────
                logger(f"  SuperAnimal: {_sa_name} / {_model_name}")
                _sa_kwargs = dict(
                    superanimal_name              = _sa_name,
                    model_name                    = _model_name,
                    detector_name                 = _detector_name,
                    scale_list                    = scale_list,
                    pcutoff                       = _pcutoff,
                    bbox_threshold                = _bbox_thr,
                    max_individuals               = _max_ind,
                    batch_size                    = inf_batch,
                    detector_batch_size           = det_batch,
                    create_labeled_video          = create_filt_v,
                    video_adapt                   = use_adapt,
                    pseudo_threshold              = pseudo_thr,
                    detector_epochs               = _det_epochs,
                    pose_epochs                   = _pose_epochs,
                    device                        = "auto",
                )
                # superanimal_transfer_learning is accepted in DLC >= 2.3
                try:
                    import inspect as _inspect
                    _sig = _inspect.signature(dlc.video_inference_superanimal)
                    if "superanimal_transfer_learning" in _sig.parameters:
                        _sa_kwargs["superanimal_transfer_learning"] = _transfer
                except Exception:
                    pass
                dlc.video_inference_superanimal([inf_path], **_sa_kwargs)
            logger(f"  ✓  Inference done: {vname}")

            # ── Post-inference: clean names, filter H5, delete byproducts ──────
            h5_new = [p for p in Path(dest_folder).glob("*.h5")
                      if p not in h5_before
                      and not p.name.startswith("BSOID_")
                      and not p.stem.endswith("_filtered")]
            if h5_new:
                # Prefer post-adapt (snapshot) H5 over pre-adapt plain H5
                snap_h5 = [p for p in h5_new if "snapshot" in p.stem]
                final_h5 = snap_h5[0] if snap_h5 else h5_new[0]

                clean_h5       = Path(dest_folder) / f"{base_noext}.h5"
                clean_filtered = Path(dest_folder) / f"{base_noext}_filtered.h5"
                if filter_types:
                    filter_dlc_h5(final_h5, filter_types, log_fn=logger,
                                  out_path=clean_filtered,
                                  fps=float(session.get("fps", 30)))
                else:
                    shutil.copy2(str(final_h5), str(clean_filtered))
                    logger(f"  Saved H5 → {clean_filtered.name}")

                # Rename primary H5 to clean unfiltered name; delete extras
                try:
                    final_h5.rename(clean_h5)
                except Exception:
                    pass
                for p in h5_new:
                    if p != final_h5:
                        try:
                            p.unlink()
                        except Exception:
                            pass

            # Delete before-adapt labeled video (obsolete; after-adapt is kept)
            for p in Path(dest_folder).glob("*_labeled_before_adapt.mp4"):
                try:
                    p.unlink()
                    logger(f"  [cleanup] Deleted: {p.name}")
                except Exception:
                    pass

            # Rename after-adapt labeled video to a short clean name.
            # Use YYYYMMDD_HHMMSS timestamp from the stem when available so the
            # filename stays unique and short; fall back to the first 50 chars.
            for p in Path(dest_folder).glob("*_labeled_after_adapt.mp4"):
                _ts_m = re.search(r"\d{8}_\d{6}", base_noext)
                _short_stem = _ts_m.group(0) if _ts_m else base_noext[:50]
                clean_vid = Path(dest_folder) / f"{_short_stem}_labeled.mp4"
                try:
                    p.rename(clean_vid)
                    logger(f"  [cleanup] Renamed → {clean_vid.name}")
                except Exception:
                    pass
        except Exception:
            msg = f"  ERROR on {vname}: {traceback.format_exc()}"
            logger.error(msg)
            errors.append(vname)

        # Incremental cleanup: remove pseudo_* dirs and DLC .json files
        # immediately after each video to prevent MAX_PATH errors and
        # keep disk usage low during long multi-video batch runs.
        try:
            cleanup_video_byproducts(Path(dest_folder), logger)
        except Exception:
            pass

        # GPU cleanup — synchronize first so async ops release their temporaries
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        if cooldown_secs and idx < total:
            logger(f"  Cooldown {cooldown_secs}s  ")
            time.sleep(cooldown_secs)

        after_fn(lambda cur=idx: pb.step_tick(cur, total))

    if run_prep:
        logger.step("Running CUBE pre-processing (Step 2)...")
        bsoid_roots = []
        for folder in folders:
            root = run_bsoid_prep(
                folder, log_fn=logger,
                min_confidence=float(session["bsoid_min_conf"]))
            if root:
                bsoid_roots.append(str(root))
        session["bsoid_ready_dirs"] = bsoid_roots

    if errors:
        logger.warn(f"DLC finished with {len(errors)} error(s): {errors}")
    else:
        logger.success(f"DLC complete: {total} video(s) processed.")


# ─────────────────────────────────────────────────────────────────────────────
#  SMART ADAPT HELPERS  (v3 — Scenario A: single adaptation, batch inference)
# ─────────────────────────────────────────────────────────────────────────────

def _find_highest_snapshot_path(search_root: Path, logger) -> "str | None":
    """
    Recursively scan *search_root* for snapshot-N.index files (TensorFlow format).
    Returns the base path (without .index) of the highest-numbered snapshot,
    or None if none are found.
    """
    import re as _re
    best_num, best_path = -1, None
    for idx_file in search_root.rglob("*.index"):
        m = _re.search(r"snapshot-(\d+)\.index$", idx_file.name)
        if m:
            num = int(m.group(1))
            if num > best_num:
                best_num, best_path = num, idx_file
    if best_path:
        try:
            rel = best_path.parent.relative_to(search_root)
        except ValueError:
            rel = best_path.parent
        logger.info(f"  Highest snapshot: snapshot-{best_num}  ({rel})")
        return str(best_path.with_suffix(""))   # strip .index extension
    logger.warn(f"  No snapshot-*.index files found under {search_root.name}/")
    return None


def _find_adapted_pt_checkpoints(
        adapt_work: Path, model_name: str, detector_name: str, logger
) -> "tuple[str | None, str | None]":
    """
    Scan *adapt_work* for DLC 3.x PyTorch adapted checkpoints (*.pt).

    DLC 3.x writes adapted weights to:
        pseudo_{video_stem}/checkpoints/snapshot-{model_name}-{N:03}.pt
        pseudo_{video_stem}/checkpoints/snapshot-{detector_name}-{N:03}.pt

    Returns (pose_checkpoint_path, detector_checkpoint_path); either may be None.
    Both 'best' and epoch-numbered variants are matched; the highest epoch wins.
    """
    import re as _re
    pose_pat = _re.compile(
        rf"^snapshot-{_re.escape(model_name)}-(?:best-)?(\d+)\.pt$")
    det_pat  = _re.compile(
        rf"^snapshot-{_re.escape(detector_name)}-(?:best-)?(\d+)\.pt$")

    best_pose_n, best_pose = -1, None
    best_det_n,  best_det  = -1, None

    for pt_file in adapt_work.rglob("*.pt"):
        m = pose_pat.match(pt_file.name)
        if m:
            n = int(m.group(1))
            if n > best_pose_n:
                best_pose_n, best_pose = n, pt_file
        m = det_pat.match(pt_file.name)
        if m:
            n = int(m.group(1))
            if n > best_det_n:
                best_det_n, best_det = n, pt_file

    if best_pose:
        try:
            rel = best_pose.relative_to(adapt_work)
        except ValueError:
            rel = best_pose
        logger.info(f"  Adapted pose checkpoint: {rel}")
    if best_det:
        try:
            rel = best_det.relative_to(adapt_work)
        except ValueError:
            rel = best_det
        logger.info(f"  Adapted detector checkpoint: {rel}")
    if not best_pose and not best_det:
        logger.warn(f"  No *.pt snapshot files found under {adapt_work.name}/")

    return (str(best_pose) if best_pose else None,
            str(best_det)  if best_det  else None)


def _run_dlc_smart_adapt_step(session: SessionState, settings: SettingsPanel,
                               logger: PipelineLogger, pb: DualProgressBar,
                               after_fn):
    """
    Smart Adapt pipeline — Scenario A (v3).

    Phase 1  Discover all videos, validate integrity via cv2, quarantine bad files,
             compute median pixel brightness; select the 1 video closest to that
             median as the representative.
    Phase 2  Run Zoo adaptation (create_video_adaptation_project / fallback) on the
             representative video only.
    Phase 3  Locate the highest-numbered snapshot from the adaptation project.
    Phase 4  Create a named DLC inference project (base-folder name), copy
             snapshots, and inject init_weights via ruamel.yaml.
    Phase 5  Batch inference via dlc.analyze_videos on all valid videos with
             adaptive OOM batchsize reduction.

    Error handling
    ──────────────
    * Corrupted / unreadable video  → quarantined to <output_root>/errors/
    * Adaptation divergence          → fallback to base Zoo weights
    * CUDA OOM during inference      → batchsize halved and retried
    * Any phase-level exception      → graceful fallback to per-video Zoo inference
    """
    import gc
    import re as _re

    os.environ["DL_LIGHT"] = "True"

    try:
        import deeplabcut as dlc
    except ImportError:
        raise ImportError(
            "DeepLabCut is not installed. "
            "Activate your DLC conda environment and relaunch.")
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not found.  pip install opencv-python-headless")

    try:
        import numpy as _np
    except ImportError:
        raise ImportError("NumPy not found.  pip install numpy")

    folders = session["video_folders"]
    if not folders:
        raise ValueError("No video folders selected.")

    # ── Read advanced DLC parameters ──────────────────────────────────────────
    _adv           = session.get("dlc_advanced_cfg", {})
    _sa_name       = str(_adv.get("dlc_superanimal_name", "superanimal_quadruped"))
    _model_name    = _validated_dlc_model_name(
                         str(_adv.get("dlc_architecture", "hrnet_w32")),
                         session, logger)
    _detector_name = str(_adv.get("dlc_detector",
                          "fasterrcnn_mobilenet_v3_large_fpn"))
    _pcutoff       = float(_adv.get("dlc_pcutoff",        0.6))
    _bbox_thr      = float(_adv.get("dlc_bbox_threshold", 0.6))
    _max_ind       = int(_adv.get("dlc_max_individuals",  1))
    _det_epochs    = int(_adv.get("dlc_det_epochs",       15))
    _pose_epochs   = int(_adv.get("dlc_pose_epochs",      15))
    _transfer      = bool(_adv.get("dlc_transfer",        True))

    n_epochs      = int(settings.get("dlc_epochs", 15))
    filter_key    = settings.get("dlc_filter", "Sequential  Median  ’ Gaussian")
    filter_types  = FILTER_OPTIONS.get(filter_key, ["median", "gaussian"])
    run_prep      = bool(settings.get("dlc_run_prep", True))
    create_filt_v = bool(settings.get("dlc_filtered_vid", True))

    work_dir   = _resolve_work_dir(session)
    errors_dir = work_dir / "errors"
    adapt_work = work_dir / "smart_adapt_workspace"
    adapt_work.mkdir(parents=True, exist_ok=True)

    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

    # GPU batch auto-sizing; capped at 85% of free VRAM to avoid OOM crashes
    inf_batch = 8
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            usable_gb = free_gb * 0.85
            inf_batch = 32 if usable_gb >= 10 else (16 if usable_gb >= 5 else 8)
    except Exception:
        pass

    # =========================================================================
    #  Phase 1 — Discovery, Validation & Representative Selection
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "Smart Adapt Phase 1/5: Discovery & Validation")
    pb.step_start("Smart Adapt: Validation", 100)

    all_raw: list = []
    for root_folder in folders:
        for sub, dirs, files in os.walk(root_folder):
            dirs[:] = [d for d in dirs if not d.endswith("_results")]
            if Path(sub).name.endswith("_results"):
                continue
            for fname in sorted(files):
                if fname.startswith("resized_"):
                    continue
                if Path(fname).suffix.lower() in VIDEO_EXTS:
                    all_raw.append((os.path.join(sub, fname), sub))

    if not all_raw:
        raise ValueError("No video files found in selected folders.")

    logger.info(f"  Found {len(all_raw)} candidate video(s). Validating …")

    valid_entries: list = []   # (path, subfolder, mean_brightness)

    for vpath, vsub in all_raw:
        vname = Path(vpath).name
        try:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                raise RuntimeError("Cannot open video")
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if nframes == 0:
                raise RuntimeError("Zero frames reported")
            # Sample 7 evenly-spaced frames for brightness estimation
            sample_idxs = [max(0, int(nframes * i / 8)) for i in range(1, 8)]
            brightnesses: list = []
            for fi in sample_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret and frame is not None:
                    brightnesses.append(float(_np.mean(frame)))
            cap.release()
            if not brightnesses:
                raise RuntimeError("Could not read any frames")
            mean_b = float(_np.mean(brightnesses))
            valid_entries.append((vpath, vsub, mean_b))
            logger.info(f"  ✓ {vname}  (brightness {mean_b:.1f})")
        except Exception as e:
            logger.warn(f"  [QUARANTINE] {vname}: {e}")
            errors_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(vpath, str(errors_dir / vname))
                logger.warn(f"    → Moved to {errors_dir.name}/")
            except Exception as mv_e:
                logger.warn(f"    → Could not move: {mv_e}")

    if not valid_entries:
        raise RuntimeError("No valid videos survived the validation pass.")

    quarantined = len(all_raw) - len(valid_entries)
    logger.success(
        f"  Phase 1: {len(valid_entries)} valid, {quarantined} quarantined  "
        f"[{datetime.now().strftime('%H:%M:%S')}]")

    # Select representative: video closest to dataset median brightness
    brightnesses_all = [e[2] for e in valid_entries]
    median_b         = float(_np.median(brightnesses_all))
    rep_entry        = min(valid_entries, key=lambda x: abs(x[2] - median_b))
    rep_video_path, rep_sub, rep_b = rep_entry

    logger.success(
        f"  Representative: {Path(rep_video_path).name}  "
        f"brightness={rep_b:.1f}  dataset_median={median_b:.1f}")
    pb.step_done()

    # =========================================================================
    #  Phase 1.5 — Convert all videos to target resolution (if enabled)
    # =========================================================================
    long_edge = RESOLUTION_PRESETS.get(settings.get("dlc_resolution"))
    if long_edge:
        logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Smart Adapt: Converting {len(valid_entries)} video(s) "
                    f"to long-edge {long_edge}px …")
        pb.step_start("Smart Adapt: Video Conversion", len(valid_entries))
        converted_entries = []
        _conv_errors = []
        for _ci, (_vpath, _vsub, _mean_b) in enumerate(valid_entries, 1):
            _vname    = Path(_vpath).name
            _base     = Path(_vpath).stem
            _dest_dir = Path(_vsub) / f"{_base}_results"
            _dest_dir.mkdir(parents=True, exist_ok=True)
            _inf_path = str(_dest_dir / f"resized_{_base}.mp4")
            try:
                if not os.path.exists(_inf_path):
                    _cap2    = cv2.VideoCapture(_vpath)
                    _ow2_raw = int(_cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                    _oh2_raw = int(_cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    _fps2    = _cap2.get(cv2.CAP_PROP_FPS) or 30
                    try:
                        _rot2 = int(_cap2.get(cv2.CAP_PROP_ORIENTATION_META))
                    except Exception:
                        _rot2 = 0
                    # Swap visual dimensions for 90°/270° rotated videos
                    if _rot2 in (90, 270):
                        _ow2, _oh2 = _oh2_raw, _ow2_raw
                    else:
                        _ow2, _oh2 = _ow2_raw, _oh2_raw
                    _sc2  = min(long_edge / max(_ow2, _oh2), 1.0)
                    _nw2  = int(_ow2 * _sc2) & ~1
                    _nh2  = int(_oh2 * _sc2) & ~1
                    if _nw2 == _ow2 and _nh2 == _oh2 and _rot2 == 0:
                        _cap2.release()
                        shutil.copy2(_vpath, _inf_path)
                        logger.info(f"  {_vname}: already at/below target — copied")
                    else:
                        logger.info(f"  Resizing {_vname}: {_ow2}x{_oh2} → {_nw2}x{_nh2} (rotation={_rot2}°)")
                        _fc2   = cv2.VideoWriter_fourcc(*"mp4v")
                        _wrt2  = cv2.VideoWriter(_inf_path, _fc2, _fps2, (_nw2, _nh2))
                        _rmap2 = {
                            90:  cv2.ROTATE_90_CLOCKWISE,
                            180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                        }
                        while True:
                            _r2, _fr2 = _cap2.read()
                            if not _r2:
                                break
                            if _rot2 in _rmap2:
                                _fr2 = cv2.rotate(_fr2, _rmap2[_rot2])
                            _wrt2.write(cv2.resize(_fr2, (_nw2, _nh2),
                                                   interpolation=cv2.INTER_AREA))
                        _cap2.release()
                        _wrt2.release()
                        logger.info(f"  Resized → {Path(_inf_path).name}")
                converted_entries.append((_inf_path, _vsub, _mean_b))
            except Exception as _conv_exc:
                logger.error(f"  Conversion failed for {_vname}: {_conv_exc}")
                _conv_errors.append(_vname)
                send_push_notification(
                    session,
                    f"Video conversion failed for {_vname}:\n{_conv_exc}",
                    title="CUBE — Conversion Error", logger=logger)
                converted_entries.append((_vpath, _vsub, _mean_b))
            after_fn(lambda cur=_ci: pb.step_tick(cur, len(valid_entries)))
        valid_entries = converted_entries
        # Update representative to point to the converted path
        rep_entry = min(valid_entries, key=lambda x: abs(x[2] - median_b))
        rep_video_path, rep_sub, rep_b = rep_entry
        if _conv_errors:
            logger.warn(
                f"  Video conversion done: {len(valid_entries) - len(_conv_errors)} "
                f"converted, {len(_conv_errors)} failed.")
            send_push_notification(
                session,
                f"Video conversion finished with {len(_conv_errors)} error(s): "
                f"{', '.join(_conv_errors)}",
                title="CUBE — Conversion Partial Failure", logger=logger)
        else:
            logger.success(
                f"  All {len(valid_entries)} video(s) converted.  "
                f"Representative: {Path(rep_video_path).name}")
            send_push_notification(
                session,
                f"All {len(valid_entries)} video(s) converted to {long_edge}px. "
                f"Starting DLC inference.",
                title="CUBE — Conversion Complete", logger=logger)
        pb.step_done()

    # =========================================================================
    #  Phase 2 — Zoo Adaptation on the representative video
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "Smart Adapt Phase 2/5: Zoo Adaptation")
    pb.step_indeterminate("Zoo Adaptation running …")
    _apply_dlc_monkeypatch(logger)

    adapt_config_path  = None
    adapt_project_dir  = None

    try:
        if hasattr(dlc, "create_video_adaptation_project"):
            # Primary path: dedicated adaptation API (DLC ≥ 2.3).
            # superanimal_name must be the family name only ("superanimal_quadruped"),
            # NOT the combined "superanimal_quadruped_hrnet_w32" string.
            logger.info(
                f"  create_video_adaptation_project → {_sa_name} / {_model_name}")
            _cvap_kw = dict(
                videos            = [rep_video_path],
                working_directory = str(adapt_work),
                superanimal_name  = _sa_name,
                num_epochs        = n_epochs,
                batch_size        = inf_batch,
            )
            # DLC ≥ 3.x uses display_iters; older builds may ignore unknown kwargs
            try:
                import inspect as _insp2
                _cvap_sig = _insp2.signature(dlc.create_video_adaptation_project)
                if "display_iters" in _cvap_sig.parameters:
                    _cvap_kw["display_iters"] = 100
                elif "displayiters" in _cvap_sig.parameters:
                    _cvap_kw["displayiters"] = 100
            except Exception:
                pass
            result = dlc.create_video_adaptation_project(**_cvap_kw)
            if result and Path(str(result)).is_file():
                adapt_config_path = str(result)
                adapt_project_dir = Path(result).parent
                logger.success(
                    f"  Adaptation project: {adapt_project_dir.name}")
            else:
                logger.warn("  create_video_adaptation_project returned no path; "
                            "switching to fallback.")
        # ── Fallback: video_inference_superanimal with video_adapt=True ────────
        if adapt_config_path is None:
            logger.warn(
                "  Fallback: video_inference_superanimal "
                "(video_adapt=True) on representative video")
            rep_dest = adapt_work / f"{Path(rep_video_path).stem}_adapt_results"
            rep_dest.mkdir(parents=True, exist_ok=True)
            rep_inf  = rep_dest / Path(rep_video_path).name
            if not rep_inf.exists():
                shutil.copy2(rep_video_path, str(rep_inf))

            sa_kw = dict(
                superanimal_name     = _sa_name,
                model_name           = _model_name,
                detector_name        = _detector_name,
                scale_list           = list(range(100, 650, 50)),
                pcutoff              = _pcutoff,
                bbox_threshold       = _bbox_thr,
                max_individuals      = _max_ind,
                batch_size           = inf_batch,
                create_labeled_video = False,
                video_adapt          = True,
                pseudo_threshold     = 0.5,
                detector_epochs      = _det_epochs,
                pose_epochs          = _pose_epochs,
                device               = "auto",
            )
            try:
                import inspect as _insp
                _vis_sig = _insp.signature(dlc.video_inference_superanimal).parameters
                if "superanimal_transfer_learning" in _vis_sig:
                    sa_kw["superanimal_transfer_learning"] = _transfer
                if "video_adapt_batch_size" in _vis_sig:
                    sa_kw["video_adapt_batch_size"] = inf_batch
            except Exception:
                pass
            dlc.video_inference_superanimal([str(rep_inf)], **sa_kw)
            # DLC writes snapshots into adapt_work subdirs (not rep_dest itself);
            # point Phase 3 at the entire workspace so rglob finds them.
            adapt_project_dir = adapt_work
            logger.success("  Fallback adaptation complete.")

    except Exception as exc:
        logger.error(f"  Adaptation FAILED: {exc}\n{traceback.format_exc()}")
        logger.warn("  FALLBACK → standard per-video Zoo inference on all videos")
        _run_dlc_zoo_per_video(
            dlc, cv2, gc, valid_entries, session, settings, logger, pb,
            after_fn, n_epochs, _adv, filter_types, run_prep,
            create_filt_v, inf_batch)
        return

    pb.step_done()
    send_push_notification(
        session,
        f"Adaptation complete on {Path(rep_video_path).name}. "
        f"Fine-tuned weights ready for extraction.",
        title="CUBE — Adaptation Over", logger=logger)

    # =========================================================================
    #  Phase 3 — Snapshot Extraction
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "Smart Adapt Phase 3/5: Extracting Fine-tuned Weights")

    adapted_pose_ckpt     = None   # DLC 3.x PyTorch (.pt)
    adapted_detector_ckpt = None   # DLC 3.x PyTorch (.pt)
    snapshot_base         = None   # legacy TensorFlow (.index)

    if adapt_project_dir and adapt_project_dir.exists():
        # DLC 3.x PyTorch: search for .pt checkpoints first
        adapted_pose_ckpt, adapted_detector_ckpt = _find_adapted_pt_checkpoints(
            adapt_project_dir, _model_name, _detector_name, logger)
        if adapted_pose_ckpt is None:
            # Legacy TF fallback: search for .index files
            snapshot_base = _find_highest_snapshot_path(adapt_project_dir, logger)

    if adapted_pose_ckpt is None and snapshot_base is None:
        logger.error("  DIVERGENCE: no snapshot found after adaptation!")
        logger.warn("  FALLBACK → base Zoo weights (no weight injection)")
        inference_config = adapt_config_path   # may still be None
        if inference_config is None:
            logger.warn("  No config available — running standard Zoo inference")
            _run_dlc_zoo_per_video(
                dlc, cv2, gc, valid_entries, session, settings, logger, pb,
                after_fn, n_epochs, _adv, filter_types, run_prep,
                create_filt_v, inf_batch)
            return
    else:
        if adapted_pose_ckpt:
            logger.success(
                f"  DLC 3.x adapted pose snapshot: {Path(adapted_pose_ckpt).name}")
            if adapted_detector_ckpt:
                logger.success(
                    f"  DLC 3.x adapted detector snapshot: "
                    f"{Path(adapted_detector_ckpt).name}")
        else:
            snap_match = _re.search(r"snapshot-(\d+)$", Path(snapshot_base).name)
            snap_num   = int(snap_match.group(1)) if snap_match else 0
            logger.success(
                f"  Snapshot extracted: snapshot-{snap_num}  "
                f"({Path(snapshot_base).parent.name}/)")
        inference_config = adapt_config_path   # updated below in Phase 4

    # =========================================================================
    #  Phase 4 — Create Named Inference Project & Inject Weights (ruamel.yaml)
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "Smart Adapt Phase 4/5: Project Creation & Weight Injection")

    base_folder_name = Path(session["video_folders"][0]).name

    if adapted_pose_ckpt:
        # DLC 3.x PyTorch: checkpoints are passed directly to video_inference_superanimal
        # in Phase 5 via customized_pose_checkpoint / customized_detector_checkpoint.
        # No separate DLC project or pose_cfg.yaml injection is needed.
        logger.success(
            "  DLC 3.x: adapted checkpoints ready — skipping legacy project creation.")
        inference_config = None   # not used in the PyTorch Zoo inference path
    else:
        try:
            # Legacy TF path: create a named project skeleton based on the input folder name
            new_cfg_path = dlc.create_new_project(
                project          = f"{base_folder_name}_CUBE_v3",
                experimenter     = "CUBE",
                videos           = [rep_video_path],
                working_directory= str(adapt_work),
                copy_videos      = False,
            )
            new_project_dir = Path(new_cfg_path).parent
            logger.success(f"  Named project created: {new_project_dir.name}")

            # ── Copy model artefacts if the new project has no dlc-models yet ──
            new_models   = new_project_dir / "dlc-models"
            adapt_models = adapt_project_dir / "dlc-models" if adapt_project_dir else None
            if not new_models.exists() and adapt_models and adapt_models.is_dir():
                shutil.copytree(str(adapt_models), str(new_models))
                logger.success("  dlc-models/ copied from adaptation project.")

            # ── ruamel.yaml: inject init_weights in every pose_cfg.yaml found ──
            _pcfg_targets = []
            if adapt_project_dir:
                _pcfg_targets += list(adapt_project_dir.rglob("pose_cfg.yaml"))
            _pcfg_targets += list(new_project_dir.rglob("pose_cfg.yaml"))

            _use_ruamel = False
            try:
                from ruamel.yaml import YAML as _YAML
                _use_ruamel = True
            except ImportError:
                logger.warn(
                    "  ruamel.yaml not found (pip install ruamel.yaml); "
                    "using PyYAML fallback — YAML comments may be stripped.")

            for pcfg_path in _pcfg_targets:
                try:
                    if _use_ruamel:
                        _ry = _YAML()
                        _ry.preserve_quotes = True
                        with open(pcfg_path, "r", encoding="utf-8") as _f:
                            _pdata = _ry.load(_f) or {}
                        if snapshot_base:
                            _pdata["init_weights"] = str(
                                Path(snapshot_base).resolve())
                        with open(pcfg_path, "w", encoding="utf-8") as _f:
                            _ry.dump(_pdata, _f)
                    else:
                        import yaml as _pyyaml
                        with open(pcfg_path, "r", encoding="utf-8") as _f:
                            _pdata = _pyyaml.safe_load(_f) or {}
                        if snapshot_base:
                            _pdata["init_weights"] = str(
                                Path(snapshot_base).resolve())
                        with open(pcfg_path, "w", encoding="utf-8") as _f:
                            _pyyaml.dump(_pdata, _f, default_flow_style=False)
                    try:
                        _rel = pcfg_path.relative_to(work_dir)
                    except ValueError:
                        _rel = pcfg_path
                    logger.success(
                        f"  pose_cfg.yaml → init_weights injected  ({_rel})")
                except Exception as pcfg_e:
                    logger.warn(
                        f"  pose_cfg.yaml injection warning for {pcfg_path}: {pcfg_e}")

            # Prefer adaptation project config for inference (fully set up);
            # fall back to newly created project config if adaptation config unavailable
            inference_config = adapt_config_path or str(new_cfg_path)
            logger.success(
                f"  Inference config: {Path(inference_config).parent.name}/config.yaml")

        except Exception as exc:
            logger.warn(f"  Phase 4 warning: {exc}")
            logger.warn("  Continuing with adaptation project config for inference.")
            if adapt_config_path:
                inference_config = adapt_config_path

    if not inference_config and not adapted_pose_ckpt:
        raise RuntimeError(
            "No inference config available after Phase 4 — "
            "cannot proceed with batch inference.")

    # =========================================================================
    #  Phase 5 — Batch Inference with adaptive OOM recovery
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Smart Adapt Phase 5/5: Batch Inference "
                f"({len(valid_entries)} video(s))")

    total            = len(valid_entries)
    pb.step_start("Smart Adapt: Batch Inference", total)
    errors: list     = []
    current_batchsize = inf_batch   # persists across videos; OOM will reduce it globally

    send_push_notification(
        session,
        f"Weight transfer complete. Starting batch inference on {total} video(s).",
        title="CUBE — Transfer Complete", logger=logger)

    for idx, (vpath, vsub, _) in enumerate(valid_entries, 1):
        vpath_obj   = Path(vpath)
        vname       = vpath_obj.name
        # If Phase 1.5 placed the video inside an existing _results dir, reuse it
        if vpath_obj.parent.name.endswith("_results"):
            dest_folder = vpath_obj.parent
            base_noext  = dest_folder.name[:-len("_results")]
        else:
            base_noext  = vpath_obj.stem
            dest_folder = Path(vsub) / f"{base_noext}_results"
        dest_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"  [{idx}/{total}]  {vname}")
        h5_before = set(dest_folder.glob("*.h5"))
        _v_t0 = time.time()
        succeeded  = False
        while not succeeded:
            try:
                if adapted_pose_ckpt:
                    # DLC 3.x PyTorch: inject adapted weights via customized checkpoints.
                    # analyze_videos cannot use Zoo weights; video_inference_superanimal must be used.
                    _sa_inf_kw = dict(
                        superanimal_name               = _sa_name,
                        model_name                     = _model_name,
                        detector_name                  = _detector_name,
                        scale_list                     = list(range(100, 650, 50)),
                        pcutoff                        = _pcutoff,
                        bbox_threshold                 = _bbox_thr,
                        max_individuals                = _max_ind,
                        batch_size                     = current_batchsize,
                        create_labeled_video           = create_filt_v,
                        video_adapt                    = False,
                        dest_folder                    = str(dest_folder),
                        device                         = "auto",
                        customized_pose_checkpoint     = adapted_pose_ckpt,
                        customized_detector_checkpoint = adapted_detector_ckpt,
                    )
                    dlc.video_inference_superanimal([vpath], **_sa_inf_kw)
                else:
                    dlc.analyze_videos(
                        inference_config,
                        [vpath],
                        save_as_csv    = True,
                        destfolder     = str(dest_folder),
                        batchsize      = current_batchsize,
                        allow_growth   = True,
                        robust_nframes = True,
                    )
                succeeded = True
                logger.info(f"    ✓ Inference done  (batchsize={current_batchsize})")
            except RuntimeError as oom_exc:
                exc_l = str(oom_exc).lower()
                if "cuda" in exc_l and (
                        "memory" in exc_l or "oom" in exc_l or "alloc" in exc_l):
                    if current_batchsize <= 1:
                        logger.error(
                            f"    OOM at batchsize=1 — cannot reduce further. "
                            f"Skipping {vname}.")
                        errors.append(vname)
                        succeeded = True
                    else:
                        current_batchsize = max(1, current_batchsize // 2)
                        logger.warn(
                            f"    OOM detected — reducing batchsize → "
                            f"{current_batchsize}, retrying …")
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    logger.error(f"    RuntimeError: {oom_exc}")
                    errors.append(vname)
                    succeeded = True
            except Exception:
                logger.error(
                    f"    ERROR on {vname}:\n{traceback.format_exc()}")
                errors.append(vname)
                succeeded = True

        # ── Post-inference cleanup: filter H5, rename labeled video ───────────
        h5_new = [p for p in dest_folder.glob("*.h5")
                  if p not in h5_before
                  and not p.name.startswith("BSOID_")
                  and not p.stem.endswith("_filtered")]
        if h5_new:
            final_h5       = h5_new[0]
            clean_h5       = dest_folder / f"{base_noext}.h5"
            clean_filtered = dest_folder / f"{base_noext}_filtered.h5"
            if filter_types:
                filter_dlc_h5(final_h5, filter_types, log_fn=logger,
                              out_path=clean_filtered,
                              fps=float(session.get("fps", 30)))
            else:
                shutil.copy2(str(final_h5), str(clean_filtered))
            try:
                final_h5.rename(clean_h5)
            except Exception:
                pass
            for p in h5_new:
                if p != final_h5:
                    try:    p.unlink()
                    except Exception: pass

        # Rename labeled video to a short clean name.
        # Use YYYYMMDD_HHMMSS timestamp when available, else truncate to 50 chars.
        if create_filt_v:
            _ts_m2 = re.search(r"\d{8}_\d{6}", base_noext)
            _short_stem2 = _ts_m2.group(0) if _ts_m2 else base_noext[:50]
            clean_vid = dest_folder / f"{_short_stem2}_labeled.mp4"
            if not clean_vid.exists():
                for p in sorted(dest_folder.glob("*.mp4")):
                    n = p.name.lower()
                    if any(x in n for x in ("_el.", "_labeled.")):
                        if "before_adapt" not in n and "pseudo" not in n:
                            try:
                                p.rename(clean_vid)
                            except Exception:
                                pass
                            break

        # Never clean inside adapt_work — the pseudo_*/checkpoints/ tree there
        # holds the adapted .pt weights used by every remaining video in this loop.
        try:
            if not str(dest_folder.resolve()).startswith(str(adapt_work.resolve())):
                cleanup_video_byproducts(dest_folder, logger)
        except Exception:
            pass

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        _elapsed = int(time.time() - _v_t0)
        _remaining = total - idx
        send_push_notification(
            session,
            f"Processed: {vname}\nTime: {_elapsed}s\nRemaining: {_remaining}/{total}",
            title="CUBE — Video Complete", logger=logger)

        after_fn(lambda cur=idx: pb.step_tick(cur, total))

    # ── Optional auto-run Step 2 ──────────────────────────────────────────────
    if run_prep:
        logger.step("Running CUBE pre-processing (Step 2) …")
        bsoid_roots = []
        for folder in folders:
            root = run_bsoid_prep(
                folder, log_fn=logger,
                min_confidence=float(session.get("bsoid_min_conf", 0.35)))
            if root:
                bsoid_roots.append(str(root))
        session["bsoid_ready_dirs"] = bsoid_roots

    if errors:
        logger.warn(
            f"Smart Adapt finished with {len(errors)} error(s): {errors}")
    else:
        logger.success(
            f"Smart Adapt complete: {total} video(s) processed  "
            f"[{datetime.now().strftime('%H:%M:%S')}]")


def _run_dlc_zoo_per_video(dlc, cv2, gc, valid_entries, session, settings,
                            logger, pb, after_fn, n_epochs, _adv,
                            filter_types, run_prep, create_filt_v, inf_batch):
    """
    Fallback used by _run_dlc_smart_adapt_step when adaptation fails.
    Runs standard Zoo inference (video_adapt=False) on pre-validated
    valid_entries list: [(path, subfolder, brightness), ...].
    """
    total = len(valid_entries)
    pb.step_start("DLC Zoo inference (fallback)", total)

    _sa_name       = str(_adv.get("dlc_superanimal_name", "superanimal_quadruped"))
    _model_name    = _validated_dlc_model_name(
                         str(_adv.get("dlc_architecture", "hrnet_w32")),
                         session, logger)
    _detector_name = str(_adv.get("dlc_detector",
                          "fasterrcnn_mobilenet_v3_large_fpn"))
    _pcutoff  = float(_adv.get("dlc_pcutoff",        0.6))
    _bbox_thr = float(_adv.get("dlc_bbox_threshold", 0.6))
    _max_ind  = int(_adv.get("dlc_max_individuals",  1))
    _det_epochs  = int(_adv.get("dlc_det_epochs",  n_epochs))
    _pose_epochs = int(_adv.get("dlc_pose_epochs", n_epochs))
    _transfer    = bool(_adv.get("dlc_transfer",   True))
    _det_batch   = int(_adv.get("dlc_det_batch",   0)) or inf_batch

    filter_key   = settings.get("dlc_filter", "Sequential  Median  ' Gaussian")
    filter_types = FILTER_OPTIONS.get(filter_key, filter_types)

    folders = session.get("video_folders", [])
    errors: list = []

    for idx, (vpath, vsub, _) in enumerate(valid_entries, 1):
        vpath_obj   = Path(vpath)
        vname       = vpath_obj.name
        if vpath_obj.parent.name.endswith("_results"):
            dest_folder = vpath_obj.parent
            base_noext  = dest_folder.name[:-len("_results")]
        else:
            base_noext  = vpath_obj.stem
            dest_folder = Path(vsub) / f"{base_noext}_results"
        dest_folder.mkdir(parents=True, exist_ok=True)
        logger(f"  [{idx}/{total}]  {vname}")
        h5_before = set(Path(dest_folder).glob("*.h5"))

        try:
            cap3  = cv2.VideoCapture(vpath)
            short = min(int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap3.release()
            centre     = max(150, int(short * 0.35))
            scale_list = list(range(max(100, centre - 150),
                                    min(1200, centre + 200), 50))
            sa_kw = dict(
                superanimal_name     = _sa_name,
                model_name           = _model_name,
                detector_name        = _detector_name,
                scale_list           = scale_list,
                pcutoff              = _pcutoff,
                bbox_threshold       = _bbox_thr,
                max_individuals      = _max_ind,
                batch_size           = inf_batch,
                detector_batch_size  = _det_batch,
                create_labeled_video = create_filt_v,
                video_adapt          = False,
                device               = "auto",
            )
            try:
                import inspect as _insp
                if "superanimal_transfer_learning" in \
                        _insp.signature(dlc.video_inference_superanimal).parameters:
                    sa_kw["superanimal_transfer_learning"] = _transfer
            except Exception:
                pass
            dlc.video_inference_superanimal([vpath], **sa_kw)
            logger.info(f"    ✓ {vname}")
        except Exception:
            logger.error(f"    ERROR on {vname}:\n{traceback.format_exc()}")
            errors.append(vname)

        h5_new = [p for p in Path(dest_folder).glob("*.h5")
                  if p not in h5_before
                  and not p.name.startswith("BSOID_")
                  and not p.stem.endswith("_filtered")]
        if h5_new:
            final_h5       = h5_new[0]
            clean_h5       = Path(dest_folder) / f"{base_noext}.h5"
            clean_filtered = Path(dest_folder) / f"{base_noext}_filtered.h5"
            if filter_types:
                filter_dlc_h5(final_h5, filter_types, log_fn=logger,
                              out_path=clean_filtered,
                              fps=float(session.get("fps", 30)))
            else:
                shutil.copy2(str(final_h5), str(clean_filtered))
            try:
                final_h5.rename(clean_h5)
            except Exception:
                pass
            for p in h5_new:
                if p != final_h5:
                    try:    p.unlink()
                    except Exception: pass

        try:
            cleanup_video_byproducts(Path(dest_folder), logger)
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        after_fn(lambda cur=idx: pb.step_tick(cur, total))

    if run_prep:
        bsoid_roots = []
        for folder in folders:
            root = run_bsoid_prep(
                folder, log_fn=logger,
                min_confidence=float(session.get("bsoid_min_conf", 0.35)))
            if root:
                bsoid_roots.append(str(root))
        session["bsoid_ready_dirs"] = bsoid_roots

    if errors:
        logger.warn(f"DLC fallback finished with {len(errors)} error(s).")
        send_push_notification(
            session,
            f"DLC inference finished with {len(errors)} error(s):\n"
            + "\n".join(errors),
            title="CUBE — DLC Errors", logger=logger)
    else:
        logger.success(f"DLC fallback complete: {total} video(s) processed.")
        send_push_notification(
            session,
            f"DLC inference complete: {total} video(s) processed successfully.",
            title="CUBE — DLC Complete", logger=logger)


def _run_bsoid_prep_step(session: SessionState, settings: SettingsPanel,
                          logger: PipelineLogger, pb: DualProgressBar,
                          after_fn):
    """Run CUBE pre-processing on all selected video folders."""
    folders = session["video_folders"]
    if not folders:
        raise ValueError("No video folders selected.")
    pb.step_start("CUBE pre-processing", len(folders))
    roots = []
    for i, folder in enumerate(folders, 1):
        logger(f"  Pre-processing folder {i}/{len(folders)}: "
               f"{Path(folder).name}")
        root = run_bsoid_prep(
            folder, log_fn=logger,
            min_confidence=float(settings.get("bsoid_min_conf", 0.35)))
        if root:
            roots.append(str(root))
        after_fn(lambda cur=i: pb.step_tick(cur, len(folders)))
    session["bsoid_ready_dirs"] = roots
    if not roots:
        raise RuntimeError("No BSOID_Project_Ready directories were created.")
    logger.success(f"CUBE pre-processing done: {len(roots)} project(s) ready.")
    send_push_notification(
        session,
        f"Pre-processing complete: {len(roots)} project(s) ready in BSOID_Project_Ready/.",
        title="CUBE — Pre-processing Complete", logger=logger)


def _run_engine_step(session: SessionState, settings: SettingsPanel,
                     logger: PipelineLogger, pb: DualProgressBar,
                     after_fn,
                     bd_min: float = 0.0, bd_max: float = 999.0):
    """Run a SINGLE combined CUBE clustering engine across ALL groups/folders.

    All BSOID_Project_Ready directories are combined into one analysis so that
    every group shares the same cluster space and results can be directly
    compared in Step 5.  Experimental group assignments (video_groups) are
    only used in Step 5 analysis and have no effect on clustering here.
    """
    # ── Discover BSOID_Project_Ready directories ─────────────────────────────
    # Merge Step-2-stored paths with a fresh recursive scan of every video
    # folder so that nested or pre-existing project dirs are found at any depth
    # even when Step 2 was skipped.
    _PROJ_NAME   = "BSOID_Project_Ready"
    _seen: set   = set()
    bsoid_roots: list = []

    def _add_bsoid_root(p: Path) -> bool:
        key = str(p.resolve())
        if key not in _seen and p.is_dir():
            _seen.add(key)
            bsoid_roots.append(str(p))
            return True
        return False

    # Step-2-stored paths come first (they are pre-validated)
    for r in session.get("bsoid_ready_dirs", []):
        _add_bsoid_root(Path(r))

    # Recursively scan every video folder for _PROJ_NAME at any depth
    n_before = len(bsoid_roots)
    for folder in session.get("video_folders", []):
        fp = Path(folder)
        if not fp.is_dir():
            continue
        if fp.name == _PROJ_NAME:
            _add_bsoid_root(fp)
        else:
            for match in sorted(fp.rglob(_PROJ_NAME)):
                if match.is_dir():
                    _add_bsoid_root(match)

    n_discovered = len(bsoid_roots) - n_before
    if n_discovered:
        logger.info(f"  Recursively discovered {n_discovered} "
                    f"{_PROJ_NAME} dir(s) from video folders.")

    if not bsoid_roots:
        raise ValueError(
            "No BSOID_Project_Ready directories found.\n"
            "Run Step 2 (CUBE Pre-processing) first,\n"
            "or ensure your video folders contain BSOID_Project_Ready "
            "subdirectories.")

    session["bsoid_ready_dirs"] = bsoid_roots

    # Merge publication defaults with any user overrides from Advanced CUBE window
    cfg = dict(BSoidEngine.DEFAULTS)
    cfg.update(session.get("engine_cfg", {}))
    # Bout duration always comes from the prominent front-panel widget
    cfg["min_epoch_dur_s"]       = float(bd_min)
    cfg["max_epoch_dur_s"]       = float(bd_max)
    cfg["delete_labeled_videos"] = True   # always delete per issue 5
    logger.info(f"  Bout duration filter: [{bd_min:.2f} s, {bd_max:.1f} s]")

    fps_val = float(settings.get("fps", 30))

    _vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    def _has_videos(d: Path) -> bool:
        return d.is_dir() and any(
            f.suffix.lower() in _vid_exts for f in d.rglob("*") if f.is_file()
        )

    # ── Collect ALL csv and video dirs from ALL bsoid roots ──────────────────
    all_csv_dirs: list = []
    all_vid_dirs: list = []
    for bsoid_root in bsoid_roots:
        rp = Path(bsoid_root)
        csv_dir = rp / "csv"
        if not csv_dir.is_dir():
            csv_dir = rp
        all_csv_dirs.append(csv_dir)
        vid_dir = rp / "videos"
        if _has_videos(vid_dir):
            all_vid_dirs.append(vid_dir)

    # Fall back to session source folders when BSOID video dirs are empty
    if not all_vid_dirs:
        for src_folder in session.get("video_folders", []):
            sp = Path(src_folder)
            if _has_videos(sp):
                all_vid_dirs.append(sp)
                logger.warn(f"    BSOID videos/ empty — using source folder: {sp.name}")

    # ── Single combined output directory (timestamped to preserve prior runs) ──
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(bsoid_roots) == 1:
        # Single root: keep original behaviour (output inside BSOID_Project_Ready)
        this_out = Path(bsoid_roots[0]) / f"cube_results_{_ts}"
    else:
        # Multiple roots: use workspace root so output is separate from any group
        this_out = _resolve_work_dir(session) / f"cube_results_{_ts}"

    logger.step(f"  CUBE Engine: {len(bsoid_roots)} project(s) → single combined analysis")
    for _i, _br in enumerate(bsoid_roots, 1):
        _p = Path(_br)
        _label = f"{_p.parent.name}/{_p.name}" if _p.parent.name else _p.name
        logger(f"    [{_i}] {_label}")
    logger(f"    CSV folders  : {len(all_csv_dirs)}")
    logger(f"    Video folders: {len(all_vid_dirs)}")
    logger(f"    Combined output: {this_out}")

    pb.step_start("BSOID engine (combined)", 1)

    def _prog(cur, tot):
        after_fn(lambda: pb.step_tick(cur, tot))

    def _stage(stage: str, detail: str = ""):
        text = f"Combined | {stage}"
        if detail:
            text += f"  —  {detail}"
        if "UMAP embedding" in stage:
            after_fn(lambda t=text: pb.step_indeterminate(t))
        elif "UMAP done" in stage:
            after_fn(lambda t=text: (pb.step_start(t, 1), pb.step_tick(0, 1)))
        else:
            after_fn(lambda t=text: pb.step_label(t))

        notify_msg = detail if detail else stage
        if "UMAP embedding" in stage:
            send_push_notification(session, f"Combined: {notify_msg}",
                                   title="CUBE — UMAP Running", logger=logger)
        elif "HDBSCAN done" in stage:
            send_push_notification(session, f"Combined: {notify_msg}",
                                   title="CUBE — Clustering Complete", logger=logger)
        elif "MLP done" in stage:
            send_push_notification(session, f"Combined: {notify_msg}",
                                   title="CUBE — Classifier Ready", logger=logger)
        elif "VALIDATION BLOCK" in stage:
            send_push_notification(session, f"Combined: {notify_msg}",
                                   title="CUBE — Validation BLOCK", logger=logger)
        elif "VALIDATION WARN" in stage:
            send_push_notification(session, f"Combined: {notify_msg}",
                                   title="CUBE — Validation Warning", logger=logger)

    engine = BSoidEngine(
        csv_folder   = all_csv_dirs,
        video_folder = all_vid_dirs if all_vid_dirs else None,
        output_dir   = this_out,
        fps          = fps_val,
        logger       = logger,
        progress_cb  = _prog,
        stage_cb     = _stage,
        cfg          = cfg,
    )
    results = engine.run()
    bout_all = [str(p) for p in results.get("bout_lengths_paths", [])]
    after_fn(lambda: pb.step_tick(1, 1))

    session["bout_lengths_paths"] = bout_all
    session["engine_out_dirs"]    = [str(this_out)]

    # ── Delete BSOID_Project_Ready/videos/ copies if user requested ──────────
    if bool(session.get("bsoid_delete_videos_folder", False)):
        for bsoid_root in bsoid_roots:
            _vd = Path(bsoid_root) / "videos"
            if _vd.exists():
                try:
                    shutil.rmtree(str(_vd))
                    logger.info(f"  [cleanup] Deleted: {_vd}")
                except Exception as _e:
                    logger.warn(f"  [cleanup] Could not delete {_vd.name}: {_e}")

    logger.success(f"Engine done: {len(bout_all)} bout-length CSV(s) created.")
    send_push_notification(
        session,
        f"Clustering complete: {len(bout_all)} bout-length CSV(s) created.",
        title="CUBE — Clustering Complete", logger=logger)


#
#  ADVANCED PARAMETER POPUP WINDOWS
#

def _adv_section(parent, text: str, colour: str = None) -> tk.Frame:
    """Coloured section divider + body frame for advanced popup windows."""
    colour = colour or C["subtext"]
    hdr = tk.Frame(parent, bg=colour, height=1)
    hdr.pack(fill="x", pady=(10, 0))
    tk.Label(parent, text=f"  {text}",
             font=("Segoe UI", 8, "bold"),
             bg=C["bg"], fg=colour).pack(anchor="w", padx=8, pady=(2, 0))
    body = tk.Frame(parent, bg=C["card"],
                    highlightbackground=C["border"],
                    highlightthickness=1)
    body.pack(fill="x", padx=8, pady=(2, 4))
    return body


def _adv_row(parent, label: str, widget_fn):
    """Label + widget row inside a section body."""
    row = tk.Frame(parent, bg=C["card"])
    row.pack(fill="x", padx=8, pady=2)
    tk.Label(row, text=label, width=26, anchor="w",
             font=("Segoe UI", 9), bg=C["card"],
             fg=C["text"]).pack(side="left")
    return widget_fn(row)


class DLCPrepSettingsWindow(tk.Toplevel):
    """Popup dialog that exposes all DLC & Prep settings from SettingsPanel."""

    def __init__(self, parent, settings_panel):
        super().__init__(parent)
        self.title("DLC & Prep Settings")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="  ⚙  DLC & Prep Settings",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["bg"], fg=C["yellow"]).pack(anchor="w", padx=10, pady=(10, 4))
        tk.Label(self,
                 text="  Changes apply immediately — close when done.",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["dim"],
                 justify="left").pack(anchor="w", padx=10, pady=(0, 6))

        inner = tk.Frame(self, bg=C["card"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        for key, label, wtype, opts, _default, tip in SettingsPanel._DLC_ROWS:
            row = tk.Frame(inner, bg=C["card"])
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=label, width=24, anchor="w",
                     font=("Segoe UI", 9), bg=C["card"],
                     fg=C["text"]).pack(side="left")
            v = settings_panel._vars[key]
            if wtype == "bool":
                tk.Checkbutton(row, variable=v, bg=C["card"],
                               fg=C["green"], selectcolor=C["card2"],
                               activebackground=C["card"]).pack(side="left")
            elif wtype == "combo":
                ttk.Combobox(row, textvariable=v, values=opts,
                             state="readonly", width=26,
                             font=("Segoe UI", 9)).pack(side="left")
            elif wtype == "int":
                lo, hi, step = opts
                tk.Spinbox(row, from_=lo, to=hi, increment=step,
                           textvariable=v, width=7,
                           bg=C["card2"], fg=C["text"],
                           buttonbackground=C["card2"],
                           font=("Segoe UI", 9)).pack(side="left")
            elif wtype == "float":
                lo, hi, step = opts
                tk.Spinbox(row, from_=lo, to=hi, increment=step,
                           format="%.2f", textvariable=v, width=7,
                           bg=C["card2"], fg=C["text"],
                           buttonbackground=C["card2"],
                           font=("Segoe UI", 9)).pack(side="left")
            else:
                tk.Entry(row, textvariable=v, width=18,
                         bg=C["card2"], fg=C["text"],
                         insertbackground=C["text"],
                         relief="flat").pack(side="left")
            if tip:
                tk.Label(row, text=tip, font=("Segoe UI", 7),
                         bg=C["card"], fg=C["dim"],
                         wraplength=280).pack(side="left", padx=4)

        btn_row = tk.Frame(self, bg=C["bg"])
        btn_row.pack(pady=(4, 10))

        def _test_notification():
            import urllib.request
            import urllib.error
            topic = settings_panel._vars["ntfy_topic"].get().strip()
            if not topic:
                messagebox.showwarning("No Topic",
                    "Enter a notification topic first.", parent=self)
                return
            try:
                url = "https://ntfy.sh/" + topic
                req = urllib.request.Request(
                    url,
                    data="CUBE test notification — connection OK!".encode("utf-8"),
                    headers={"Title": "CUBE - Test", "Priority": "default"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=10):
                    pass
                messagebox.showinfo("Sent",
                    f"Test notification sent to topic:\n{topic}", parent=self)
            except urllib.error.HTTPError as e:
                messagebox.showerror("HTTP Error",
                    f"ntfy.sh returned {e.code}: {e.reason}", parent=self)
            except urllib.error.URLError as e:
                messagebox.showerror("Network Error",
                    f"Could not reach ntfy.sh:\n{e.reason}", parent=self)
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self)

        tk.Button(btn_row, text="Test Notification", font=("Segoe UI", 9),
                  bg=C["card2"], fg=C["cyan"], relief="flat",
                  padx=14, pady=5, cursor="hand2",
                  command=_test_notification).pack(side="left", padx=6)
        tk.Button(btn_row, text="Close", font=("Segoe UI", 9, "bold"),
                  bg=C["card2"], fg=C["text"], relief="flat",
                  padx=20, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left", padx=6)

        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        w  = self.winfo_reqwidth()
        h  = self.winfo_reqheight()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")


class AdvancedDLCWindow(tk.Toplevel):
    """Modal popup for advanced DLC inference parameters."""

    SUPERANIMAL_MODELS = [
        "superanimal_quadruped",
        "superanimal_topviewmouse",
    ]
    ARCHITECTURES = ["hrnet_w32", "resnet_50", "rtmpose_s", "rtmpose_x"]
    DETECTORS     = [
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_resnet50_fpn",
        "ssd300_vgg16",
    ]
    DEFAULTS = dict(
        dlc_use_custom       = False,
        dlc_superanimal_name = "superanimal_quadruped",
        dlc_architecture     = "hrnet_w32",
        dlc_detector         = "fasterrcnn_mobilenet_v3_large_fpn",
        dlc_transfer         = True,
        dlc_custom_config    = "",
        dlc_pcutoff          = 0.6,
        dlc_bbox_threshold   = 0.6,
        dlc_max_individuals  = 1,
        dlc_inf_batch        = 0,
        dlc_det_batch        = 0,
        dlc_det_epochs       = 15,
        dlc_pose_epochs      = 15,
        dlc_scale_mode       = "Auto",
        dlc_scale_min        = 100,
        dlc_scale_max        = 600,
        dlc_scale_step       = 50,
    )

    def __init__(self, parent, session: "SessionState"):
        super().__init__(parent)
        self.title("⚙  Advanced DLC Parameters")
        self.configure(bg=C["bg"])
        self.geometry("540x720")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self._session = session
        self._vars: dict = {}

        # ── Bottom buttons (packed first so they stay fixed) ──────────────────
        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(side="bottom", fill="x", pady=8, padx=12)
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 9),
                  bg=C["card2"], fg=C["subtext"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Restore Defaults", font=("Segoe UI", 9),
                  bg=C["card2"], fg=C["yellow"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self._restore).pack(side="left", padx=6)
        tk.Button(btn_f, text="Apply & Close", font=("Segoe UI", 10, "bold"),
                  bg=C["green"], fg="white", relief="flat",
                  padx=16, pady=5, cursor="hand2",
                  command=self._apply).pack(side="right")

        # ── Scrollable canvas ─────────────────────────────────────────────────
        canvas = tk.Canvas(self, bg=C["bg"], highlightthickness=0)
        sb = tk.Scrollbar(self, orient="vertical", command=canvas.yview,
                          bg=C["card"], troughcolor=C["bg"])
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=C["bg"])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-(e.delta // 120), "units"))

        def _unbind_mw(e):
            try:
                canvas.unbind_all("<MouseWheel>")
            except Exception:
                pass
        self.bind("<Destroy>", _unbind_mw)

        self._build(inner)
        self._load()

    def _v(self, key, var):
        self._vars[key] = var
        return var

    def _build(self, p):
        tk.Label(p, text="  ⚙  Advanced DLC Parameters",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["bg"], fg=C["green"]).pack(anchor="w", padx=10, pady=(10, 4))
        tk.Label(p,
                 text="  Basic settings (resolution, adapt, filter) are in the\n"
                      "  main DLC & Prep panel. These control the underlying model.",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["dim"],
                 justify="left").pack(anchor="w", padx=10, pady=(0, 4))

        # ── Model type ────────────────────────────────────────────────────────
        sec = _adv_section(p, "MODEL TYPE", C["green"])
        self._v("dlc_use_custom", tk.BooleanVar(value=False))
        def _mtrow(text, val):
            r = tk.Frame(sec, bg=C["card"])
            r.pack(anchor="w", padx=8, pady=2)
            tk.Radiobutton(r, text=text, variable=self._vars["dlc_use_custom"],
                           value=val, bg=C["card"], fg=C["text"],
                           selectcolor=C["card2"],
                           activebackground=C["card"],
                           font=("Segoe UI", 9),
                           command=self._toggle_model_type).pack(side="left")
        _mtrow("SuperAnimal Zoo model  (recommended)", False)
        _mtrow("Custom DLC project  (user-supplied config.yaml)", True)

        # ── SuperAnimal settings ──────────────────────────────────────────────
        self._sa_sec = _adv_section(p, "SUPERANIMAL SETTINGS", C["cyan"])

        def _combo(parent, key, values, default):
            v = self._v(key, tk.StringVar(value=default))
            ttk.Combobox(parent, textvariable=v, values=values,
                         state="readonly", width=34,
                         font=("Segoe UI", 9)).pack(side="left")
            return v

        _adv_row(self._sa_sec, "Zoo model",
                 lambda r: _combo(r, "dlc_superanimal_name",
                                  self.SUPERANIMAL_MODELS,
                                  "superanimal_quadruped"))
        _adv_row(self._sa_sec, "Architecture (pose)",
                 lambda r: _combo(r, "dlc_architecture",
                                  self.ARCHITECTURES, "hrnet_w32"))
        _adv_row(self._sa_sec, "Detector backbone",
                 lambda r: _combo(r, "dlc_detector",
                                  self.DETECTORS,
                                  "fasterrcnn_mobilenet_v3_large_fpn"))
        _adv_row(self._sa_sec, "SuperAnimal transfer learning",
                 lambda r: tk.Checkbutton(
                     r, variable=self._v("dlc_transfer", tk.BooleanVar(value=True)),
                     bg=C["card"], fg=C["green"],
                     selectcolor=C["card2"],
                     activebackground=C["card"]).pack(side="left"))

        # ── Custom model ──────────────────────────────────────────────────────
        self._cust_sec = _adv_section(p, "CUSTOM MODEL  (when Custom is selected)", C["yellow"])
        cust_row = tk.Frame(self._cust_sec, bg=C["card"])
        cust_row.pack(fill="x", padx=8, pady=4)
        tk.Label(cust_row, text="config.yaml path", width=26, anchor="w",
                 font=("Segoe UI", 9), bg=C["card"],
                 fg=C["text"]).pack(side="left")
        self._v("dlc_custom_config", tk.StringVar(value=""))
        tk.Entry(cust_row, textvariable=self._vars["dlc_custom_config"],
                 width=22, bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Consolas", 8)).pack(side="left", padx=(0, 4))
        tk.Button(cust_row, text="Browse…", font=("Segoe UI", 8),
                  bg=C["card2"], fg=C["cyan"], relief="flat",
                  padx=6, cursor="hand2",
                  command=self._browse_config).pack(side="left")

        # ── Detection thresholds ──────────────────────────────────────────────
        sec2 = _adv_section(p, "DETECTION THRESHOLDS", C["cyan"])

        def _spin_f(row, key, lo, hi, step, default):
            v = self._v(key, tk.DoubleVar(value=default))
            tk.Spinbox(row, from_=lo, to=hi, increment=step,
                       format="%.2f", textvariable=v, width=7,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left")

        def _spin_i(row, key, lo, hi, step, default):
            v = self._v(key, tk.IntVar(value=default))
            tk.Spinbox(row, from_=lo, to=hi, increment=step,
                       textvariable=v, width=7,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left")

        _adv_row(sec2, "Pose confidence (pcutoff)",
                 lambda r: _spin_f(r, "dlc_pcutoff", 0.0, 1.0, 0.05, 0.6))
        _adv_row(sec2, "Bounding-box threshold",
                 lambda r: _spin_f(r, "dlc_bbox_threshold", 0.0, 1.0, 0.05, 0.6))
        _adv_row(sec2, "Max individuals per frame",
                 lambda r: _spin_i(r, "dlc_max_individuals", 1, 20, 1, 1))

        # ── Batch sizes ───────────────────────────────────────────────────────
        sec3 = _adv_section(p, "BATCH SIZES  (0 = auto from GPU memory)", C["orange"])

        def _spin_i_tip(row, key, tip, default):
            v = self._v(key, tk.IntVar(value=default))
            tk.Spinbox(row, from_=0, to=128, increment=4,
                       textvariable=v, width=6,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left")
            tk.Label(row, text=tip, font=("Segoe UI", 7),
                     bg=C["card"], fg=C["dim"]).pack(side="left", padx=4)

        _adv_row(sec3, "Inference (pose) batch",
                 lambda r: _spin_i_tip(r, "dlc_inf_batch",
                                       "0=auto  8/16/32 typical", 0))
        _adv_row(sec3, "Detector batch",
                 lambda r: _spin_i_tip(r, "dlc_det_batch",
                                       "0=same as inference batch", 0))

        # ── Epochs (when video_adapt=True) ────────────────────────────────────
        sec4 = _adv_section(p, "ADAPT EPOCHS  (used when Video Adapt is enabled)", C["purple"])
        _adv_row(sec4, "Detector epochs",
                 lambda r: _spin_i(r, "dlc_det_epochs", 1, 200, 1, 15))
        _adv_row(sec4, "Pose epochs",
                 lambda r: _spin_i(r, "dlc_pose_epochs", 1, 200, 1, 15))

        # ── Scale list ────────────────────────────────────────────────────────
        sec5 = _adv_section(p, "DETECTOR SCALE LIST  (input crop sizes for detection)", C["accent"])
        self._v("dlc_scale_mode", tk.StringVar(value="Auto"))
        self._scale_body = tk.Frame(sec5, bg=C["card"])

        def _scale_mode_row(text, val):
            r = tk.Frame(sec5, bg=C["card"])
            r.pack(anchor="w", padx=8, pady=1)
            tk.Radiobutton(r, text=text, variable=self._vars["dlc_scale_mode"],
                           value=val, bg=C["card"], fg=C["text"],
                           selectcolor=C["card2"], activebackground=C["card"],
                           font=("Segoe UI", 9),
                           command=self._toggle_scale).pack(side="left")

        _scale_mode_row("Auto  (derived from video resolution)", "Auto")
        _scale_mode_row("Manual  (set range below)", "Manual")

        self._scale_body = tk.Frame(sec5, bg=C["card"])
        self._scale_body.pack(fill="x", padx=8, pady=(0, 4))

        srow = tk.Frame(self._scale_body, bg=C["card"])
        srow.pack(anchor="w", padx=0, pady=2)
        for label, key, default in [
            ("Min:", "dlc_scale_min", 100),
            ("Max:", "dlc_scale_max", 600),
            ("Step:", "dlc_scale_step", 50),
        ]:
            tk.Label(srow, text=label, font=("Segoe UI", 9),
                     bg=C["card"], fg=C["text"]).pack(side="left", padx=(6, 0))
            v = self._v(key, tk.IntVar(value=default))
            tk.Spinbox(srow, from_=50, to=2000, increment=50,
                       textvariable=v, width=6,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left", padx=(2, 0))

    def _toggle_model_type(self):
        if self._vars["dlc_use_custom"].get():
            self._sa_sec.pack_forget()
            self._cust_sec.pack(fill="x", padx=8, pady=(2, 4))
        else:
            self._cust_sec.pack_forget()
            self._sa_sec.pack(fill="x", padx=8, pady=(2, 4))

    def _toggle_scale(self):
        if self._vars["dlc_scale_mode"].get() == "Manual":
            self._scale_body.pack(fill="x", padx=8, pady=(0, 4))
        else:
            self._scale_body.pack_forget()

    def _browse_config(self):
        p = filedialog.askopenfilename(
            title="Select DLC config.yaml",
            filetypes=[("YAML config", "*.yaml *.yml"), ("All", "*")])
        if p:
            self._vars["dlc_custom_config"].set(p)

    def _load(self):
        cfg = self._session.get("dlc_advanced_cfg", {})
        for k, default in self.DEFAULTS.items():
            val = cfg.get(k, default)
            if k in self._vars:
                try:
                    self._vars[k].set(val)
                except Exception:
                    pass
        self._toggle_model_type()
        self._toggle_scale()

    def _restore(self):
        for k, default in self.DEFAULTS.items():
            if k in self._vars:
                try:
                    self._vars[k].set(default)
                except Exception:
                    pass
        self._toggle_model_type()
        self._toggle_scale()

    def _apply(self):
        cfg = {}
        for k, var in self._vars.items():
            try:
                cfg[k] = var.get()
            except Exception:
                pass
        self._session["dlc_advanced_cfg"] = cfg
        self.destroy()


class AdvancedCUBEWindow(tk.Toplevel):
    """Modal popup for advanced CUBE engine / analysis parameters."""

    DEFAULTS = dict(
        body_normalise        = True,
        pca_pre_reduce        = "auto",
        likelihood_thresh     = 0.30,
        boxcar_win_sec        = 0.07,
        train_frac            = 0.30,
        umap_full_thresh      = 10_000,
        umap_n_neighbors      = 60,
        umap_n_components     = 3,
        umap_min_dist         = 0.10,
        umap_random_state     = 42,
        hdbscan_metric        = "euclidean",
        hdbscan_method        = "eom",
        hdbscan_methods_to_try = "eom,leaf",
        target_n_clusters     = 0,
        preferred_clusters_lo = 8,
        preferred_clusters_hi = 30,
        mlp_hidden            = "100,50",
        mlp_max_iter          = 1000,
        cv_folds              = 5,
        output_fps            = 15,
        max_clips_per_cluster = 3,
        save_plots            = True,
        save_videos           = True,
    )

    def __init__(self, parent, session: "SessionState"):
        super().__init__(parent)
        self.title("⚙  Advanced CUBE Analysis Parameters")
        self.configure(bg=C["bg"])
        self.geometry("520x720")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self._session = session
        self._vars: dict = {}

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(side="bottom", fill="x", pady=8, padx=12)
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 9),
                  bg=C["card2"], fg=C["subtext"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Restore Defaults", font=("Segoe UI", 9),
                  bg=C["card2"], fg=C["yellow"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self._restore).pack(side="left", padx=6)
        tk.Button(btn_f, text="Apply & Close", font=("Segoe UI", 10, "bold"),
                  bg=C["purple"], fg="white", relief="flat",
                  padx=16, pady=5, cursor="hand2",
                  command=self._apply).pack(side="right")

        # ── Scrollable canvas ─────────────────────────────────────────────────
        canvas = tk.Canvas(self, bg=C["bg"], highlightthickness=0)
        sb = tk.Scrollbar(self, orient="vertical", command=canvas.yview,
                          bg=C["card"], troughcolor=C["bg"])
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=C["bg"])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-(e.delta // 120), "units"))

        def _unbind_mw(e):
            try:
                canvas.unbind_all("<MouseWheel>")
            except Exception:
                pass
        self.bind("<Destroy>", _unbind_mw)

        self._build(inner)
        self._load()

    def _v(self, key, var):
        self._vars[key] = var
        return var

    def _build(self, p):
        tk.Label(p, text="  ⚙  Advanced CUBE Analysis Parameters",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["bg"], fg=C["purple"]).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(p,
                 text="  Publication defaults are pre-set. Only change these if\n"
                      "  you have a specific reason (e.g. high-fps data, noisy DLC).",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["dim"],
                 justify="left").pack(anchor="w", padx=10, pady=(0, 4))

        def _spin_f(row, key, lo, hi, step, default):
            v = self._v(key, tk.DoubleVar(value=default))
            tk.Spinbox(row, from_=lo, to=hi, increment=step,
                       format="%.3f", textvariable=v, width=8,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left")

        def _spin_i(row, key, lo, hi, step, default):
            v = self._v(key, tk.IntVar(value=default))
            tk.Spinbox(row, from_=lo, to=hi, increment=step,
                       textvariable=v, width=8,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left")

        def _check(row, key, default):
            v = self._v(key, tk.BooleanVar(value=default))
            tk.Checkbutton(row, variable=v, bg=C["card"], fg=C["green"],
                           selectcolor=C["card2"],
                           activebackground=C["card"]).pack(side="left")

        def _combo(row, key, values, default):
            v = self._v(key, tk.StringVar(value=default))
            ttk.Combobox(row, textvariable=v, values=values,
                         state="readonly", width=18,
                         font=("Segoe UI", 9)).pack(side="left")

        # ── Feature extraction ────────────────────────────────────────────────
        s = _adv_section(p, "FEATURE EXTRACTION", C["cyan"])
        _adv_row(s, "Body normalisation",
                 lambda r: _check(r, "body_normalise", True))
        tk.Label(s,
                 text="    Divide all distances by nose-to-tailbase spine length.\n"
                      "    Requires 'nose' and 'tailbase' bodypart names in DLC output.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "PCA pre-reduction",
                 lambda r: _combo(r, "pca_pre_reduce",
                                  ["auto", "on", "off"], "auto"))
        tk.Label(s,
                 text="    auto = reduce when features ≥ samples/5  |  on = always  |  off = never\n"
                      "    Reduces dimensionality before UMAP when features ≈ samples.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Likelihood threshold",
                 lambda r: _spin_f(r, "likelihood_thresh", 0.0, 1.0, 0.05, 0.30))
        _adv_row(s, "Boxcar smooth (s)",
                 lambda r: _spin_f(r, "boxcar_win_sec", 0.0, 0.5, 0.01, 0.07))

        # ── UMAP ─────────────────────────────────────────────────────────────
        s2 = _adv_section(p, "UMAP EMBEDDING  (Hsu & Bhatt 2021 defaults)", C["cyan"])
        _adv_row(s2, "Full-data threshold",
                 lambda r: _spin_i(r, "umap_full_thresh", 1000, 100_000, 1000, 10_000))
        tk.Label(s2,
                 text="    Use all bins for UMAP when total bins ≤ this value.\n"
                      "    Subsamples at 'Train fraction' only for larger recordings.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s2, "Train fraction",
                 lambda r: _spin_f(r, "train_frac", 0.05, 1.0, 0.05, 0.30))
        _adv_row(s2, "n_neighbors",
                 lambda r: _spin_i(r, "umap_n_neighbors", 5, 300, 5, 60))
        _adv_row(s2, "n_components",
                 lambda r: _spin_i(r, "umap_n_components", 2, 10, 1, 3))
        _adv_row(s2, "min_dist",
                 lambda r: _spin_f(r, "umap_min_dist", 0.0, 1.0, 0.05, 0.10))
        _adv_row(s2, "Random seed",
                 lambda r: _spin_i(r, "umap_random_state", 0, 9999, 1, 42))

        # ── HDBSCAN ───────────────────────────────────────────────────────────
        s3 = _adv_section(p, "HDBSCAN CLUSTERING", C["orange"])
        _adv_row(s3, "Distance metric",
                 lambda r: _combo(r, "hdbscan_metric",
                                  ["euclidean", "manhattan", "cosine"],
                                  "euclidean"))
        _adv_row(s3, "Cluster selection method",
                 lambda r: _combo(r, "hdbscan_method",
                                  ["eom", "leaf"], "eom"))
        tk.Label(s3,
                 text="    Both eom and leaf are tried automatically; DBCV score selects\n"
                      "    the best result (most internally cohesive + maximally separated).\n"
                      "    min_cluster_size is swept adaptively (anchored to full dataset size).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── Cluster count guidance ────────────────────────────────────────────
        s3b = _adv_section(p, "CLUSTER COUNT GUIDANCE", C["cyan"])
        _adv_row(s3b, "Target cluster count",
                 lambda r: _spin_i(r, "target_n_clusters", 0, 200, 1, 0))
        tk.Label(s3b,
                 text="    0 = auto (prefers 8–30 clusters by default).\n"
                      "    Set to a positive number to re-run Step 3 and steer the\n"
                      "    analysis toward that cluster count while keeping DBCV quality.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3b, "Preferred range — low",
                 lambda r: _spin_i(r, "preferred_clusters_lo", 2, 100, 1, 8))
        _adv_row(s3b, "Preferred range — high",
                 lambda r: _spin_i(r, "preferred_clusters_hi", 2, 200, 1, 30))
        tk.Label(s3b,
                 text="    Auto-mode selects the best DBCV solution inside this range.\n"
                      "    Ignored when Target cluster count > 0.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── MLP classifier ────────────────────────────────────────────────────
        def _adv_entry(row, key, default):
            v = self._v(key, tk.StringVar(value=default))
            tk.Entry(row, textvariable=v, width=14,
                     bg=C["card2"], fg=C["text"],
                     insertbackground=C["text"],
                     relief="flat", font=("Segoe UI", 9)).pack(side="left")

        s4 = _adv_section(p, "MLP CLASSIFIER", C["purple"])
        _adv_row(s4, "Hidden layer sizes",
                 lambda r: _adv_entry(r, "mlp_hidden", "100,50"))
        _adv_row(s4, "Max iterations",
                 lambda r: _spin_i(r, "mlp_max_iter", 100, 10000, 100, 1000))
        _adv_row(s4, "Cross-validation folds",
                 lambda r: _spin_i(r, "cv_folds", 2, 10, 1, 5))
        tk.Label(s4,
                 text="    Hidden layers: comma-separated sizes, e.g. '100,50' or '256,128,64'",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── Output options ─────────────────────────────────────────────────────
        s5 = _adv_section(p, "OUTPUT OPTIONS", C["accent"])
        _adv_row(s5, "Example clip FPS",
                 lambda r: _spin_i(r, "output_fps", 1, 60, 1, 15))
        _adv_row(s5, "Max clips per cluster",
                 lambda r: _spin_i(r, "max_clips_per_cluster", 1, 20, 1, 3))
        _adv_row(s5, "Save plots",
                 lambda r: _check(r, "save_plots", True))
        _adv_row(s5, "Save labeled videos",
                 lambda r: _check(r, "save_videos", True))
        _adv_row(s5, "Plot theme",
                 lambda r: _combo(r, "plot_theme", ["dark", "light"], "dark"))
        tk.Label(s5,
                 text="    dark = white-on-black figures  |  light = publication-ready white background",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

    def _load(self):
        cfg = self._session.get("engine_cfg", {})
        for k, default in self.DEFAULTS.items():
            val = cfg.get(k, default)
            if k in self._vars:
                try:
                    self._vars[k].set(val)
                except Exception:
                    pass

    def _restore(self):
        for k, default in self.DEFAULTS.items():
            if k in self._vars:
                try:
                    self._vars[k].set(default)
                except Exception:
                    pass

    def _apply(self):
        cfg = {}
        for k, var in self._vars.items():
            try:
                cfg[k] = var.get()
            except Exception:
                pass
        self._session["engine_cfg"] = cfg
        self.destroy()


#
#  MAIN APPLICATION
#

class PipelineApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CUBE: Comprehensive Unsupervised Behavioral Explorer  v3.0")
        self.configure(bg=C["bg"])
        self.geometry("1300x820")
        self.minsize(1100, 660)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # core state — output_root is left empty until video folders are added;
        # _resolve_work_dir() will derive it from the data drive at that point.
        self._session  = SessionState()
        # Temporary logger in the script directory (same drive as CUBE install)
        # until a video folder is added, at which point it migrates to the data drive.
        _init_log_dir = HERE / "CUBE_logs"
        _init_log_dir.mkdir(parents=True, exist_ok=True)
        self._logger   = PipelineLogger(_init_log_dir)
        self._running  = False
        self._chain_engine_after_prep = False

        self._build_ui()
        self._initial_log()
        self._tick_timer()

    #  " "  timer  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 
    def _tick_timer(self):
        self.after(1000, self._tick_timer)

    #  " "  UI  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 

    def _build_ui(self):
        self._build_header()
        body = tk.PanedWindow(self, orient="vertical",
                              bg=C["bg"], sashwidth=6, sashrelief="flat")
        body.pack(fill="both", expand=True)
        self._build_top_pane(body)
        self._build_log_pane(body)

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["log_bg"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="  CUBE",
                 font=("Segoe UI", 18, "bold"),
                 bg=C["log_bg"], fg=C["accent"]).pack(side="left", padx=(16, 0), pady=10)
        tk.Label(hdr, text="  Comprehensive Unsupervised Behavioral Explorer",
                 font=("Segoe UI", 13),
                 bg=C["log_bg"], fg=C["text"]).pack(side="left")

        right = tk.Frame(hdr, bg=C["log_bg"])
        right.pack(side="right", padx=12)
        for txt, cmd in [
            ("Help",          lambda: show_help(self)),
            ("Theme",         self._toggle_theme),
            ("Save",          self._save_session),
            ("Load",          self._load_session),
            ("Output",        self._set_output),
        ]:
            tk.Button(right, text=txt, font=("Segoe UI", 9),
                      bg=C["card2"], fg=C["subtext"],
                      relief="flat", padx=10, pady=5,
                      cursor="hand2", command=cmd).pack(
                side="left", padx=3, pady=8)

        # flow bar
        flow = tk.Frame(self, bg=C["card2"], pady=5)
        flow.pack(fill="x")
        self._flow_labels = []
        self._flow_bar_frame = flow
        for label, colour in [
            ("1 DLC",        C["green"]),
            (" ▸",           C["dim"]),
            ("2 Pre-process", C["cyan"]),
            (" ▸",           C["dim"]),
            ("3 Clustering", C["purple"]),
            (" ▸",           C["dim"]),
            ("4 Annotate",   C["orange"]),
            (" ▸",           C["dim"]),
            ("5 Analyse",    C["accent"]),
        ]:
            is_step = " ▸" not in label
            lbl = tk.Label(flow, text=f"  {label}  ",
                           font=("Segoe UI", 9,
                                 "bold" if is_step else "normal"),
                           bg=C["card2"], fg=colour)
            lbl.pack(side="left")
            self._flow_labels.append((lbl, colour))


    def _update_flow_bar(self, active_index: int):
        """Highlight the active step with inverted colours; dim all others."""
        if not hasattr(self, "_flow_labels"):
            return
        for i, (lbl, orig_col) in enumerate(self._flow_labels):
            if i % 2 != 0:                     # separator arrow — leave as-is
                continue
            step_i = i // 2
            if active_index != -1 and step_i == active_index:
                # Active step: white text on coloured background
                lbl.configure(fg="white", bg=orig_col,
                              font=("Segoe UI", 9, "bold"))
            elif active_index == -1:
                # Idle: restore all steps to normal colour on card2 background
                lbl.configure(fg=orig_col, bg=C["card2"],
                              font=("Segoe UI", 9, "bold"))
            else:
                # Non-active during a run: dim
                lbl.configure(fg=C["dim"], bg=C["card2"],
                              font=("Segoe UI", 9, "bold"))

    def _build_top_pane(self, parent):
        top = tk.Frame(parent, bg=C["bg"])
        parent.add(top, minsize=360)

        # left column
        left = tk.Frame(top, bg=C["bg"])
        left.pack(side="left", fill="both", padx=(10,4), pady=8)

        # output path display
        op_f = tk.Frame(left, bg=C["card"])
        op_f.pack(fill="x", pady=(0,4))
        tk.Label(op_f, text="  Output:",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["card"], fg=C["subtext"]).pack(side="left", padx=8, pady=4)
        self._out_lbl = tk.Label(op_f,
                                  text="(auto — set when folders are added)",
                                  font=("Segoe UI", 8),
                                  bg=C["card"], fg=C["cyan"],
                                  wraplength=280, justify="left")
        self._out_lbl.pack(side="left", padx=4)

        self._folder_list = FolderList(left,
                                        on_change=self._folders_changed)
        self._folder_list.pack(fill="x", pady=(0,4))

        # ── Experimental Group Assignment (Step 5 — split analyses by group) ─
        eg_frame = tk.Frame(left, bg=C["card"],
                            highlightbackground=C["cyan"],
                            highlightthickness=1)
        eg_frame.pack(fill="x", pady=(0, 4))
        tk.Label(eg_frame,
                 text="  Experimental Groups  (Step 5 — optional)",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["card"], fg=C["cyan"]).pack(anchor="w", padx=8, pady=(5, 2))
        eg_row = tk.Frame(eg_frame, bg=C["card"])
        eg_row.pack(fill="x", padx=8, pady=(0, 5))
        tk.Label(eg_row, text="Selected folder →  Group:",
                 font=("Segoe UI", 8), bg=C["card"],
                 fg=C["subtext"]).pack(side="left")
        self._eg_var = tk.StringVar(value="")
        tk.Entry(eg_row, textvariable=self._eg_var, width=14,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left", padx=(4, 4))
        tk.Button(eg_row, text="Apply", font=("Segoe UI", 8),
                  bg=C["cyan"], fg="white", relief="flat", padx=6,
                  cursor="hand2",
                  command=self._apply_exp_group).pack(side="left")
        tk.Label(eg_row,
                 text="  (select a folder above first)",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(side="left", padx=4)
        self._eg_lbl = tk.Label(eg_frame, text="No groups set.",
                                 font=("Segoe UI", 7), bg=C["card"],
                                 fg=C["dim"])
        self._eg_lbl.pack(anchor="w", padx=8, pady=(0, 4))
        # When a folder is selected in the listbox, populate the group entry
        self._folder_list._lb.bind(
            "<<ListboxSelect>>", self._on_folder_select)

        # ── Bout Duration Panel (only required user input per publication) ────
        bd_frame = tk.Frame(left, bg=C["card"],
                            highlightbackground=C["purple"],
                            highlightthickness=2)
        bd_frame.pack(fill="x", pady=(0, 4))
        tk.Label(bd_frame,
                 text="  Bout Duration Filter  (Step 3 — required input)",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["card"], fg=C["purple"]).pack(anchor="w", padx=8,
                                                     pady=(6, 2))
        bd_row = tk.Frame(bd_frame, bg=C["card"])
        bd_row.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(bd_row, text="Min (s):", font=("Segoe UI", 9),
                 bg=C["card"], fg=C["text"]).pack(side="left")
        self._bd_min = tk.DoubleVar(value=0.3)
        tk.Spinbox(bd_row, from_=0.0, to=60.0, increment=0.1,
                   format="%.2f", textvariable=self._bd_min, width=7,
                   bg=C["card2"], fg=C["text"],
                   buttonbackground=C["card2"],
                   font=("Segoe UI", 9)).pack(side="left", padx=(2, 14))
        tk.Label(bd_row, text="Max (s):", font=("Segoe UI", 9),
                 bg=C["card"], fg=C["text"]).pack(side="left")
        self._bd_max = tk.DoubleVar(value=5.0)
        tk.Spinbox(bd_row, from_=0.1, to=600.0, increment=1.0,
                   format="%.1f", textvariable=self._bd_max, width=7,
                   bg=C["card2"], fg=C["text"],
                   buttonbackground=C["card2"],
                   font=("Segoe UI", 9)).pack(side="left", padx=(2, 0))
        tk.Label(bd_row, text="  (per publication methodology)",
                 font=("Segoe UI", 8), bg=C["card"],
                 fg=C["dim"]).pack(side="left", padx=6)

        self._settings = SettingsPanel(left)
        self._settings.pack(fill="x")

        adv_row = tk.Frame(left, bg=C["bg"])
        adv_row.pack(fill="x", pady=(4, 2))
        tk.Button(adv_row, text="⚙  DLC & Prep Settings...",
                  font=("Segoe UI", 8, "bold"), bg=C["card2"], fg=C["yellow"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_dlc_prep).pack(side="left", padx=(0, 4), pady=2)
        tk.Button(adv_row, text="⚙  Advanced DLC Parameters...",
                  font=("Segoe UI", 8, "bold"), bg=C["card2"], fg=C["green"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_dlc_advanced).pack(side="left", padx=(0, 4), pady=2)
        tk.Button(adv_row, text="⚙  Advanced CUBE Analysis...",
                  font=("Segoe UI", 8, "bold"), bg=C["card2"], fg=C["purple"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_cube_advanced).pack(side="left", pady=2)

        # right column  " step cards
        right = tk.Frame(top, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True,
                   padx=(4,10), pady=8)
        for i in range(5):
            right.columnconfigure(i, weight=1, uniform="card")
        right.rowconfigure(0, weight=1)

        self._cards: dict[str, StepCard] = {}
        CMDS = {
            "dlc":       self._launch_dlc,
            "bsoid_prep":self._launch_bsoid_prep,
            "bsoid_run": self._launch_bsoid_run,
            "annotate":  self._launch_annotate,
            "analyse":   self._launch_analyse,
        }
        for i, meta in enumerate(STEP_META):
            card = StepCard(right, meta, CMDS[meta["key"]])
            card.grid(row=0, column=i, sticky="nsew", padx=4)
            self._cards[meta["key"]] = card

        # progress + status
        pb_outer = tk.Frame(right, bg=C["panel"],
                            highlightbackground=C["border"],
                            highlightthickness=1)
        pb_outer.grid(row=1, column=0, columnspan=5, sticky="ew",
                      padx=4, pady=(6,0))
        right.rowconfigure(1, minsize=90)
        self._pb = DualProgressBar(pb_outer)
        self._pb.pack(fill="x")

        sb = tk.Frame(right, bg=C["card2"])
        sb.grid(row=2, column=0, columnspan=5, sticky="ew", padx=4, pady=(4,0))
        self._status_lbl = tk.Label(sb, text="Ready.",
                                     font=("Segoe UI", 9),
                                     bg=C["card2"], fg=C["subtext"])
        self._status_lbl.pack(side="left", padx=10, pady=4)
        self._timer_lbl = tk.Label(sb, text="00:00:00",
                                    font=("Consolas", 9, "bold"),
                                    bg=C["card2"], fg=C["yellow"])
        self._timer_lbl.pack(side="right", padx=10)
        self._step_start_t = 0.0

    def _build_log_pane(self, parent):
        log_f = tk.Frame(parent, bg=C["panel"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        parent.add(log_f, minsize=200)
        self._log_panel = LogPanel(log_f)
        self._log_panel.pack(fill="both", expand=True)
        self._log_panel.attach(self._logger)

    #  " "  initial log  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 

    def _initial_log(self):
        self._logger.step("=" * 60)
        self._logger.step("  CUBE: Comprehensive Unsupervised Behavioral Explorer  v3.0")
        self._logger.step("=" * 60)
        if CORE_OK:
            self._logger.success("  ✓  cube_core v2 loaded")
        else:
            self._logger.error(f"  ✗  cube_core NOT found: {_CORE_ERR}")
        if _MOD_ANALYSER:
            self._logger.success(f"  ✓  Analyser loaded: {_PATH_ANALYSER.name}")
        else:
            self._logger.warn("      Analyser script not found")
        if _MOD_VIDEO:
            self._logger.success(f"  ✓  Video Explorer loaded: {_PATH_VIDEO.name}")
        else:
            self._logger.warn("      Video Explorer script not found")
        if not CTK_OK:
            self._logger.warn("      customtkinter not installed (needed for Step 5)")
        self._logger.info(f"  Log: {self._logger.log_path}")
        self._logger.info("  Add video folders and click Step 1 to begin.")

    #  " "  timer  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 

    def _start_step_timer(self):
        self._step_start_t = time.time()
        self._update_timer()

    def _update_timer(self):
        if not self._running:
            return
        el   = int(time.time() - self._step_start_t)
        h, r = divmod(el, 3600)
        m, s = divmod(r, 60)
        self._timer_lbl.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.after(1000, self._update_timer)

    #  " "  helpers  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 

    def _status(self, msg: str):
        self._status_lbl.configure(text=msg)
        self.update_idletasks()

    def _folders_changed(self):
        folders = self._folder_list.get_folders()
        self._session["video_folders"] = folders

        # Auto-derive output root from the data drive whenever it is not yet
        # explicitly set (empty) AND at least one video folder exists.
        if folders and not (self._session.get("output_root") or "").strip():
            new_root = str(_resolve_work_dir(self._session))
            self._session["output_root"] = new_root
            if hasattr(self, "_out_lbl"):
                self._out_lbl.configure(text=new_root)
            # Migrate logger to the data-drive location
            self._logger.close()
            _log_dir = Path(new_root) / "logs"
            _log_dir.mkdir(parents=True, exist_ok=True)
            self._logger = PipelineLogger(_log_dir)
            if hasattr(self, "_log_panel"):
                self._log_panel.attach(self._logger)
            self._logger.info(f"Output root auto-set to data drive: {new_root}")

        # Prune group assignments for removed folders
        groups = {k: v for k, v in self._session.get("video_groups", {}).items()
                  if k in folders}
        self._session["video_groups"] = groups
        if hasattr(self, "_eg_lbl"):
            self._update_eg_label()

    def _on_folder_select(self, _event=None):
        """Populate group entry when a folder is selected in the listbox."""
        sel = self._folder_list._lb.curselection()
        if not sel:
            return
        folder = self._folder_list._lb.get(sel[0])
        groups = self._session.get("video_groups", {})
        self._eg_var.set(groups.get(folder, ""))

    def _apply_exp_group(self):
        """Store the group name for the currently selected folder."""
        sel = self._folder_list._lb.curselection()
        if not sel:
            messagebox.showinfo("Select folder",
                "Select a folder in the list above, then type a group name and click Apply.")
            return
        folder = self._folder_list._lb.get(sel[0])
        group  = self._eg_var.get().strip()
        groups = dict(self._session.get("video_groups", {}))
        if group:
            groups[folder] = group
        else:
            groups.pop(folder, None)
        self._session["video_groups"] = groups
        self._update_eg_label()

    def _update_eg_label(self):
        groups = self._session.get("video_groups", {})
        if not groups:
            self._eg_lbl.configure(text="No groups set.")
        else:
            parts = sorted(set(groups.values()))
            self._eg_lbl.configure(
                text=f"{len(groups)} folder(s) assigned: " + ", ".join(parts))

    def _after(self, fn):
        """Thread-safe: run fn on main thread."""
        self.after(0, fn)

    #  " "  common step runner  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 

    def _run_step(self, key: str, fn, *args):
        """Launch fn(*args) in a daemon thread; update card/pb/session."""
        if self._running:
            messagebox.showwarning("Busy", "Another step is already running.")
            return
        self._running = True
        self._session.set_status(key, "running")
        self._settings.export_to_session(self._session)
        self._session["video_folders"] = self._folder_list.get_folders()
        self._save_session_auto()
        self._cards[key].set_status("running")
        self._status(f"Running step: {key} ")
        self._start_step_timer()
        step_idx = {"dlc":0, "bsoid_prep":1, "bsoid_run":2, "annotate":3, "analyse":4}.get(key, -1)
        self._update_flow_bar(step_idx)

        def _worker():
            try:
                fn(*args)
                self._session.set_status(key, "done")
                self._session.save()
                n_done = sum(1 for s in STEP_META
                             if self._session.is_done(s["key"]))
                self._after(lambda: self._pb.set_overall(n_done))
                self._after(lambda: self._cards[key].set_status("done"))
                self._after(lambda: self._status(f"Step '{key}' complete v"))
                self._after(lambda: self._pb.step_done())
                self._logger.success(f"Step '{key}' complete.")
                # Auto-chain Steps 2+3 after DLC if user enabled it
                if key == "dlc" and self._settings.get("auto_bsoid", False):
                    self._logger.step("Auto-run: launching Steps 2+3...")
                    # Clear _running on the main-thread queue BEFORE the launch
                    # callback so _run_step doesn't see it still True (race condition).
                    self._after(lambda: setattr(self, '_running', False))
                    self._after(self._auto_launch_bsoid_chain)
                # If prep was auto-chained, trigger engine next
                elif key == "bsoid_prep" and getattr(self, "_chain_engine_after_prep", False):
                    self._chain_engine_after_prep = False
                    self._logger.step("Auto-run: launching Step 3 (clustering)...")
                    self._after(lambda: setattr(self, '_running', False))
                    self._after(self._launch_bsoid_run)
            except Exception:
                tb = traceback.format_exc()
                self._session.set_status(key, "error")
                self._session.save()
                self._logger.error(f"Step '{key}' failed:\n{tb}")
                self._after(lambda: self._cards[key].set_status("error"))
                self._after(lambda: self._status(f"Step '{key}' FAILED  -"))
                self._after(lambda: self._pb.step_done())
                self._after(lambda: messagebox.showerror(
                    f"Step {key} Error",
                    f"{tb[:1000]}\n\nSee log panel for details."))
            finally:
                self._running = False
                self._after(self._save_session_auto)
                self._after(lambda: self._update_flow_bar(-1))

        threading.Thread(target=_worker, daemon=True).start()

    def _auto_launch_bsoid_chain(self):
        """Chain Step 2 (prep) then Step 3 (clustering) after auto-run DLC."""
        # Set flag so prep completion will trigger the engine
        self._chain_engine_after_prep = True
        self._launch_bsoid_prep()

    #   ADVANCED POPUP OPENERS

    def _open_dlc_prep(self):
        DLCPrepSettingsWindow(self, self._settings)

    def _open_dlc_advanced(self):
        AdvancedDLCWindow(self, self._session)

    def _open_cube_advanced(self):
        AdvancedCUBEWindow(self, self._session)

    #   STEP LAUNCHERS

    def _launch_dlc(self):
        if not CORE_OK:
            messagebox.showerror("Missing", "cube_core.py not found.")
            return
        if not self._session["video_folders"]:
            messagebox.showwarning("No folders",
                "Add at least one video folder first.")
            return
        self._run_step("dlc", _run_dlc_step,
                       self._session, self._settings, self._logger,
                       self._pb, self._after)

    def _launch_bsoid_prep(self):
        if not CORE_OK:
            messagebox.showerror("Missing", "cube_core.py not found.")
            return
        if not self._session["video_folders"]:
            messagebox.showwarning("No folders",
                "Add at least one video folder first.")
            return
        self._run_step("bsoid_prep", _run_bsoid_prep_step,
                       self._session, self._settings, self._logger,
                       self._pb, self._after)

    def _launch_bsoid_run(self):
        if not CORE_OK:
            messagebox.showerror("Missing", "cube_core.py not found.")
            return
        # Ensure bsoid_ready_dirs is populated before launching the engine.
        # Try a recursive scan of loaded video folders first; only prompt the
        # user manually if nothing is found automatically.
        if not self._session.get("bsoid_ready_dirs"):
            _PROJ = "BSOID_Project_Ready"
            _found: list = []
            _seen:  set  = set()
            for _vf in self._session.get("video_folders", []):
                _fp = Path(_vf)
                if not _fp.is_dir():
                    continue
                if _fp.name == _PROJ:
                    _k = str(_fp.resolve())
                    if _k not in _seen:
                        _seen.add(_k); _found.append(str(_fp))
                else:
                    for _m in sorted(_fp.rglob(_PROJ)):
                        if _m.is_dir():
                            _k = str(_m.resolve())
                            if _k not in _seen:
                                _seen.add(_k); _found.append(str(_m))
            if _found:
                self._session["bsoid_ready_dirs"] = _found
            else:
                d = filedialog.askdirectory(
                    title="Select BSOID_Project_Ready folder "
                          "(or any folder containing CSV/H5 files)")
                if not d:
                    return
                self._session["bsoid_ready_dirs"] = [d]

        # Ask once per session whether to delete the BSOID_Project_Ready/videos/
        # copies after clustering completes (they duplicate source videos).
        if "bsoid_delete_videos_folder" not in self._session._d:
            ans = messagebox.askyesno(
                "Delete copied videos after analysis?",
                "BSOID_Project_Ready/videos/ contains copies of your source\n"
                "videos used for example-clip generation.\n\n"
                "Delete these copies once clustering completes?\n"
                "(Your original source videos are NOT affected.)",
                parent=self)
            self._session["bsoid_delete_videos_folder"] = ans

        bd_min = float(self._bd_min.get()) if hasattr(self, "_bd_min") else 0.0
        bd_max = float(self._bd_max.get()) if hasattr(self, "_bd_max") else 999.0
        self._run_step("bsoid_run", _run_engine_step,
                       self._session, self._settings, self._logger,
                       self._pb, self._after, bd_min, bd_max)

    def _launch_annotate(self):
        if _MOD_VIDEO is None:
            messagebox.showerror(
                "Script missing",
                f"cube_video_explorer.py not found in:\n{HERE}\n\n"
                "Place it in the same folder as this launcher.")
            return
        if self._running:
            messagebox.showwarning("Busy", "Another step is already running.")
            return
        self._running = True

        self._cards["annotate"].set_status("running")
        self._status("Step 4: Video Annotation — window opening")
        self._logger.step("Launching Video Explorer ")

        # Find example_clips folder.
        # engine_out_dirs points to BSOID_Project_Ready/cube_results_TIMESTAMP;
        # bsoid_ready_dirs points to BSOID_Project_Ready itself (Step 2 output).
        clip_folder = None
        for d in self._session.get("engine_out_dirs", []) + \
                 self._session.get("bsoid_ready_dirs", []):
            dp = Path(d)
            # Direct subpaths (covers engine_out_dirs which already IS cube_results_*)
            for sub in ("videos/example_clips", "example_clips", "videos", "output"):
                p = dp / sub
                if p.is_dir() and any(p.rglob("*.mp4")):
                    clip_folder = p
                    break
            if clip_folder:
                break
            # Glob for cube_results_* subdirectories (covers bsoid_ready_dirs parent)
            for cr in sorted(dp.glob("cube_results*"), reverse=True):
                for sub in ("videos/example_clips", "example_clips", "videos"):
                    p = cr / sub
                    if p.is_dir() and any(p.rglob("*.mp4")):
                        clip_folder = p
                        break
                if clip_folder:
                    break
            if clip_folder:
                break

        def _run():
            try:
                app = _MOD_VIDEO.BSoidAnnotator(auto_open=False)
                if clip_folder:
                    try:
                        clusters = _MOD_VIDEO.discover_clusters(clip_folder)
                        if clusters:
                            sd = app.sd
                            sd.folder_path   = clip_folder
                            sd.clusters      = clusters
                            sd.cluster_order = sorted(clusters.keys())
                            sd.current_index = 0
                            app.title(f"BSOID Annotator — {clip_folder.name}")
                            app._refresh_cluster_list()
                            app._refresh_group_panel()
                            app._refresh_assign_buttons()
                            app._load_current()
                            self._logger.success(
                                f"Loaded {len(clusters)} clusters from {clip_folder}")
                    except Exception:
                        self._logger.warn(
                            f"Auto-load failed: {traceback.format_exc()}")
                if not clip_folder:
                    # No clips found automatically — prompt the user once
                    self._logger.warn("No example clips found automatically. "
                                      "Use Open Folder to select the output directory.")
                    app.after(200, app._open_folder)
                # add menu
                mb = tk.Menu(app)
                sm = tk.Menu(mb, tearoff=0)
                sm.add_command(label="Open Folder ", command=app._open_folder)
                sm.add_command(label="Save Session",  command=app._save_session)
                sm.add_separator()
                sm.add_command(label="Export TSVs ",  command=app._export_tsv)
                sm.add_separator()
                sm.add_command(label="Quit",           command=app._on_close)
                mb.add_cascade(label="Session", menu=sm)
                app.configure(menu=mb)
                app.mainloop()
                self._running = False
                self._session.set_status("annotate", "done")
                self._session.save()
                self._after(lambda: self._cards["annotate"].set_status("done"))
                self._after(lambda: self._status("Step 4 complete v"))
            except Exception:
                self._running = False
                tb = traceback.format_exc()
                self._logger.error(f"Step 4 error:\n{tb}")
                self._after(lambda: self._cards["annotate"].set_status("error"))
                self._after(lambda: messagebox.showerror("Step 4 Error", tb[:800]))

        # Must run on main thread: BSoidAnnotator is a tk.Tk root window.
        # Creating or calling mainloop() on a Tk root from a background thread
        # triggers a STATUS_BREAKPOINT in tcl86t.dll (Tcl threading assertion).
        self.after(0, _run)

    def _launch_analyse(self):
        if _MOD_ANALYSER is None:
            messagebox.showerror(
                "Script missing",
                f"cube_analyser not found in:\n{HERE}\n\n"
                "Place it in the same folder as this launcher.")
            return
        if not CTK_OK:
            messagebox.showerror("Missing dependency",
                "customtkinter is required for the Analyser.\n"
                "Run:  pip install customtkinter")
            return
        if self._running:
            messagebox.showwarning("Busy", "Another step is already running.")
            return
        self._running = True

        # Load mapping file?
        mapping = self._session.get("mapping_file", "")
        if not mapping or not Path(mapping).is_file():
            ans = messagebox.askyesno(
                "Load mapping?",
                "Load a cluster 'behaviour TSV from Step 4?")
            if ans:
                p = filedialog.askopenfilename(
                    title="Select mapping TSV",
                    filetypes=[("TSV/JSON","*.tsv *.json"),("All","*")])
                if p:
                    self._session["mapping_file"] = p

        # Ask where to save group comparison plots (once per session)
        _comp_plot_dir = self._session.get("comparison_plot_dir", "")
        if not _comp_plot_dir:
            if messagebox.askyesno(
                "Comparison plots",
                "Choose a folder to save group comparison plots?\n\n"
                "(Skip to use the default location next to each data file.)"):
                _d = filedialog.askdirectory(
                    title="Select folder for group comparison plots")
                if _d:
                    self._session["comparison_plot_dir"] = _d
                    _comp_plot_dir = _d

        # Capture group assignments for injection into the analyser
        _video_groups = dict(self._session.get("video_groups", {}))

        self._cards["analyse"].set_status("running")
        self._status("Step 5: Analysis — window opening")
        self._logger.step("Launching CUBE Analyser")

        # Find bout_lengths folder — search session paths then video folders
        bout_root = None
        search_paths = (self._session.get("engine_out_dirs", []) +
                        self._session.get("bsoid_ready_dirs", []))
        for d in search_paths:
            p = Path(d)
            for sub in ("bout_lengths", "output", "BSOID", ""):
                candidate = p / sub if sub else p
                if candidate.is_dir():
                    files = (list(candidate.glob("*bout_lengths*.csv")) +
                             list(candidate.glob("*bout_lengths*.tsv")))
                    if files:
                        bout_root = candidate
                        break
            if bout_root:
                break

        if not bout_root:
            # Fallback: search video folders for cube_results_*/bout_lengths
            for folder in self._session.get("video_folders", []):
                for cr in sorted(Path(folder).glob("cube_results*"), reverse=True):
                    candidate = cr / "bout_lengths"
                    if candidate.is_dir():
                        files = (list(candidate.glob("*bout_lengths*.csv")) +
                                 list(candidate.glob("*bout_lengths*.tsv")))
                        if files:
                            bout_root = candidate
                            break
                if bout_root:
                    break

        _plot_theme = self._session.get("engine_cfg", {}).get("plot_theme", "dark")

        def _run():
            try:
                app = _MOD_ANALYSER.BSOiDApp()
                # Apply plot theme chosen in Advanced CUBE settings
                try:
                    app._toggle_theme(_plot_theme)
                except Exception:
                    pass
                # Inject comparison plot output directory if user specified one
                if _comp_plot_dir:
                    app._comparison_plot_dir = Path(_comp_plot_dir)
                if bout_root:
                    try:
                        files = _MOD_ANALYSER.find_bsoid_files(bout_root)
                        if files:
                            app._root_dir = bout_root
                            try:
                                app._folder_lbl.configure(text=str(bout_root))
                            except Exception:
                                pass
                            app._csv_paths = files
                            try:
                                app._csv_combo.configure(
                                    values=[f.name for f in files])
                                app._csv_combo.set(files[0].name)
                            except Exception:
                                pass
                            try:
                                app._load_csv(files[0])
                            except Exception:
                                pass
                            self._logger.success(
                                f"Auto-loaded {len(files)} file(s) from {bout_root}")
                            # Populate the Combined Analysis animal panel
                            try:
                                ap = getattr(app, "_animal_panel", None)
                                if ap is not None:
                                    ap.add_files_from_paths(files)
                            except Exception:
                                pass
                            # Inject experimental group assignments if set in session
                            if _video_groups:
                                try:
                                    ap = getattr(app, "_animal_panel", None)
                                    panel_animals = getattr(ap, "_animals", []) if ap else []
                                    assigned = 0
                                    for animal in panel_animals:
                                        apath = str(animal.get("path", ""))
                                        for folder, grp in _video_groups.items():
                                            if apath.startswith(folder):
                                                eg_var = animal.get("exp_group")
                                                if eg_var is not None:
                                                    eg_var.set(grp)
                                                    assigned += 1
                                                break
                                    if assigned:
                                        self._logger.success(
                                            f"  Injected exp_group for "
                                            f"{assigned} animal(s) from session groups")
                                except Exception:
                                    pass
                    except Exception:
                        self._logger.warn(
                            f"Auto-load failed: {traceback.format_exc()}")

                app.mainloop()
                self._running = False
                if messagebox.askyesno(
                        "Start another analysis?",
                        "Would you like to start a new analysis session?\n\n"
                        "Yes — resets this step so you can launch it again.\n"
                        "No  — marks Step 5 complete.",
                        default="no"):
                    self._session.set_status("analyse", "idle")
                    self._session.save()
                    self._after(lambda: self._cards["analyse"].set_status("idle"))
                    self._after(lambda: self._status("Step 5 ready — click to start another analysis."))
                else:
                    self._session.set_status("analyse", "done")
                    self._session.save()
                    self._after(lambda: self._cards["analyse"].set_status("done"))
                    self._after(lambda: self._status("Step 5 complete v"))
            except Exception:
                self._running = False
                tb = traceback.format_exc()
                self._logger.error(f"Step 5 error:\n{tb}")
                self._after(lambda: self._cards["analyse"].set_status("error"))
                self._after(lambda: messagebox.showerror("Step 5 Error", tb[:800]))

        # Must run on main thread: BSOiDApp is a ctk.CTk (tk.Tk) root window.
        self.after(0, _run)

    #  " "  session management  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 


    def _toggle_theme(self):
        global C
        if C["bg"] == "#09090f":
            # Dark to Light
            C["bg"] = "#f0f2f5"
            C["panel"] = "#ffffff"
            C["card"] = "#f8f9fa"
            C["card2"] = "#e9ecef"
            C["border"] = "#dee2e6"
            C["text"] = "#222222"
            C["subtext"] = "#666666"
            C["log_bg"] = "#ffffff"
            C["log_fg"] = "#333333"
            with open(HERE / "theme.txt", "w", encoding="utf-8") as f: f.write("light")
        else:
            # Light to Dark
            C["bg"] = "#09090f"
            C["panel"] = "#111120"
            C["card"] = "#16162a"
            C["card2"] = "#1e1e35"
            C["border"] = "#2a2a4a"
            C["text"] = "#eaeaea"
            C["subtext"] = "#7788aa"
            C["log_bg"] = "#07070d"
            C["log_fg"] = "#00ff88"
            with open(HERE / "theme.txt", "w", encoding="utf-8") as f: f.write("dark")
        messagebox.showinfo("Theme",
                            "Theme updated. Please restart CUBE for the "
                            "changes to take effect.")
        
    def _save_session(self):

        p = filedialog.asksaveasfilename(
            title="Save session",
            defaultextension=SESSION_EXT,
            filetypes=[("Session", f"*{SESSION_EXT}"), ("All","*")])
        if not p:
            return
        self._settings.export_to_session(self._session)
        self._session["video_folders"] = self._folder_list.get_folders()
        self._session.save(Path(p))
        self._logger.success(f"Session saved: {p}")
        messagebox.showinfo("Saved", f"Session saved:\n{p}")

    def _save_session_auto(self):
        out = _resolve_work_dir(self._session)
        self._settings.export_to_session(self._session)
        self._session["video_folders"] = self._folder_list.get_folders()
        self._session.save(out / f"autosave{SESSION_EXT}")

    def _load_session(self):
        p = filedialog.askopenfilename(
            title="Load session",
            filetypes=[("Session", f"*{SESSION_EXT}"), ("All","*")])
        if not p:
            return
        self._session = SessionState.load(Path(p))
        self._folder_list.set_folders(self._session["video_folders"])
        self._settings.apply_session(self._session)
        # Clear any C-drive output_root saved in the session file so
        # _resolve_work_dir() will re-derive from the data drive.
        _sys_drive = Path.home().drive.upper()
        _saved_root = self._session.get("output_root", "")
        if _saved_root and _sys_drive and Path(_saved_root).drive.upper() == _sys_drive:
            self._session["output_root"] = ""
            _saved_root = ""
        out = _saved_root
        if out:
            self._out_lbl.configure(text=out)
        else:
            # Re-derive from video folders that were just loaded
            new_root = str(_resolve_work_dir(self._session))
            self._session["output_root"] = new_root
            self._out_lbl.configure(text=new_root)
            self._logger.close()
            _log_dir = Path(new_root) / "logs"
            _log_dir.mkdir(parents=True, exist_ok=True)
            self._logger = PipelineLogger(_log_dir)
            self._log_panel.attach(self._logger)
        if hasattr(self, "_eg_lbl"):
            self._update_eg_label()
        for meta in STEP_META:
            st = self._session["step_status"].get(meta["key"], "idle")
            self._cards[meta["key"]].set_status(st)
        n = sum(1 for m in STEP_META if self._session.is_done(m["key"]))
        self._pb.set_overall(n)
        self._logger.success(f"Session loaded: {p}")
        self._status("Session loaded — completed steps shown as ✓.")

    def _set_output(self):
        d = filedialog.askdirectory(title="Select output root folder")
        if not d:
            return
        _sys_drive = Path.home().drive.upper()
        if _sys_drive and Path(d).drive.upper() == _sys_drive:
            messagebox.showwarning(
                "Wrong drive",
                f"Output folder is on the system drive ({_sys_drive}).\n"
                "All CUBE output must stay on the data drive to avoid\n"
                "filling up the system disk.\n\n"
                "Please choose a folder on the drive where your videos are.")
            return
        self._session["output_root"] = d
        self._out_lbl.configure(text=d)
        self._logger.close()
        self._logger = PipelineLogger(Path(d) / "logs")
        self._log_panel.attach(self._logger)
        self._logger.info(f"Output root set: {d}")

    #  " "  close  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno("Running",
                    "A step is still running.  Quit anyway?"):
                return
        if messagebox.askyesno("Quit", "Save session and exit?"):
            self._save_session_auto()
        self._logger.close()
        self.destroy()


#  
#  DEPENDENCY CHECK  (shown once at startup)
#  

def _check_and_warn():
    missing_scripts = []
    if not CORE_OK:
        missing_scripts.append("cube_core.py")
    if not _MOD_ANALYSER:
        missing_scripts.append("cube_analyser.py")
    if not _MOD_VIDEO:
        missing_scripts.append("cube_video_explorer.py")

    missing_pkgs = []
    for pkg, install in [
        ("umap",          "pip install umap-learn"),
        ("hdbscan",       "conda install -c conda-forge hdbscan"),
        ("sklearn",       "pip install scikit-learn"),
        ("PIL",           "pip install pillow"),
        ("cv2",           "pip install opencv-python-headless"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(f"{pkg}   '  {install}")

    if not CTK_OK:
        missing_pkgs.append("customtkinter   '  pip install customtkinter")

    msgs = []
    if missing_scripts:
        msgs.append("Scripts not found (place in same folder):")
        msgs += [f"   -  {s}" for s in missing_scripts]
    if missing_pkgs:
        msgs.append("\nMissing Python packages:")
        msgs += [f"   -  {p}" for p in missing_pkgs]

    if msgs:
        msgs.append(f"\nExpected folder:\n  {HERE}")
        msgs.append("Affected steps will be disabled; others work normally.")
        root_tmp = tk.Tk(); root_tmp.withdraw()
        messagebox.showwarning("CUBE — Setup incomplete", "\n".join(msgs))
        root_tmp.destroy()


#  
#  ENTRY POINT
#  

def main():
    _check_and_warn()
    app = PipelineApp()
    app.mainloop()


if __name__ == "__main__":
    main()
