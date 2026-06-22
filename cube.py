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
    pip install pillow opencv-python-headless scipy scikit-learn umap-learn customtkinter plotly
    conda install -c conda-forge hdbscan
"""

# Force single-threaded BLAS/MKL before any numpy import so loky workers
# spawned from this process inherit the correct threading config on Windows.
# cube_analyser.py sets the same vars at its own module level for standalone
# use; this block covers cube.py as the primary entry point.
import os as _os_env
for _k_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os_env.environ[_k_env] = "1"
del _os_env, _k_env

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
# Deferred to _deferred_imports() so the loading splash renders first.
CORE_OK      = False
_CORE_ERR    = ""
PipelineLogger = BSoidEngine = run_bsoid_prep = None
filter_dlc_h5 = cleanup_video_byproducts = create_umap_evolution_video = None

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

_MOD_VIDEO = _PATH_VIDEO = None
_MOD_ANALYSER = _PATH_ANALYSER = None

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
# Button-specific colour slots (allow light mode to make flat buttons visible)
C["btn"]    = C["card2"]   # button background
C["btn_fg"] = C["subtext"] # text on secondary/muted buttons

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
            # Light-mode button overrides: flat buttons need a mid-gray bg to
            # be visible against the near-white page; dark amber replaces the
            # bright yellow that is invisible on light backgrounds.
            C["btn"]    = "#8892a0"  # mid-gray — clearly distinct from #f0f2f5
            C["btn_fg"] = "#1c1c30"  # near-black — readable on mid-gray
            C["yellow"] = "#7a4e00"  # dark amber — replaces #ffd60a in light mode
except Exception:
    pass

# Resolved once at import so the splash and any future theme-aware widgets can
# read it without re-parsing theme.txt.
_DARK_THEME: bool = C["bg"] == "#09090f"

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
        bsoid_min_conf  = 0.30,
        bsoid_conf_metric    = "median",
        bsoid_min_sess_frac  = 0.6,
        bsoid_min_keep       = 6,
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
                  font=("Segoe UI", 8), bg=C["btn"], fg=C["btn_fg"],
                  relief="flat", padx=6, cursor="hand2",
                  command=self.clear).pack(side="right", padx=2)
        tk.Button(tb, text="  Open log",
                  font=("Segoe UI", 8), bg=C["btn"], fg=C["btn_fg"],
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
        ("bsoid_min_conf",   "Min BP confidence",   "float",(0.1,0.9,0.05),0.30,
         "Bodyparts below this confidence are excluded"),
        ("bsoid_conf_metric","BP conf metric",      "combo",
         ["median","mean"], "median",
         "median resists brief occlusion dropouts (single-view cameras)"),
        ("bsoid_min_sess_frac","BP keep if passes ≥","float",(0.1,1.0,0.1),0.6,
         "Keep a bodypart if it passes in >= this fraction of sessions (not all)"),
        ("bsoid_min_keep",   "Min bodyparts kept",  "int",  (2,40,1), 6,
         "Floor: fall back to top-N by confidence if fewer pass"),
        ("ntfy_topic",       "Notification Topic",  "str",  None, "",
         "ntfy.sh topic name for push alerts"),
    ]
    _ENGINE_ROWS = [
        ("body_normalise",       "Body normalisation",   "bool", None, False,
         "Divide distances by nose-to-tailbase length"),
        ("likelihood_thresh",    "Likelihood threshold", "float",(0.1,0.9,0.05),0.30,""),
        ("max_interp_gap_sec",   "Max interp gap (s)",   "float",(0.0,5.0,0.1),0.50,
         "Occlusions longer than this are held flat, not ramped (0 = legacy)"),
        ("boxcar_win_sec",       "Boxcar smooth (s)",    "float",(0.0,0.5,0.01),0.07,""),
        ("train_frac",           "UMAP train fraction",  "float",(0.05,1.0,0.05),0.30,""),
        ("umap_n_neighbors",     "UMAP n_neighbors",     "int",  (5,200,5),   60,""),
        ("umap_n_components",    "UMAP n_components",    "int",  (2,8,1),      3,""),
        ("umap_min_dist",        "UMAP min_dist",        "float",(0.0,1.0,0.05),0.1,""),
        ("umap_random_state",    "UMAP random seed",     "int",  (0,9999,1),  42,""),
        ("hdbscan_metric",       "HDBSCAN metric",       "combo",
         ["euclidean","manhattan","cosine"],"euclidean",""),
        ("hdbscan_method",       "HDBSCAN method",       "combo",
         ["both","eom","leaf"],"both","both=DBCV picks best; eom=larger; leaf=finer"),
        ("mlp_hidden",           "MLP layers",           "str",  None, "100,50",""),
        ("mlp_max_iter",         "MLP max iter",         "int",  (100,5000,100),1000,""),
        ("mlp_confidence_thresh","MLP conf threshold",   "float",(0.0,1.0,0.05),0.0,
         "Bins below this top-class probability become unclassified (0 = off)"),
        ("cv_folds",             "CV folds",             "int",  (2,10,1),     5,""),
        ("min_epoch_dur_s",      "Min epoch dur (s)",    "float",(0.0,60.0,0.1),0.0,""),
        ("max_epoch_dur_s",      "Max epoch dur (s)",    "float",(0.0,9999.0,1.0),300.0,""),
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
                        bg=C["btn"], fg=C["yellow"],
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


def _ffmpeg_transcode(src: str, dst: str, vf: str) -> None:
    """Transcode src → dst; tries h264_nvenc first, falls back to libx264."""
    _ff   = _resolve_ffmpeg()
    _base = [_ff, "-y", "-noautorotate", "-i", src, "-vf", vf, "-an"]
    try:
        subprocess.run(_base + ["-c:v", "h264_nvenc", "-preset", "p4", dst],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(_base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", dst],
                       check=True, capture_output=True)


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


def _validated_dlc_detector_name(detector_name: str, session, logger,
                                 fallback: str = "fasterrcnn_mobilenet_v3_large_fpn") -> str:
    """Return detector_name if DLC has a model_config YAML for it; otherwise fall back and warn."""
    try:
        import os as _os
        import deeplabcut as _dlc_tmp
        cfg_dir = _os.path.join(_os.path.dirname(_dlc_tmp.__file__),
                                "modelzoo", "model_configs")
        if not _os.path.isfile(_os.path.join(cfg_dir, f"{detector_name}.yaml")):
            avail = sorted(
                _os.path.splitext(f)[0]
                for f in _os.listdir(cfg_dir)
                if f.endswith(".yaml")
            )
            msg = (f"Detector '{detector_name}' has no model config in the installed "
                   f"DeepLabCut version (available: {', '.join(avail)}). "
                   f"Falling back to '{fallback}'.")
            logger.warn(f"  [DLC] {msg}")
            send_push_notification(session, msg, title="CUBE — Unsupported Detector", logger=logger)
            return fallback
    except Exception:
        pass
    return detector_name


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
    _detector_name  = _validated_dlc_detector_name(
                          str(_adv.get("dlc_detector",
                              "fasterrcnn_mobilenet_v3_large_fpn")),
                          session, logger)
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
    _crop_enable    = bool(_adv.get("dlc_crop_enable",     False))
    _crop_x         = int(_adv.get("dlc_crop_x",           0))
    _crop_y         = int(_adv.get("dlc_crop_y",           0))
    _crop_w         = int(_adv.get("dlc_crop_w",           0))
    _crop_h         = int(_adv.get("dlc_crop_h",           0))
    _do_crop        = _crop_enable and _crop_w > 0 and _crop_h > 0

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

        # resize / crop
        if long_edge or _do_crop:
            inf_path = os.path.join(dest_folder, f"resized_{base_noext}.mp4")
            if not os.path.exists(inf_path):
                _cap = cv2.VideoCapture(video_path)
                _ow_raw = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _oh_raw = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                try:
                    _rot = int(_cap.get(cv2.CAP_PROP_ORIENTATION_META))
                except Exception:
                    _rot = 0
                # Swap visual dimensions for 90°/270° rotated videos
                if _rot in (90, 270):
                    _ow, _oh = _oh_raw, _ow_raw
                else:
                    _ow, _oh = _ow_raw, _oh_raw
                # Post-crop source dimensions drive the resize target
                _src_w = _crop_w if _do_crop else _ow
                _src_h = _crop_h if _do_crop else _oh
                if long_edge:
                    _scale = min(long_edge / max(_src_w, _src_h), 1.0)
                    _nw = int(_src_w * _scale) & ~1
                    _nh = int(_src_h * _scale) & ~1
                else:
                    _nw = _src_w & ~1
                    _nh = _src_h & ~1
                if _nw == _ow and _nh == _oh and _rot == 0 and not _do_crop:
                    _cap.release()
                    shutil.copy2(video_path, inf_path)
                    logger(f"  Video already at/below target — copied to workspace")
                else:
                    _cap.release()
                    _msg = f"  Processing {_ow}x{_oh} → {_nw}x{_nh} (rotation={_rot}°)"
                    if _do_crop:
                        _msg += f" [crop {_crop_w}x{_crop_h} @ {_crop_x},{_crop_y}]"
                    logger(_msg)
                    _vf_parts = []
                    if _rot in (90, 270, 180):
                        _vf_parts.append(
                            {90: "transpose=1", 180: "transpose=2,transpose=2",
                             270: "transpose=2"}[_rot])
                    if _do_crop:
                        _vf_parts.append(
                            f"crop={_crop_w}:{_crop_h}:{_crop_x}:{_crop_y}")
                    _vf_parts.append(f"scale={_nw}:{_nh}:flags=area")
                    _ffmpeg_transcode(video_path, inf_path, ",".join(_vf_parts))
                    logger(f"  Processed video saved (long edge {long_edge})")
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
                min_confidence=float(session["bsoid_min_conf"]),
                conf_metric=str(session.get("bsoid_conf_metric", "median")),
                min_session_frac=float(session.get("bsoid_min_sess_frac", 0.6)),
                min_keep=int(session.get("bsoid_min_keep", 6)))
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
    _detector_name = _validated_dlc_detector_name(
                         str(_adv.get("dlc_detector",
                             "fasterrcnn_mobilenet_v3_large_fpn")),
                         session, logger)
    _pcutoff       = float(_adv.get("dlc_pcutoff",        0.6))
    _bbox_thr      = float(_adv.get("dlc_bbox_threshold", 0.6))
    _max_ind       = int(_adv.get("dlc_max_individuals",  1))
    _det_epochs    = int(_adv.get("dlc_det_epochs",       15))
    _pose_epochs   = int(_adv.get("dlc_pose_epochs",      15))
    _transfer      = bool(_adv.get("dlc_transfer",        True))
    _crop_enable   = bool(_adv.get("dlc_crop_enable",    False))
    _crop_x        = int(_adv.get("dlc_crop_x",          0))
    _crop_y        = int(_adv.get("dlc_crop_y",          0))
    _crop_w        = int(_adv.get("dlc_crop_w",          0))
    _crop_h        = int(_adv.get("dlc_crop_h",          0))
    _do_crop       = _crop_enable and _crop_w > 0 and _crop_h > 0

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
    #  Phase 1.5 — Convert all videos to target resolution / crop (if enabled)
    # =========================================================================
    long_edge = RESOLUTION_PRESETS.get(settings.get("dlc_resolution"))
    if long_edge or _do_crop:
        if long_edge and _do_crop:
            _phase_msg = f"cropping to {_crop_w}x{_crop_h} and resizing to long-edge {long_edge}px"
        elif long_edge:
            _phase_msg = f"resizing to long-edge {long_edge}px"
        else:
            _phase_msg = f"cropping to {_crop_w}x{_crop_h}"
        logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Smart Adapt: Converting {len(valid_entries)} video(s) "
                    f"({_phase_msg}) …")
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
                    try:
                        _rot2 = int(_cap2.get(cv2.CAP_PROP_ORIENTATION_META))
                    except Exception:
                        _rot2 = 0
                    # Swap visual dimensions for 90°/270° rotated videos
                    if _rot2 in (90, 270):
                        _ow2, _oh2 = _oh2_raw, _ow2_raw
                    else:
                        _ow2, _oh2 = _ow2_raw, _oh2_raw
                    # Post-crop source dimensions drive the resize target
                    _src2_w = _crop_w if _do_crop else _ow2
                    _src2_h = _crop_h if _do_crop else _oh2
                    if long_edge:
                        _sc2 = min(long_edge / max(_src2_w, _src2_h), 1.0)
                        _nw2 = int(_src2_w * _sc2) & ~1
                        _nh2 = int(_src2_h * _sc2) & ~1
                    else:
                        _nw2 = _src2_w & ~1
                        _nh2 = _src2_h & ~1
                    if _nw2 == _ow2 and _nh2 == _oh2 and _rot2 == 0 and not _do_crop:
                        _cap2.release()
                        shutil.copy2(_vpath, _inf_path)
                        logger.info(f"  {_vname}: already at/below target — copied")
                    else:
                        _cap2.release()
                        _log2 = (f"  Processing {_vname}: {_ow2}x{_oh2} → {_nw2}x{_nh2}"
                                 f"{f' [crop {_crop_w}x{_crop_h} @ {_crop_x},{_crop_y}]' if _do_crop else ''}"
                                 f" (rotation={_rot2}°)")
                        logger.info(_log2)
                        _vf2_parts = []
                        if _rot2 in (90, 270, 180):
                            _vf2_parts.append(
                                {90: "transpose=1", 180: "transpose=2,transpose=2",
                                 270: "transpose=2"}[_rot2])
                        if _do_crop:
                            _vf2_parts.append(
                                f"crop={_crop_w}:{_crop_h}:{_crop_x}:{_crop_y}")
                        _vf2_parts.append(f"scale={_nw2}:{_nh2}:flags=area")
                        _ffmpeg_transcode(_vpath, _inf_path, ",".join(_vf2_parts))
                        logger.info(f"  Processed → {Path(_inf_path).name}")
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
                f"All {len(valid_entries)} video(s) converted ({_phase_msg}). "
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
                min_confidence=float(session.get("bsoid_min_conf", 0.30)),
                conf_metric=str(session.get("bsoid_conf_metric", "median")),
                min_session_frac=float(session.get("bsoid_min_sess_frac", 0.6)),
                min_keep=int(session.get("bsoid_min_keep", 6)))
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
    _detector_name = _validated_dlc_detector_name(
                         str(_adv.get("dlc_detector",
                             "fasterrcnn_mobilenet_v3_large_fpn")),
                         session, logger)
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
                min_confidence=float(session.get("bsoid_min_conf", 0.30)),
                conf_metric=str(session.get("bsoid_conf_metric", "median")),
                min_session_frac=float(session.get("bsoid_min_sess_frac", 0.6)),
                min_keep=int(session.get("bsoid_min_keep", 6)))
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
            min_confidence=float(settings.get("bsoid_min_conf", 0.30)),
            conf_metric=str(settings.get("bsoid_conf_metric", "median")),
            min_session_frac=float(settings.get("bsoid_min_sess_frac", 0.6)),
            min_keep=int(settings.get("bsoid_min_keep", 6)))
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


def _find_video_by_stem(stem: str, search_dirs: list):
    """Locate a video file named '<stem>.<ext>' under any of search_dirs
    (recursively).  Returns the first match as a Path, or None."""
    exts = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
    for d in search_dirs or []:
        try:
            base = Path(d)
            if not base.exists():
                continue
            for ext in exts:
                hit = next(base.rglob(f"{stem}{ext}"), None)
                if hit is not None and hit.is_file():
                    return hit
        except Exception:
            continue
    return None


def _export_umap_evolution_videos(out_dir, n_req: int, source_fps: float,
                                  logger, output_fps: float = 15.0,
                                  seed=None, search_dirs: list = None) -> list:
    """Export up to ``n_req`` side-by-side UMAP-evolution videos for a finished
    run directory.  Reusable by both the automatic post-Step-3 export and the
    manual launcher.  Returns the list of produced paths; logs and returns []
    on any problem rather than raising (never breaks the surrounding step).

    Reads the per-session 3-D embedding (model/umap_embedding.npy), the bin
    ranges + embedded video paths (model/session_bin_ranges.json), and the
    per-frame cluster labels (bout_lengths/<stem>_frame_labels[_hmm].csv).
    """
    if not CORE_OK:
        return []
    import numpy as _np, pandas as _pd, random as _rnd
    out_dir   = Path(out_dir)
    model_dir = out_dir / "model"
    emb_p     = model_dir / "umap_embedding.npy"
    lab_p     = model_dir / "umap_labels.npy"
    sbr_p     = model_dir / "session_bin_ranges.json"
    if not (emb_p.is_file() and sbr_p.is_file()):
        logger.warn("  [umap-evo] umap_embedding.npy / session_bin_ranges.json "
                    "missing — cannot export.")
        return []
    try:
        embedding   = _np.load(str(emb_p))
        umap_labels = _np.load(str(lab_p)) if lab_p.is_file() else None
        sbr         = json.loads(sbr_p.read_text())
    except Exception as e:
        logger.warn(f"  [umap-evo] cannot load UMAP data: {e}")
        return []

    ready = []
    missing = []
    for k, v in sbr.items():
        if k == "_total_bins" or not isinstance(v, list) or len(v) < 3:
            continue
        vp = str(v[2]) if v[2] else None
        if vp and Path(vp).is_file():
            ready.append((k, int(v[0]), int(v[1]), vp))
            continue
        # Embedded path missing (e.g. the BSOID_Project_Ready/videos copy was
        # deleted after the run, or files moved) — try to locate by name.
        alt = _find_video_by_stem(k, search_dirs)
        if alt is not None:
            ready.append((k, int(v[0]), int(v[1]), str(alt)))
        else:
            missing.append(k)
    if missing:
        logger.warn(f"  [umap-evo] {len(missing)} session(s) had no locatable "
                    f"source video (searched embedded path + provided folders): "
                    f"{', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}")
    if not ready:
        logger.warn("  [umap-evo] no sessions with an available source video — "
                    "skipped. (If you enabled 'delete BSOID_Project_Ready/videos', "
                    "the source copies were removed; keep them or point CUBE at the "
                    "original videos to enable this export.)")
        return []

    chosen   = _rnd.Random(seed).sample(ready, min(int(n_req), len(ready)))
    evo_dir  = out_dir / "videos" / "umap_evolution"
    evo_dir.mkdir(parents=True, exist_ok=True)
    bout_dir = out_dir / "bout_lengths"
    produced = []
    for i, (stem, sb, eb, vp) in enumerate(chosen, 1):
        try:
            if eb > len(embedding):
                logger.warn(f"  [umap-evo] {stem}: bin range exceeds embedding — "
                            f"skipped.")
                continue
            # Per-frame labels: prefer HMM-smoothed, then raw.  These CSVs have a
            # header row (frame,time_s,label) — read with the header and select
            # the 'label' column (NOT iloc[:,0], which is the frame index).
            frame_labels = None
            for suffix in (f"{stem}_frame_labels_hmm.csv",
                           f"{stem}_frame_labels.csv"):
                cand = bout_dir / suffix
                if cand.is_file():
                    dfl = _pd.read_csv(str(cand))
                    col = "label" if "label" in dfl.columns else dfl.columns[-1]
                    frame_labels = (_pd.to_numeric(dfl[col], errors="coerce")
                                    .dropna().to_numpy(dtype=int))
                    break
            if frame_labels is None or frame_labels.size == 0:
                logger.warn(f"  [umap-evo] {stem}: no usable frame-label CSV — "
                            f"skipped.")
                continue
            out_p = evo_dir / f"{stem}_umap_evolution.mp4"
            logger.info(f"  [umap-evo] {i}/{len(chosen)}  '{stem}' -> {out_p.name}")
            res = create_umap_evolution_video(
                video_path=Path(vp),
                embedding=embedding[sb:eb],
                umap_labels=(umap_labels[sb:eb] if umap_labels is not None
                             else _np.zeros(eb - sb, dtype=int)),
                frame_labels=frame_labels,
                source_fps=source_fps,
                out_path=out_p,
                output_fps=output_fps,
            )
            if res is not None:
                produced.append(res)
                logger.success(f"  [umap-evo] saved -> {res}")
            else:
                logger.warn(f"  [umap-evo] {stem}: export returned no output "
                            f"(is opencv installed and the video readable?).")
        except Exception:
            logger.warn(f"  [umap-evo] {stem}: {traceback.format_exc()}")
    return produced


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
        else:
            # Per-root fallback: BSOID_Project_Ready lives directly inside the
            # source folder, so rp.parent IS the source folder.  The old global
            # "if not all_vid_dirs" guard caused all roots to be skipped whenever
            # at least one sibling root had a populated videos/ directory.
            _parent = rp.parent
            if _has_videos(_parent):
                all_vid_dirs.append(_parent)
                logger.warn(
                    f"    [{rp.name}] videos/ empty — "
                    f"using parent folder: {_parent.name}")

    # ── Build stem→group mapping so analyser can assign exp_group per file ──────
    # video_groups maps source-folder-path → group name; the bout CSVs live in a
    # completely different output tree, so we map by DLC-file stem instead.
    _video_groups_session = session.get("video_groups", {})
    if _video_groups_session:
        _stem_to_group: dict = {}
        for _bsoid_root, _csv_d in zip(bsoid_roots, all_csv_dirs):
            _bsoid_res = Path(_bsoid_root).resolve()
            for _fg_str, _fg_grp in _video_groups_session.items():
                try:
                    _bsoid_res.relative_to(Path(_fg_str).resolve())
                    # This bsoid_root lives inside this source folder
                    for _ext in ("*.csv", "*.h5"):
                        for _cf in sorted(Path(_csv_d).glob(_ext)):
                            _stem_to_group[_cf.stem] = _fg_grp
                    break
                except ValueError:
                    continue
        if _stem_to_group:
            session["stem_to_group"] = _stem_to_group
            logger.info(f"  Group mapping: {len(_stem_to_group)} DLC stem(s) → group "
                        f"({len(set(_stem_to_group.values()))} group(s))")
        else:
            logger.warn("  Group mapping: no DLC stems could be matched to source folders — "
                        "exp_group will not be auto-populated in Analyser.")

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

    # Persist group assignments so future sessions can inject them without
    # needing the original session file to be loaded.
    if session.get("stem_to_group"):
        try:
            _ga_path = this_out / "model" / "group_assignments.json"
            _ga_path.parent.mkdir(parents=True, exist_ok=True)
            _ga_path.write_text(
                json.dumps(session.get("stem_to_group", {}), indent=2),
                encoding="utf-8")
            logger.info(f"  Group assignments saved to {_ga_path.name}")
        except Exception:
            pass

    # ── UMAP evolution videos: fallback export ────────────────────────────────
    # The engine (cube_core.run) already auto-exports umap_evolution videos in
    # the normal case.  Only retry here if it produced nothing AND the user has
    # not disabled it — this adds the by-name video search (recovers sessions
    # whose embedded path is missing) without double-rendering when the engine
    # already succeeded.  MUST run BEFORE the video-folder cleanup below so the
    # source videos still exist.
    _evo_n = cfg.get("umap_evolution_n",
                     session.get("engine_cfg", {}).get("umap_evolution_n", 1))
    try:
        _evo_n = int(_evo_n or 0)
    except (TypeError, ValueError):
        _evo_n = 1
    _evo_dir = Path(this_out) / "videos" / "umap_evolution"
    _engine_made = _evo_dir.exists() and any(_evo_dir.glob("*.mp4"))
    if _evo_n > 0 and not _engine_made:
        try:
            logger.step(f"UMAP evolution: engine produced none — retrying export "
                        f"(up to {_evo_n}) with by-name video search...")
            _evo_search = (list(all_vid_dirs or []) +
                           list(session.get("video_folders", [])))
            _vids = _export_umap_evolution_videos(
                this_out, _evo_n, fps_val, logger, search_dirs=_evo_search)
            if _vids:
                logger.success(
                    f"UMAP evolution: {len(_vids)} video(s) saved to {_evo_dir}.")
            else:
                logger.warn("UMAP evolution: no videos produced (see [umap-evo] "
                            "messages above).")
        except Exception:
            logger.warn(f"  [umap-evo] fallback export failed:\n"
                        f"{traceback.format_exc()}")

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
                  bg=C["btn"], fg=C["cyan"], relief="flat",
                  padx=14, pady=5, cursor="hand2",
                  command=_test_notification).pack(side="left", padx=6)
        tk.Button(btn_row, text="Close", font=("Segoe UI", 9, "bold"),
                  bg=C["btn"], fg=C["text"], relief="flat",
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
        "fasterrcnn_resnet50_fpn_v2",
        "ssdlite",
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
        dlc_crop_enable      = False,
        dlc_crop_x           = 0,
        dlc_crop_y           = 0,
        dlc_crop_w           = 0,
        dlc_crop_h           = 0,
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
                  bg=C["btn"], fg=C["btn_fg"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Restore Defaults", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["yellow"], relief="flat",
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

        # ── Video crop ────────────────────────────────────────────────────────
        sec6 = _adv_section(p, "VIDEO CROP  (spatial region applied before inference)", C["yellow"])
        crop_en_row = tk.Frame(sec6, bg=C["card"])
        crop_en_row.pack(anchor="w", padx=8, pady=4)
        self._v("dlc_crop_enable", tk.BooleanVar(value=False))
        tk.Checkbutton(
            crop_en_row,
            text="Enable spatial crop  (trim each video to a defined pixel region)",
            variable=self._vars["dlc_crop_enable"],
            bg=C["card"], fg=C["text"],
            selectcolor=C["card2"], activebackground=C["card"],
            font=("Segoe UI", 9),
        ).pack(side="left")
        for key, default in [("dlc_crop_x", 0), ("dlc_crop_y", 0),
                              ("dlc_crop_w", 0), ("dlc_crop_h", 0)]:
            self._v(key, tk.IntVar(value=default))
        tk.Label(sec6,
                 text="  Current region (pixels, 0 = not set — use Preview button to set):",
                 font=("Segoe UI", 8), bg=C["card"], fg=C["dim"]).pack(anchor="w", padx=8)
        crop_coord_row = tk.Frame(sec6, bg=C["card"])
        crop_coord_row.pack(anchor="w", padx=8, pady=(0, 4))
        for lbl, key in [("X:", "dlc_crop_x"), ("Y:", "dlc_crop_y"),
                          ("W:", "dlc_crop_w"), ("H:", "dlc_crop_h")]:
            tk.Label(crop_coord_row, text=lbl, font=("Segoe UI", 9),
                     bg=C["card"], fg=C["text"]).pack(side="left", padx=(6, 0))
            tk.Spinbox(crop_coord_row, from_=0, to=9999, increment=1,
                       textvariable=self._vars[key], width=6,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left", padx=(2, 0))
        tk.Button(sec6,
                  text="  Preview / Set Crop Region…",
                  font=("Segoe UI", 9, "bold"),
                  bg=C["yellow"], fg=C["bg"],
                  activebackground="#e6c200", relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self._open_crop_preview).pack(anchor="w", padx=8, pady=(0, 6))

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

    def _open_crop_preview(self):
        cfg = {}
        for k, var in self._vars.items():
            try:
                cfg[k] = var.get()
            except Exception:
                pass
        self._session["dlc_advanced_cfg"] = cfg
        dlg = CropPreviewDialog(self, self._session)
        self.wait_window(dlg)
        if dlg.confirmed:
            adv = self._session.get("dlc_advanced_cfg", {})
            for k in ("dlc_crop_x", "dlc_crop_y", "dlc_crop_w", "dlc_crop_h"):
                if k in self._vars:
                    try:
                        self._vars[k].set(int(adv.get(k, 0)))
                    except Exception:
                        pass

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

    # GUI-managed parameters.  This baseline is OVERLAID by BSoidEngine.DEFAULTS
    # (the canonical source) so the GUI can never drift from the engine — see
    # DEFAULTS below.  The baseline values are used only as a fallback when the
    # core engine failed to import (in which case no run can happen anyway).
    # Keys here that are NOT in the engine (e.g. umap_evolution_n) are GUI-only.
    _BASELINE = dict(
        body_normalise        = False,
        pca_pre_reduce        = "auto",
        likelihood_thresh     = 0.30,
        max_interp_gap_sec    = 0.50,
        boxcar_win_sec        = 0.07,
        train_frac            = 0.30,
        umap_full_thresh      = 10_000,
        umap_n_neighbors      = 0,     # 0 = auto (scales with recording length)
        umap_n_components     = 3,
        umap_min_dist         = 0.10,
        umap_random_state     = 42,
        hdbscan_metric        = "euclidean",
        hdbscan_method        = "both",
        hdbscan_methods_to_try = "eom,leaf",
        target_n_clusters     = 0,
        preferred_clusters_lo = 5,
        preferred_clusters_hi = 30,
        min_cluster_freq      = 0.2,   # percentage of total bins; 0 = disabled
        mlp_hidden            = "100,50",
        mlp_max_iter          = 1000,
        mlp_confidence_thresh = 0.0,
        cv_folds              = 5,
        output_fps            = 15,
        max_clips_per_cluster = 3,
        save_plots            = True,
        save_videos           = True,
        umap_evolution_n      = 1,     # GUI-only: videos auto-exported after Step 3
        hmm_enabled           = True,
        hmm_n_states          = 0,     # 0 = auto (= n_clusters)
        hmm_n_iter            = 100,
        hmm_min_prob          = 0.05,
        compat_mode           = "current",  # "current" or "legacy_v2"
        seed_sweep_n          = 0,     # >0 = run cluster-stability seed sweep
    )
    try:
        # Engine defaults win for every shared key; GUI-only keys persist.
        DEFAULTS = {**_BASELINE, **dict(BSoidEngine.DEFAULTS)}
    except Exception:
        DEFAULTS = dict(_BASELINE)

    def __init__(self, parent, session: "SessionState"):
        super().__init__(parent)
        self.title("⚙  Advanced CUBE Analysis Parameters")
        self.configure(bg=C["bg"])
        self.geometry("520x760")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self._session = session
        self._vars: dict = {}

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(side="bottom", fill="x", pady=8, padx=12)
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Restore Defaults", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["yellow"], relief="flat",
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

        # Widget defaults are sourced from self.DEFAULTS (which is the canonical
        # BSoidEngine.DEFAULTS overlaid on the GUI baseline) so the seeded value
        # can never drift from the engine.  The literal passed at the call site
        # is only a fallback for keys absent from DEFAULTS.
        def _spin_f(row, key, lo, hi, step, default):
            _dv = self.DEFAULTS.get(key, default)
            if _dv is None:          # engine default may be None (e.g. auto) — use literal
                _dv = default
            v = self._v(key, tk.DoubleVar(value=float(_dv)))
            tk.Spinbox(row, from_=lo, to=hi, increment=step,
                       format="%.3f", textvariable=v, width=8,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left")

        def _spin_i(row, key, lo, hi, step, default):
            _dv = self.DEFAULTS.get(key, default)
            if _dv is None:          # engine default may be None (e.g. hmm_n_states) — use literal
                _dv = default
            v = self._v(key, tk.IntVar(value=int(_dv)))
            tk.Spinbox(row, from_=lo, to=hi, increment=step,
                       textvariable=v, width=8,
                       bg=C["card2"], fg=C["text"],
                       buttonbackground=C["card2"],
                       font=("Segoe UI", 9)).pack(side="left")

        def _check(row, key, default):
            v = self._v(key, tk.BooleanVar(value=bool(self.DEFAULTS.get(key, default))))
            tk.Checkbutton(row, variable=v, bg=C["card"], fg=C["green"],
                           selectcolor=C["card2"],
                           activebackground=C["card"]).pack(side="left")

        def _combo(row, key, values, default):
            v = self._v(key, tk.StringVar(value=str(self.DEFAULTS.get(key, default))))
            ttk.Combobox(row, textvariable=v, values=values,
                         state="readonly", width=18,
                         font=("Segoe UI", 9)).pack(side="left")

        # ── Feature extraction ────────────────────────────────────────────────
        s = _adv_section(p, "FEATURE EXTRACTION", C["cyan"])
        _adv_row(s, "Body normalisation",
                 lambda r: _check(r, "body_normalise", False))
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
        s2 = _adv_section(p, "UMAP EMBEDDING  (Hsu & Yttri 2021 reference)", C["cyan"])
        _adv_row(s2, "Full-data threshold",
                 lambda r: _spin_i(r, "umap_full_thresh", 1000, 100_000, 1000, 10_000))
        tk.Label(s2,
                 text="    Use all bins for UMAP when total bins ≤ this value.\n"
                      "    Subsamples at 'Train fraction' only for larger recordings.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s2, "Train fraction",
                 lambda r: _spin_f(r, "train_frac", 0.05, 1.0, 0.05, 0.30))
        _adv_row(s2, "n_neighbors  (0 = auto)",
                 lambda r: _spin_i(r, "umap_n_neighbors", 0, 300, 5, 0))
        tk.Label(s2,
                 text="    0 = auto: scales with recording length, clip(n_bins/25, 15, 60).\n"
                      "    Set a positive value to fix it (B-SOiD reference = 60).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s2, "n_components",
                 lambda r: _spin_i(r, "umap_n_components", 2, 10, 1, 3))
        _adv_row(s2, "min_dist",
                 lambda r: _spin_f(r, "umap_min_dist", 0.0, 1.0, 0.05, 0.10))
        tk.Label(s2,
                 text="    Recommended: 0.1.  Values < 0.05 pack UMAP points so tightly\n"
                      "    that HDBSCAN's density graph degenerates (DBCV becomes non-finite)\n"
                      "    and noise fraction rises sharply.  Use 0.0 only with legacy_v2.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 4))
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
                                  ["both", "eom", "leaf"], "both"))
        tk.Label(s3,
                 text="    both = tries eom and leaf at every step; DBCV score picks best.\n"
                      "    eom  = larger, more stable clusters (overlapping densities).\n"
                      "    leaf = finer, denser clusters (stereotyped/brief events).\n"
                      "    min_cluster_size is swept adaptively (anchored to clustered points).",
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
        _adv_row(s3b, "Min cluster frequency (%)",
                 lambda r: _spin_f(r, "min_cluster_freq", 0.0, 10.0, 0.1, 0.5))
        tk.Label(s3b,
                 text="    Clusters whose share of total analysis time is below this\n"
                      "    percentage are removed before MLP training (reassigned to noise).\n"
                      "    0.2 % = default.  Set to 0 to disable pruning.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── MLP classifier ────────────────────────────────────────────────────
        def _adv_entry(row, key, default):
            v = self._v(key, tk.StringVar(value=str(self.DEFAULTS.get(key, default))))
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

        # ── HMM smoothing ─────────────────────────────────────────────────────
        s_hmm = _adv_section(p, "HMM SMOOTHING  (post-hoc temporal filter)", C["green"])
        _adv_row(s_hmm, "Enable HMM smoothing",
                 lambda r: _check(r, "hmm_enabled", True))
        _adv_row(s_hmm, "HMM states  (0 = auto)",
                 lambda r: _spin_i(r, "hmm_n_states", 0, 200, 1, 0))
        _adv_row(s_hmm, "Baum-Welch iterations",
                 lambda r: _spin_i(r, "hmm_n_iter", 10, 500, 10, 100))
        _adv_row(s_hmm, "Min edge prob (syntax graph)",
                 lambda r: _spin_f(r, "hmm_min_prob", 0.01, 0.50, 0.01, 0.05))
        tk.Label(s_hmm,
                 text="    Wraps B-SOiD MLP output with a Multinomial HMM (Baum-Welch + Viterbi).\n"
                      "    Eliminates single-frame state flickers due to tracking jitter.\n"
                      "    States = 0 uses n_clusters from HDBSCAN (smoothing-only mode).\n"
                      "    States < n_clusters groups motifs into behavioral macro-states.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── Reproducibility & methodology ─────────────────────────────────────
        s_rep = _adv_section(p, "REPRODUCIBILITY & METHODOLOGY", C["cyan"])
        _adv_row(s_rep, "Compatibility mode",
                 lambda r: _combo(r, "compat_mode",
                                  ["current", "legacy_v2"], "current"))
        tk.Label(s_rep,
                 text="    current = v2.1 corrected behaviour (recommended).\n"
                      "    legacy_v2 = reproduce pre-2.1 runs exactly (min_dist=0,\n"
                      "    full-dataset mcs anchor, evenly-spaced angular fallback).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_rep, "Cluster-stability seed sweep  (0 = off)",
                 lambda r: _spin_i(r, "seed_sweep_n", 0, 50, 1, 0))
        tk.Label(s_rep,
                 text="    >0 re-runs UMAP+HDBSCAN over this many seeds to measure\n"
                      "    cluster-count / partition stability (plots cluster_stability.png).\n"
                      "    Adds runtime proportional to the number of seeds.",
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
        _adv_row(s5, "UMAP evolution videos  (0 = off)",
                 lambda r: _spin_i(r, "umap_evolution_n", 0, 50, 1, 1))
        tk.Label(s5,
                 text="    Auto-export this many side-by-side evolution videos\n"
                      "    at the end of Step 3.  Sessions are chosen randomly.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s5, "Plot theme",
                 lambda r: _combo(r, "plot_theme", ["dark", "light"], "dark"))
        tk.Label(s5,
                 text="    dark = white-on-black figures  |  light = publication-ready white background",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

    def _load(self):
        # Recompute at open-time so BSoidEngine.DEFAULTS is available (deferred import
        # has run by the time the user clicks to open this window).
        try:
            effective = {**self._BASELINE, **dict(BSoidEngine.DEFAULTS)}
        except Exception:
            effective = dict(self._BASELINE)
        cfg = self._session.get("engine_cfg", {})
        for k, default in effective.items():
            val = cfg.get(k, default)
            if k in self._vars:
                try:
                    self._vars[k].set(val)
                except Exception:
                    pass

    def _restore(self):
        try:
            effective = {**self._BASELINE, **dict(BSoidEngine.DEFAULTS)}
        except Exception:
            effective = dict(self._BASELINE)
        for k, default in effective.items():
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
#  CROP PREVIEW DIALOG
#

class CropPreviewDialog(tk.Toplevel):
    """Interactive crop-region picker shown before DLC Step 1.

    Loads random frames from ≥50 % of all queued videos and shows them on an
    interactive canvas where the user can drag to draw / move / resize the crop
    rectangle.  A thumbnail strip below shows the same overlay on all sampled
    frames so the user can check that nothing critical is cut off.
    """

    _HANDLE_R = 6    # half-side of corner / midpoint handle squares (display px)
    _THUMB_H  = 110  # thumbnail height in the preview strip

    def __init__(self, parent, session: "SessionState"):
        super().__init__(parent)
        self.title("Set Video Crop Region")
        self.configure(bg=C["bg"])
        self.geometry("960x760")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self._session  = session
        self.confirmed = False

        adv = session.get("dlc_advanced_cfg", {})
        self._rx = int(adv.get("dlc_crop_x", 0))
        self._ry = int(adv.get("dlc_crop_y", 0))
        self._rw = int(adv.get("dlc_crop_w", 0))
        self._rh = int(adv.get("dlc_crop_h", 0))

        self._vx = tk.IntVar(value=self._rx)
        self._vy = tk.IntVar(value=self._ry)
        self._vw = tk.IntVar(value=self._rw)
        self._vh = tk.IntVar(value=self._rh)

        self._frames: list       = []   # [(PIL.Image, video_path), ...]
        self._main_photo         = None
        self._thumb_photos: list = []   # [[label_widget, orig_img, photo_ref], ...]
        self._scale              = 1.0
        self._main_w             = 1
        self._main_h             = 1
        self._canvas_offset      = (0, 0)

        self._drag_mode     = None  # None | "draw" | "move" | "handle_XX"
        self._drag_ox       = 0
        self._drag_oy       = 0
        self._rect_snapshot = None

        self._build_ui()
        self.after(50, self._load_frames)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        tk.Label(self, text="  Set Video Crop Region",
                 font=("Segoe UI", 13, "bold"),
                 bg=C["bg"], fg=C["yellow"]).pack(anchor="w", padx=12, pady=(10, 0))
        tk.Label(self,
                 text="  Drag on the preview to draw a crop box.  "
                      "Drag corner / edge handles to resize.  "
                      "Drag inside the box to move it.",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["dim"],
                 justify="left").pack(anchor="w", padx=12, pady=(0, 4))

        self._canvas = tk.Canvas(self, bg="#111111",
                                 cursor="crosshair", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=12)
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>",          self._on_hover)
        self._canvas.bind("<Configure>",       lambda _e: self._refresh_main())

        coord_f = tk.Frame(self, bg=C["bg"])
        coord_f.pack(fill="x", padx=12, pady=4)
        tk.Label(coord_f, text="Crop region (original pixels):",
                 font=("Segoe UI", 9), bg=C["bg"], fg=C["text"]).pack(side="left")
        for lbl, var in [("  X:", self._vx), ("  Y:", self._vy),
                          ("  W:", self._vw), ("  H:", self._vh)]:
            tk.Label(coord_f, text=lbl,
                     font=("Segoe UI", 9), bg=C["bg"], fg=C["dim"]).pack(side="left")
            ent = tk.Entry(coord_f, textvariable=var, width=6,
                           bg=C["card2"], fg=C["text"],
                           font=("Segoe UI", 9),
                           insertbackground=C["text"], relief="flat")
            ent.pack(side="left", padx=(1, 0))
            ent.bind("<Return>",   self._on_entry_commit)
            ent.bind("<FocusOut>", self._on_entry_commit)

        tk.Label(self,
                 text="  Sample frames (≥50 % of queued videos) — verify nothing critical is cut off:",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["dim"]).pack(anchor="w", padx=12)

        strip_outer = tk.Frame(self, bg=C["bg"],
                               height=self._THUMB_H + 24)
        strip_outer.pack(fill="x", padx=12, pady=(2, 4))
        strip_outer.pack_propagate(False)
        self._strip_cv = tk.Canvas(strip_outer, bg=C["card"],
                                   height=self._THUMB_H + 10,
                                   highlightthickness=0)
        h_sb = tk.Scrollbar(strip_outer, orient="horizontal",
                             command=self._strip_cv.xview,
                             bg=C["card"], troughcolor=C["bg"])
        self._strip_cv.configure(xscrollcommand=h_sb.set)
        h_sb.pack(side="bottom", fill="x")
        self._strip_cv.pack(fill="both", expand=True)
        self._strip_inner = tk.Frame(self._strip_cv, bg=C["card"])
        self._strip_cv.create_window((0, 0), window=self._strip_inner, anchor="nw")
        self._strip_inner.bind(
            "<Configure>",
            lambda _e: self._strip_cv.configure(
                scrollregion=self._strip_cv.bbox("all")))

        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(fill="x", padx=12, pady=(4, 10))
        tk.Button(btn_f, text="Cancel",
                  font=("Segoe UI", 9), bg=C["btn"], fg=C["btn_fg"],
                  relief="flat", padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Reset to Full Frame",
                  font=("Segoe UI", 9), bg=C["btn"], fg=C["yellow"],
                  relief="flat", padx=10, pady=5, cursor="hand2",
                  command=self._reset).pack(side="left", padx=6)
        tk.Button(btn_f,
                  text="  Accept — proceed with this crop  ",
                  font=("Segoe UI", 10, "bold"),
                  bg=C["green"], fg="white",
                  relief="flat", padx=16, pady=5, cursor="hand2",
                  command=self._accept).pack(side="right")

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _load_frames(self):
        import random
        import math
        try:
            import cv2
            from PIL import Image
        except ImportError:
            self._canvas.create_text(
                10, 10, anchor="nw",
                text="OpenCV / Pillow not available — cannot show preview.",
                fill="white", font=("Segoe UI", 10))
            return

        VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
        folders    = self._session.get("video_folders", [])
        all_videos = []
        for root_folder in folders:
            for sub, dirs, files in os.walk(root_folder):
                dirs[:] = [d for d in dirs if not d.endswith("_results")]
                if Path(sub).name.endswith("_results"):
                    continue
                for fname in sorted(files):
                    if fname.startswith("resized_"):
                        continue
                    if Path(fname).suffix.lower() in VIDEO_EXTS:
                        all_videos.append(os.path.join(sub, fname))

        if not all_videos:
            self._canvas.create_text(
                self._canvas.winfo_width() // 2 or 300,
                self._canvas.winfo_height() // 2 or 200,
                text="No videos found in the selected folders.",
                fill="white", font=("Segoe UI", 11))
            return

        n_sample = max(1, math.ceil(len(all_videos) / 2))
        sampled  = random.sample(all_videos, min(n_sample, len(all_videos)))

        pil_frames = []
        for vpath in sampled:
            cap   = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                cap.release()
                continue
            lo  = int(total * 0.2)
            hi  = max(lo + 1, int(total * 0.8))
            idx = random.randint(lo, hi - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_frames.append((img, vpath))

        if not pil_frames:
            self._canvas.create_text(
                self._canvas.winfo_width() // 2 or 300,
                self._canvas.winfo_height() // 2 or 200,
                text="Could not read frames from any video.",
                fill="white", font=("Segoe UI", 11))
            return

        self._frames  = pil_frames
        self._main_w  = pil_frames[0][0].width
        self._main_h  = pil_frames[0][0].height

        if self._rw == 0 and self._rh == 0:
            self._rx, self._ry = 0, 0
            self._rw, self._rh = self._main_w, self._main_h
            self._sync_vars()

        self._refresh_main()
        self._rebuild_thumbs()

    # ── Canvas drawing ────────────────────────────────────────────────────────

    def _refresh_main(self, *_):
        if not self._frames:
            return
        try:
            from PIL import ImageTk, ImageDraw, Image
        except ImportError:
            return

        cw = max(1, self._canvas.winfo_width())
        ch = max(1, self._canvas.winfo_height())
        img_orig, _ = self._frames[0]
        ow, oh      = img_orig.width, img_orig.height

        scale       = min(cw / ow, ch / oh, 1.0)
        dw          = max(1, int(ow * scale))
        dh          = max(1, int(oh * scale))
        self._scale = scale

        off_x = (cw - dw) // 2
        off_y = (ch - dh) // 2
        self._canvas_offset = (off_x, off_y)

        img_disp = img_orig.resize((dw, dh), Image.LANCZOS).convert("RGBA")

        if self._rw > 0 and self._rh > 0:
            x0 = max(0, min(dw - 1, int(self._rx * scale)))
            y0 = max(0, min(dh - 1, int(self._ry * scale)))
            x1 = max(0, min(dw - 1, int((self._rx + self._rw) * scale)))
            y1 = max(0, min(dh - 1, int((self._ry + self._rh) * scale)))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 110))
            ovd = ImageDraw.Draw(overlay)
            ovd.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 0))
            img_disp = Image.alpha_composite(img_disp, overlay)

            draw = ImageDraw.Draw(img_disp)
            draw.rectangle([x0,     y0,     x1,     y1    ], outline="#FFD700", width=2)
            draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline="#000000", width=1)

            R = self._HANDLE_R
            for hx, hy in self._get_handle_positions_display().values():
                draw.rectangle([hx - R, hy - R, hx + R, hy + R],
                                fill="#FFD700", outline="#000000")

        self._main_photo = ImageTk.PhotoImage(img_disp.convert("RGB"))
        self._canvas.delete("all")
        self._canvas.create_image(off_x, off_y, anchor="nw",
                                  image=self._main_photo)

    def _rebuild_thumbs(self):
        for w in self._strip_inner.winfo_children():
            w.destroy()
        self._thumb_photos = []
        for img_orig, vpath in self._frames:
            cell = tk.Frame(self._strip_inner, bg=C["card2"])
            cell.pack(side="left", padx=2, pady=4)
            lbl  = tk.Label(cell, bg=C["card2"])
            lbl.pack()
            tk.Label(cell, text=Path(vpath).name[:22],
                     font=("Segoe UI", 7), bg=C["card2"],
                     fg=C["dim"]).pack()
            self._thumb_photos.append([lbl, img_orig, None])
        self._update_thumbs()

    def _update_thumbs(self):
        try:
            from PIL import ImageTk, ImageDraw, Image
        except ImportError:
            return
        for entry in self._thumb_photos:
            lbl, img_orig, _ = entry
            ow, oh = img_orig.width, img_orig.height
            s  = self._THUMB_H / oh if oh > 0 else 1.0
            tw = max(1, int(ow * s))
            th = max(1, int(oh * s))
            img_d = img_orig.resize((tw, th), Image.LANCZOS).convert("RGBA")

            if self._rw > 0 and self._rh > 0:
                x0 = max(0, min(tw - 1, int(self._rx * s)))
                y0 = max(0, min(th - 1, int(self._ry * s)))
                x1 = max(0, min(tw - 1, int((self._rx + self._rw) * s)))
                y1 = max(0, min(th - 1, int((self._ry + self._rh) * s)))
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                ov = Image.new("RGBA", (tw, th), (0, 0, 0, 110))
                ImageDraw.Draw(ov).rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 0))
                img_d = Image.alpha_composite(img_d, ov)
                ImageDraw.Draw(img_d).rectangle([x0, y0, x1, y1],
                                                outline="#FFD700", width=1)

            photo = ImageTk.PhotoImage(img_d.convert("RGB"))
            entry[2] = photo
            lbl.configure(image=photo)

    # ── Handle positions ──────────────────────────────────────────────────────

    def _get_handle_positions_display(self) -> dict:
        s  = self._scale or 1.0
        ox, oy = self._canvas_offset
        x0 = int(self._rx * s) + ox
        y0 = int(self._ry * s) + oy
        x1 = int((self._rx + self._rw) * s) + ox
        y1 = int((self._ry + self._rh) * s) + oy
        mx = (x0 + x1) // 2
        my = (y0 + y1) // 2
        return {
            "nw": (x0, y0), "n":  (mx, y0), "ne": (x1, y0),
            "w":  (x0, my),                   "e":  (x1, my),
            "sw": (x0, y1), "s":  (mx, y1), "se": (x1, y1),
        }

    def _hit_handle(self, cx: int, cy: int):
        if self._rw == 0 or self._rh == 0:
            return None
        R = self._HANDLE_R + 3
        for key, (hx, hy) in self._get_handle_positions_display().items():
            if abs(cx - hx) <= R and abs(cy - hy) <= R:
                return key
        return None

    def _inside_rect(self, cx: int, cy: int) -> bool:
        s  = self._scale or 1.0
        ox, oy = self._canvas_offset
        x0 = int(self._rx * s) + ox
        y0 = int(self._ry * s) + oy
        x1 = int((self._rx + self._rw) * s) + ox
        y1 = int((self._ry + self._rh) * s) + oy
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _canvas_to_orig(self, cx: int, cy: int):
        ox, oy = self._canvas_offset
        s = self._scale or 1.0
        return (cx - ox) / s, (cy - oy) / s

    # ── Mouse events ─────────────────────────────────────────────────────────

    def _on_press(self, event):
        cx, cy = event.x, event.y
        h = self._hit_handle(cx, cy)
        if h:
            self._drag_mode = f"handle_{h}"
        elif self._rw > 0 and self._rh > 0 and self._inside_rect(cx, cy):
            self._drag_mode = "move"
        else:
            self._drag_mode = "draw"
        self._drag_ox       = cx
        self._drag_oy       = cy
        self._rect_snapshot = (self._rx, self._ry, self._rw, self._rh)

    def _on_drag(self, event):
        if self._drag_mode is None:
            return
        cx, cy = event.x, event.y
        s = self._scale or 1.0
        dx = (cx - self._drag_ox) / s
        dy = (cy - self._drag_oy) / s
        W, H = self._main_w, self._main_h
        rx0, ry0, rw0, rh0 = self._rect_snapshot

        def clamp(v, lo, hi):
            return max(lo, min(hi, int(round(v))))

        if self._drag_mode == "draw":
            ox_o, oy_o = self._canvas_to_orig(self._drag_ox, self._drag_oy)
            nx_o, ny_o = self._canvas_to_orig(cx, cy)
            lx, rx_ = sorted([ox_o, nx_o])
            ty, by  = sorted([oy_o, ny_o])
            self._rx = clamp(lx,  0, W - 1)
            self._ry = clamp(ty,  0, H - 1)
            self._rw = clamp(rx_ - lx, 1, W - self._rx)
            self._rh = clamp(by  - ty, 1, H - self._ry)

        elif self._drag_mode == "move":
            self._rx = clamp(rx0 + dx, 0, W - rw0)
            self._ry = clamp(ry0 + dy, 0, H - rh0)
            self._rw, self._rh = rw0, rh0

        elif self._drag_mode.startswith("handle_"):
            h  = self._drag_mode[len("handle_"):]
            lx, ty  = float(rx0),        float(ry0)
            rx_, by = float(rx0 + rw0),  float(ry0 + rh0)
            if "w" in h:
                lx  = clamp(lx + dx,  0,     rx_ - 1)
            if "e" in h:
                rx_ = clamp(rx_ + dx, lx + 1, W)
            if "n" in h:
                ty  = clamp(ty + dy,  0,     by - 1)
            if "s" in h:
                by  = clamp(by + dy,  ty + 1, H)
            self._rx = int(lx);  self._ry = int(ty)
            self._rw = int(rx_ - lx);  self._rh = int(by - ty)

        self._sync_vars()
        self._refresh_main()
        self._update_thumbs()

    def _on_release(self, _event):
        self._drag_mode     = None
        self._rect_snapshot = None

    _CURSOR_MAP = {
        "nw": "top_left_corner",    "n":  "top_side",
        "ne": "top_right_corner",   "w":  "left_side",
        "e":  "right_side",         "sw": "bottom_left_corner",
        "s":  "bottom_side",        "se": "bottom_right_corner",
    }

    def _on_hover(self, event):
        if self._drag_mode is not None:
            return
        h = self._hit_handle(event.x, event.y)
        if h:
            self._canvas.configure(cursor=self._CURSOR_MAP.get(h, "crosshair"))
        elif self._rw > 0 and self._rh > 0 and self._inside_rect(event.x, event.y):
            self._canvas.configure(cursor="fleur")
        else:
            self._canvas.configure(cursor="crosshair")

    # ── Entry sync ────────────────────────────────────────────────────────────

    def _sync_vars(self):
        self._vx.set(self._rx)
        self._vy.set(self._ry)
        self._vw.set(self._rw)
        self._vh.set(self._rh)

    def _on_entry_commit(self, *_):
        try:
            rx = max(0, min(self._main_w - 1, int(self._vx.get())))
            ry = max(0, min(self._main_h - 1, int(self._vy.get())))
            rw = max(1, min(self._main_w - rx, int(self._vw.get())))
            rh = max(1, min(self._main_h - ry, int(self._vh.get())))
            self._rx, self._ry, self._rw, self._rh = rx, ry, rw, rh
            self._sync_vars()
            self._refresh_main()
            self._update_thumbs()
        except (tk.TclError, ValueError):
            pass

    # ── Buttons ───────────────────────────────────────────────────────────────

    def _reset(self):
        self._rx, self._ry = 0, 0
        self._rw, self._rh = self._main_w or 0, self._main_h or 0
        self._sync_vars()
        self._refresh_main()
        self._update_thumbs()

    def _accept(self):
        adv = dict(self._session.get("dlc_advanced_cfg", {}))
        adv["dlc_crop_x"] = self._rx
        adv["dlc_crop_y"] = self._ry
        # store 0 when the rect covers the full frame (= no crop needed)
        adv["dlc_crop_w"] = self._rw if not (
            self._rx == 0 and self._ry == 0 and
            self._rw == self._main_w and self._rh == self._main_h) else 0
        adv["dlc_crop_h"] = self._rh if not (
            self._rx == 0 and self._ry == 0 and
            self._rw == self._main_w and self._rh == self._main_h) else 0
        self._session["dlc_advanced_cfg"] = adv
        self.confirmed = True
        self.destroy()


#
#  MAIN APPLICATION
#

class CubeSplash(tk.Toplevel):
    """
    Borderless startup splash shown briefly before the main window.

    Extensibility note — publication citation
    -----------------------------------------
    Set `_citation_text` to a non-empty string when a publication is ready
    (e.g. "Valiathan et al. 2026, Nature Methods").  The citation label in the
    centre-bottom of the splash will appear automatically; no other code changes
    are needed.  Keep the string short (<80 chars) so it fits on one line.
    """

    DISPLAY_MS  = 2400   # ms the splash is fully visible before fade begins
    FADE_STEPS  = 20     # alpha increments during fade-out
    FADE_MS     = 15     # ms between fade steps  (20 × 15 = 300 ms total fade)

    # Future: assign the citation string here once the paper is published.
    _citation_text: str = ""

    def __init__(self, parent: tk.Tk, is_dark: bool, on_done):
        super().__init__(parent)
        self._on_done = on_done

        bg  = "#09090f" if is_dark else "#f0f2f5"
        sub = "#4a4a6a" if is_dark else "#aaaaaa"
        bdr = "#2a2a4a" if is_dark else "#dee2e6"

        self.overrideredirect(True)
        self.configure(bg=bg)
        self.attributes("-alpha", 1.0)
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass

        W, H = 560, 300

        logo_file = "CUBE_logo dark theme.png" if is_dark else "CUBE_logo.png"
        self._logo_img = None
        try:
            from PIL import Image, ImageTk
            p = HERE / logo_file
            if p.is_file():
                img = Image.open(p).convert("RGBA")
                img.thumbnail((440, 200), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(img)
        except Exception:
            pass

        cv = tk.Canvas(self, width=W, height=H, bg=bg,
                       highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)

        if self._logo_img:
            cv.create_image(W // 2, H // 2 - 22, image=self._logo_img, anchor="center")
        else:
            fg = "#eaeaea" if is_dark else "#222222"
            cv.create_text(W // 2, H // 2 - 20, text="CUBE",
                           fill=fg, font=("Helvetica", 52, "bold"), anchor="center")

        # subtle 1-px border
        cv.create_rectangle(1, 1, W - 2, H - 2, outline=bdr, width=1)

        pad = 16
        bot = H - pad

        # bottom-left: author / year
        cv.create_text(pad, bot, text="P.Valiathan · 2026",
                       fill=sub, font=("Helvetica", 9), anchor="sw")

        # bottom-right: institution
        cv.create_text(W - pad, bot, text="Karolinska Institutet",
                       fill=sub, font=("Helvetica", 9), anchor="se")

        # centre-bottom: future publication citation (invisible when empty)
        if self._citation_text:
            cv.create_text(W // 2, bot, text=self._citation_text,
                           fill=sub, font=("Helvetica", 8, "italic"), anchor="s")

        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")

        hold_ms = max(0, self.DISPLAY_MS - self.FADE_STEPS * self.FADE_MS)
        self.after(hold_ms, self._begin_fade)

    def _begin_fade(self):
        self._fade(1.0)

    def _fade(self, alpha: float):
        alpha -= 1.0 / self.FADE_STEPS
        if alpha <= 0.05:
            self._finish()
            return
        try:
            self.attributes("-alpha", alpha)
        except Exception:
            self._finish()
            return
        self.after(self.FADE_MS, lambda: self._fade(alpha))

    def _finish(self):
        try:
            self.destroy()
        except Exception:
            pass
        self._on_done()


class PipelineApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.withdraw()   # hidden while building UI; shown at end of __init__
        self.title("CUBE: Comprehensive Unsupervised Behavioral Explorer  v3.0")

        _ico = HERE / "CUBE.ico"
        if _ico.is_file():
            try:
                # default= stamps the icon on every window in this interpreter,
                # not just the root, so child Toplevels also get the CUBE icon.
                self.iconbitmap(default=str(_ico))
            except Exception:
                pass

        self.configure(bg=C["bg"])
        self.geometry("1440x820")
        self.minsize(1100, 660)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Show splash as a Toplevel of this — the one and only — Tk root.
        # Creating a second tk.Tk() for the splash steals _default_root, which
        # causes every tk.Var created later (BooleanVar, IntVar, etc.) to bind
        # to the splash's Tcl interpreter.  When that splash is later destroyed
        # the interpreter dies and every settings field goes blank.
        _splash = CubeSplash(self, _DARK_THEME, on_done=lambda: None)
        self.update()   # render splash before blocking on imports

        # Heavy imports while splash is visible (main thread blocks; splash frozen)
        _deferred_imports()   # must run before PipelineLogger is instantiated below
        _check_and_warn()

        self._session  = SessionState()
        _init_log_dir = HERE / "CUBE_logs"
        _init_log_dir.mkdir(parents=True, exist_ok=True)
        self._logger   = PipelineLogger(_init_log_dir)
        self._running  = False
        self._chain_engine_after_prep = False

        self._build_ui()
        self._initial_log()
        self._tick_timer()

        # Hold ~2 s so splash stays visible after imports finish; animation plays
        _deadline = time.monotonic() + 2.0
        while time.monotonic() < _deadline:
            try:
                self.update()
            except Exception:
                break
            time.sleep(0.05)

        try:
            _splash.destroy()
        except Exception:
            pass
        self.deiconify()

    #  " "  timer  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "
    def _tick_timer(self):
        self.after(1000, self._tick_timer)

    #  " "  UI  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " 

    def _build_ui(self):
        self._build_menubar()
        self._build_header()
        body = tk.PanedWindow(self, orient="vertical",
                              bg=C["bg"], sashwidth=6, sashrelief="flat")
        body.pack(fill="both", expand=True)
        self._build_top_pane(body)
        self._build_log_pane(body)

    def _build_menubar(self):
        """Native menu bar.  Utility actions (e.g. the manual UMAP-evolution
        video export) live here rather than as buttons cluttering the main panel.
        The evolution video is produced automatically after Step 3; this menu item
        is only for exporting additional sessions on demand."""
        menubar = tk.Menu(self)

        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Export Extra UMAP Evolution Videos...",
                          command=self._launch_umap_evolution_video)
        menubar.add_cascade(label="Tools", menu=tools)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="Help", command=lambda: show_help(self))
        menubar.add_cascade(label="Help", menu=helpm)

        self.config(menu=menubar)

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
                      bg=C["btn"], fg=C["btn_fg"],
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
        self._bd_max = tk.DoubleVar(value=10.0)
        tk.Spinbox(bd_row, from_=0.1, to=9999.0, increment=1.0,
                   format="%.1f", textvariable=self._bd_max, width=7,
                   bg=C["card2"], fg=C["text"],
                   buttonbackground=C["card2"],
                   font=("Segoe UI", 9)).pack(side="left", padx=(2, 0))
        tk.Label(bd_row, text="  (raise to capture longer sustained bouts)",
                 font=("Segoe UI", 8), bg=C["card"],
                 fg=C["dim"]).pack(side="left", padx=6)

        self._settings = SettingsPanel(left)
        self._settings.pack(fill="x")

        adv_row = tk.Frame(left, bg=C["bg"])
        adv_row.pack(fill="x", pady=(4, 2))
        tk.Button(adv_row, text="⚙  DLC & Prep Settings...",
                  font=("Segoe UI", 8, "bold"), bg=C["btn"], fg=C["yellow"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_dlc_prep).pack(side="left", padx=(0, 4), pady=2)
        tk.Button(adv_row, text="⚙  Advanced DLC Parameters...",
                  font=("Segoe UI", 8, "bold"), bg=C["btn"], fg=C["green"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_dlc_advanced).pack(side="left", padx=(0, 4), pady=2)
        tk.Button(adv_row, text="⚙  Advanced CUBE Analysis...",
                  font=("Segoe UI", 8, "bold"), bg=C["btn"], fg=C["purple"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_cube_advanced).pack(side="left", pady=2)

        # "Export Extra UMAP Evolution Videos" lives in the Tools menu bar
        # (see _build_menubar) rather than as a button in the main panel.

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
        adv = self._session.get("dlc_advanced_cfg", {})
        if adv.get("dlc_crop_enable", False):
            dlg = CropPreviewDialog(self, self._session)
            self.wait_window(dlg)
            if not dlg.confirmed:
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
        # When auto_bsoid is on (Steps 2+3 run automatically after DLC) we default
        # to deleting to keep disk usage low without interrupting the unattended run.
        if "bsoid_delete_videos_folder" not in self._session._d:
            if bool(self._settings.get("auto_bsoid", False)):
                self._session["bsoid_delete_videos_folder"] = True
            else:
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
        _video_groups  = dict(self._session.get("video_groups", {}))
        _stem_to_group = dict(self._session.get("stem_to_group", {}))

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

        # If the session is missing group assignments (e.g. fresh GUI run without
        # loading the old session), look for the file written after Step 3.
        if not _stem_to_group and bout_root is not None:
            for _ga_p in (
                bout_root.parent / "model" / "group_assignments.json",
                bout_root / "group_assignments.json",
            ):
                if _ga_p.is_file():
                    try:
                        _stem_to_group = json.loads(
                            _ga_p.read_text(encoding="utf-8"))
                        self._logger.info(
                            f"  Loaded {len(_stem_to_group)} group assignment(s) "
                            f"from {_ga_p.name}")
                    except Exception:
                        pass
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
                            # Auto-detect behaviour groups from Phase 4 output
                            try:
                                app._auto_load_groups()
                            except Exception:
                                pass
                            # Load UMAP embedding/labels.  _select_folder is
                            # bypassed during auto-launch, so we do this explicitly.
                            # model/ is a sibling of bout_lengths/ under cube_results_*
                            # so search from the parent directory.
                            try:
                                _umap_root = bout_root.parent \
                                    if bout_root.name == "bout_lengths" \
                                    else bout_root
                                _emb_p, _lbl_p = _MOD_ANALYSER.find_umap_data(
                                    _umap_root)
                                if _emb_p and _lbl_p:
                                    import numpy as _np_umap
                                    app._umap_embedding = _np_umap.load(str(_emb_p))
                                    app._umap_labels    = _np_umap.load(str(_lbl_p))
                                    self._logger.info(
                                        f"  UMAP data loaded: {_emb_p.parent}")
                            except Exception:
                                pass
                            self._logger.success(
                                f"Auto-loaded {len(files)} file(s) from {bout_root}")
                            # Populate the Combined Analysis animal panel
                            try:
                                ap = getattr(app, "_animal_panel", None)
                                if ap is not None:
                                    ap.add_files_from_paths(files)
                                    if ap.animal_count():
                                        try:
                                            app._tabs.set("Combined Analysis")
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            # Inject experimental group assignments if set in session.
                            # Primary method: stem_to_group maps each DLC-file stem
                            # to a group; the bout CSV stem is <dlc_stem>_bout_lengths[_hmm]
                            # so we strip the known suffix to recover the DLC stem.
                            # Fallback: folder-prefix matching (works when analyser loads
                            # files directly from the source folder tree).
                            if _video_groups or _stem_to_group:
                                try:
                                    _BOUT_SUFFIXES = (
                                        "_bout_lengths_hmm", "_bout_lengths",
                                        "_frame_labels_hmm", "_frame_labels",
                                    )
                                    ap = getattr(app, "_animal_panel", None)
                                    panel_animals = getattr(ap, "_animals", []) if ap else []
                                    assigned = 0
                                    for animal in panel_animals:
                                        apath = str(animal.get("path", ""))
                                        # Derive DLC stem by stripping bout-CSV suffixes
                                        astem = Path(apath).stem
                                        for _sfx in _BOUT_SUFFIXES:
                                            if astem.endswith(_sfx):
                                                astem = astem[: -len(_sfx)]
                                                break
                                        grp = _stem_to_group.get(astem)
                                        if grp is None:
                                            # Fallback: path-prefix check
                                            for folder, g in _video_groups.items():
                                                if apath.startswith(folder):
                                                    grp = g
                                                    break
                                        if grp is not None:
                                            eg_var = animal.get("exp_group")
                                            if eg_var is not None:
                                                eg_var.set(grp)
                                                assigned += 1
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

    def _launch_umap_evolution_video(self):
        """Export side-by-side UMAP evolution videos to the CUBE analysis folder.

        Only user input required: how many videos to export.  Sessions with
        embedded video paths are auto-discovered; videos are randomly sampled
        from the full pool and saved to <cube_results>/videos/umap_evolution/.
        """
        if not CORE_OK:
            messagebox.showerror("Missing", "cube_core.py not found.")
            return
        if self._running:
            messagebox.showwarning("Busy", "Another step is already running.")
            return

        # ── Locate model/ directory automatically ────────────────────────────
        model_dir = None
        search_roots = (self._session.get("engine_out_dirs", []) +
                        self._session.get("bsoid_ready_dirs", []))
        for d in search_roots:
            candidate = Path(d) / "model"
            if (candidate / "umap_embedding.npy").is_file():
                model_dir = candidate
                break
        if model_dir is None:
            for folder in self._session.get("video_folders", []):
                for cr in sorted(Path(folder).glob("cube_results*"), reverse=True):
                    candidate = cr / "model"
                    if (candidate / "umap_embedding.npy").is_file():
                        model_dir = candidate
                        break
                if model_dir:
                    break
        if model_dir is None or not (model_dir / "umap_embedding.npy").is_file():
            messagebox.showerror(
                "Model not found",
                "Could not locate umap_embedding.npy automatically.\n\n"
                "Run Step 3 (CUBE Clustering) first, then try again.")
            return

        # ── Load session_bin_ranges.json ─────────────────────────────────────
        sbr_path = model_dir / "session_bin_ranges.json"
        if not sbr_path.is_file():
            messagebox.showerror(
                "Missing data",
                "session_bin_ranges.json not found.\n\n"
                "Re-run Step 3 to generate it.")
            return
        try:
            sbr = json.loads(sbr_path.read_text())
        except Exception as e:
            messagebox.showerror("Load error", f"Cannot read session_bin_ranges.json:\n{e}")
            return

        def _parse_sbr(entry):
            if isinstance(entry, list) and len(entry) >= 2:
                start, end = int(entry[0]), int(entry[1])
                vpath = str(entry[2]) if len(entry) >= 3 and entry[2] else None
                return start, end, vpath
            return None, None, None

        sessions = {k: _parse_sbr(v) for k, v in sbr.items()
                    if k != "_total_bins"}
        # Embedded path first; if it is missing (e.g. the BSOID_Project_Ready
        # video copies were deleted after the run) fall back to searching the
        # configured video folders by session name.
        _evo_search = list(self._session.get("video_folders", []))
        ready = []
        for k, (s, e, v) in sessions.items():
            if v and Path(v).is_file():
                ready.append((k, s, e, v))
            else:
                alt = _find_video_by_stem(k, _evo_search)
                if alt is not None:
                    ready.append((k, s, e, str(alt)))

        if not ready:
            messagebox.showerror(
                "No sessions ready",
                "No source videos could be located for this run.\n\n"
                "session_bin_ranges.json points at videos that no longer exist "
                "(if you enabled 'delete BSOID_Project_Ready/videos', the copies "
                "were removed). Keep the source videos, or add their folder to "
                "the video sources, then try again.")
            return

        # ── Ask how many videos to export ────────────────────────────────────
        from tkinter import simpledialog as _sd
        n_req = _sd.askinteger(
            "UMAP Evolution Videos",
            f"How many evolution videos to export?\n"
            f"({len(ready)} session(s) available)",
            initialvalue=min(1, len(ready)),
            minvalue=1, maxvalue=len(ready),
            parent=self,
        )
        if n_req is None:
            return

        import random as _rnd
        chosen = _rnd.sample(ready, min(n_req, len(ready)))

        # ── Load embedding + labels ───────────────────────────────────────────
        try:
            import numpy as _np_ev
            embedding   = _np_ev.load(str(model_dir / "umap_embedding.npy"))
            umap_labels = _np_ev.load(str(model_dir / "umap_labels.npy"))
        except Exception as e:
            messagebox.showerror("Load error", f"Cannot load UMAP data:\n{e}")
            return

        fps_val = float(self._settings.get("fps", 30))
        out_dir = model_dir.parent / "videos" / "umap_evolution"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"  Output folder: {out_dir}")

        n_exports = len(chosen)
        self._running = True
        self._logger.step(f"Exporting {n_exports} UMAP evolution video(s)...")

        def _progress_factory(label: str):
            def _progress(phase: str, pct: float):
                self._logger.info(f"  [{label}] {phase}: {int(pct * 100)} %")
            return _progress

        def _worker():
            try:
                import pandas as _pd_ev
                produced = []
                for i, (key, start_bin, end_bin, vpath_str) in enumerate(chosen, 1):
                    vid_path = Path(vpath_str)
                    stem     = vid_path.stem
                    self._logger.info(
                        f"Exporting UMAP evolution video {i}/{n_exports} "
                        f"for '{stem}'...")

                    if end_bin > len(embedding):
                        self._logger.error(
                            f"  Skipping '{stem}': bin range [{start_bin}:{end_bin}] "
                            f"exceeds embedding length {len(embedding)}.")
                        continue

                    session_embedding   = embedding[start_bin:end_bin]
                    session_umap_labels = umap_labels[start_bin:end_bin]

                    bout_dir = model_dir.parent / "bout_lengths"
                    frame_labels_path = None
                    for suffix in (f"{stem}_frame_labels_hmm.csv",
                                   f"{stem}_frame_labels.csv"):
                        candidate = bout_dir / suffix
                        if candidate.is_file():
                            frame_labels_path = candidate
                            break
                    if frame_labels_path is None:
                        self._logger.error(
                            f"  Skipping '{stem}': no frame_labels CSV in {bout_dir}")
                        continue
                    try:
                        # frame_labels CSVs have a header (frame,time_s,label);
                        # select the 'label' column, not iloc[:,0] (frame index).
                        fl_df = _pd_ev.read_csv(str(frame_labels_path))
                        _lc   = "label" if "label" in fl_df.columns else fl_df.columns[-1]
                        frame_labels = (_pd_ev.to_numeric(fl_df[_lc], errors="coerce")
                                        .dropna().to_numpy(dtype=int))
                    except Exception as e:
                        self._logger.error(
                            f"  Skipping '{stem}': cannot load frame labels: {e}")
                        continue

                    out_path = out_dir / f"{stem}_umap_evolution.mp4"
                    result = create_umap_evolution_video(
                        video_path=vid_path,
                        embedding=session_embedding,
                        umap_labels=session_umap_labels,
                        frame_labels=frame_labels,
                        source_fps=fps_val,
                        out_path=out_path,
                        output_fps=15.0,
                        progress_cb=_progress_factory(stem),
                    )
                    if result is not None:
                        self._logger.success(f"  Saved → {result}")
                        produced.append(result)
                    else:
                        self._logger.error(
                            f"  Export failed for '{stem}' (no output written).")

                if produced:
                    summary = "\n".join(str(p) for p in produced)
                    self._after(lambda s=summary: messagebox.showinfo(
                        "Export complete",
                        f"UMAP evolution video(s) saved to:\n{out_dir}\n\n{s}"))
                else:
                    self._after(lambda: messagebox.showerror(
                        "Export failed",
                        "No UMAP evolution videos were produced.  "
                        "Check the log for details."))
            except Exception:
                tb = traceback.format_exc()
                self._logger.error(f"UMAP evolution video error:\n{tb}")
                self._after(lambda: messagebox.showerror(
                    "Export error", tb[:800]))
            finally:
                self._running = False

        threading.Thread(target=_worker, daemon=True).start()

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
#  DEFERRED HEAVY IMPORTS  (called from main() after loading splash is visible)
#

def _deferred_imports():
    """Import cube_core and companion scripts. Called after the loading splash renders."""
    global CORE_OK, _CORE_ERR
    global PipelineLogger, BSoidEngine, run_bsoid_prep
    global filter_dlc_h5, cleanup_video_byproducts, create_umap_evolution_video
    global _MOD_VIDEO, _PATH_VIDEO, _MOD_ANALYSER, _PATH_ANALYSER

    try:
        from cube_core import (
            PipelineLogger, BSoidEngine, run_bsoid_prep, filter_dlc_h5,
            cleanup_video_byproducts, create_umap_evolution_video,
        )
        CORE_OK = True
    except ImportError as _ce:
        CORE_OK = False
        _CORE_ERR = str(_ce)
        def cleanup_video_byproducts(*_a, **_kw): pass
        def filter_dlc_h5(h5_path, *_a, out_path=None, **_kw):
            import shutil as _sh
            dst = out_path or h5_path.with_name(h5_path.stem + "_filtered.h5")
            _sh.copy2(str(h5_path), str(dst))
            return dst

    _MOD_VIDEO,    _PATH_VIDEO    = _load_script(["cube_video_explorer.py",
                                                   "BSOID_VIDEO_EXPLR.py"])
    _MOD_ANALYSER, _PATH_ANALYSER = _load_script(["cube_analyser.py"])


#
#  ENTRY POINT
#

def main():
    # Tell Windows who this process is before any window is created.
    # This ensures the taskbar always uses the CUBE icon, not Python's feather.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Cube.BehaviouralExplorer.v3")
    except Exception:
        pass

    app = PipelineApp()
    app.mainloop()


if __name__ == "__main__":
    main()
