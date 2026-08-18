# -*- coding: utf-8 -*-
"""
CUBE: Comprehensive Unsupervised Behavioral Explorer
====================================================
v5.0  —  DeepLabCut  ▸  CUBE engine  ▸  Annotator  ▸  Analyser

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
    cube_core.py           (required — V5 analysis engine)
    cube_analyser.py       (required for Step 5)
    cube_video_explorer.py (required for Step 4)

Step 1 — Run DLC inference       (DeepLabCut SuperAnimal)
Step 2 — CUBE pre-processing     (bodypart filtering, H5/CSV export)
Step 3 — CUBE clustering engine  (V5 features · UMAP · HDBSCAN · MLP)
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
# Intel's MKL loads libiomp5md.dll while scikit-learn's compiled extensions
# load VCOMP140.DLL (MSVC OpenMP) into the same process. When both runtimes
# initialise, Intel's OpenMP detects the "duplicate runtime" condition and
# calls abort() (OMP Error #15), which Windows surfaces as an unrecoverable
# native crash (ucrtbase.dll, exception 0xc0000409) with no Python traceback
# - typically during the HDBSCAN sweep, where MKL-heavy and sklearn-heavy
# code run concurrently across threads. Single-threading above does not
# prevent this; only disabling the duplicate-runtime abort does.
_os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# DO NOT set NUMBA_THREADING_LAYER=workqueue here (tried and reverted, Aug
# 2026): it was meant to fix a video-export crash later traced to an
# unrelated pynndescent bug (see _patch_pynndescent_thread_safety() in
# cube_core.py), but workqueue itself is documented as unsafe under
# concurrent access from multiple threads -- and this codebase's nested
# dispatch (seed_sweep_stability/consensus_cluster/split_impure_clusters)
# does exactly that, reproducibly crashing with "Numba workqueue threading
# layer is terminating: Concurrent access has been detected." numba's own
# default (TBB) is the thread-safe choice here; leave it unset.
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

# On Windows, an app that hasn't declared itself DPI-aware gets bitmap-scaled
# by the OS on non-100%-scaled or mixed-DPI multi-monitor setups: Tk renders
# widgets at one (logical) coordinate space while Windows displays them
# stretched, so a click that visually lands on one widget is delivered to a
# DIFFERENT widget at the underlying unscaled coordinates (e.g. clicking a
# "3x" radio button in cube_video_explorer.py's annotator actually invokes
# the "1x" button). This must be set once, process-wide, before any Tk
# window is created -- BSoidAnnotator/AnalyserApp are created later as
# nested tk.Tk() roots within this same process, so declaring it here covers
# them too.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

#  " "  local engine  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "
# Deferred to _deferred_imports() so the loading splash renders first.
CORE_OK      = False
_CORE_ERR    = ""
PipelineLogger = BSoidEngine = run_bsoid_prep = run_bsoid_prep_batch = None
filter_dlc_h5 = cleanup_video_byproducts = create_umap_evolution_video = None
find_dlc_files = peek_dlc_bodyparts = group_bodyparts_by_region = None
# v6 part 2 (Environmental_Context_v6_Implementation_Plan.md Step 2):
# resolve_env_shapes + the paradigm/role vocabulary constants are the single
# source of truth shared with cube_core.py's compute_session_env_context();
# EnvContextWindow's paradigm screen, role dropdowns, and naming suggestions
# read directly from these rather than duplicating the vocabulary here.
resolve_env_shapes = None
ENV_PARADIGMS = ENV_PARADIGM_ROLE_VOCAB = ENV_PARADIGM_MIN_ROLES = None
# Used by EnvParadigmWindow's auto-threshold preview (same functions
# compute_session_env_context itself uses for the real auto-derivation).
load_dlc_file = _find_spine_indices = _spine_norm_factor = None

#  " "   optional companion scripts  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "
HERE = Path(__file__).resolve().parent

#  " "   crash diagnostics  " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "
# Two real hard crashes in one session (Aug 2026) left no usable forensics:
# the first was a native access-violation inside python312.dll itself (found
# only via Windows Event Viewer, after the fact); the second left NO trace
# anywhere -- no WER report, no System-log entry, no Reliability Monitor
# entry -- python.exe just stopped existing. Neither is a catchable Python
# exception, so ordinary try/except logging around the pipeline stages can
# never see them. Two complementary safety nets, installed once, as early as
# possible, before any heavy numeric/native code runs:
#
#   1. faulthandler -- a stdlib fault handler that intercepts fatal native
#      signals (SIGSEGV/access-violation, SIGABRT, SIGFPE, SIGILL) and, unlike
#      WER, prints exactly which Python thread and source line was executing
#      at the moment of the fault. This is the only way to localise a crash
#      that happens inside a compiled extension (MKL/numba/HDBSCAN/OpenCV)
#      to an actual pipeline stage.
#   2. sys.excepthook / threading.excepthook -- covers the other failure
#      mode: a plain uncaught Python exception that exits the process
#      without ever reaching a dialog (e.g. raised in a background thread,
#      whose default behaviour is to print to stderr and vanish with no
#      GUI trace at all).
#
# Both write to a single persistent, append-only log (survives across runs
# and app restarts, unlike the per-run PipelineLogger file) so a crash that
# kills the process mid-write is still captured on disk immediately before.
_CRASH_LOG_DIR = HERE / "CUBE_logs"
_CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
_crash_fh = open(_CRASH_LOG_DIR / "crash_diagnostics.log", "a", buffering=1,
                  encoding="utf-8")
_crash_fh.write(
    f"\n{'='*78}\n[{datetime.now():%Y-%m-%d %H:%M:%S}] CUBE session start "
    f"(pid={os.getpid()}, python={sys.version.split()[0]})\n{'='*78}\n")

import faulthandler as _faulthandler
_faulthandler.enable(file=_crash_fh, all_threads=True)

def _crash_log_exception(exc_type, exc_value, exc_tb, thread_name="MainThread"):
    _crash_fh.write(
        f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] UNCAUGHT EXCEPTION "
        f"(thread: {thread_name})\n")
    traceback.print_exception(exc_type, exc_value, exc_tb, file=_crash_fh)
    _crash_fh.flush()

def _sys_excepthook(exc_type, exc_value, exc_tb):
    _crash_log_exception(exc_type, exc_value, exc_tb, "MainThread")
    sys.__excepthook__(exc_type, exc_value, exc_tb)

def _threading_excepthook(args):
    _crash_log_exception(
        args.exc_type, args.exc_value, args.exc_traceback,
        args.thread.name if args.thread is not None else "unknown")

sys.excepthook = _sys_excepthook
threading.excepthook = _threading_excepthook
sys._cube_crash_diag_installed = True
del _CRASH_LOG_DIR

# Global (cross-project) sidecar for the last-applied Body-Region Weights, so
# a brand-new project starts from the user's last customisation instead of
# uniform every time -- same write-on-change/read-on-next-use pattern as
# theme.txt, just JSON instead of a single value. Per-project SessionState
# (bodypart_weights in the .pipeline_session.json) always wins over this when
# a project already has its own explicit weights set; this is only the
# fallback initial value for an otherwise-uniform/new project.
_BODY_REGION_WEIGHTS_FILE = HERE / "body_region_weights.json"

def _load_saved_body_region_weights() -> dict:
    try:
        return json.loads(_BODY_REGION_WEIGHTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_body_region_weights(weights: dict):
    try:
        _BODY_REGION_WEIGHTS_FILE.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    except Exception:
        pass

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
C["cube_title"] = "#4fc3f7"  # CUBE header text — blue, distinct from "accent" (overridden to teal in light mode)

# Single source of truth for the keys that differ between dark/light — used
# both at startup (parsing theme.txt) and by the live theme toggle, so the
# two paths can never drift apart. Keys not listed here (accent, green, cyan,
# yellow's dark-mode source value, orange, purple, red, dim, ...) stay fixed
# across themes.
_THEME_DARK = dict(
    bg="#09090f", panel="#111120", card="#16162a", card2="#1e1e35",
    border="#2a2a4a", text="#eaeaea", subtext="#7788aa",
    log_bg="#07070d", log_fg="#00ff88",
    btn="#1e1e35", btn_fg="#7788aa", yellow="#ffd60a", cube_title="#4fc3f7",
)
_THEME_LIGHT = dict(
    bg="#f0f2f5", panel="#ffffff", card="#ffffff", card2="#e9ecef",
    border="#dee2e6", text="#222222", subtext="#666666",
    log_bg="#ffffff", log_fg="#333333",
    # Flat buttons need a visible bg; dark amber replaces the bright yellow
    # that is invisible on light backgrounds; teal-blue replaces red for the
    # CUBE header text.
    btn="#c5ccd6", btn_fg="#1c1c30", yellow="#7a4e00", cube_title="#0077a8",
)

try:
    with open(HERE / "theme.txt", "r", encoding="utf-8") as _f:
        if _f.read().strip() == "light":
            C.update(_THEME_LIGHT)
except Exception:
    pass

# Resolved once at import so the splash and any future theme-aware widgets can
# read it without re-parsing theme.txt.
_DARK_THEME: bool = C["bg"] == "#09090f"

# Per-step card background — switches between dark/light tints. Kept as an
# explicit dark/light pair (rather than resolved once via a lambda) so a live
# theme toggle can look up "the other" value for each step later.
_STEP_BG_DARK = {
    "dlc": "#1a3a1a", "dlc_3d": "#102030", "bsoid_prep": "#1a1a3a",
    "bsoid_run": "#2a1a4a", "annotate": "#4a2a1a", "analyse": "#1a1a4a",
}
_STEP_BG_LIGHT = {
    "dlc": "#edf7ed", "dlc_3d": "#e8f4fd", "bsoid_prep": "#e0f7fa",
    "bsoid_run": "#f3e5f5", "annotate": "#fff3e0", "analyse": "#fce4ec",
}
_sbg = lambda dark, light: dark if _DARK_THEME else light
STEP_META = [
    dict(num=1, key="dlc",        icon=" ", title="DLC Inference",
         subtitle="DeepLabCut SuperAnimal on raw videos",
         bg=_sbg(_STEP_BG_DARK["dlc"], _STEP_BG_LIGHT["dlc"]), accent="#4caf50"),
    dict(num=2, key="dlc_3d",     icon="⬡", title="3D DLC + Anipose",
         subtitle="Per-camera tracking · triangulate · fuse cams",
         note="Multi-camera recordings only",
         bg=_sbg(_STEP_BG_DARK["dlc_3d"], _STEP_BG_LIGHT["dlc_3d"]), accent="#4fc3f7"),
    dict(num=3, key="bsoid_prep", icon="", title="CUBE Pre-processing",
         subtitle="Filter bodyparts · export H5/CSV",
         bg=_sbg(_STEP_BG_DARK["bsoid_prep"], _STEP_BG_LIGHT["bsoid_prep"]), accent="#00b4d8"),
    dict(num=4, key="bsoid_run",  icon=" ",  title="CUBE Clustering",
         subtitle="V2 features · UMAP · HDBSCAN · MLP",
         bg=_sbg(_STEP_BG_DARK["bsoid_run"], _STEP_BG_LIGHT["bsoid_run"]), accent="#9c27b0"),
    dict(num=5, key="annotate",   icon=" ", title="Video Annotation",
         subtitle="Label clusters via example clips",
         bg=_sbg(_STEP_BG_DARK["annotate"], _STEP_BG_LIGHT["annotate"]), accent="#ff9800"),
    dict(num=6, key="analyse",    icon=" ", title="Behaviour Analysis",
         subtitle="Metrics, ethograms, statistics",
         bg=_sbg(_STEP_BG_DARK["analyse"], _STEP_BG_LIGHT["analyse"]), accent="#e94560"),
]


# Widget color options that may hold a literal C[...] (or step-card bg)
# value baked in at construction time.
_THEMED_WIDGET_OPTIONS = (
    "bg", "fg", "background", "foreground",
    "activebackground", "activeforeground",
    "highlightbackground", "highlightcolor",
    "insertbackground", "selectbackground", "selectforeground",
    "troughcolor", "disabledforeground",
)


def _flat_theme_dict(dark: bool) -> dict:
    """Flatten the current C-dict state + per-step card backgrounds into one
    dict of synthetic keys -> colour values, for before/after comparison when
    live-toggling the theme (see _rethread_widget_colors)."""
    flat = dict(C)
    for key in _STEP_BG_DARK:
        flat[f"__stepbg_{key}"] = _STEP_BG_DARK[key] if dark else _STEP_BG_LIGHT[key]
    return flat


def _rethread_widget_colors(widget, old_theme: dict, new_theme: dict):
    """Recursively re-theme an already-built widget subtree in place.

    Most widgets in this app are constructed once with a literal color
    pulled from the global ``C`` dict at build time (e.g. ``bg=C["card"]``);
    flipping ``C`` afterwards does nothing for them since the string was
    already resolved. This walks the widget tree (including any open
    Toplevel settings windows, which appear as descendants via
    winfo_children()) and, for any color option whose current value exactly
    matches a value from *old_theme*, swaps in the corresponding value from
    *new_theme* — restyling in place without rebuilding anything.
    """
    for opt in _THEMED_WIDGET_OPTIONS:
        try:
            cur = widget.cget(opt)
        except Exception:
            continue
        if not isinstance(cur, str) or not cur:
            continue
        for key, old_val in old_theme.items():
            if cur == old_val:
                new_val = new_theme.get(key)
                if new_val is not None and new_val != cur:
                    try:
                        widget.configure(**{opt: new_val})
                    except Exception:
                        pass
                break
    try:
        children = widget.winfo_children()
    except Exception:
        children = []
    for child in children:
        _rethread_widget_colors(child, old_theme, new_theme)

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
        dlc_smart_adapt = True,
        # BSOID prep
        bsoid_min_conf  = 0.30,
        bsoid_conf_metric    = "median",
        bsoid_min_sess_frac  = 0.85,
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
        # 3D DLC + Anipose
        dlc_3d_enabled          = False,
        dlc_3d_calib_folder     = "",
        dlc_3d_input_folder     = "",
        dlc_3d_output_folder    = "",
        dlc_3d_cam_labels       = ["cam0", "cam1", "cam2", "cam3"],
        dlc_3d_models           = {},
        dlc_3d_ll_agg           = "min",
        dlc_3d_delete_orig_videos = False,
        dlc_3d_delete_cam_h5s        = False,
        dlc_3d_export_skeleton_video = False,
        dlc_3d_use_ransac            = True,
        dlc_3d_ransac_threshold      = 0.5,
        dlc_3d_ll_threshold          = 0.0,
        dlc_3d_ll_gate               = 0.6,
        dlc_3d_median_window         = 3,
        dlc_3d_source_folders        = [],  # saved before video_folders is replaced by session dirs
        pca_n_components             = "auto",
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

    def refresh_theme(self):
        """Called after a live theme toggle. _COLOURS is a class-level dict
        resolved once at import time, and tk.Text tag colours aren't plain
        widget options, so neither is touched by the generic bg/fg walk —
        both need an explicit re-apply here."""
        colours = dict(self._COLOURS)
        colours.update(INFO=C["text"], WARN=C["yellow"],
                       ERROR=C["red"], SUCCESS=C["green"], DEBUG=C["dim"])
        for tag, colour in colours.items():
            self._txt.tag_config(tag, foreground=colour)
        self._txt.tag_config("TS", foreground=C["dim"])


#  
#  PROGRESS BAR  (dual: overall + per-step)
#  

class DualProgressBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        style = ttk.Style()
        style.configure("Step.Horizontal.TProgressbar",
                        troughcolor=C["card"], background=C["green"],
                        thickness=8)
        tk.Label(self, text="Steps", font=("Segoe UI", 8),
                 bg=C["panel"], fg=C["subtext"]).pack(anchor="w", padx=8)
        chips_row = tk.Frame(self, bg=C["panel"])
        chips_row.pack(fill="x", padx=8, pady=(0, 4))
        self._chips: dict[str, tk.Label] = {}
        for meta in STEP_META:
            chip = tk.Label(chips_row, text=f"{meta['num']} {meta['icon']}",
                             font=("Segoe UI", 8, "bold"),
                             bg=C["card"], fg=C["subtext"],
                             padx=6, pady=2)
            chip.pack(side="left", padx=(0, 3))
            self._chips[meta["key"]] = chip
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

    def set_step_status(self, key: str, status: str):
        """Recolour a single step's chip to reflect its own status —
        independent of any other step, since the pipeline can be entered
        at any step (existing DLC output, resuming mid-way, etc.)."""
        chip = self._chips.get(key)
        if chip is None:
            return
        colour, _ = _BADGE_STATES.get(status, _BADGE_STATES["idle"])
        fg = "white" if status in ("running", "done", "error") else C["subtext"]
        chip.configure(bg=colour, fg=fg)

    def sync_step_statuses(self, statuses: dict):
        """Bulk-refresh every chip, e.g. after loading a saved session."""
        for key, status in statuses.items():
            self.set_step_status(key, status)

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
    "idle":    ("#555566",    " -   Waiting"),
    "ready":   ("#00b4d8",    " -   Ready"),
    "running": (C["yellow"],  "   Running "),
    "done":    ("#4caf50",    "v  Complete"),
    "error":   ("#f44336",    " -  Error"),
    "skipped": ("#888899",    "   Skipped"),
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
            anchor="w", padx=10, pady=(0, 6))

        if meta.get("note"):
            tk.Label(self, text=f"⚠  {meta['note']}",
                     font=("Segoe UI", 7, "italic"),
                     bg=bg, fg=meta["accent"],
                     wraplength=195, justify="left").pack(
                anchor="w", padx=10, pady=(0, 8))
        else:
            tk.Frame(self, bg=bg, height=8).pack()

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

    def refresh_theme(self, new_bg: str):
        """Called after a live theme toggle. The generic widget-color walk
        (_rethread_widget_colors) already recolours every child widget by
        matching against the old/new step-bg pair; this just keeps the
        cached attribute in sync for any future construction that reads it.
        """
        self._bg = new_bg


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
        ("dlc_epochs",       "Adapt epochs",        "int",  (4,200,2),  15,
         "15=recommended  4=fast  (Advanced dialog's Det/Pose epochs share this ceiling)"),
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
         "Run Step 3 pre-processing inline within DLC (Step 1)"),
        ("auto_bsoid",       "Auto-run analysis",   "bool", None, False,
         "After DLC: auto-launch Step 3 (pre-processing) then Step 4 (BSoid analysis)"),
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
        ("bsoid_min_sess_frac","BP keep if passes ≥","float",(0.1,1.0,0.1),0.85,
         "Keep a bodypart if it passes in ≥ this fraction of sessions"),
        ("bsoid_min_keep",   "Min bodyparts kept",  "int",  (2,40,1), 6,
         "Floor: fall back to top-N by confidence if fewer pass"),
        ("ntfy_topic",       "Notification Topic",  "str",  None, "",
         "ntfy.sh topic name for push alerts"),
    ]
    # NOTE: engine-level analysis parameters (body_normalise, train_frac,
    # UMAP/HDBSCAN/MLP settings, long_lag_drift, long_scale_bins,
    # pca_n_components, etc.) are configured via AdvancedCUBEWindow, which
    # writes directly to session["engine_cfg"] -- NOT via this panel. A
    # previous _ENGINE_ROWS list duplicating those same keys here was dead
    # code (never rendered into widgets, get_engine_cfg() always returned
    # {}) and has been removed; see AdvancedCUBEWindow for the real controls.

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

    def apply_session(self, session: SessionState):
        for key in self._vars:
            val = session[key] if key in session._d else None
            if val is not None:
                self.set_val(key, val)

    def export_to_session(self, session: SessionState):
        dlc_keys = [r[0] for r in self._DLC_ROWS]
        for k in dlc_keys:
            session[k] = self.get(k)
        # engine_cfg is written directly by AdvancedCUBEWindow._apply() --
        # this panel never touches it, so it's left untouched here too.


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
    """Dynamically unfreeze BatchNorm statistics in DeepLabCut PyTorch engine during adaptation training.

    Applied by both the regular per-video "Video adapt" path and Smart
    Adapt v3, since both drive the same underlying DLC adaptation-training
    code. Idempotent: safe to call more than once per process (e.g. a
    regular-path run followed by a Smart Adapt run in the same session) --
    re-wrapping an already-patched COCOLoader.update_model_cfg is skipped
    rather than nesting another wrapper layer.
    """
    try:
        from deeplabcut.pose_estimation_pytorch.modelzoo.train_from_coco import COCOLoader
        if getattr(COCOLoader.update_model_cfg, "_cube_bn_unfreeze_patch", False):
            return
        original_update = COCOLoader.update_model_cfg

        def custom_update(self, updates):
            if "model.backbone.freeze_bn_stats" in updates:
                updates["model.backbone.freeze_bn_stats"] = False
                logger.info("  [MONKEYPATCH] Unfreezing pose model backbone BatchNorm stats.")
            if "detector.model.freeze_bn_stats" in updates:
                updates["detector.model.freeze_bn_stats"] = False
                logger.info("  [MONKEYPATCH] Unfreezing detector model backbone BatchNorm stats.")
            original_update(self, updates)

        custom_update._cube_bn_unfreeze_patch = True
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

    # Same BatchNorm-unfreeze patch Smart Adapt applies before its adaptation
    # training call -- both paths drive the same underlying DLC
    # video_adapt=True training code, so both should get consistent behavior.
    if use_adapt and not _use_custom:
        _apply_dlc_monkeypatch(logger)

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
                                  fps=float(session.get("fps", 30)),
                                  likelihood_thresh=_pcutoff)
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
        logger.step("Running CUBE pre-processing (Step 3)...")
        # run_bsoid_prep_batch() (Aug 2026): pools the conserved-bodyparts
        # decision once across every folder's sessions instead of each
        # folder choosing its own conserved set independently -- see
        # _run_bsoid_prep_step's docstring for why this matters (avoids
        # compounding several independently-conservative cuts into one
        # overly aggressive one).
        bsoid_roots = run_bsoid_prep_batch(
            folders, log_fn=logger,
            min_confidence=float(session["bsoid_min_conf"]),
            conf_metric=str(session.get("bsoid_conf_metric", "median")),
            min_session_frac=float(session.get("bsoid_min_sess_frac", 0.85)),
            min_keep=int(session.get("bsoid_min_keep", 6)))
        session["bsoid_ready_dirs"] = [str(r) for r in bsoid_roots]

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

    n_epochs      = int(settings.get("dlc_epochs", 15))
    pseudo_thr    = float(settings.get("dlc_pseudo_thr", 0.5))
    filter_key    = settings.get("dlc_filter", "Sequential  Median  ’ Gaussian")
    filter_types  = FILTER_OPTIONS.get(filter_key, ["median", "gaussian"])
    run_prep      = bool(settings.get("dlc_run_prep", True))
    create_filt_v = bool(settings.get("dlc_filtered_vid", True))

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
    _det_epochs    = int(_adv.get("dlc_det_epochs",       n_epochs))
    _pose_epochs   = int(_adv.get("dlc_pose_epochs",      n_epochs))
    _transfer      = bool(_adv.get("dlc_transfer",        True))
    _scale_mode    = str(_adv.get("dlc_scale_mode",       "Auto"))
    _scale_min     = int(_adv.get("dlc_scale_min",        100))
    _scale_max     = int(_adv.get("dlc_scale_max",        600))
    _scale_step    = int(_adv.get("dlc_scale_step",       50))
    _inf_batch_ov  = int(_adv.get("dlc_inf_batch",        0))
    _det_batch_ov  = int(_adv.get("dlc_det_batch",        0))
    _crop_enable   = bool(_adv.get("dlc_crop_enable",    False))
    _crop_x        = int(_adv.get("dlc_crop_x",          0))
    _crop_y        = int(_adv.get("dlc_crop_y",          0))
    _crop_w        = int(_adv.get("dlc_crop_w",          0))
    _crop_h        = int(_adv.get("dlc_crop_h",          0))
    _do_crop       = _crop_enable and _crop_w > 0 and _crop_h > 0

    work_dir   = _resolve_work_dir(session)
    errors_dir = work_dir / "errors"
    adapt_work = work_dir / "smart_adapt_workspace"
    adapt_work.mkdir(parents=True, exist_ok=True)

    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

    # GPU batch auto-sizing; capped at 85% of free VRAM to avoid OOM crashes.
    # A user-forced dlc_inf_batch override (Advanced DLC settings) takes
    # precedence over the VRAM-based auto-calc, matching the regular
    # per-video path's behavior (cube.py _run_dlc_step).
    if _inf_batch_ov > 0:
        inf_batch = _inf_batch_ov
    else:
        inf_batch = 8
        try:
            import torch
            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                usable_gb = free_gb * 0.85
                inf_batch = 32 if usable_gb >= 10 else (16 if usable_gb >= 5 else 8)
        except Exception:
            pass
    det_batch = _det_batch_ov if _det_batch_ov > 0 else inf_batch

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

            # ── Scale list for SuperAnimal detector (mirrors the regular
            # per-video path's Auto/Manual logic instead of a fixed range) ──
            if _scale_mode == "Manual":
                _scale_list = list(range(_scale_min,
                                         _scale_max + _scale_step,
                                         _scale_step))
            else:
                try:
                    _cap_sc = cv2.VideoCapture(str(rep_inf))
                    _short  = min(int(_cap_sc.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(_cap_sc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    _cap_sc.release()
                except Exception:
                    _short = 720
                _centre     = max(150, int(_short * 0.35))
                _scale_list = list(range(max(100, _centre - 150),
                                         min(1200, _centre + 200), 50))

            sa_kw = dict(
                superanimal_name     = _sa_name,
                model_name           = _model_name,
                detector_name        = _detector_name,
                scale_list           = _scale_list,
                pcutoff              = _pcutoff,
                bbox_threshold       = _bbox_thr,
                max_individuals      = _max_ind,
                batch_size           = inf_batch,
                detector_batch_size  = det_batch,
                create_labeled_video = False,
                video_adapt          = True,
                pseudo_threshold     = pseudo_thr,
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
                project          = f"{base_folder_name}_CUBE_v5",
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

        # ── Scale list for SuperAnimal detector (mirrors the regular
        # per-video path's Auto/Manual logic instead of a fixed range) ──
        if _scale_mode == "Manual":
            _scale_list = list(range(_scale_min,
                                     _scale_max + _scale_step,
                                     _scale_step))
        else:
            try:
                _cap_sc = cv2.VideoCapture(str(vpath))
                _short  = min(int(_cap_sc.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(_cap_sc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                _cap_sc.release()
            except Exception:
                _short = 720
            _centre     = max(150, int(_short * 0.35))
            _scale_list = list(range(max(100, _centre - 150),
                                     min(1200, _centre + 200), 50))

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
                        scale_list                     = _scale_list,
                        pcutoff                        = _pcutoff,
                        bbox_threshold                 = _bbox_thr,
                        max_individuals                = _max_ind,
                        batch_size                     = current_batchsize,
                        # bounded by current_batchsize so a prior OOM-driven
                        # reduction (see the retry loop below) is respected
                        # even when dlc_det_batch was set to a larger value
                        detector_batch_size            = min(det_batch, current_batchsize),
                        create_labeled_video           = create_filt_v,
                        video_adapt                    = False,
                        dest_folder                    = str(dest_folder),
                        device                         = "auto",
                        customized_pose_checkpoint     = adapted_pose_ckpt,
                        customized_detector_checkpoint = adapted_detector_ckpt,
                    )
                    try:
                        import inspect as _insp3
                        _vis_sig3 = _insp3.signature(dlc.video_inference_superanimal).parameters
                        if "superanimal_transfer_learning" in _vis_sig3:
                            _sa_inf_kw["superanimal_transfer_learning"] = _transfer
                    except Exception:
                        pass
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
                              fps=float(session.get("fps", 30)),
                              likelihood_thresh=_pcutoff)
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
        logger.step("Running CUBE pre-processing (Step 3) …")
        # run_bsoid_prep_batch() (Aug 2026): pools the conserved-bodyparts
        # decision once across every folder's sessions instead of each
        # folder choosing its own conserved set independently -- see
        # _run_bsoid_prep_step's docstring for why this matters.
        bsoid_roots = run_bsoid_prep_batch(
            folders, log_fn=logger,
            min_confidence=float(session.get("bsoid_min_conf", 0.30)),
            conf_metric=str(session.get("bsoid_conf_metric", "median")),
            min_session_frac=float(session.get("bsoid_min_sess_frac", 0.85)),
            min_keep=int(session.get("bsoid_min_keep", 6)))
        session["bsoid_ready_dirs"] = [str(r) for r in bsoid_roots]

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
                              fps=float(session.get("fps", 30)),
                              likelihood_thresh=_pcutoff)
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
        # run_bsoid_prep_batch() (Aug 2026): pools the conserved-bodyparts
        # decision once across every folder's sessions instead of each
        # folder choosing its own conserved set independently -- see
        # _run_bsoid_prep_step's docstring for why this matters.
        bsoid_roots = run_bsoid_prep_batch(
            folders, log_fn=logger,
            min_confidence=float(session.get("bsoid_min_conf", 0.30)),
            conf_metric=str(session.get("bsoid_conf_metric", "median")),
            min_session_frac=float(session.get("bsoid_min_sess_frac", 0.85)),
            min_keep=int(session.get("bsoid_min_keep", 6)))
        session["bsoid_ready_dirs"] = [str(r) for r in bsoid_roots]

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
    """Run CUBE pre-processing on all selected video folders.

    Uses run_bsoid_prep_batch() (Aug 2026) rather than one run_bsoid_prep()
    call per folder: when 2+ folders are selected (e.g. one per experimental
    group), the conserved-bodyparts decision is made ONCE, pooled across every
    session in every folder, instead of each folder choosing its own
    conserved set independently and relying on BSoidEngine's later cross-group
    intersection to reconcile them. Both approaches guarantee every group ends
    up sharing the exact same bodypart set (comparability is unaffected) --
    pooling just avoids compounding several independently-conservative cuts
    into one overly aggressive one, so more real tracking signal survives.
    """
    folders = session["video_folders"]
    if not folders:
        raise ValueError("No video folders selected.")
    pb.step_start("CUBE pre-processing", len(folders))
    logger(f"  Pre-processing {len(folders)} folder(s) "
           f"(pooled bodypart conservation across all of them):")
    bsoid_roots = run_bsoid_prep_batch(
        folders, log_fn=logger,
        min_confidence=float(settings.get("bsoid_min_conf", 0.30)),
        conf_metric=str(settings.get("bsoid_conf_metric", "median")),
        min_session_frac=float(settings.get("bsoid_min_sess_frac", 0.85)),
        min_keep=int(settings.get("bsoid_min_keep", 6)))
    roots = [str(r) for r in bsoid_roots]
    after_fn(lambda: pb.step_tick(len(folders), len(folders)))
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

    # User overrides from the Advanced CUBE window ONLY -- do NOT pre-merge
    # BSoidEngine.DEFAULTS here (Aug 2026 fix). BSoidEngine.__init__ already
    # merges cfg over its own DEFAULTS internally, so the merged VALUES are
    # identical either way -- but it also records self._explicit_cfg_keys =
    # set(cfg.keys()) to distinguish "user deliberately set this" from "just
    # inherited the default", used by e.g. the consensus-clustering
    # auto-trigger's opt-out check. Pre-merging the full DEFAULTS dict here
    # made every single key look "explicitly set", which silently defeated
    # that check for every GUI-driven run (confirmed: a real run with mean
    # ARI=0.427, well under the 0.6 auto-trigger threshold, never triggered
    # consensus because consensus_clustering_enabled=False looked deliberate
    # even though the user never touched it).
    cfg = dict(session.get("engine_cfg", {}))
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
        def _on_close():
            topic = settings_panel._vars["ntfy_topic"].get().strip()
            try:
                (HERE / "ntfy_topic.txt").write_text(topic, encoding="utf-8")
            except Exception:
                pass
            self.destroy()

        self.protocol("WM_DELETE_WINDOW", _on_close)
        tk.Button(btn_row, text="Close", font=("Segoe UI", 9, "bold"),
                  bg=C["btn"], fg=C["text"], relief="flat",
                  padx=20, pady=5, cursor="hand2",
                  command=_on_close).pack(side="left", padx=6)

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
    ARCHITECTURES = ["hrnet_w32", "resnet_50", "rtmpose_s", "rtmpose_x", "dlcrnet"]
    # dlcrnet is DLC's bottom-up TensorFlow-engine architecture (no detector
    # pairing needed); the other four are PyTorch top-down architectures.
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
                 text="  Basic settings live in the main DLC & Prep panel — these control the model.",
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

        def _pcutoff_row(r):
            _spin_f(r, "dlc_pcutoff", 0.0, 1.0, 0.05, 0.6)
            tk.Label(r, text="also gates which frames the jitter filter denoises",
                     font=("Segoe UI", 7), bg=C["card"], fg=C["dim"]
                     ).pack(side="left", padx=4)

        _adv_row(sec2, "Pose confidence (pcutoff)", _pcutoff_row)
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
        preferred_clusters_hi = 20,
        hdbscan_selection_mode    = "floor_soft_cap",
        hdbscan_overshoot_penalty = 0.01,
        cluster_hierarchy_enabled = True,
        cluster_hierarchy_linkage = "ward",
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
        seed_sweep_n          = 6,     # >0 = run cluster-stability seed sweep
        seed_sweep_n_jobs     = 1,     # T1.P: 1 = sequential; >1/-1 = parallel
        # Body-region weighting (issue 1b) / adaptive visibility (issue 2) /
        # iterative split+merge refinement (issue 4).  Engine DEFAULTS below
        # win when the core import succeeds; these are only the offline
        # fallback (see DEFAULTS overlay just below).
        visibility_features_enabled = True,
        visibility_adaptive_pct     = 10,
        hdbscan_merge_thresh         = 0.08,
        hdbscan_leaf_bonus           = 0.03,
        hdbscan_fine_bias            = 0.05,
        hdbscan_split_silhouette_thresh = 0.2,
        hdbscan_split_max_subclusters = 3,
        hdbscan_split_min_points     = 250,
        recluster_max_iterations     = 2,
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
        self._bodypart_weights: dict = {}

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
                 text="  Publication defaults are pre-set — change only for a specific reason.",
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
                 text="    Divide distances by nose-to-tailbase length (needs those bodyparts).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "PCA pre-reduction",
                 lambda r: _combo(r, "pca_pre_reduce",
                                  ["auto", "on", "off"], "auto"))
        tk.Label(s,
                 text="    auto = reduce when features ≥ samples/5  |  on = always  |  off = never",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "PCA pre-UMAP (inner)",
                 lambda r: _combo(r, "pca_n_components",
                                  ["auto", "off", "30", "50", "100"], "auto"))
        tk.Label(s,
                 text="    Separate inner PCA gate used by run_umap() itself. Independent of\n"
                      "    'PCA pre-reduction' above — leave at auto unless you need both.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Long lag drift (2-3 s)",
                 lambda r: _check(r, "long_lag_drift", False))
        tk.Label(s,
                 text="    Lag-offset features for sustained states (freezing, guarding). Off by default.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Long scale bins (500ms/1s)",
                 lambda r: _check(r, "long_scale_bins", False))
        tk.Label(s,
                 text="    Coarse temporal bins for slow sustained behaviours. Off by default.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Likelihood threshold",
                 lambda r: _spin_f(r, "likelihood_thresh", 0.0, 1.0, 0.05, 0.30))
        _adv_row(s, "Boxcar smooth (s)",
                 lambda r: _spin_f(r, "boxcar_win_sec", 0.0, 0.5, 0.01, 0.07))
        _adv_row(s, "Adaptive visibility features",
                 lambda r: _check(r, "visibility_features_enabled", True))
        tk.Label(s,
                 text="    ON: turned-away frames form their own cluster instead of polluting real behaviours.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Visibility adaptive percentile",
                 lambda r: _spin_f(r, "visibility_adaptive_pct", 1.0, 50.0, 1.0, 10.0))
        tk.Label(s,
                 text="    Per-bodypart low-confidence floor, layered on 'Likelihood threshold'.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Exclude turned-away frames from clustering",
                 lambda r: _check(r, "exclude_turned_away", True))
        tk.Label(s,
                 text="    ON: excludes turned-away bins from training and labels them 'Turned Away'.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Turned-away confidence threshold",
                 lambda r: _spin_f(r, "turned_away_conf_thresh", 0.0, 1.0, 0.05, 0.30))
        tk.Label(s,
                 text="    Low-confidence fraction that flags a bin as turned-away. 0.30 = validated default.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Auto-flag impure clusters  (opt-in)",
                 lambda r: _check(r, "auto_flag_impure_clusters", False))
        tk.Label(s,
                 text="    OFF by default. Folds low-silhouette clusters into 'Turned Away' — always\n"
                      "    check example clips before trusting a flagged cluster.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s, "Impure-cluster silhouette threshold  (0 = same as split)",
                 lambda r: _spin_f(r, "auto_flag_impure_silhouette_thresh", 0.0, 1.0, 0.05, 0.0))
        tk.Label(s,
                 text="    0 reuses 'Split silhouette thresh' below (0.20 default).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        self._bpw_status = tk.Label(s, text="Body-region weighting: disabled (uniform)",
                                     font=("Segoe UI", 7, "italic"),
                                     bg=C["card"], fg=C["dim"])
        tk.Button(s, text="Body-Region Weights (optional)...",
                  font=("Segoe UI", 9), bg=C["btn"], fg=C["yellow"],
                  relief="flat", padx=10, pady=4, cursor="hand2",
                  command=self._open_bodypart_weights).pack(anchor="w", padx=8, pady=(4, 0))
        self._bpw_status.pack(anchor="w", padx=8, pady=(0, 4))

        # Environmental context / object interaction / behavioral paradigms
        # now live in their own standalone EnvParadigmWindow (main window's
        # "Environments, Objects, Paradigms..." button), not here -- this
        # window no longer reads or writes env_features_enabled,
        # kinematic_directedness_enabled, env_arena_cfg, or
        # env_interaction_threshold, so there is only one place that can
        # change them.

        # ── UMAP ─────────────────────────────────────────────────────────────
        s2 = _adv_section(p, "UMAP EMBEDDING  (Hsu & Yttri 2021 reference)", C["cyan"])
        _adv_row(s2, "Full-data threshold",
                 lambda r: _spin_i(r, "umap_full_thresh", 1000, 100_000, 1000, 10_000))
        tk.Label(s2,
                 text="    Below this bin count, use all bins; larger recordings subsample at 'Train fraction'.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s2, "Train fraction",
                 lambda r: _spin_f(r, "train_frac", 0.05, 1.0, 0.05, 0.30))
        _adv_row(s2, "n_neighbors  (0 = auto)",
                 lambda r: _spin_i(r, "umap_n_neighbors", 0, 300, 5, 0))
        tk.Label(s2,
                 text="    0 = auto (scales with recording length). B-SOiD reference = 60.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s2, "n_components",
                 lambda r: _spin_i(r, "umap_n_components", 2, 10, 1, 3))
        _adv_row(s2, "min_dist",
                 lambda r: _spin_f(r, "umap_min_dist", 0.0, 1.0, 0.05, 0.10))
        tk.Label(s2,
                 text="    Recommended 0.1 — below 0.05 can destabilise HDBSCAN's density graph.",
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
                 text="    both = tries eom + leaf, DBCV picks best  |  eom = larger clusters  |  leaf = finer",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── Cluster count guidance ────────────────────────────────────────────
        s3b = _adv_section(p, "CLUSTER COUNT GUIDANCE", C["cyan"])
        _adv_row(s3b, "Target cluster count",
                 lambda r: _spin_i(r, "target_n_clusters", 0, 200, 1, 0))
        tk.Label(s3b,
                 text="    0 = auto (prefers 8–30 clusters). >0 steers Step 3 toward that count.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3b, "Preferred range — low",
                 lambda r: _spin_i(r, "preferred_clusters_lo", 2, 100, 1, 8))
        _adv_row(s3b, "Preferred range — high",
                 lambda r: _spin_i(r, "preferred_clusters_hi", 2, 200, 1, 30))
        tk.Label(s3b,
                 text="    Auto-mode picks the best DBCV solution in this range. Ignored if target > 0.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3b, "Selection mode",
                 lambda r: _combo(r, "hdbscan_selection_mode",
                                  ["legacy", "floor_soft_cap"], "floor_soft_cap"))
        tk.Label(s3b,
                 text="    floor_soft_cap (default): avoids the low bound, lightly penalises overshoot.\n"
                      "    legacy: pre-Aug-2026 rule, can collapse to very few clusters.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3b, "Overshoot penalty",
                 lambda r: _spin_f(r, "hdbscan_overshoot_penalty", 0.0, 1.0, 0.01, 0.01))
        tk.Label(s3b,
                 text="    Score penalty per cluster above the high bound (floor_soft_cap only).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3b, "Cluster hierarchy plot",
                 lambda r: _check(r, "cluster_hierarchy_enabled", True))
        _adv_row(s3b, "Hierarchy linkage method",
                 lambda r: _combo(r, "cluster_hierarchy_linkage",
                                  ["ward", "average", "complete"], "ward"))
        tk.Label(s3b,
                 text="    Saves a dendrogram (plots/cluster_hierarchy.png) to guide manual merging.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3b, "Min cluster frequency (%)",
                 lambda r: _spin_f(r, "min_cluster_freq", 0.0, 10.0, 0.1, 0.5))
        tk.Label(s3b,
                 text="    Clusters below this share of total time are pruned before MLP training.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── Iterative split + merge refinement (issue 4) ────────────────────────
        s3c = _adv_section(p, "ITERATIVE SPLIT / MERGE REFINEMENT  (advanced, on by default)", C["orange"])
        _adv_row(s3c, "Merge threshold  (0 = off)",
                 lambda r: _spin_f(r, "hdbscan_merge_thresh", 0.0, 1.0, 0.01, 0.0))
        tk.Label(s3c,
                 text="    0 = merge pass off. >0 merges clusters only weakly split from a shared parent.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3c, "Leaf method bonus",
                 lambda r: _spin_f(r, "hdbscan_leaf_bonus", 0.0, 0.5, 0.01, 0.03))
        tk.Label(s3c,
                 text="    Boosts leaf-method scores when merge threshold > 0 (merge then cleans up).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3c, "Fine-partition bias",
                 lambda r: _spin_f(r, "hdbscan_fine_bias", 0.0, 0.5, 0.01, 0.05))
        tk.Label(s3c,
                 text="    Nudges cluster count finer when merge threshold > 0, trusting merge to consolidate.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3c, "Split silhouette thresh  (0 = off)",
                 lambda r: _spin_f(r, "hdbscan_split_silhouette_thresh", -1.0, 1.0, 0.05, 0.0))
        tk.Label(s3c,
                 text="    Clusters below this mean silhouette are candidates for local re-clustering.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3c, "Max sub-clusters per split",
                 lambda r: _spin_i(r, "hdbscan_split_max_subclusters", 2, 20, 1, 3))
        tk.Label(s3c,
                 text="    Cap on pieces one impure cluster may split into per pass.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3c, "Min points to attempt a split",
                 lambda r: _spin_i(r, "hdbscan_split_min_points", 20, 2000, 10, 250))
        tk.Label(s3c,
                 text="    Smaller candidate clusters are left untouched (too few points to re-embed).",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s3c, "Max refine iterations",
                 lambda r: _spin_i(r, "recluster_max_iterations", 0, 10, 1, 2))
        tk.Label(s3c,
                 text="    Cap on the split → merge → repeat loop.",
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
                 text="    Smooths MLP output with an HMM, removing single-frame flicker from jitter.\n"
                      "    States = 0 uses n_clusters (smoothing only); fewer states = macro-behaviours.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── Reproducibility & methodology ─────────────────────────────────────
        s_rep = _adv_section(p, "REPRODUCIBILITY & METHODOLOGY", C["cyan"])
        _adv_row(s_rep, "Compatibility mode",
                 lambda r: _combo(r, "compat_mode",
                                  ["current", "legacy_v2"], "current"))
        tk.Label(s_rep,
                 text="    current = v2.1 corrected behaviour (recommended)  |  legacy_v2 = reproduce pre-2.1 runs",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_rep, "Cluster-stability seed sweep  (0 = off, default 6)",
                 lambda r: _spin_i(r, "seed_sweep_n", 0, 50, 1, 6))
        tk.Label(s_rep,
                 text="    >0 re-runs over this many seeds to measure partition stability. Adds runtime.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _cpu_n = os.cpu_count() or 32
        _adv_row(s_rep, "Seed sweep parallel jobs",
                 lambda r: _spin_i(r, "seed_sweep_n_jobs", -1, _cpu_n, 1, -1))
        tk.Label(s_rep,
                 text="    -1 = auto-managed (default, see System Resources below)  |  1 = sequential (safest)  |  N = pin exact count.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_rep, "Consensus auto-trigger threshold  (0 = off, default 0.55)",
                 lambda r: _spin_f(r, "consensus_auto_threshold", 0.0, 1.0, 0.01, 0.55))
        tk.Label(s_rep,
                 text="    Below this seed-sweep ARI, consensus clustering auto-enables. 0 = never auto.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_rep, "Enable consensus clustering  (force on)",
                 lambda r: _check(r, "consensus_clustering_enabled", False))
        tk.Label(s_rep,
                 text="    Manual override — always use consensus, regardless of the threshold above.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))

        # ── System resources ────────────────────────────────────────────────────
        # The primary HDBSCAN min_cluster_size sweep used to run sequentially
        # regardless of core count — most cores sat idle for most of a run's
        # wall-clock time. These settings let the engine detect this machine's
        # cores/RAM at run start and size its own parallelism within a safe
        # band, re-checking RAM before each heavy stage so it only ever shrinks
        # under memory pressure, never risks an OOM crash.
        s_sys = _adv_section(p, "SYSTEM RESOURCES", C["green"])
        _adv_row(s_sys, "Auto-manage CPU/memory usage",
                 lambda r: _check(r, "auto_resource_management", True))
        tk.Label(s_sys,
                 text="    Detects cores/RAM at run start; re-checks before each heavy stage. Recommended on.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_sys, "Target utilization  (fraction of cores, default 0.65)",
                 lambda r: _spin_f(r, "system_resource_target_pct", 0.50, 0.70, 0.01, 0.65))
        tk.Label(s_sys,
                 text="    Ideal sustained load during the HDBSCAN sweep — the 60-70% band.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_sys, "Hard cap  (fraction of cores, default 0.80)",
                 lambda r: _spin_f(r, "system_resource_cap_pct", 0.70, 0.90, 0.01, 0.80))
        tk.Label(s_sys,
                 text="    Never exceeded regardless of core count, even under auto-management.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_sys, "HDBSCAN sweep parallel jobs",
                 lambda r: _spin_i(r, "hdbscan_sweep_n_jobs", -1, _cpu_n, 1, -1))
        tk.Label(s_sys,
                 text="    -1 = auto-managed (recommended)  |  1 = sequential (safest)  |  N = pin exact worker count.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s_sys, "Recluster split parallel jobs",
                 lambda r: _spin_i(r, "hdbscan_split_n_jobs", -1, _cpu_n, 1, -1))
        tk.Label(s_sys,
                 text="    Same -1/1/N semantics, for the impure-cluster re-split pass.",
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
                 text="    Auto-exports this many random evolution videos at the end of Step 3.",
                 font=("Segoe UI", 7), bg=C["card"],
                 fg=C["dim"]).pack(anchor="w", padx=8, pady=(0, 2))
        _adv_row(s5, "Plot theme",
                 lambda r: _combo(r, "plot_theme", ["dark", "light"], "light"))
        tk.Label(s5,
                 text="    dark = white-on-black figures  |  light = publication-ready background",
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
        self._bodypart_weights = dict(cfg.get("bodypart_weights") or {})
        self._refresh_bpw_status()

    def _refresh_bpw_status(self):
        if not hasattr(self, "_bpw_status"):
            return
        if self._bodypart_weights:
            n = len(self._bodypart_weights)
            self._bpw_status.configure(
                text=f"Body-region weighting: enabled ({n} bodypart weight(s) set)")
        else:
            self._bpw_status.configure(
                text="Body-region weighting: disabled (uniform)")

    def _open_bodypart_weights(self):
        try:
            # Pre-fill from this project's own weights if it has any;
            # otherwise offer the last weights applied in ANY project as a
            # starting point (still requires the user to click Apply in the
            # editor below before it's committed to this project's cfg).
            _prefill = self._bodypart_weights or _load_saved_body_region_weights()
            win = BodyPartWeightWindow(self, self._session, _prefill)
            self.wait_window(win)
            result = getattr(win, "result", None)
            if result is not None:      # None = Cancel; keep previous weights
                self._bodypart_weights = dict(result)
            self._refresh_bpw_status()
        except Exception as e:
            messagebox.showerror("Body-Region Weights",
                                  f"Could not open the body-region weight editor:\n{e}")

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
        # Only persist keys whose value differs from the effective default
        # (same {baseline, then BSoidEngine.DEFAULTS} merge _load() seeds the
        # widgets from). Previously every widget was written unconditionally,
        # so engine_cfg always contained all ~80 keys -- including untouched
        # ones like consensus_clustering_enabled=False -- which made
        # BSoidEngine's self._explicit_cfg_keys (used by the consensus
        # auto-trigger's opt-out check) treat every default as "the user
        # deliberately set this," permanently defeating that auto-trigger.
        try:
            effective = {**self._BASELINE, **dict(BSoidEngine.DEFAULTS)}
        except Exception:
            effective = dict(self._BASELINE)
        cfg = {}
        for k, var in self._vars.items():
            try:
                val = var.get()
            except Exception:
                continue
            default = effective.get(k, val)
            if default is None or val != default:
                cfg[k] = val
        cfg["bodypart_weights"] = getattr(self, "_bodypart_weights", {}) or {}
        # This window rebuilds engine_cfg from scratch above, but the four
        # environments/objects/paradigms keys are now owned exclusively by
        # EnvParadigmWindow (not by self._vars) -- carry over whatever it
        # last wrote so Apply here doesn't silently erase that config.
        _prior = self._session.get("engine_cfg", {}) or {}
        for _k in ("env_features_enabled", "kinematic_directedness_enabled",
                   "env_arena_cfg", "env_interaction_threshold"):
            if _k in _prior:
                cfg[_k] = _prior[_k]
        self._session["engine_cfg"] = cfg
        self.destroy()


#
#  ENVIRONMENTAL CONTEXT & OBJECT INTERACTION WINDOW  (v6 part 2)
#

_ENV_PARADIGM_LABELS = {
    "open_field":         "Open Field",
    "novel_object":       "Novel Object / Animal Recognition",
    "y_maze":             "Y-Maze",
    "elevated_plus_maze": "Elevated Plus Maze",
    "three_chamber":      "Three-Chamber Test",
    "place_preference":   "Place Preference / CPP",
    "custom":             "Custom / Other Arena",
}
# Primary tool(s) shown expanded by default; the other tool is tucked under
# an "Advanced" toggle (still fully usable, never hidden). "both" = neither
# is secondary (three_chamber, custom).
_ENV_PRIMARY_TOOL = {
    "open_field": "region", "novel_object": "object", "y_maze": "region",
    "elevated_plus_maze": "region", "three_chamber": "both",
    "place_preference": "region", "custom": "both",
}
_ENV_NAME_SUGGESTIONS = {
    "open_field":         {"region": ["Center"], "object": []},
    "novel_object":       {"region": [], "object": ["Object A", "Object B"]},
    "y_maze":             {"region": ["Arm A", "Arm B", "Arm C", "Center"], "object": []},
    "elevated_plus_maze": {"region": ["Open Arm A", "Open Arm B", "Closed Arm A",
                                       "Closed Arm B", "Center"], "object": []},
    "three_chamber":      {"region": ["Left Chamber", "Center Chamber", "Right Chamber"],
                            "object": ["Cup A", "Cup B"]},
    "place_preference":   {"region": ["Chamber A", "Chamber B", "Middle/Neutral"], "object": []},
    "custom":             {"region": [], "object": []},
}
_ENV_BOUNDARY_HINT = {
    "open_field": "circle (typical open-field arena)",
    "y_maze": "polygon (angular multi-arm footprint)",
    "elevated_plus_maze": "polygon (angular multi-arm footprint)",
    "three_chamber": "polygon (angular multi-chamber footprint)",
    "place_preference": "polygon (angular multi-chamber footprint)",
    "novel_object": "circle or polygon -- whatever matches your arena",
    "custom": "circle or polygon -- whatever matches your arena",
}
_ENV_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
_ENV_PARADIGM_DESCRIPTIONS = {
    "open_field":         "General region/object time -- no paradigm-specific index.",
    "novel_object":       "Discrimination index from time near 'novel' vs 'familiar' objects.",
    "y_maze":             "Spontaneous alternation % from entries into 3 named 'arm' regions.",
    "elevated_plus_maze": "% time and entries into 'open_arm' vs 'closed_arm' regions.",
    "three_chamber":      "Sociability/social-novelty indices from 'stranger'/'empty'/'novel_stranger' objects.",
    "place_preference":   "Preference index from time in 'paired' vs 'unpaired' regions.",
    "custom":             "Generic per-region/per-object time and entries, no derived index.",
}


class EnvParadigmWindow(tk.Toplevel):
    """
    Standalone hub for environmental context, object interaction, and
    behavioral paradigms -- one click from the main window rather than
    nested inside AdvancedCUBEWindow. Owns env_features_enabled,
    kinematic_directedness_enabled, env_arena_cfg, and
    env_interaction_threshold exclusively: AdvancedCUBEWindow no longer
    reads or writes any of these four engine_cfg keys, so there is only one
    place that can change them.

    Arena/region/object tracing itself is unchanged -- this window just
    opens the existing EnvContextWindow for that part.
    """

    def __init__(self, parent, session: "SessionState"):
        super().__init__(parent)
        self.title("Environments, Objects, Paradigms")
        self.configure(bg=C["bg"])
        self.geometry("560x420")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._session = session
        cfg = session.get("engine_cfg", {}) or {}
        self._env_arena_cfg = cfg.get("env_arena_cfg") or None

        pad = dict(padx=10, pady=4)

        tk.Label(self, text="  Environments, Objects & Paradigms",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["bg"], fg=C["cyan"]).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(self,
                 text="  Both opt-in and off by default -- with both unticked, output is "
                      "byte-identical to a run without this feature.",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["subtext"],
                 justify="left").pack(anchor="w", padx=10, pady=(0, 8))

        # ── Environmental context ───────────────────────────────────────────
        self._env_v = tk.BooleanVar(value=bool(cfg.get("env_features_enabled", False)))
        row = tk.Frame(self, bg=C["bg"])
        row.pack(fill="x", **pad)
        tk.Checkbutton(row, text="Enable environmental context (arena / regions / objects)",
                       variable=self._env_v, bg=C["bg"], fg=C["green"],
                       selectcolor=C["card"], activebackground=C["bg"],
                       command=self._refresh_status).pack(side="left")
        tk.Label(self,
                 text="  Trace an arena and pick a paradigm below for region/object time "
                      "and paradigm-specific indices (alternation %, discrimination index, etc).",
                 font=("Segoe UI", 7), bg=C["bg"],
                 fg=C["dim"]).pack(anchor="w", padx=10, pady=(0, 4))

        # ── Kinematic directedness ──────────────────────────────────────────
        self._kin_v = tk.BooleanVar(value=bool(cfg.get("kinematic_directedness_enabled", False)))
        row = tk.Frame(self, bg=C["bg"])
        row.pack(fill="x", **pad)
        tk.Checkbutton(row, text="Enable kinematic directedness (per-bout straightness/speed/heading)",
                       variable=self._kin_v, bg=C["bg"], fg=C["green"],
                       selectcolor=C["card"], activebackground=C["bg"]).pack(side="left")
        tk.Label(self,
                 text="  Enabling this together with environmental context also unlocks "
                      "approach/avoid object-interaction events.",
                 font=("Segoe UI", 7), bg=C["bg"],
                 fg=C["dim"]).pack(anchor="w", padx=10, pady=(0, 4))

        # ── Interaction threshold ───────────────────────────────────────────
        thr_row = tk.Frame(self, bg=C["bg"])
        thr_row.pack(fill="x", **pad)
        tk.Label(thr_row, text="Interaction threshold (px, blank = auto):",
                 font=("Segoe UI", 9), bg=C["bg"], fg=C["text"]).pack(side="left")
        _thr = cfg.get("env_interaction_threshold")
        self._env_thresh_var = tk.StringVar(value="" if _thr is None else str(_thr))
        tk.Entry(thr_row, textvariable=self._env_thresh_var, width=8,
                 bg=C["card2"], fg=C["text"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(6, 8))
        tk.Button(thr_row, text="Preview auto value", font=("Segoe UI", 8),
                  bg=C["btn"], fg=C["subtext"], relief="flat", padx=8, pady=2,
                  cursor="hand2", command=self._preview_threshold).pack(side="left")
        self._thr_preview_lbl = tk.Label(self, text="", font=("Segoe UI", 7, "italic"),
                                          bg=C["bg"], fg=C["dim"])
        self._thr_preview_lbl.pack(anchor="w", padx=10, pady=(0, 4))

        # ── Arena / paradigm configuration ──────────────────────────────────
        _hdr = tk.Frame(self, bg=C["bg"])
        _hdr.pack(fill="x", pady=(8, 0), padx=10)
        tk.Frame(_hdr, bg=C["dim"], height=1).pack(fill="x")
        tk.Button(self, text="Configure Arena, Regions & Objects...",
                  font=("Segoe UI", 9), bg=C["btn"], fg=C["yellow"],
                  relief="flat", padx=10, pady=4, cursor="hand2",
                  command=self._open_arena_editor).pack(anchor="w", padx=10, pady=(8, 0))
        self._status_lbl = tk.Label(self, text="", font=("Segoe UI", 7, "italic"),
                                     bg=C["bg"], fg=C["dim"])
        self._status_lbl.pack(anchor="w", padx=10, pady=(2, 4))

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(side="bottom", fill="x", pady=8, padx=12)
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Apply & Close", font=("Segoe UI", 10, "bold"),
                  bg=C["purple"], fg="white", relief="flat",
                  padx=16, pady=5, cursor="hand2",
                  command=self._apply).pack(side="right")

        self._refresh_status()

    def _refresh_status(self):
        if not self._env_v.get():
            self._status_lbl.configure(text="Arena: not configured (enable the checkbox above)")
            return
        cfg = self._env_arena_cfg or {}
        n_reg = len((cfg.get("reference_shapes") or {}).get("regions") or [])
        n_obj = len((cfg.get("reference_shapes") or {}).get("objects") or [])
        has_bound = bool((cfg.get("reference_shapes") or {}).get("boundary"))
        if not cfg or (n_reg == 0 and n_obj == 0 and not has_bound):
            self._status_lbl.configure(text="Arena: not configured yet")
        else:
            self._status_lbl.configure(
                text=f"Arena: {cfg.get('paradigm', 'custom')} paradigm, "
                     f"{n_reg} region(s), {n_obj} object(s)"
                     f"{' + boundary' if has_bound else ''}")

    def _open_arena_editor(self):
        try:
            win = EnvContextWindow(self, self._session, self._env_arena_cfg)
            self.wait_window(win)
            result = getattr(win, "result", None)
            if result is not None:      # None = Cancel; keep previous config
                self._env_arena_cfg = result
            self._refresh_status()
        except Exception as e:
            messagebox.showerror("Environments, Objects, Paradigms",
                                  f"Could not open the arena editor:\n{e}")

    def _find_first_dlc_file(self):
        """Mirrors BodyPartWeightWindow._find_first_dlc_file's search order."""
        candidates = []
        try:
            candidates.extend(self._session.get("bsoid_ready_dirs") or [])
        except Exception:
            pass
        try:
            candidates.extend(self._session.get("video_folders") or [])
        except Exception:
            pass
        for folder in candidates:
            try:
                if not folder or not Path(folder).is_dir():
                    continue
                files = find_dlc_files(folder) if find_dlc_files else []
                if files:
                    return files[0]
            except Exception:
                continue
        return None

    def _preview_threshold(self):
        if load_dlc_file is None or _find_spine_indices is None or _spine_norm_factor is None:
            self._thr_preview_lbl.configure(text="Preview requires cube_core (not loaded).")
            return
        dlc_path = self._find_first_dlc_file()
        if dlc_path is None:
            self._thr_preview_lbl.configure(
                text="No DLC file found yet -- run Step 1/2 first, then reopen to preview.")
            return
        try:
            xy, bodyparts, _fps = load_dlc_file(dlc_path)
            head_idx, tail_idx = _find_spine_indices(bodyparts)
            xs, ys = xy[:, 0::2], xy[:, 1::2]
            if head_idx is not None and tail_idx is not None:
                import numpy as _np
                val = float(_np.nanmedian(_spine_norm_factor(xs, ys, head_idx, tail_idx)))
            else:
                val = 50.0
            self._thr_preview_lbl.configure(
                text=f"Auto-derived value for this session would be ~{val:.1f} px.")
        except Exception as e:
            self._thr_preview_lbl.configure(text=f"Could not compute preview: {e}")

    def _apply(self):
        cfg = dict(self._session.get("engine_cfg", {}) or {})

        if self._env_v.get():
            cfg["env_features_enabled"] = True
        else:
            cfg.pop("env_features_enabled", None)

        if self._kin_v.get():
            cfg["kinematic_directedness_enabled"] = True
        else:
            cfg.pop("kinematic_directedness_enabled", None)

        if self._env_arena_cfg:
            cfg["env_arena_cfg"] = self._env_arena_cfg
        else:
            cfg.pop("env_arena_cfg", None)

        _thr_raw = self._env_thresh_var.get().strip()
        if _thr_raw:
            try:
                _thr_val = float(_thr_raw)
                if _thr_val <= 0:
                    raise ValueError("threshold must be positive")
                cfg["env_interaction_threshold"] = _thr_val
            except ValueError:
                messagebox.showwarning(
                    "Interaction Threshold",
                    f"'{_thr_raw}' is not a valid positive number -- ignoring it. "
                    "The threshold will be auto-derived instead.")
                cfg.pop("env_interaction_threshold", None)
        else:
            cfg.pop("env_interaction_threshold", None)

        self._session["engine_cfg"] = cfg
        self.destroy()


class EnvContextWindow(tk.Toplevel):
    """
    Step 2 (Environmental_Context_v6_Implementation_Plan.md): paradigm-first,
    progressively-disclosed arena/region/object tracing window. Opened from
    AdvancedCUBEWindow, gated on the env_features_enabled checkbox.

    Leads with a mandatory paradigm choice (_build_paradigm_screen); once
    chosen, Tab 1 ("Define Reference Arena") lets the user trace a boundary/
    regions/objects on a reference video frame, and Tab 2 ("Apply to Other
    Videos") lets them align those reference shapes (translate, or a full
    per-shape override) onto every other paired video.

    Coordinate-space note (ground rule 8 / user-confirmed design decision,
    see cube_core.compute_session_env_context's docstring): frames are read
    here through the SAME dlc_crop_x/y/w/h rectangle DLC tracked on (see
    _load_frame_for), so traced vertices land in the same pixel coordinate
    space as the pose data by construction -- this window IS the structural
    correctness guarantee, not a runtime assertion elsewhere.

    Scope trims relative to the plan's full interaction spec (documented
    explicitly, not silent): shapes are traced by clicking vertices in order
    then clicking "Finish Shape" (no free-hand drag-drawing); the Tab 2
    "Edit individual shapes" override mode supports independently
    click-dragging each whole shape (translate per shape) rather than
    per-vertex handle reshaping. Both are real, working interactions -- the
    trim is in interaction richness, not in whether the feature works.
    """

    def __init__(self, parent, session: "SessionState", initial_cfg: "dict | None"):
        super().__init__(parent)
        self.title("Environmental Context & Object Interaction")
        self.configure(bg=C["bg"])
        self.geometry("1100x800")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self.result = None

        self._session = session
        self._cfg: "dict | None" = self._clone_cfg(initial_cfg) if initial_cfg else None

        # Tab 1 canvas/drawing state
        self._ref_frame_img = None      # PIL.Image, crop-applied, full-res
        self._photo = None
        self._scale = 1.0
        self._canvas_offset = (0, 0)
        self._drawing_kind = None       # None | "boundary" | "region" | "object"
        self._draw_points: list = []    # original-pixel-space points while drawing
        self._advanced_shown = False
        self._selected_shape_key = None  # ("boundary"|"regions"|"objects", index)

        # Tab 2 state
        self._other_videos: list = []   # [(stem, path), ...]
        self._active_video_stem = None
        self._tab2_photo = None
        self._tab2_editing_shapes = False
        self._tab2_drag_last = None
        self._tab2_drag_shape_idx = None  # (list_key, idx) when dragging one shape in override mode

        self._tabs_built = False

        if self._cfg and self._cfg.get("paradigm"):
            self._build_main_ui()
        else:
            self._build_paradigm_screen()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clone_cfg(cfg: dict) -> dict:
        import copy
        return copy.deepcopy(cfg)

    def _new_cfg(self, paradigm: str, reference_stem: str) -> dict:
        return {
            "schema_version": 3,
            "paradigm": paradigm,
            "reference_stem": reference_stem,
            "coord_space": "post_crop",
            "reference_shapes": {"boundary": None, "regions": [], "objects": []},
            "per_video": {},
        }

    def _dlc_crop_rect(self):
        adv = self._session.get("dlc_advanced_cfg", {}) or {}
        if not bool(adv.get("dlc_crop_enable", False)):
            return 0, 0, 0, 0
        rx, ry = int(adv.get("dlc_crop_x", 0)), int(adv.get("dlc_crop_y", 0))
        rw, rh = int(adv.get("dlc_crop_w", 0)), int(adv.get("dlc_crop_h", 0))
        return rx, ry, rw, rh

    def _list_videos(self):
        folders = self._session.get("video_folders", [])
        out = []
        seen = set()
        for root_folder in folders:
            for sub, dirs, files in os.walk(root_folder):
                dirs[:] = [d for d in dirs if not d.endswith("_results")]
                if Path(sub).name.endswith("_results"):
                    continue
                for fname in sorted(files):
                    if fname.startswith("resized_"):
                        continue
                    p = Path(fname)
                    if p.suffix.lower() in _ENV_VIDEO_EXTS:
                        stem = p.stem
                        if stem not in seen:
                            seen.add(stem)
                            out.append((stem, os.path.join(sub, fname)))
        return out

    def _load_frame_for(self, video_path: str):
        """Read one representative frame, CROP-APPLIED (dlc_crop_x/y/w/h) so
        pixel (0,0) here matches DLC's own tracked coordinate origin -- the
        structural coordinate-space guarantee this window is responsible for.
        Returns a PIL.Image or None."""
        try:
            import cv2
            from PIL import Image
        except ImportError:
            return None
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = max(0, total // 3) if total > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                return None
        finally:
            cap.release()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rx, ry, rw, rh = self._dlc_crop_rect()
        if rw > 0 and rh > 0:
            img = img.crop((rx, ry, min(rx + rw, img.width), min(ry + rh, img.height)))
        return img

    # ── paradigm screen ─────────────────────────────────────────────────────

    def _build_paradigm_screen(self):
        for w in self.winfo_children():
            w.destroy()
        tk.Label(self, text="  Choose the behavioral paradigm for this project",
                 font=("Segoe UI", 13, "bold"), bg=C["bg"], fg=C["cyan"]
                 ).pack(anchor="w", padx=10, pady=(16, 4))
        tk.Label(self,
                 text="  Only one paradigm is configured per project -- this determines which\n"
                      "  drawing tools, naming prompts, and derived metrics are available below.\n"
                      "  A batch mixing experiment types needs separate CUBE projects.",
                 font=("Segoe UI", 9), bg=C["bg"], fg=C["subtext"]
                 ).pack(anchor="w", padx=10, pady=(0, 12))

        card = tk.Frame(self, bg=C["card"])
        card.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        # Pre-select the CURRENT paradigm when re-entering via "Change
        # Paradigm" on an existing config, rather than always defaulting to
        # "open_field" -- a user who re-opens this screen just to look
        # should see their actual current choice highlighted, not a reset.
        _current_paradigm = (self._cfg.get("paradigm") if self._cfg else None) or "open_field"
        var = tk.StringVar(value=_current_paradigm)
        for key in ENV_PARADIGMS or list(_ENV_PARADIGM_LABELS.keys()):
            row = tk.Frame(card, bg=C["card"])
            row.pack(anchor="w", fill="x", padx=12, pady=4)
            tk.Radiobutton(row, text=_ENV_PARADIGM_LABELS.get(key, key), variable=var,
                           value=key, bg=C["card"], fg=C["text"],
                           selectcolor=C["card2"], activebackground=C["card"],
                           font=("Segoe UI", 10)).pack(side="left")
            _desc = _ENV_PARADIGM_DESCRIPTIONS.get(key, "")
            if _desc:
                tk.Label(row, text=_desc, font=("Segoe UI", 7),
                         bg=C["card"], fg=C["dim"]).pack(side="left", padx=(10, 0))

        others = self._list_videos()
        ref_stem = others[0][0] if others else "reference"

        def _confirm():
            # Switching paradigm on an EXISTING config must never delete
            # already-traced reference_shapes/per_video (per the plan's Step 1
            # non-destructive-switch requirement, and this screen's own
            # confirmation dialog text in _change_paradigm below) -- only
            # build a fresh config from scratch on true first-time setup.
            if self._cfg:
                self._cfg["paradigm"] = var.get()
            else:
                self._cfg = self._new_cfg(var.get(), ref_stem)
            self._build_main_ui()

        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(side="bottom", fill="x", padx=16, pady=10)
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat", padx=10, pady=5,
                  cursor="hand2", command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Continue  ->", font=("Segoe UI", 10, "bold"),
                  bg=C["green"], fg="white", relief="flat", padx=16, pady=5,
                  cursor="hand2", command=_confirm).pack(side="right")

    def _change_paradigm(self):
        if messagebox.askyesno(
                "Change Paradigm",
                "Switching paradigms does not delete anything already traced -- "
                "it only changes which shapes/tools are shown and which derived "
                "metrics are computed. Continue?"):
            self._build_paradigm_screen()

    # ── main UI (paradigm chosen) ───────────────────────────────────────────

    def _build_main_ui(self):
        for w in self.winfo_children():
            w.destroy()
        paradigm = self._cfg.get("paradigm", "custom")

        header = tk.Frame(self, bg=C["bg"])
        header.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(header, text=f"  {_ENV_PARADIGM_LABELS.get(paradigm, paradigm)} session",
                 font=("Segoe UI", 11, "bold"), bg=C["bg"], fg=C["yellow"]).pack(side="left")
        tk.Button(header, text="Change Paradigm", font=("Segoe UI", 8),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat", padx=8, pady=2,
                  cursor="hand2", command=self._change_paradigm).pack(side="left", padx=10)

        tabbar = tk.Frame(self, bg=C["bg"])
        tabbar.pack(fill="x", padx=10, pady=(6, 0))
        self._tab1_btn = tk.Button(tabbar, text="1. Define Reference Arena",
                                    font=("Segoe UI", 9, "bold"), relief="flat",
                                    padx=10, pady=5, cursor="hand2",
                                    command=lambda: self._show_tab(1))
        self._tab2_btn = tk.Button(tabbar, text="2. Apply to Other Videos",
                                    font=("Segoe UI", 9, "bold"), relief="flat",
                                    padx=10, pady=5, cursor="hand2",
                                    command=lambda: self._show_tab(2))
        self._tab1_btn.pack(side="left", padx=(0, 4))
        self._tab2_btn.pack(side="left")

        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=(6, 0))
        self._tab1_frame = tk.Frame(body, bg=C["bg"])
        self._tab2_frame = tk.Frame(body, bg=C["bg"])

        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(side="bottom", fill="x", padx=16, pady=10)
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat", padx=10, pady=5,
                  cursor="hand2", command=self.destroy).pack(side="left")
        tk.Button(btn_f, text="Save as Template...", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["yellow"], relief="flat", padx=10, pady=5,
                  cursor="hand2", command=self._save_template).pack(side="left", padx=6)
        tk.Button(btn_f, text="Load Template...", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["yellow"], relief="flat", padx=10, pady=5,
                  cursor="hand2", command=self._load_template).pack(side="left", padx=6)
        tk.Button(btn_f, text="  Apply  ", font=("Segoe UI", 10, "bold"),
                  bg=C["green"], fg="white", relief="flat", padx=16, pady=5,
                  cursor="hand2", command=self._apply_and_close).pack(side="right")

        self._build_tab1(self._tab1_frame)
        self._build_tab2(self._tab2_frame)
        self._show_tab(1)

    def _show_tab(self, n: int):
        self._tab1_frame.pack_forget()
        self._tab2_frame.pack_forget()
        active, inactive = (C["green"], C["btn"])
        if n == 1:
            self._tab1_frame.pack(fill="both", expand=True)
            self._tab1_btn.configure(bg=active, fg="white")
            self._tab2_btn.configure(bg=inactive, fg=C["btn_fg"])
        else:
            self._tab2_frame.pack(fill="both", expand=True)
            self._tab2_btn.configure(bg=active, fg="white")
            self._tab1_btn.configure(bg=inactive, fg=C["btn_fg"])
            self._refresh_tab2_list()

    # ── Tab 1: Define Reference Arena ───────────────────────────────────────

    def _build_tab1(self, parent):
        left = tk.Frame(parent, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(parent, bg=C["card"], width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        paradigm = self._cfg.get("paradigm", "custom")
        primary = _ENV_PRIMARY_TOOL.get(paradigm, "both")
        boundary_hint = _ENV_BOUNDARY_HINT.get(paradigm, "circle or polygon")

        toolbar = tk.Frame(left, bg=C["bg"])
        toolbar.pack(fill="x", pady=(0, 4))
        tk.Button(toolbar, text="Add Boundary", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["cyan"], relief="flat", padx=8, pady=4,
                  cursor="hand2", command=lambda: self._start_drawing("boundary")
                  ).pack(side="left", padx=(0, 4))
        tk.Label(toolbar, text=f"(suggested: {boundary_hint})",
                 font=("Segoe UI", 7), bg=C["bg"], fg=C["dim"]).pack(side="left", padx=(0, 10))

        self._primary_btn_frame = tk.Frame(toolbar, bg=C["bg"])
        self._primary_btn_frame.pack(side="left")

        def _mk_tool_btn(f, kind, label):
            tk.Button(f, text=label, font=("Segoe UI", 9),
                      bg=C["btn"], fg=C["green"], relief="flat", padx=8, pady=4,
                      cursor="hand2", command=lambda: self._start_drawing(kind)
                      ).pack(side="left", padx=(0, 4))

        if primary in ("region", "both"):
            _mk_tool_btn(self._primary_btn_frame, "region", "Add Region")
        if primary in ("object", "both"):
            _mk_tool_btn(self._primary_btn_frame, "object", "Add Object")

        self._adv_toggle_btn = None
        self._adv_btn_frame = tk.Frame(toolbar, bg=C["bg"])
        if primary not in ("both",):
            self._adv_toggle_btn = tk.Button(
                toolbar, text="Advanced ▾", font=("Segoe UI", 8),
                bg=C["bg"], fg=C["subtext"], relief="flat", padx=6, pady=4,
                cursor="hand2", command=self._toggle_advanced)
            self._adv_toggle_btn.pack(side="left", padx=(6, 0))
            self._adv_btn_frame.pack(side="left")
            secondary = "object" if primary == "region" else "region"
            _mk_tool_btn(self._adv_btn_frame, secondary,
                         f"Add {'Object' if secondary == 'object' else 'Region'}")
            self._adv_btn_frame.pack_forget()

        finish_row = tk.Frame(left, bg=C["bg"])
        finish_row.pack(fill="x")
        self._draw_status = tk.Label(finish_row, text="", font=("Segoe UI", 8),
                                      bg=C["bg"], fg=C["yellow"])
        self._draw_status.pack(side="left")
        tk.Button(finish_row, text="Finish Shape", font=("Segoe UI", 8),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat", padx=6, pady=2,
                  cursor="hand2", command=self._finish_shape).pack(side="left", padx=6)
        tk.Button(finish_row, text="Cancel Drawing", font=("Segoe UI", 8),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat", padx=6, pady=2,
                  cursor="hand2", command=self._cancel_drawing).pack(side="left")

        self._tab1_canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self._tab1_canvas.pack(fill="both", expand=True, pady=(4, 0))
        self._tab1_canvas.bind("<Configure>", lambda e: self._refresh_tab1_canvas())
        self._tab1_canvas.bind("<Button-1>", self._on_tab1_click)

        others = self._list_videos()
        ref_video = next((p for s, p in others if s == self._cfg.get("reference_stem")),
                          others[0][1] if others else None)
        self._ref_frame_img = self._load_frame_for(ref_video) if ref_video else None

        tk.Label(right, text="Traced Shapes", font=("Segoe UI", 10, "bold"),
                 bg=C["card"], fg=C["cyan"]).pack(anchor="w", padx=8, pady=(8, 4))
        self._sidebar_canvas = tk.Canvas(right, bg=C["card"], highlightthickness=0)
        _sb_scroll = tk.Scrollbar(right, orient="vertical", command=self._sidebar_canvas.yview)
        self._sidebar_inner = tk.Frame(self._sidebar_canvas, bg=C["card"])
        self._sidebar_canvas.configure(yscrollcommand=_sb_scroll.set)
        self._sidebar_canvas.pack(side="left", fill="both", expand=True, padx=(4, 0))
        _sb_scroll.pack(side="right", fill="y")
        self._sidebar_canvas.create_window((0, 0), window=self._sidebar_inner, anchor="nw")
        self._sidebar_inner.bind(
            "<Configure>",
            lambda e: self._sidebar_canvas.configure(scrollregion=self._sidebar_canvas.bbox("all")))

        self._refresh_sidebar()
        self._refresh_tab1_canvas()

    def _toggle_advanced(self):
        self._advanced_shown = not self._advanced_shown
        if self._advanced_shown:
            self._adv_btn_frame.pack(side="left")
            self._adv_toggle_btn.configure(text="Advanced ▴")
        else:
            self._adv_btn_frame.pack_forget()
            self._adv_toggle_btn.configure(text="Advanced ▾")

    def _start_drawing(self, kind: str):
        if kind == "boundary" and self._cfg["reference_shapes"].get("boundary"):
            if not messagebox.askyesno("Replace Boundary",
                                        "A boundary is already traced. Replace it?"):
                return
        self._drawing_kind = kind
        self._draw_points = []
        self._draw_status.configure(
            text=f"Drawing {kind}: click points on the frame, then 'Finish Shape' "
                 f"(need >= 3 points).")
        self._refresh_tab1_canvas()

    def _cancel_drawing(self):
        self._drawing_kind = None
        self._draw_points = []
        self._draw_status.configure(text="")
        self._refresh_tab1_canvas()

    def _canvas_to_orig(self, cx, cy):
        ox, oy = self._canvas_offset
        s = self._scale or 1.0
        return (cx - ox) / s, (cy - oy) / s

    def _orig_to_canvas(self, x, y):
        ox, oy = self._canvas_offset
        s = self._scale or 1.0
        return x * s + ox, y * s + oy

    def _on_tab1_click(self, event):
        if not self._drawing_kind or self._ref_frame_img is None:
            return
        ox, oy = self._canvas_to_orig(event.x, event.y)
        self._draw_points.append((ox, oy))
        self._refresh_tab1_canvas()

    def _suggested_name(self, kind: str) -> str:
        paradigm = self._cfg.get("paradigm", "custom")
        existing = {s["name"] for s in (self._cfg["reference_shapes"].get(
            "regions" if kind == "region" else "objects") or [])}
        suggestions = _ENV_NAME_SUGGESTIONS.get(paradigm, {}).get(kind, [])
        for s in suggestions:
            if s not in existing:
                return s
        return f"{'Region' if kind == 'region' else 'Object'} {len(existing) + 1}"

    def _role_vocab(self, kind: str):
        paradigm = self._cfg.get("paradigm", "custom")
        vocab = (ENV_PARADIGM_ROLE_VOCAB or {}).get(paradigm, {})
        key = "regions" if kind == "region" else "objects"
        return vocab.get(key)

    def _finish_shape(self):
        if not self._drawing_kind:
            return
        if len(self._draw_points) < 3:
            messagebox.showwarning(
                "Finish Shape",
                "Need at least 3 points to form a closed shape (2 points make a line, "
                "which region/boundary/object-interaction detection can't use).")
            return
        kind = self._drawing_kind
        default_name = ("Arena" if kind == "boundary" else self._suggested_name(kind))
        role_vocab = self._role_vocab(kind) if kind != "boundary" else None
        name, role = self._ask_name_role(default_name, role_vocab)
        if name is None:
            return
        shape = {"name": name, "kind": kind, "vertices": list(self._draw_points), "role": role}
        rs = self._cfg["reference_shapes"]
        if kind == "boundary":
            rs["boundary"] = shape
        elif kind == "region":
            rs["regions"].append(shape)
        else:
            rs["objects"].append(shape)
        self._drawing_kind = None
        self._draw_points = []
        self._draw_status.configure(text="")
        self._refresh_sidebar()
        self._refresh_tab1_canvas()

    def _ask_name_role(self, default_name: str, role_vocab):
        dlg = tk.Toplevel(self)
        dlg.title("Name Shape")
        dlg.configure(bg=C["bg"])
        dlg.transient(self)
        dlg.grab_set()
        result = {"name": None, "role": None}

        tk.Label(dlg, text="Name:", bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, padx=8, pady=8, sticky="w")
        name_var = tk.StringVar(value=default_name)
        tk.Entry(dlg, textvariable=name_var, width=24, bg=C["card2"], fg=C["text"],
                 font=("Segoe UI", 9)).grid(row=0, column=1, padx=8, pady=8)

        role_var = tk.StringVar(value=(role_vocab[0] if role_vocab else ""))
        if role_vocab:
            tk.Label(dlg, text="Role:", bg=C["bg"], fg=C["text"],
                     font=("Segoe UI", 9)).grid(row=1, column=0, padx=8, pady=(0, 8), sticky="w")
            ttk.Combobox(dlg, textvariable=role_var, values=list(role_vocab),
                         state="readonly", width=22).grid(row=1, column=1, padx=8, pady=(0, 8))

        def _ok():
            result["name"] = name_var.get().strip() or default_name
            result["role"] = role_var.get() if role_vocab else None
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg=C["bg"])
        btn_row.grid(row=2, column=0, columnspan=2, pady=(0, 8))
        tk.Button(btn_row, text="Cancel", command=_cancel, bg=C["btn"], fg=C["btn_fg"],
                  relief="flat", padx=8, pady=3).pack(side="left", padx=4)
        tk.Button(btn_row, text="OK", command=_ok, bg=C["green"], fg="white",
                  relief="flat", padx=8, pady=3).pack(side="left", padx=4)
        dlg.wait_window()
        return result["name"], result["role"]

    def _refresh_tab1_canvas(self):
        cv = getattr(self, "_tab1_canvas", None)
        if cv is None:
            return
        try:
            from PIL import ImageTk, ImageDraw
        except ImportError:
            return
        cw = max(1, cv.winfo_width())
        ch = max(1, cv.winfo_height())
        cv.delete("all")
        if self._ref_frame_img is None:
            cv.create_text(cw // 2, ch // 2, text="No reference video frame available.",
                            fill="white", font=("Segoe UI", 10))
            return
        ow, oh = self._ref_frame_img.width, self._ref_frame_img.height
        scale = min(cw / ow, ch / oh, 1.0) or 1.0
        dw, dh = max(1, int(ow * scale)), max(1, int(oh * scale))
        off_x, off_y = (cw - dw) // 2, (ch - dh) // 2
        self._scale = scale
        self._canvas_offset = (off_x, off_y)

        from PIL import Image
        disp = self._ref_frame_img.resize((dw, dh), Image.LANCZOS).convert("RGB")
        draw = ImageDraw.Draw(disp)

        def _dp(x, y):
            return (x * scale, y * scale)

        rs = self._cfg["reference_shapes"]
        _COLORS = {"boundary": "#00E5FF", "region": "#4CAF50", "object": "#FF9800"}
        if rs.get("boundary"):
            pts = [_dp(*p) for p in rs["boundary"]["vertices"]]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=_COLORS["boundary"], width=2)
        for s in rs.get("regions") or []:
            pts = [_dp(*p) for p in s["vertices"]]
            if len(pts) >= 3:
                draw.polygon(pts, outline=_COLORS["region"], width=2)
        for s in rs.get("objects") or []:
            pts = [_dp(*p) for p in s["vertices"]]
            if len(pts) >= 3:
                draw.polygon(pts, outline=_COLORS["object"], width=2)

        if self._drawing_kind and self._draw_points:
            pts = [_dp(*p) for p in self._draw_points]
            col = _COLORS.get(self._drawing_kind, "#FFFFFF")
            for px, py in pts:
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)
            if len(pts) >= 2:
                draw.line(pts, fill=col, width=2)

        self._photo = ImageTk.PhotoImage(disp)
        cv.create_image(off_x, off_y, anchor="nw", image=self._photo)

    def _refresh_sidebar(self):
        for w in self._sidebar_inner.winfo_children():
            w.destroy()
        rs = self._cfg["reference_shapes"]
        role_vocab_r = self._role_vocab("region")
        role_vocab_o = self._role_vocab("object")

        def _shape_row(shape, on_delete, role_vocab, on_role_change):
            row = tk.Frame(self._sidebar_inner, bg=C["card2"])
            row.pack(fill="x", padx=4, pady=2)
            tk.Label(row, text=shape["name"], bg=C["card2"], fg=C["text"],
                     font=("Segoe UI", 8)).pack(side="left", padx=4)
            if role_vocab:
                rv = tk.StringVar(value=shape.get("role") or role_vocab[0])
                cb = ttk.Combobox(row, textvariable=rv, values=list(role_vocab),
                                   state="readonly", width=10, font=("Segoe UI", 7))
                cb.pack(side="left", padx=2)
                cb.bind("<<ComboboxSelected>>", lambda e: on_role_change(rv.get()))
            tk.Button(row, text="x", font=("Segoe UI", 7), bg=C["card2"], fg=C["red"],
                      relief="flat", padx=4, cursor="hand2",
                      command=on_delete).pack(side="right", padx=4)

        if rs.get("boundary"):
            tk.Label(self._sidebar_inner, text="Boundary", font=("Segoe UI", 8, "bold"),
                     bg=C["card"], fg=C["cyan"]).pack(anchor="w", padx=4, pady=(6, 0))
            _shape_row(rs["boundary"], lambda: self._delete_shape("boundary", None),
                       None, lambda r: None)

        if rs.get("regions"):
            tk.Label(self._sidebar_inner, text="Regions", font=("Segoe UI", 8, "bold"),
                     bg=C["card"], fg=C["cyan"]).pack(anchor="w", padx=4, pady=(6, 0))
            for i, s in enumerate(rs["regions"]):
                _shape_row(s, (lambda i=i: self._delete_shape("regions", i)),
                           role_vocab_r,
                           (lambda role, i=i: self._set_role("regions", i, role)))

        if rs.get("objects"):
            tk.Label(self._sidebar_inner, text="Objects", font=("Segoe UI", 8, "bold"),
                     bg=C["card"], fg=C["cyan"]).pack(anchor="w", padx=4, pady=(6, 0))
            for i, s in enumerate(rs["objects"]):
                _shape_row(s, (lambda i=i: self._delete_shape("objects", i)),
                           role_vocab_o,
                           (lambda role, i=i: self._set_role("objects", i, role)))

        self._refresh_min_shape_warning()

    def _delete_shape(self, key, idx):
        rs = self._cfg["reference_shapes"]
        if key == "boundary":
            rs["boundary"] = None
        else:
            del rs[key][idx]
        self._refresh_sidebar()
        self._refresh_tab1_canvas()

    def _set_role(self, key, idx, role):
        self._cfg["reference_shapes"][key][idx]["role"] = role
        self._refresh_min_shape_warning()

    def _refresh_min_shape_warning(self):
        # NOTE: self._sidebar_inner's children (including any previous
        # _warn_label) are destroyed wholesale at the top of _refresh_sidebar
        # on every call, so a cached widget reference would go stale --
        # always create a fresh label here rather than caching across calls.
        self._warn_label = tk.Label(self._sidebar_inner, text="", font=("Segoe UI", 7),
                                     bg=C["card"], fg=C["yellow"], wraplength=230,
                                     justify="left")
        self._warn_label.pack(anchor="w", padx=4, pady=(8, 4))
        paradigm = self._cfg.get("paradigm", "custom")
        mins = (ENV_PARADIGM_MIN_ROLES or {}).get(paradigm)
        if not mins:
            self._warn_label.configure(text="")
            return
        rs = self._cfg["reference_shapes"]
        all_shapes = (rs.get("regions") or []) + (rs.get("objects") or [])
        missing = []
        for role, need in mins.items():
            have = sum(1 for s in all_shapes if s.get("role") == role)
            if have < need:
                missing.append(f"{role} (have {have}, need {need})")
        if missing:
            self._warn_label.configure(
                text="Not enough role-tagged shapes for this paradigm's specialized "
                     "metric yet -- missing: " + ", ".join(missing) +
                     ". Generic per-region/per-object output still works.")
        else:
            self._warn_label.configure(text="")

    def _save_template(self):
        path = filedialog.asksaveasfilename(
            title="Save Arena Template", defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            payload = {"paradigm": self._cfg.get("paradigm"),
                       "reference_shapes": self._cfg.get("reference_shapes")}
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save Template", str(e))

    def _load_template(self):
        path = filedialog.askopenfilename(
            title="Load Arena Template", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self._cfg["paradigm"] = payload.get("paradigm", self._cfg.get("paradigm"))
            self._cfg["reference_shapes"] = payload.get(
                "reference_shapes", self._cfg["reference_shapes"])
            self._build_main_ui()
        except Exception as e:
            messagebox.showerror("Load Template", str(e))

    # ── Tab 2: Apply to Other Videos ────────────────────────────────────────

    def _build_tab2(self, parent):
        left = tk.Frame(parent, bg=C["card"], width=260)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side="right", fill="both", expand=True)

        tk.Label(left, text="Other Videos", font=("Segoe UI", 10, "bold"),
                 bg=C["card"], fg=C["cyan"]).pack(anchor="w", padx=8, pady=(8, 4))
        self._tab2_list_frame = tk.Frame(left, bg=C["card"])
        self._tab2_list_frame.pack(fill="both", expand=True, padx=4)

        toolbar2 = tk.Frame(right, bg=C["bg"])
        toolbar2.pack(fill="x")
        self._tab2_active_label = tk.Label(toolbar2, text="No video selected",
                                            font=("Segoe UI", 9, "bold"),
                                            bg=C["bg"], fg=C["yellow"])
        self._tab2_active_label.pack(side="left")
        self._tab2_edit_var = tk.BooleanVar(value=False)
        tk.Checkbutton(toolbar2, text="Edit individual shapes", variable=self._tab2_edit_var,
                       bg=C["bg"], fg=C["green"], selectcolor=C["card2"],
                       activebackground=C["bg"],
                       command=self._on_tab2_edit_toggle).pack(side="left", padx=10)
        tk.Button(toolbar2, text="Reset to Reference", font=("Segoe UI", 8),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat", padx=6, pady=2,
                  cursor="hand2", command=self._tab2_reset).pack(side="left")

        tk.Label(right, text="Drag anywhere on the frame to nudge the whole overlay "
                              "(or, with 'Edit individual shapes' on, drag near one "
                              "shape to move just that shape).",
                 font=("Segoe UI", 7), bg=C["bg"], fg=C["dim"], wraplength=600,
                 justify="left").pack(anchor="w", pady=(2, 0))

        self._tab2_canvas = tk.Canvas(right, bg="black", highlightthickness=0)
        self._tab2_canvas.pack(fill="both", expand=True, pady=(4, 0))
        self._tab2_canvas.bind("<Configure>", lambda e: self._refresh_tab2_canvas())
        self._tab2_canvas.bind("<ButtonPress-1>", self._on_tab2_press)
        self._tab2_canvas.bind("<B1-Motion>", self._on_tab2_drag)
        self._tab2_canvas.bind("<ButtonRelease-1>", self._on_tab2_release)

    def _refresh_tab2_list(self):
        for w in self._tab2_list_frame.winfo_children():
            w.destroy()
        self._other_videos = [(s, p) for s, p in self._list_videos()
                               if s != self._cfg.get("reference_stem")]
        per_video = self._cfg.get("per_video") or {}
        for stem, path in self._other_videos:
            status = "Adjusted" if stem in per_video else "Default"
            colour = C["yellow"] if status == "Adjusted" else C["dim"]
            row = tk.Frame(self._tab2_list_frame, bg=C["card2"])
            row.pack(fill="x", pady=1)
            tk.Button(row, text=stem[:26], font=("Segoe UI", 8), bg=C["card2"],
                      fg=C["text"], relief="flat", anchor="w", cursor="hand2",
                      command=lambda s=stem, p=path: self._select_tab2_video(s, p)
                      ).pack(side="left", fill="x", expand=True, padx=(4, 0))
            tk.Label(row, text=status, font=("Segoe UI", 7), bg=C["card2"],
                     fg=colour).pack(side="right", padx=4)

    def _select_tab2_video(self, stem, path):
        self._active_video_stem = stem
        self._tab2_active_label.configure(text=stem)
        self._tab2_frame_img = self._load_frame_for(path)
        pv = (self._cfg.get("per_video") or {}).get(stem)
        self._tab2_edit_var.set(bool(pv and pv.get("mode") == "override"))
        self._refresh_tab2_canvas()

    def _current_tab2_shapes(self):
        """Effective flat shape list for the active video (mirrors
        resolve_env_shapes, kept local so Tab 2 can render live drag
        feedback before it's committed to self._cfg)."""
        if resolve_env_shapes is not None and self._active_video_stem:
            return resolve_env_shapes(self._cfg, self._active_video_stem)
        return []

    def _refresh_tab2_canvas(self):
        cv = getattr(self, "_tab2_canvas", None)
        if cv is None:
            return
        try:
            from PIL import ImageTk, ImageDraw, Image
        except ImportError:
            return
        cv.delete("all")
        img = getattr(self, "_tab2_frame_img", None)
        cw, ch = max(1, cv.winfo_width()), max(1, cv.winfo_height())
        if img is None:
            cv.create_text(cw // 2, ch // 2, text="Select a video on the left.",
                            fill="white", font=("Segoe UI", 10))
            return
        ow, oh = img.width, img.height
        scale = min(cw / ow, ch / oh, 1.0) or 1.0
        dw, dh = max(1, int(ow * scale)), max(1, int(oh * scale))
        off_x, off_y = (cw - dw) // 2, (ch - dh) // 2
        self._tab2_scale = scale
        self._tab2_offset = (off_x, off_y)
        disp = img.resize((dw, dh), Image.LANCZOS).convert("RGB")
        draw = ImageDraw.Draw(disp)
        _COLORS = {"boundary": "#00E5FF", "region": "#4CAF50", "object": "#FF9800"}
        for s in self._current_tab2_shapes():
            pts = [(x * scale, y * scale) for x, y in s["vertices"]]
            col = _COLORS.get(s["kind"], "#FFFFFF")
            if s["kind"] == "boundary" and len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=col, width=2)
            elif len(pts) >= 3:
                draw.polygon(pts, outline=col, width=2)
        self._tab2_photo = ImageTk.PhotoImage(disp)
        cv.create_image(off_x, off_y, anchor="nw", image=self._tab2_photo)

    def _tab2_canvas_to_orig(self, cx, cy):
        ox, oy = getattr(self, "_tab2_offset", (0, 0))
        s = getattr(self, "_tab2_scale", 1.0) or 1.0
        return (cx - ox) / s, (cy - oy) / s

    def _on_tab2_edit_toggle(self):
        if not self._active_video_stem:
            return
        stem = self._active_video_stem
        pv = self._cfg.setdefault("per_video", {})
        if self._tab2_edit_var.get():
            # snapshot the CURRENT effective shapes as an independent override set.
            shapes = resolve_env_shapes(self._cfg, stem) if resolve_env_shapes else []
            boundary = next((s for s in shapes if s["kind"] == "boundary"), None)
            regions = [s for s in shapes if s["kind"] == "region"]
            objects = [s for s in shapes if s["kind"] == "object"]
            existing_roles = (pv.get(stem, {}) or {}).get("role_overrides")
            pv[stem] = {"mode": "override",
                        "shapes": {"boundary": boundary, "regions": regions, "objects": objects},
                        "role_overrides": existing_roles}
        else:
            # Unchecking discards this video's independent per-shape edits
            # (reverting to the reference/default alignment) -- confirm
            # first, same as the explicit "Reset to Reference" button,
            # rather than silently losing dragged-into-place shapes.
            if stem in pv and pv[stem].get("mode") == "override":
                if messagebox.askyesno(
                        "Discard Individual Shape Edits",
                        "Turning this off discards this video's independently-adjusted "
                        "shapes and reverts to the default (reference) alignment. Continue?"):
                    del pv[stem]
                else:
                    self._tab2_edit_var.set(True)
                    return
        self._refresh_tab2_canvas()
        self._refresh_tab2_list()

    def _tab2_reset(self):
        if not self._active_video_stem:
            return
        pv = self._cfg.get("per_video") or {}
        pv.pop(self._active_video_stem, None)
        self._tab2_edit_var.set(False)
        self._refresh_tab2_canvas()
        self._refresh_tab2_list()

    def _on_tab2_press(self, event):
        if not self._active_video_stem:
            return
        self._tab2_drag_last = (event.x, event.y)
        self._tab2_drag_shape_idx = None
        if self._tab2_edit_var.get():
            ox, oy = self._tab2_canvas_to_orig(event.x, event.y)
            pv = self._cfg.setdefault("per_video", {})
            entry = pv.get(self._active_video_stem)
            if not entry or entry.get("mode") != "override":
                return
            for key in ("regions", "objects"):
                for i, s in enumerate(entry["shapes"].get(key) or []):
                    verts = s["vertices"]
                    cx = sum(v[0] for v in verts) / len(verts)
                    cy = sum(v[1] for v in verts) / len(verts)
                    if abs(cx - ox) < 25 and abs(cy - oy) < 25:
                        self._tab2_drag_shape_idx = (key, i)
                        return

    def _on_tab2_drag(self, event):
        if self._tab2_drag_last is None or not self._active_video_stem:
            return
        dx_c = event.x - self._tab2_drag_last[0]
        dy_c = event.y - self._tab2_drag_last[1]
        scale = getattr(self, "_tab2_scale", 1.0) or 1.0
        dx, dy = dx_c / scale, dy_c / scale
        self._tab2_drag_last = (event.x, event.y)
        pv = self._cfg.setdefault("per_video", {})
        stem = self._active_video_stem

        if self._tab2_edit_var.get() and self._tab2_drag_shape_idx:
            key, i = self._tab2_drag_shape_idx
            entry = pv.get(stem)
            if entry and entry.get("mode") == "override":
                shp = entry["shapes"][key][i]
                shp["vertices"] = [(x + dx, y + dy) for x, y in shp["vertices"]]
        elif not self._tab2_edit_var.get():
            entry = pv.setdefault(stem, {"mode": "transform", "translate": [0, 0],
                                          "role_overrides": None})
            if entry.get("mode") != "transform":
                entry = pv[stem] = {"mode": "transform", "translate": [0, 0],
                                    "role_overrides": entry.get("role_overrides")}
            entry["translate"] = [entry["translate"][0] + dx, entry["translate"][1] + dy]
        self._refresh_tab2_canvas()

    def _on_tab2_release(self, event):
        self._tab2_drag_last = None
        self._tab2_drag_shape_idx = None
        self._refresh_tab2_list()

    # ── finish ───────────────────────────────────────────────────────────────

    def _apply_and_close(self):
        self.result = self._cfg
        self.destroy()


#
#  BODY-REGION FEATURE WEIGHTING WINDOW  (issue 1b)
#


class BodyPartWeightWindow(tk.Toplevel):
    """
    Optional per-body-region feature weighting editor, opened from
    AdvancedCUBEWindow.  Peeks bodyparts from the first DLC file found in the
    session's folders (cube_core.peek_dlc_bodyparts — header-only, no full
    load), groups them via cube_core.group_bodyparts_by_region, and renders
    one bordered card per region (plain tk widgets, not ttk.LabelFrame — see
    __init__) with a single 0.1-3.0 weight slider (default 1.0 = uniform, no
    separate on/off checkbox).

    A region is "enabled" implicitly: any slider left at 1.0 is treated as
    not customised, and any slider moved away from 1.0 is treated as an
    explicit weight for that region's bodyparts. So is the window as a
    whole -> Apply produces an empty {} weights dict iff every slider is
    still at 1.0 (uniform weighting = today's exact behaviour).

    On Apply, self.result is set to the expanded per-bodypart dict
    ({bodypart_name: weight} for every bodypart in a region whose slider
    was moved off 1.0) and the window closes.  On Cancel, self.result stays
    None so the caller knows to keep whatever weights it already had.
    """

    def __init__(self, parent, session: "SessionState", initial_weights: dict = None):
        super().__init__(parent)
        self.title("Body-Region Weights (optional)")
        self.configure(bg=C["bg"])
        self.geometry("480x600")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self.result = None

        self._session = session
        self._initial_weights = dict(initial_weights or {})
        self._region_vars: dict = {}   # region -> {"slider": DoubleVar, "bps": [..]}

        # NOTE: region boxes below are plain tk.Frame/tk.Label, not
        # ttk.LabelFrame — Windows' default "vista" ttk theme ignores
        # ttk.Style background/foreground configuration for TLabelframe, so a
        # ttk-based box would silently keep rendering with native (non-theme)
        # colours regardless of C[] here.  Classic tk widgets always honour
        # explicit bg/fg, so they're used instead to guarantee both dark and
        # light themes render correctly.

        # ── Header ───────────────────────────────────────────────────────────
        tk.Label(self, text="  Body-Region Feature Weights",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["bg"], fg=C["cyan"]).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(self,
                 text="  Weight anatomical regions in the feature set before UMAP.\n"
                      "  Move a slider off 1.0 to customise; 1.0 = uniform (default).",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["subtext"],
                 justify="left").pack(anchor="w", padx=10, pady=(0, 8))

        # ── Scrollable region list ──────────────────────────────────────────
        canvas = tk.Canvas(self, bg=C["bg"], highlightthickness=0)
        sb = tk.Scrollbar(self, orient="vertical", command=canvas.yview,
                          bg=C["card"], troughcolor=C["bg"])
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True, padx=(10, 0))
        inner = tk.Frame(canvas, bg=C["bg"])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        self._status_lbl = tk.Label(inner, text="Loading bodyparts...",
                                     font=("Segoe UI", 9), bg=C["bg"], fg=C["dim"])
        self._status_lbl.pack(anchor="w", padx=4, pady=8)

        # ── Bottom buttons ───────────────────────────────────────────────────
        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(side="bottom", fill="x", pady=8, padx=12)
        tk.Button(btn_f, text="Cancel", font=("Segoe UI", 9),
                  bg=C["btn"], fg=C["btn_fg"], relief="flat",
                  padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")
        self._apply_btn = tk.Button(btn_f, text="Apply", font=("Segoe UI", 10, "bold"),
                                    bg=C["purple"], fg="white", relief="flat",
                                    padx=16, pady=5, cursor="hand2",
                                    command=self._apply)
        self._apply_btn.pack(side="right")

        self._build_regions(inner)

    # -- bodypart discovery ---------------------------------------------------
    def _find_first_dlc_file(self):
        """Look in bsoid_ready_dirs first (post-prep DLC output), then the raw
        video_folders, mirroring the pipeline's own preference order."""
        candidates = []
        try:
            candidates.extend(self._session.get("bsoid_ready_dirs") or [])
        except Exception:
            pass
        try:
            candidates.extend(self._session.get("video_folders") or [])
        except Exception:
            pass
        for folder in candidates:
            try:
                if not folder or not Path(folder).is_dir():
                    continue
                files = find_dlc_files(folder) if find_dlc_files else []
                if files:
                    return files[0]
            except Exception:
                continue
        return None

    def _build_regions(self, parent):
        if find_dlc_files is None or peek_dlc_bodyparts is None or group_bodyparts_by_region is None:
            self._status_lbl.configure(
                text="Body-region weighting requires cube_core (not loaded).")
            self._apply_btn.configure(state="disabled")
            return

        dlc_path = self._find_first_dlc_file()
        if dlc_path is None:
            self._status_lbl.configure(
                text="No DLC file found in this session's folders yet.\n"
                     "Run Step 1/2 first, then reopen this window to customise weights.")
            self._apply_btn.configure(state="disabled")
            return

        try:
            bodyparts = peek_dlc_bodyparts(dlc_path)
        except Exception as e:
            bodyparts = []
            self._status_lbl.configure(text=f"Could not read bodyparts from:\n{dlc_path}\n({e})")

        if not bodyparts:
            self._status_lbl.configure(
                text=f"Could not read any bodyparts from:\n{dlc_path}")
            self._apply_btn.configure(state="disabled")
            return

        self._status_lbl.configure(
            text=f"Bodyparts detected from:\n{Path(dlc_path).name}  ({len(bodyparts)} bodyparts)",
            justify="left")

        try:
            regions = group_bodyparts_by_region(bodyparts)
        except Exception as e:
            self._status_lbl.configure(text=f"group_bodyparts_by_region failed: {e}")
            self._apply_btn.configure(state="disabled")
            return

        for region, bps in regions.items():
            if not bps:
                continue
            # Bordered "card" via highlightthickness rather than ttk.LabelFrame
            # (see note in __init__ re: the vista ttk theme).
            frame = tk.Frame(parent, bg=C["card"], highlightthickness=1,
                             highlightbackground=C["border"],
                             highlightcolor=C["border"])
            frame.pack(fill="x", padx=4, pady=4)
            tk.Label(frame, text=region, font=("Segoe UI", 9, "bold"),
                    bg=C["card"], fg=C["cyan"]).pack(anchor="w", padx=4, pady=(4, 0))

            # Pre-fill from initial_weights if any bodypart in this region was
            # previously customised (average of its set weights).
            prior_vals = [self._initial_weights[bp] for bp in bps
                          if bp in self._initial_weights]
            init_val = float(sum(prior_vals) / len(prior_vals)) if prior_vals else 1.0

            slider_var = tk.DoubleVar(value=round(init_val, 1))

            slider = tk.Scale(frame, from_=0.1, to=3.0, resolution=0.1,
                              orient="horizontal", variable=slider_var,
                              bg=C["card"], fg=C["text"],
                              troughcolor=C["card2"], highlightthickness=0,
                              activebackground=C["cyan"],
                              font=("Segoe UI", 8))
            slider.pack(fill="x", padx=4, pady=(2, 2))

            bp_text = ", ".join(bps)
            if len(bp_text) > 90:
                bp_text = bp_text[:87] + "..."
            tk.Label(frame, text=bp_text, font=("Segoe UI", 7),
                    bg=C["card"], fg=C["subtext"], wraplength=420,
                    justify="left").pack(anchor="w", padx=4, pady=(0, 4))

            self._region_vars[region] = {"slider": slider_var, "bps": bps}

    def _apply(self):
        weights = {}
        for info in self._region_vars.values():
            w = float(info["slider"].get())
            if abs(w - 1.0) < 1e-9:
                continue
            for bp in info["bps"]:
                weights[bp] = w
        self.result = weights
        _save_body_region_weights(weights)
        self.destroy()


#
#  3D DLC SETTINGS WINDOW


class ThreeDSettingsWindow(tk.Toplevel):
    """
    Modal popup for 3D DLC + Anipose settings.
    Mirrors the AdvancedDLCWindow / AdvancedCUBEWindow pattern.
    """

    def __init__(self, parent, session):
        super().__init__(parent)
        self._session = session
        self.title("3D DLC + Anipose Settings")
        self.resizable(False, False)
        self.configure(bg=C["bg"])
        self.grab_set()

        pad = dict(padx=10, pady=4)

        def _hdr(text, color=C["cyan"]):
            tk.Label(self, text=text, font=("Segoe UI", 9, "bold"),
                     bg=C["bg"], fg=color).pack(anchor="w", padx=10, pady=(8, 2))

        def _row(label_text):
            row = tk.Frame(self, bg=C["bg"])
            row.pack(fill="x", **pad)
            tk.Label(row, text=label_text, width=28, anchor="w",
                     font=("Segoe UI", 9), bg=C["bg"],
                     fg=C["text"]).pack(side="left")
            return row

        # ── Enable 3D mode ────────────────────────────────────────────────────
        _hdr("3D Mode")
        self._enabled = tk.BooleanVar(
            value=bool(session.get("dlc_3d_enabled", False)))
        row = _row("Enable 3D DLC + Anipose:")
        tk.Checkbutton(row, variable=self._enabled, bg=C["bg"],
                       fg=C["text"], selectcolor=C["card"],
                       activebackground=C["bg"]).pack(side="left")

        # ── Calibration folder ───────────────────────────────────────────────
        _hdr("Folders")
        self._calib = tk.StringVar(
            value=str(session.get("dlc_3d_calib_folder", "")))
        row = _row("Calibration folder (→ calibration.toml):")
        tk.Entry(row, textvariable=self._calib, width=38,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")
        tk.Button(row, text="Browse…", bg=C["btn"], fg=C["subtext"],
                  relief="flat", font=("Segoe UI", 8), cursor="hand2",
                  command=lambda: self._browse_dir(self._calib)
                  ).pack(side="left", padx=(4, 0))

        # Resolved TOML label
        self._toml_lbl = tk.Label(self, text="", font=("Segoe UI", 7),
                                   bg=C["bg"], fg=C["dim"])
        self._toml_lbl.pack(anchor="w", padx=36, pady=(0, 2))
        self._calib.trace_add("write",
                               lambda *_: self._refresh_toml_label())
        self._refresh_toml_label()

        # ── Output folder ─────────────────────────────────────────────────────
        self._output = tk.StringVar(
            value=str(session.get("dlc_3d_output_folder", "")))
        row = _row("Output folder (blank = source folder):")
        tk.Entry(row, textvariable=self._output, width=38,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")
        tk.Button(row, text="Browse…", bg=C["btn"], fg=C["subtext"],
                  relief="flat", font=("Segoe UI", 8), cursor="hand2",
                  command=lambda: self._browse_dir(self._output)
                  ).pack(side="left", padx=(4, 0))

        # ── Camera labels ─────────────────────────────────────────────────────
        _hdr("Camera Configuration")
        labels_raw = session.get("dlc_3d_cam_labels", ["cam0","cam1","cam2","cam3"])
        if isinstance(labels_raw, list):
            labels_raw = ",".join(labels_raw)
        self._cam_labels = tk.StringVar(value=str(labels_raw))
        row = _row("Camera labels (comma-separated):")
        tk.Entry(row, textvariable=self._cam_labels, width=28,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")

        # ── Likelihood aggregation ────────────────────────────────────────────
        _hdr("Triangulation Options")
        self._ll_agg = tk.StringVar(
            value=str(session.get("dlc_3d_ll_agg", "min")))
        row = _row("Confidence aggregation across cams:")
        for val, lbl in [("min","min (safest)"), ("mean","mean"),
                          ("max","max (most permissive)")]:
            tk.Radiobutton(row, text=lbl, variable=self._ll_agg, value=val,
                           bg=C["bg"], fg=C["text"],
                           selectcolor=C["card"],
                           activebackground=C["bg"],
                           font=("Segoe UI", 9)).pack(side="left", padx=4)

        self._use_ransac = tk.BooleanVar(
            value=bool(session.get("dlc_3d_use_ransac", True)))
        row = _row("RANSAC triangulation (robust for occlusions):")
        tk.Checkbutton(row, variable=self._use_ransac, bg=C["bg"],
                       fg=C["text"], selectcolor=C["card"],
                       activebackground=C["bg"]).pack(side="left")

        self._ransac_thr = tk.StringVar(
            value=str(session.get("dlc_3d_ransac_threshold", 0.5)))
        row = _row("RANSAC reprojection threshold (px):")
        tk.Entry(row, textvariable=self._ransac_thr, width=8,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")

        self._ll_thr = tk.StringVar(
            value=str(session.get("dlc_3d_ll_threshold", 0.0)))
        row = _row("Point confidence threshold (0 = off, e.g. 0.3):")
        tk.Entry(row, textvariable=self._ll_thr, width=8,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")

        self._ll_gate = tk.StringVar(
            value=str(session.get("dlc_3d_ll_gate", 0.6)))
        row = _row("Pre-triangulation likelihood gate (0 = off, default 0.6):")
        tk.Entry(row, textvariable=self._ll_gate, width=8,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")

        self._median_win = tk.StringVar(
            value=str(session.get("dlc_3d_median_window", 3)))
        row = _row("Temporal median filter — frames (0 = off, default 3):")
        tk.Entry(row, textvariable=self._median_win, width=8,
                 bg=C["card2"], fg=C["text"],
                 insertbackground=C["text"],
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")

        # ── Post-processing options ───────────────────────────────────────────
        _hdr("Post-processing")
        self._del_orig = tk.BooleanVar(
            value=bool(session.get("dlc_3d_delete_orig_videos", False)))
        row = _row("Delete camera videos after quad composite:")
        tk.Checkbutton(row, variable=self._del_orig, bg=C["bg"],
                       fg=C["text"], selectcolor=C["card"],
                       activebackground=C["bg"]).pack(side="left")

        self._del_h5s = tk.BooleanVar(
            value=bool(session.get("dlc_3d_delete_cam_h5s", False)))
        row = _row("Delete per-camera H5s after 3D H5 is created:")
        tk.Checkbutton(row, variable=self._del_h5s, bg=C["bg"],
                       fg=C["text"], selectcolor=C["card"],
                       activebackground=C["bg"]).pack(side="left")

        self._skel_vid = tk.BooleanVar(
            value=bool(session.get("dlc_3d_export_skeleton_video", False)))
        row = _row("Export 3D skeleton visualization video:")
        tk.Checkbutton(row, variable=self._skel_vid, bg=C["bg"],
                       fg=C["text"], selectcolor=C["card"],
                       activebackground=C["bg"]).pack(side="left")

        # ── Session scan ──────────────────────────────────────────────────────
        _hdr("Session Discovery")
        scan_row = tk.Frame(self, bg=C["bg"])
        scan_row.pack(fill="x", padx=10, pady=(2, 4))
        tk.Button(scan_row, text="Scan video source folders for sessions",
                  font=("Segoe UI", 9, "bold"), bg=C["btn"], fg=C["cyan"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._scan_sessions).pack(side="left")
        self._scan_lbl = tk.Label(scan_row, text="", font=("Segoe UI", 8),
                                   bg=C["bg"], fg=C["subtext"])
        self._scan_lbl.pack(side="left", padx=8)

        self._sess_text = tk.Text(self, height=6, width=72,
                                   bg=C["card2"], fg=C["text"],
                                   font=("Consolas", 8),
                                   state="disabled", relief="flat")
        self._sess_text.pack(fill="x", padx=10, pady=(0, 4))

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=C["bg"])
        btn_row.pack(fill="x", padx=10, pady=(6, 10))
        tk.Button(btn_row, text="Apply & Close",
                  font=("Segoe UI", 9, "bold"), bg=C["cyan"], fg="white",
                  relief="flat", padx=12, pady=5, cursor="hand2",
                  command=self._apply).pack(side="right")
        tk.Button(btn_row, text="Cancel",
                  font=("Segoe UI", 9), bg=C["btn"], fg=C["subtext"],
                  relief="flat", padx=10, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="right", padx=(0, 6))

        self.update_idletasks()
        self.geometry(
            f"+{parent.winfo_rootx() + 60}+{parent.winfo_rooty() + 60}")

    def _browse_dir(self, var: tk.StringVar):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select folder")
        if d:
            var.set(d)

    def _refresh_toml_label(self):
        calib = self._calib.get().strip()
        if not calib:
            self._toml_lbl.configure(text="")
            return
        toml = Path(calib) / "calibration.toml"
        if toml.exists():
            self._toml_lbl.configure(
                text=f"✓  {toml}", fg=C["green"])
        else:
            self._toml_lbl.configure(
                text=f"✗  calibration.toml not found in that folder", fg=C["red"])

    def _scan_sessions(self):
        from cube_3d_dlc import find_step0_sessions
        _raw = (self._session.get("dlc_3d_source_folders") or
                self._session.get("video_folders", []))
        source_folders = [Path(f) for f in _raw if Path(f).is_dir()]
        if not source_folders:
            self._scan_lbl.configure(
                text="Add video source folders in the main panel first.")
            return
        try:
            all_sessions: list = []
            seen: set = set()
            for vf in source_folders:
                for s in find_step0_sessions(vf):
                    if s["session_id"] not in seen:
                        seen.add(s["session_id"])
                        all_sessions.append(s)
        except Exception as e:
            self._scan_lbl.configure(text=f"Error: {e}")
            return
        self._scan_lbl.configure(
            text=f"{len(all_sessions)} session(s) across {len(source_folders)} folder(s)")
        lines = []
        for s in all_sessions:
            cams = ", ".join(sorted(s["cameras"].keys()))
            n_frames = ""
            if s["report"] and "cameras" in s["report"]:
                counts = [c.get("n_output_frames", 0)
                          for c in s["report"]["cameras"]]
                if counts:
                    n_frames = f"  {counts[0]} frames"
            lines.append(f"{s['session_id']}  [{cams}]{n_frames}")
        self._sess_text.configure(state="normal")
        self._sess_text.delete("1.0", "end")
        self._sess_text.insert("end", "\n".join(lines))
        self._sess_text.configure(state="disabled")

    def _apply(self):
        s = self._session
        s["dlc_3d_enabled"]          = bool(self._enabled.get())
        s["dlc_3d_calib_folder"]     = self._calib.get().strip()
        s["dlc_3d_output_folder"]    = self._output.get().strip()
        raw = self._cam_labels.get().strip()
        s["dlc_3d_cam_labels"]       = [c.strip() for c in raw.split(",")
                                         if c.strip()]
        s["dlc_3d_ll_agg"]             = self._ll_agg.get()
        s["dlc_3d_use_ransac"]         = bool(self._use_ransac.get())
        try:
            s["dlc_3d_ransac_threshold"] = float(self._ransac_thr.get())
        except ValueError:
            s["dlc_3d_ransac_threshold"] = 0.5
        try:
            s["dlc_3d_ll_threshold"]     = float(self._ll_thr.get())
        except ValueError:
            s["dlc_3d_ll_threshold"]     = 0.0
        try:
            s["dlc_3d_ll_gate"]          = float(self._ll_gate.get())
        except ValueError:
            s["dlc_3d_ll_gate"]          = 0.6
        try:
            _mw = int(self._median_win.get())
            if _mw > 1 and _mw % 2 == 0:
                _mw += 1  # enforce odd kernel silently
            s["dlc_3d_median_window"]    = _mw
        except ValueError:
            s["dlc_3d_median_window"]    = 3
        s["dlc_3d_delete_orig_videos"] = bool(self._del_orig.get())
        s["dlc_3d_delete_cam_h5s"]   = bool(self._del_h5s.get())
        s["dlc_3d_export_skeleton_video"] = bool(self._skel_vid.get())
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

        bg  = "#09090f" if is_dark else "#ffffff"
        sub = "#4a4a6a" if is_dark else "#888888"
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
        self.title("CUBE: Comprehensive Unsupervised Behavioral Explorer  v5.0")

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
        # Restore ntfy topic saved from a previous session
        _ntfy_file = HERE / "ntfy_topic.txt"
        if _ntfy_file.is_file():
            try:
                _saved_topic = _ntfy_file.read_text(encoding="utf-8").strip()
                if _saved_topic:
                    self._settings.set_val("ntfy_topic", _saved_topic)
            except Exception:
                pass
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
                 bg=C["log_bg"], fg=C["cube_title"]).pack(
            side="left", padx=(16, 4), pady=8, anchor="s")
        tk.Label(hdr, text="Comprehensive Unsupervised Behavioral Explorer",
                 font=("Segoe UI", 13),
                 bg=C["log_bg"], fg=C["text"]).pack(
            side="left", pady=8, anchor="s")

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
        tk.Button(adv_row, text="⬡  3D DLC Settings...",
                  font=("Segoe UI", 8, "bold"), bg=C["btn"], fg="#4fc3f7",
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_3d_settings).pack(side="left",
                                                        padx=(4, 0), pady=2)
        tk.Button(adv_row, text="🌐  Environments, Objects, Paradigms...",
                  font=("Segoe UI", 8, "bold"), bg=C["btn"], fg=C["cyan"],
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._open_env_paradigms).pack(side="left",
                                                          padx=(4, 0), pady=2)

        # "Export Extra UMAP Evolution Videos" lives in the Tools menu bar
        # (see _build_menubar) rather than as a button in the main panel.

        # right column  " step cards
        right = tk.Frame(top, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True,
                   padx=(4,10), pady=8)
        _n_cards = len(STEP_META)
        for i in range(_n_cards):
            right.columnconfigure(i, weight=1, uniform="card")
        right.rowconfigure(0, weight=1)

        self._cards: dict[str, StepCard] = {}
        CMDS = {
            "dlc":        self._launch_dlc,
            "dlc_3d":     self._launch_3d_dlc,
            "bsoid_prep": self._launch_bsoid_prep,
            "bsoid_run":  self._launch_bsoid_run,
            "annotate":   self._launch_annotate,
            "analyse":    self._launch_analyse,
        }
        for i, meta in enumerate(STEP_META):
            card = StepCard(right, meta, CMDS[meta["key"]])
            card.grid(row=0, column=i, sticky="nsew", padx=4)
            self._cards[meta["key"]] = card

        # progress + status
        pb_outer = tk.Frame(right, bg=C["panel"],
                            highlightbackground=C["border"],
                            highlightthickness=1)
        pb_outer.grid(row=1, column=0, columnspan=_n_cards, sticky="ew",
                      padx=4, pady=(6,0))
        right.rowconfigure(1, minsize=90)
        self._pb = DualProgressBar(pb_outer)
        self._pb.pack(fill="x")

        sb = tk.Frame(right, bg=C["card2"])
        sb.grid(row=2, column=0, columnspan=_n_cards, sticky="ew", padx=4, pady=(4,0))
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
        self._logger.step("  CUBE: Comprehensive Unsupervised Behavioral Explorer  v5.0")
        self._logger.step("=" * 60)
        if CORE_OK:
            self._logger.success("  ✓  cube_core v5 loaded")
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
        self._pb.set_step_status(key, "running")
        self._status(f"Running step: {key} ")
        self._start_step_timer()
        step_idx = {"dlc":0, "dlc_3d":1, "bsoid_prep":2,
                    "bsoid_run":3, "annotate":4, "analyse":5}.get(key, -1)
        self._update_flow_bar(step_idx)

        def _worker():
            try:
                fn(*args)
                self._session.set_status(key, "done")
                self._session.save()
                self._after(lambda: self._pb.set_step_status(key, "done"))
                self._after(lambda: self._cards[key].set_status("done"))
                self._after(lambda: self._status(f"Step '{key}' complete v"))
                self._after(lambda: self._pb.step_done())
                self._logger.success(f"Step '{key}' complete.")
                # Auto-chain Steps 2+3 after DLC (2D or 3D) if user enabled it
                if key in ("dlc", "dlc_3d") and self._settings.get("auto_bsoid", False):
                    self._logger.step("Auto-run: launching Step 3 (pre-processing) → Step 4 (BSoid analysis)...")
                    # Clear _running on the main-thread queue BEFORE the launch
                    # callback so _run_step doesn't see it still True (race condition).
                    self._after(lambda: setattr(self, '_running', False))
                    self._after(self._auto_launch_bsoid_chain)
                # If prep was auto-chained, trigger engine next
                elif key == "bsoid_prep" and getattr(self, "_chain_engine_after_prep", False):
                    self._chain_engine_after_prep = False
                    self._logger.step("Auto-run: launching Step 4 (BSoid analysis / clustering)...")
                    self._after(lambda: setattr(self, '_running', False))
                    self._after(self._launch_bsoid_run)
            except Exception:
                tb = traceback.format_exc()
                self._session.set_status(key, "error")
                self._session.save()
                self._logger.error(f"Step '{key}' failed:\n{tb}")
                self._after(lambda: self._pb.set_step_status(key, "error"))
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

    def _open_3d_settings(self):
        ThreeDSettingsWindow(self, self._session)

    def _open_env_paradigms(self):
        EnvParadigmWindow(self, self._session)

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

    def _launch_3d_dlc(self):
        if not CORE_OK:
            messagebox.showerror("Missing", "cube_core.py not found.")
            return
        if not self._session.get("dlc_3d_enabled", False):
            messagebox.showinfo("3D DLC",
                "Enable 3D Mode via '⬡  3D DLC Settings…' first.")
            return
        calib_folder = self._session.get("dlc_3d_calib_folder", "")
        if not calib_folder or not (Path(calib_folder) / "calibration.toml").exists():
            messagebox.showerror("Calibration missing",
                f"calibration.toml not found in:\n{calib_folder}\n\n"
                "Please set the calibration folder in 3D DLC Settings.")
            return
        if not self._session.get("video_folders", []):
            messagebox.showwarning("No source folders",
                "Add at least one video source folder in the main panel "
                "before running 3D DLC.")
            return
        try:
            from cube_3d_dlc import _run_3d_dlc_step
        except ImportError as e:
            messagebox.showerror("Import Error",
                f"Could not import cube_3d_dlc.py:\n{e}")
            return
        self._run_step("dlc_3d", _run_3d_dlc_step,
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
        self._pb.set_step_status("annotate", "running")
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
                self._after(lambda: self._pb.set_step_status("annotate", "done"))
                self._after(lambda: self._cards["annotate"].set_status("done"))
                self._after(lambda: self._status("Step 4 complete v"))
            except Exception:
                self._running = False
                tb = traceback.format_exc()
                self._logger.error(f"Step 4 error:\n{tb}")
                self._after(lambda: self._pb.set_step_status("annotate", "error"))
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
        self._pb.set_step_status("analyse", "running")
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

        _plot_theme = self._session.get("engine_cfg", {}).get("plot_theme", "light")

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
                    self._after(lambda: self._pb.set_step_status("analyse", "idle"))
                    self._after(lambda: self._cards["analyse"].set_status("idle"))
                    self._after(lambda: self._status("Step 5 ready — click to start another analysis."))
                else:
                    self._session.set_status("analyse", "done")
                    self._session.save()
                    self._after(lambda: self._pb.set_step_status("analyse", "done"))
                    self._after(lambda: self._cards["analyse"].set_status("done"))
                    self._after(lambda: self._status("Step 5 complete v"))
            except Exception:
                self._running = False
                tb = traceback.format_exc()
                self._logger.error(f"Step 5 error:\n{tb}")
                self._after(lambda: self._pb.set_step_status("analyse", "error"))
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
        global C, _DARK_THEME
        old_flat = _flat_theme_dict(_DARK_THEME)

        going_light = _DARK_THEME  # currently dark -> switching to light
        C.update(_THEME_LIGHT if going_light else _THEME_DARK)
        _DARK_THEME = not going_light
        try:
            with open(HERE / "theme.txt", "w", encoding="utf-8") as f:
                f.write("light" if going_light else "dark")
        except Exception:
            pass

        new_flat = _flat_theme_dict(_DARK_THEME)

        # Re-colour every already-built widget in place (main window, step
        # cards, settings panel, log console, and any currently-open
        # Toplevel settings window — all reachable via winfo_children()).
        _rethread_widget_colors(self, old_flat, new_flat)

        # ttk widgets use named styles rather than per-instance colour
        # options, so the progress bars need a separate style refresh.
        style = ttk.Style()
        style.configure("Overall.Horizontal.TProgressbar",
                        troughcolor=C["card"], background=C["cyan"])
        style.configure("Step.Horizontal.TProgressbar",
                        troughcolor=C["card"], background=C["green"])

        # StepCard.bg / accent live outside the widget-option walk above
        # (they're plain instance attributes used e.g. by set_status), and
        # the Log panel's tag colours are baked into a class-level dict at
        # import time — both need an explicit refresh.
        for key, card in getattr(self, "_cards", {}).items():
            card.refresh_theme(_STEP_BG_DARK[key] if _DARK_THEME else _STEP_BG_LIGHT[key])
        if hasattr(self, "_log_panel"):
            self._log_panel.refresh_theme()

        if getattr(self, "_logger", None):
            self._logger.info(f"Theme switched to {'dark' if _DARK_THEME else 'light'}.")
        
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
        # If the loaded session has no ntfy topic, fall back to the saved file
        if not self._settings.get("ntfy_topic", ""):
            _ntfy_file = HERE / "ntfy_topic.txt"
            if _ntfy_file.is_file():
                try:
                    _saved_topic = _ntfy_file.read_text(encoding="utf-8").strip()
                    if _saved_topic:
                        self._settings.set_val("ntfy_topic", _saved_topic)
                except Exception:
                    pass
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
            self._pb.set_step_status(meta["key"], st)
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
    global PipelineLogger, BSoidEngine, run_bsoid_prep, run_bsoid_prep_batch
    global filter_dlc_h5, cleanup_video_byproducts, create_umap_evolution_video
    global find_dlc_files, peek_dlc_bodyparts, group_bodyparts_by_region
    global resolve_env_shapes, ENV_PARADIGMS, ENV_PARADIGM_ROLE_VOCAB, ENV_PARADIGM_MIN_ROLES
    global load_dlc_file, _find_spine_indices, _spine_norm_factor
    global _MOD_VIDEO, _PATH_VIDEO, _MOD_ANALYSER, _PATH_ANALYSER

    try:
        from cube_core import (
            PipelineLogger, BSoidEngine, run_bsoid_prep, run_bsoid_prep_batch,
            filter_dlc_h5, cleanup_video_byproducts, create_umap_evolution_video,
            find_dlc_files, peek_dlc_bodyparts, group_bodyparts_by_region,
            resolve_env_shapes, ENV_PARADIGMS, ENV_PARADIGM_ROLE_VOCAB,
            ENV_PARADIGM_MIN_ROLES,
            load_dlc_file, _find_spine_indices, _spine_norm_factor,
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
            "Cube.BehaviouralExplorer.v5")
    except Exception:
        pass

    app = PipelineApp()
    app.mainloop()


if __name__ == "__main__":
    main()
