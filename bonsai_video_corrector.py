"""
CUBE Step 0 — Bonsai Video Corrector
Aligns multi-camera MKV recordings to a common time grid using Bonsai timestamp CSVs,
inserts duplicate frames at drop sites, and re-encodes to MP4 for 3D DLC analysis.
"""

from __future__ import annotations
import ctypes
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent

# Every Step 0 run splits a scanned root into two subfolders: the original
# recordings (RAW_DIRNAME) and the corrected, constant-fps output CUBE 1+
# reads from (CUBE0_DIRNAME). Both now live inside a single, timestamped
# "CUBE results <date>_<time>" run-root folder (see _resolve_or_create_run_root
# below), itself created inside the scanned input folder -- regardless of any
# custom output folder the user picks, since this app has no such setting
# (the old "Choose output folder..." picker was deliberately removed).
RAW_DIRNAME = "raw"
CUBE0_DIRNAME = "CUBE 0 - Corrected 3D vids"

# Prefix for the single shared per-run results folder. cube.py's own
# _resolve_cube_run_root() (D:\CUBE\cube.py) implements the SAME matching
# rule (non-recursive glob for this exact prefix, newest-by-embedded-
# timestamp tie-break) so that whichever of the two separate apps (this one,
# or cube.py's Step 1a/1b) runs first, the other adopts its run root instead
# of minting a second, differently-timestamped one. Keep the two in sync if
# this prefix or the timestamp format ever changes.
CUBE_RUN_ROOT_PREFIX = "CUBE results"
CUBE_RUN_ROOT_GLOB = f"{CUBE_RUN_ROOT_PREFIX}*"
_RUN_ROOT_TS_FMT = "%Y-%m-%d_%H-%M-%S"


def _resolve_or_create_run_root(folder: Path) -> Path:
    """
    Return the single "CUBE results <date>_<time>" run-root folder for a
    scanned recording folder -- reusing an existing one if present (so
    re-scanning/re-running Step 0 against the same folder, or a later
    cube.py Step 1a/1b run against it, all converge on one shared folder),
    else creating a fresh one.

    Matching rule (kept identical to cube.py's own _resolve_cube_run_root()
    so the two apps agree regardless of which runs first): a non-recursive
    glob for CUBE_RUN_ROOT_GLOB directly inside `folder`. If more than one
    match exists (e.g. Step 0 was run more than once against the same
    folder), the newest one by its embedded timestamp wins (falling back to
    mtime if the suffix doesn't parse as a timestamp) -- this codebase's
    existing "newest wins" convention for ambiguous candidate discovery.
    """
    folder = Path(folder)
    candidates = sorted(p for p in folder.glob(CUBE_RUN_ROOT_GLOB) if p.is_dir())
    if not candidates:
        run_root = folder / f"{CUBE_RUN_ROOT_PREFIX} {datetime.now().strftime(_RUN_ROOT_TS_FMT)}"
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root

    def _sort_key(p: Path):
        suffix = p.name[len(CUBE_RUN_ROOT_PREFIX):].strip()
        try:
            return (0, datetime.strptime(suffix, _RUN_ROOT_TS_FMT))
        except ValueError:
            return (0, datetime.fromtimestamp(p.stat().st_mtime))

    candidates.sort(key=_sort_key)
    return candidates[-1]

# ---------------------------------------------------------------------------
# Encoder label ↔ codec name
# ---------------------------------------------------------------------------
_ENC_OPTIONS = ["h264_nvenc", "h264_qsv", "libx264"]
_ENC_LABELS  = {
    "h264_nvenc": "NVENC · RTX 4000",
    "h264_qsv":   "QSV · Intel UHD",
    "libx264":    "libx264 · CPU",
}
_LABEL_TO_ENC = {v: k for k, v in _ENC_LABELS.items()}


def encoder_label(enc: str) -> str:
    return _ENC_LABELS.get(enc, enc)


def label_to_encoder(label: str, fallback: str = "libx264") -> str:
    return _LABEL_TO_ENC.get(label, fallback)


# ---------------------------------------------------------------------------
# Theme / Colors
# ---------------------------------------------------------------------------
DARK = dict(
    bg="#09090f", panel="#111120", card="#0a2a20", card2="#1e1e35",
    border="#2a2a4a", accent="#1dd1a1",
    text="#eaeaea", subtext="#7788aa", dim="#444466",
    log_bg="#07070d", log_fg="#00ff88",
    red="#f44336", yellow="#ffd60a", warn_bg="#2a1a00",
)
LIGHT = dict(
    bg="#f0f2f5", panel="#ffffff", card="#e8f8f3", card2="#e9ecef",
    border="#dee2e6", accent="#0fa87e",
    text="#222222", subtext="#666666", dim="#aaaaaa",
    log_bg="#f8f9fa", log_fg="#1a7a50",
    red="#c0392b", yellow="#7a4e00", warn_bg="#fff3cd",
)

FLOWSTEPS = [
    ("0", "Video Prep", "#1dd1a1"),
    ("1", "DLC",        "#4caf50"),
    ("2", "Pre-process","#00b4d8"),
    ("3", "Clustering", "#9c27b0"),
    ("4", "Annotate",   "#ff9800"),
    ("5", "Analyse",    "#e94560"),
]


def theme() -> dict:
    try:
        t = (HERE / "theme.txt").read_text().strip().lower()
    except Exception:
        t = "dark"
    return DARK if t != "light" else LIGHT


def is_dark() -> bool:
    try:
        return (HERE / "theme.txt").read_text().strip().lower() != "light"
    except Exception:
        return True


# ---------------------------------------------------------------------------
# FFmpeg / encoder helpers
# ---------------------------------------------------------------------------
def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(exe).exists():
            return str(exe)
    except Exception:
        pass
    fallback = Path(r"C:\Users\param\AppData\Local\Bonsai\Extensions\ffmpeg.exe")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError("FFmpeg not found. Install imageio-ffmpeg or Bonsai.")


_ENCODER_CACHE: Optional[str] = None


def detect_encoder(ffmpeg_exe: str) -> str:
    global _ENCODER_CACHE
    if _ENCODER_CACHE:
        return _ENCODER_CACHE
    for enc in _ENC_OPTIONS:
        try:
            r = subprocess.run(
                [ffmpeg_exe, "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=16x16",
                 "-t", "0.001", "-c:v", enc, "-f", "null", "-"],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                _ENCODER_CACHE = enc
                return enc
        except Exception:
            continue
    raise RuntimeError("No usable H.264 encoder found (tried nvenc, qsv, libx264).")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CamInfo:
    csv_path: Path
    mkv_path: Path
    cam_label: str
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    mapping: list = field(default_factory=list)   # [(src_idx, is_fill)]
    drops: list = field(default_factory=list)
    fill_count: int = 0
    n_out: int = 0
    status: str = "pending"   # pending / running / done / failed / cancelled / skipped
    error: str = ""


@dataclass
class SessionInfo:
    session_id: str
    cams: list[CamInfo] = field(default_factory=list)
    selected: bool = True
    valid: bool = True
    skip_reason: str = ""
    n_frames: int = 0
    duration_s: float = 0.0
    n_out_frames: int = 0
    common_start: float = 0.0   # POSIX seconds — stored for worker use
    fps: float = 0.0            # auto-detected native frame rate


# ---------------------------------------------------------------------------
# CSV / timestamp parsing
# ---------------------------------------------------------------------------
BONSAI_HEADER = 'ToString("o")'


def _parse_iso_ts(line: str) -> float:
    """Parse a Bonsai ISO-8601 timestamp (up to 7 fractional digits) to POSIX float."""
    line = line.strip()
    m = re.match(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d+)([+-]\d{2}:\d{2}|Z)$", line
    )
    if m:
        base, frac, tz = m.groups()
        frac = frac[:6].ljust(6, "0")   # Python datetime supports up to microseconds
        line = f"{base}.{frac}{tz}"
    dt = datetime.fromisoformat(line)
    return dt.timestamp()


def _parse_doric_created(value) -> float:
    """
    Parse a .doric file's top-level 'Created' attribute (e.g.
    'Wed Apr 15 17:08:49 2026', naive local time) to a POSIX timestamp.
    Returns 0.0 if missing/unparseable, in which case Pass 2 falls back to
    leaving that file's timestamps relative-to-session-start.
    """
    if not value:
        return 0.0
    try:
        return datetime.strptime(str(value), "%a %b %d %H:%M:%S %Y").timestamp()
    except Exception:
        return 0.0


def parse_csv(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != BONSAI_HEADER:
        raise ValueError(f"Not a Bonsai timestamp CSV: {path.name}")
    timestamps = [_parse_iso_ts(ln) for ln in lines[1:] if ln.strip()]
    return np.array(timestamps, dtype=np.float64)


# ---------------------------------------------------------------------------
# Scan folder for sessions
# ---------------------------------------------------------------------------
def _is_byproduct_path(p: Path, root: Path) -> bool:
    try:
        rel = p.relative_to(root)
    except ValueError:
        return False
    for part in rel.parts[:-1]:
        pl = part.lower()
        if pl in ("_calibration", "raw", "_prep", "logs") or "cube results" in pl or "cube 0" in pl:
            return True
    return False


def scan_folder(root: Path) -> list[SessionInfo]:
    sessions: dict[str, SessionInfo] = {}

    for csv_path in sorted(root.rglob("*.csv")):
        if _is_byproduct_path(csv_path, root):
            continue
        try:
            first = csv_path.read_text(encoding="utf-8", errors="ignore").split("\n")[0]
        except Exception:
            continue
        if first.strip() != BONSAI_HEADER:
            continue

        mkv_path = csv_path.with_suffix(".mkv")
        if not mkv_path.exists():
            continue

        stem = csv_path.stem
        m = re.match(r"^(cam\d+)_(\d+fps)(.*)", stem)
        if m:
            cam_label = m.group(1)
            session_id = m.group(2) + m.group(3)
        else:
            m2 = re.match(r"^(cam\d+)(.*)", stem)
            if m2:
                cam_label = m2.group(1)
                session_id = m2.group(2).lstrip("_") or "default"
            else:
                cam_label = stem
                session_id = "default"

        if session_id not in sessions:
            sessions[session_id] = SessionInfo(session_id=session_id)
        sessions[session_id].cams.append(
            CamInfo(csv_path=csv_path, mkv_path=mkv_path, cam_label=cam_label)
        )

    result = []
    for sid, sess in sessions.items():
        sess.cams.sort(key=lambda c: c.cam_label)
        _validate_session(sess)
        result.append(sess)

    result.extend(_scan_doric_sessions(root))

    result.sort(key=lambda s: s.session_id)
    return result


# ---------------------------------------------------------------------------
# Calibration-recording detection
# ---------------------------------------------------------------------------
def split_calibration_sessions(
        sessions: list[SessionInfo]) -> tuple[list[SessionInfo], list[SessionInfo]]:
    """
    Partition off any session whose session_id contains "calibration"
    (case-insensitive). Returns (calibration_sessions, normal_sessions).
    """
    calib, normal = [], []
    for s in sessions:
        (calib if "calibration" in s.session_id.lower() else normal).append(s)
    return calib, normal


def copy_calibration_sessions(calib_sessions: list[SessionInfo], out_dir: Path) -> Path:
    """
    Copy each calibration session's raw source files (unmodified) into
    {out_dir}/_calibration/{session_id}/. Handles both Bonsai (csv+mkv) and
    Doric (.doric+video) originals.

    Each session gets its own subfolder rather than a flat destination:
    Doric camera video filenames are NOT unique across sessions (every
    session's _Videos folder reuses e.g. "Cam1BCF-28C4-01_Series0001.mp4"
    verbatim, since "Series0001" only numbers recordings within a single
    session) — a flat destination would silently overwrite one
    calibration recording with another whenever more than one session
    matches the "calibration" keyword.
    """
    dest_root = out_dir / "_calibration"
    copied_doric: set[Path] = set()

    for sess in calib_sessions:
        dest = dest_root / sess.session_id
        dest.mkdir(parents=True, exist_ok=True)

        for cam in sess.cams:
            # camN-prefixed so the copied file's camera identity is
            # unambiguous even for Doric originals, whose own filenames
            # (e.g. "Cam1BCF-28C4-01_Series0001.mp4") don't encode camN and
            # are reused verbatim across every session.
            try:
                shutil.copy2(cam.mkv_path, dest / f"{cam.cam_label}_{cam.mkv_path.name}")
            except Exception:
                pass

            if cam.csv_path.exists():
                try:
                    shutil.copy2(cam.csv_path, dest / cam.csv_path.name)
                except Exception:
                    pass
            else:
                # Doric-origin camera: csv_path is synthetic — copy the
                # parent .doric file itself instead (once per session).
                doric_path = cam.mkv_path.parent.parent / f"{sess.session_id}.doric"
                if doric_path.exists() and doric_path not in copied_doric:
                    try:
                        shutil.copy2(doric_path, dest / doric_path.name)
                    except Exception:
                        pass
                    copied_doric.add(doric_path)

    return dest_root


def move_session_raw_files(sessions: list, root: Path, raw_dir: Path,
                            log=None) -> None:
    """
    Move each session's original recording files (Bonsai .mkv+.csv, or
    Doric .doric+resolved video) out of `root` and into `raw_dir`,
    mirroring their path relative to `root` so nothing collides. Updates
    each CamInfo's csv_path/mkv_path in place to point at the new location
    (CameraWorker.run() opens cam.mkv_path directly, so this must happen
    before correction starts, not after).

    Best-effort per file: a single failed move is logged and skipped
    rather than aborting the whole session, matching this module's
    existing crash-protection convention.
    """
    def _relocate(src: Path) -> Path:
        try:
            rel = src.relative_to(root)
        except ValueError:
            rel = Path(src.name)
        dest = raw_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return dest  # already moved (e.g. shared Doric file across cams)
        try:
            src.rename(dest)
        except OSError:
            shutil.move(str(src), str(dest))
        return dest

    moved_doric: set[Path] = set()

    for sess in sessions:
        for cam in sess.cams:
            # Compute the Doric parent path from the ORIGINAL (pre-move)
            # location before cam.mkv_path is reassigned below.
            doric_path = None
            if not cam.csv_path.exists():
                doric_path = cam.mkv_path.parent.parent / f"{sess.session_id}.doric"

            try:
                if cam.mkv_path.exists():
                    cam.mkv_path = _relocate(cam.mkv_path)
            except Exception as e:
                if log:
                    log(f"Could not move {cam.mkv_path.name} to raw/: {e}")

            if cam.csv_path.exists():
                try:
                    cam.csv_path = _relocate(cam.csv_path)
                except Exception as e:
                    if log:
                        log(f"Could not move {cam.csv_path.name} to raw/: {e}")
            elif doric_path is not None and doric_path not in moved_doric \
                    and doric_path.exists():
                # Doric-origin camera: csv_path is synthetic — the real
                # timestamp source is the session's .doric file, shared
                # across every camera in the session, so relocate it once.
                try:
                    _relocate(doric_path)
                except Exception as e:
                    if log:
                        log(f"Could not move {doric_path.name} to raw/: {e}")
                moved_doric.add(doric_path)


# ---------------------------------------------------------------------------
# Doric Neuroscience Studio session discovery
# ---------------------------------------------------------------------------
def _scan_doric_sessions(root: Path) -> list[SessionInfo]:
    """
    Recursively discover Doric sessions under *root*.

    Each session is a `{stem}.doric` HDF5 file sitting next to a
    `{stem}_Videos/` folder.  Per-camera frame timestamps live inside the
    .doric file at `.../Video/Series*/<serial>/Time`; the parent (serial)
    group's `RelativeFilePath` attribute names the actual video file,
    relative to the .doric file's own directory.  `*incomplete*.doric`
    files are never used as a timestamp source.

    camN labels are assigned once globally (sorted serial → cam0, cam1, ...)
    across every valid .doric file found under root, so the same physical
    camera always maps to the same camN regardless of which subset of
    cameras any individual session recorded.
    """
    if not _HAS_H5PY:
        return []

    doric_paths = sorted(
        p for p in root.rglob("*.doric")
        if "incomplete" not in p.name.lower() and not _is_byproduct_path(p, root)
    )
    if not doric_paths:
        return []

    # Pass 1 — collect every (doric_path, serial, dataset_path, rel_video_path)
    # tuple, the global set of distinct serials, and each file's absolute
    # session-start epoch (Doric's Time datasets are seconds *relative* to
    # this, not POSIX time — added back in Pass 2 so the corrected
    # timestamps CSV carries real, human-meaningful UTC times like the
    # Bonsai path does, instead of dates near the 1970 epoch).
    entries: list[tuple[Path, str, str, str]] = []
    all_serials: set[str] = set()
    doric_created: dict[Path, float] = {}

    for doric_path in doric_paths:
        try:
            with h5py.File(doric_path, "r") as f:
                doric_created[doric_path] = _parse_doric_created(f.attrs.get("Created"))
                found = []

                def _visit(name, obj):
                    if (isinstance(obj, h5py.Dataset) and obj.name.endswith("/Time")
                            and "/Video/Series" in obj.name):
                        parent = obj.parent
                        serial = parent.name.rsplit("/", 1)[-1]
                        rel_path = parent.attrs.get("RelativeFilePath")
                        if rel_path:
                            found.append((serial, obj.name, str(rel_path)))

                f.visititems(_visit)
        except Exception:
            continue

        for serial, dset_path, rel_path in found:
            entries.append((doric_path, serial, dset_path, rel_path))
            all_serials.add(serial)

    serial_to_cam = {serial: f"cam{i}" for i, serial in enumerate(sorted(all_serials))}

    # Pass 2 — build one SessionInfo per .doric file.
    by_session: dict[Path, SessionInfo] = {}
    for doric_path, serial, dset_path, rel_path in entries:
        session_id = doric_path.stem
        sess = by_session.setdefault(doric_path, SessionInfo(session_id=session_id))

        video_path = (doric_path.parent / rel_path.lstrip("/\\")).resolve()
        if not video_path.exists():
            continue

        try:
            with h5py.File(doric_path, "r") as f:
                timestamps = np.asarray(f[dset_path][:], dtype=np.float64)
        except Exception:
            continue

        created_epoch = doric_created.get(doric_path, 0.0)
        if created_epoch:
            timestamps = timestamps + created_epoch

        cam_label = serial_to_cam[serial]
        csv_path = doric_path.parent / f"{cam_label}_{session_id}.csv"
        sess.cams.append(CamInfo(csv_path=csv_path, mkv_path=video_path,
                                  cam_label=cam_label, timestamps=timestamps))

    result = []
    for sess in by_session.values():
        sess.cams.sort(key=lambda c: c.cam_label)
        if len(sess.cams) < 2:
            sess.valid = False
            sess.skip_reason = "Fewer than 2 cameras with resolvable video/timestamps"
            sess.selected = False
        else:
            _validate_session(sess)
        result.append(sess)

    return result


def _validate_session(sess: SessionInfo):
    for cam in sess.cams:
        if len(cam.timestamps):
            ts = cam.timestamps
        else:
            try:
                ts = parse_csv(cam.csv_path)
            except Exception as e:
                sess.valid = False
                sess.skip_reason = f"CSV parse error: {e}"
                sess.selected = False
                return
        if len(ts) < 2:
            sess.valid = False
            sess.skip_reason = "Aborted — 0 usable frames"
            sess.selected = False
            return
        cam.timestamps = ts

    frame_counts = [len(c.timestamps) for c in sess.cams]
    sess.n_frames = min(frame_counts)
    sess.duration_s = min(c.timestamps[-1] - c.timestamps[0] for c in sess.cams)
    sess.fps = detect_session_fps(sess)


def detect_session_fps(sess: SessionInfo) -> float:
    """
    Auto-detect the session's native frame rate from its cameras' own
    timestamps (median of each camera's (n-1)/duration), rounded to the
    nearest whole number since real camera rates are effectively always
    integral (15/24/30/60...).
    """
    native = []
    for c in sess.cams:
        if len(c.timestamps) >= 2:
            dur = c.timestamps[-1] - c.timestamps[0]
            if dur > 0:
                native.append((len(c.timestamps) - 1) / dur)
    return round(float(np.median(native))) if native else 30.0


# ---------------------------------------------------------------------------
# Frame mapping
# ---------------------------------------------------------------------------
def build_mapping(timestamps: np.ndarray, fps: float,
                  common_start: float, common_end: float) -> list[tuple[int, bool]]:
    """
    Map each output grid slot to the nearest source frame index.
    Uses a monotonic forward scan — never seeks backwards.
    A slot is marked as 'fill' when the nearest source frame is >½ frame period away.
    """
    n_out = max(0, int(round((common_end - common_start) * fps)) + 1)
    half_frame = 0.5 / fps
    n_src = len(timestamps)
    src = 0
    mapping: list[tuple[int, bool]] = []

    for k in range(n_out):
        grid_t = common_start + k / fps
        while (src + 1 < n_src and
               abs(timestamps[src + 1] - grid_t) < abs(timestamps[src] - grid_t)):
            src += 1
        is_fill = abs(timestamps[src] - grid_t) > half_frame
        mapping.append((min(src, n_src - 1), is_fill))

    return mapping


def compute_session_mappings(sess: SessionInfo, fps: float, drop_threshold_s: float):
    common_start = max(c.timestamps[0] for c in sess.cams)
    common_end   = min(c.timestamps[-1] for c in sess.cams)
    sess.common_start = common_start   # stored for worker use

    for cam in sess.cams:
        cam.mapping = build_mapping(cam.timestamps, fps, common_start, common_end)
        cam.fill_count = sum(1 for _, f in cam.mapping if f)
        cam.drops = [i for i, dt in enumerate(np.diff(cam.timestamps))
                     if dt > drop_threshold_s]
        cam.n_out = len(cam.mapping)

    sess.n_out_frames = sess.cams[0].n_out if sess.cams else 0


# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------
def estimate_output_bytes(sess: SessionInfo, fps: float) -> int:
    n_out = sess.n_out_frames or sess.n_frames
    duration = n_out / fps
    return int(duration * 1_500_000 / 8 * 1.2)   # ~1.5 Mbps at 720p, 20% headroom


def check_disk_space(output_dir: Path, needed_bytes: int) -> bool:
    return shutil.disk_usage(output_dir).free >= needed_bytes


# ---------------------------------------------------------------------------
# Camera worker (runs in thread pool)
# ---------------------------------------------------------------------------
class CameraWorker:
    def __init__(self, cam: CamInfo, session_id: str, common_start: float,
                 output_dir: Path, fps: float, crf: int,
                 encoder: str, ffmpeg_exe: str,
                 progress_q: queue.Queue, cancel_event: threading.Event):
        self.cam = cam
        self.session_id = session_id
        self.common_start = common_start
        self.output_dir = output_dir
        self.fps = fps
        self.crf = crf
        self.encoder = encoder
        self.ffmpeg_exe = ffmpeg_exe
        self.q = progress_q
        self.cancel = cancel_event

    def _put(self, msg_type: str, **kwargs):
        self.q.put({"type": msg_type, "cam": self.cam.cam_label,
                    "session": self.session_id, **kwargs})

    def run(self):
        cam = self.cam
        cam.status = "running"
        self._put("status", status="running")

        cap = cv2.VideoCapture(str(cam.mkv_path))
        if not cap.isOpened():
            cam.status = "failed"
            cam.error = f"Cannot open {cam.mkv_path.name}"
            self._put("status", status="failed", error=cam.error)
            return

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_out = len(cam.mapping)

        stem = cam.csv_path.stem
        out_mp4 = self.output_dir / f"{stem}_corrected.mp4"
        out_csv = self.output_dir / f"{stem}_corrected_timestamps.csv"
        tmp_mp4 = self.output_dir / f"{stem}_corrected.tmp.mp4"

        cmd = [
            self.ffmpeg_exe, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}", "-r", str(self.fps),
            "-i", "pipe:0",
            *self._encoder_args(),
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),          # enforce constant FPS in output container
            "-vsync", "cfr",              # strictly constant frame rate for DLC
            "-movflags", "+faststart",
            str(tmp_mp4),
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
        except Exception as e:
            cam.status = "failed"
            cam.error = str(e)
            self._put("status", status="failed", error=cam.error)
            cap.release()
            return

        ts_rows = ["frame,grid_time_posix,grid_time_iso,fill"]
        prev_frame: Optional[np.ndarray] = None
        src_cursor = -1
        fills = 0
        frame: Optional[np.ndarray] = None   # explicit init avoids unbound-var warning

        try:
            for k, (src_idx, is_fill) in enumerate(cam.mapping):
                if self.cancel.is_set():
                    proc.terminate()
                    cam.status = "cancelled"
                    self._put("status", status="cancelled")
                    cap.release()
                    _safe_delete(tmp_mp4)
                    return

                if is_fill and prev_frame is not None:
                    frame = prev_frame
                    fills += 1
                else:
                    # Advance VideoCapture sequentially — never seek backwards
                    while src_cursor < src_idx:
                        ret, f = cap.read()
                        src_cursor += 1
                        if ret:
                            frame = f
                        else:
                            # Video ended early — hold last valid frame
                            if frame is None:
                                frame = np.zeros((H, W, 3), np.uint8)
                            break

                if frame is None:
                    frame = np.zeros((H, W, 3), np.uint8)

                proc.stdin.write(frame.tobytes())
                prev_frame = frame.copy()   # copy so pipe writes can't alias it

                # Write the GRID time (not the source capture time) for DLC sync
                grid_t_posix = self.common_start + k / self.fps
                grid_t_iso = datetime.fromtimestamp(
                    grid_t_posix, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S.%f")
                ts_rows.append(f"{k},{grid_t_posix:.6f},{grid_t_iso},{int(is_fill)}")

                if k % 30 == 0 or k == n_out - 1:
                    self._put("progress", done=k + 1, total=n_out, fills=fills)

            proc.stdin.close()
            _, stderr = proc.communicate(timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode(errors="replace")[-600:])

            # Atomic rename
            if out_mp4.exists():
                out_mp4.unlink()
            tmp_mp4.rename(out_mp4)

            out_csv.write_text("\n".join(ts_rows), encoding="utf-8")

            cam.status = "done"
            cam.fill_count = fills
            self._put("status", status="done", fills=fills, out=str(out_mp4))
            self._put("log", msg=f"{cam.cam_label}: done — {fills} fill(s), "
                                  f"{n_out} frames → {out_mp4.name}")

        except Exception as e:
            try:
                proc.terminate()
            except Exception:
                pass
            cam.status = "failed"
            cam.error = str(e)[:400]
            self._put("status", status="failed", error=cam.error)
            self._put("log", msg=f"{cam.cam_label}: FAILED — {cam.error}")
            _safe_delete(tmp_mp4)
        finally:
            cap.release()

    def _encoder_args(self) -> list[str]:
        enc = self.encoder
        crf = str(self.crf)
        if enc == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", crf]
        elif enc == "h264_qsv":
            # QSV does not support CRF; use quality-based VBR closest equivalent
            return ["-c:v", "h264_qsv", "-preset", "faster",
                    "-global_quality", crf, "-look_ahead", "1"]
        else:
            return ["-c:v", "libx264", "-preset", "fast", "-crf", crf]


def _safe_delete(p: Path):
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Post-processing verification and report
# ---------------------------------------------------------------------------
def probe_mp4(path: Path) -> dict:
    """Return frame count and FPS from an MP4 via OpenCV — no full decode."""
    if not path.exists():
        return {"ok": False, "n": -1, "fps": 0.0}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "n": -1, "fps": 0.0}
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"ok": True, "n": n, "fps": fps}


def build_verification(sessions: list[SessionInfo], out_dir: Path,
                        enc: str, crf: int,
                        thresh_s: float) -> list[dict]:
    """Build the full post-processing report dict. Runs in the orchestrator thread."""
    result = []
    for sess in sessions:
        cams = sess.cams
        fps  = sess.fps

        # per-camera stats
        cam_data = []
        for cam in cams:
            ts    = cam.timestamps
            n_src = len(ts)
            if n_src >= 2:
                ivs        = np.diff(ts) * 1000          # ms
                mean_iv    = float(np.mean(ivs))
                jitter     = float(np.std(ivs))
                drop_durs  = [float(ivs[d]) for d in cam.drops if d < len(ivs)]
                max_drop_ms = max(drop_durs) if drop_durs else 0.0
            else:
                mean_iv = jitter = max_drop_ms = 0.0

            stem    = cam.csv_path.stem
            out_mp4 = out_dir / f"{stem}_corrected.mp4"
            probe   = probe_mp4(out_mp4)

            cam_data.append({
                "cam":              cam.cam_label,
                "src_frames":       n_src,
                "mean_interval_ms": mean_iv,
                "jitter_ms":        jitter,
                "drops":            len(cam.drops),
                "max_drop_ms":      max_drop_ms,
                "fills":            cam.fill_count,
                "out_frames":       cam.n_out,
                "fill_rate_pct":    100.0 * cam.fill_count / max(cam.n_out, 1),
                "status":           cam.status,
                "verified_frames":  probe["n"],
                "verified_fps":     probe["fps"],
                "out_path":         str(out_mp4) if out_mp4.exists() else "",
            })

        # session-level stats
        valid_ts   = [c.timestamps for c in cams if len(c.timestamps) > 0]
        first_ts   = [ts[0] for ts in valid_ts]
        start_spr  = (max(first_ts) - min(first_ts)) * 1000 if len(first_ts) > 1 else 0.0
        common_win = (sess.n_out_frames - 1) / fps if sess.n_out_frames > 1 else 0.0

        # alignment via corrected timestamp CSVs
        grid_jitter_ms   = -1.0
        cross_cam_off_ms = -1.0
        grid_times_all: list[np.ndarray] = []

        for cam in cams:
            if cam.status != "done":
                continue
            csv_out = out_dir / f"{cam.csv_path.stem}_corrected_timestamps.csv"
            if not csv_out.exists():
                continue
            try:
                rows = np.genfromtxt(str(csv_out), delimiter=",", skip_header=1)
                if rows.ndim == 1:
                    rows = rows.reshape(1, -1)
                gt  = rows[:, 1]
                grid_times_all.append(gt)
                ivs = np.diff(gt) * 1000
                jit = float(np.max(np.abs(ivs - np.mean(ivs))))
                if grid_jitter_ms < 0 or jit > grid_jitter_ms:
                    grid_jitter_ms = jit   # worst-case across cameras
            except Exception:
                pass

        if len(grid_times_all) >= 2:
            min_len = min(len(g) for g in grid_times_all)
            diffs: list[np.ndarray] = []
            for i in range(len(grid_times_all)):
                for j in range(i + 1, len(grid_times_all)):
                    diffs.append(
                        np.abs(grid_times_all[i][:min_len]
                               - grid_times_all[j][:min_len]) * 1000)
            cross_cam_off_ms = float(np.max(np.concatenate(diffs)))

        # frame-count alignment from probed MP4s
        done_data  = [cd for cd in cam_data if cd["status"] == "done"]
        ver_counts = [cd["verified_frames"] for cd in done_data]
        fc_ok      = (len(set(ver_counts)) == 1
                      and bool(ver_counts) and ver_counts[0] > 0)
        frame_count = ver_counts[0] if fc_ok else -1

        # outcome
        n_done  = sum(1 for c in cams if c.status == "done")
        n_total = len(cams)
        if n_done == n_total and fc_ok:
            outcome = "OK"
        elif n_done > 0:
            outcome = "PARTIAL"
        else:
            outcome = "FAIL"

        result.append({
            "session_id":      sess.session_id,
            "fps":             fps,
            "encoder":         enc,
            "crf":             crf,
            "thresh_ms":       round(thresh_s * 1000, 1),
            "common_window_s": common_win,
            "start_spread_ms": start_spr,
            "cameras":         cam_data,
            "alignment": {
                "frame_count_ok":      fc_ok,
                "frame_count":         frame_count,
                "grid_jitter_ms":      grid_jitter_ms,
                "cross_cam_offset_ms": cross_cam_off_ms,
            },
            "outcome": outcome,
        })

    return result


def _write_report_txt(sessions_data: list[dict], path: Path):
    """Write a plain-text correction report."""
    W = 70
    lines = ["=" * W,
             "  CUBE Step 0 -- Correction Report",
             f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * W, ""]

    for sd in sessions_data:
        cams = sd["cameras"]
        al   = sd.get("alignment", {})

        lines += [
            f"SESSION: {sd['session_id']}",
            (f"  Duration: {sd['common_window_s']:.2f} s"
             f"  |  {int(sd['fps'])} fps"
             f"  |  {sd['encoder']}  CRF {sd['crf']}"),
            ""
        ]

        lines += [
            "INPUT DATA QUALITY",
            (f"  {'Camera':<8}  {'Src frames':>10}  {'Mean intv':>10}"
             f"  {'Jitter':>8}  {'Drops':>5}  {'Max drop':>10}"),
        ]
        for cam in cams:
            if cam["status"] not in ("done", "failed"):
                continue
            md = f"{cam['max_drop_ms']:.0f} ms" if cam["drops"] > 0 else "--"
            lines.append(
                f"  {cam['cam']:<8}  {cam['src_frames']:>10}"
                f"  {cam['mean_interval_ms']:>8.2f} ms"
                f"  {cam['jitter_ms']:>6.2f} ms"
                f"  {cam['drops']:>5}  {md:>10}")
        lines.append(f"  Camera start-time spread: {sd['start_spread_ms']:.1f} ms")
        lines.append("")

        lines += [
            "CORRECTION IMPACT",
            (f"  {'Camera':<8}  {'Src frames':>10}  {'Out frames':>10}"
             f"  {'Fills':>6}  {'Fill %':>7}  {'Status':>8}"),
        ]
        for cam in cams:
            lines.append(
                f"  {cam['cam']:<8}  {cam['src_frames']:>10}"
                f"  {cam['out_frames']:>10}"
                f"  {cam['fills']:>6}"
                f"  {cam['fill_rate_pct']:>6.1f}%"
                f"  {cam['status'].upper():>8}")
        lines.append(
            f"  Common window: {sd['common_window_s']:.2f} s"
            f"  ({al.get('frame_count', '?')} frames)")
        lines.append("")

        def _r(ok: bool) -> str: return "OK  " if ok else "FAIL"
        fc_ok    = al.get("frame_count_ok", False)
        gj       = al.get("grid_jitter_ms", -1.0)
        cc       = al.get("cross_cam_offset_ms", -1.0)
        fps_devs = [abs(c.get("verified_fps", 0) - sd["fps"])
                    for c in cams if c["status"] == "done"]
        fps_ok   = all(d < 0.5 for d in fps_devs) if fps_devs else False

        lines += [
            "VERIFICATION",
            (f"  [{_r(fc_ok)}]  Frame count alignment : "
             f"{al.get('frame_count', '?')} frames on all cameras"),
            f"  [{_r(fps_ok)}]  Container FPS         : {int(sd['fps'])} fps",
            ((f"  [{_r(0 <= gj < 1.0)}]  Grid uniformity       : +/- {gj:.4f} ms")
             if gj >= 0 else "  [----]  Grid uniformity       : CSVs not found"),
            ((f"  [{_r(0 <= cc < 0.1)}]  Cross-camera offset   : {cc:.4f} ms")
             if cc >= 0 else "  [----]  Cross-camera offset   : CSVs not found"),
            "",
            f"OUTCOME: {sd.get('outcome', '?')}",
            "",
            "OUTPUT FILES",
        ]
        for cam in cams:
            if cam.get("out_path"):
                lines.append(f"  {cam['out_path']}")
        lines += ["", "=" * W, ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------
def _font(size: int = 10, bold: bool = False) -> ctk.CTkFont:
    return ctk.CTkFont(family="Segoe UI", size=size,
                       weight="bold" if bold else "normal")


def _load_logo(dark: bool) -> Optional[ctk.CTkImage]:
    name = "CUBE_logo dark theme.png" if dark else "CUBE_logo.png"
    p = HERE / "assets" / name
    if not p.exists():
        return None
    try:
        img = Image.open(p)
        h = 48
        w = int(img.width * h / img.height)
        img = img.resize((w, h), Image.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
class VideoPrep(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._dark = is_dark()
        self._C = theme()
        ctk.set_appearance_mode("dark" if self._dark else "light")

        self.title("CUBE: Step 0 — Video Prep")
        self.geometry("1200x880")
        self.minsize(920, 680)
        ico = HERE / "assets" / "CUBE.ico"
        if ico.exists():
            try:
                self.iconbitmap(str(ico))
            except Exception:
                pass

        self._ffmpeg:   Optional[str] = None
        self._encoder:  Optional[str] = None
        self._sessions: list[SessionInfo] = []
        self._session_vars: dict[str, ctk.BooleanVar] = {}
        self._progress_q: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._running = False
        self._start_time: Optional[float] = None

        self._cam_progress: dict[str, dict] = {}
        self._overall_done = 0
        self._overall_total = 0
        self._logo_img: Optional[ctk.CTkImage] = None

        self._build_ui()
        self._detect_ffmpeg_async()
        self.after(100, self._poll_queue)

        # Optional CLI arg: a folder path, pre-fills Input Folder and
        # auto-triggers a scan — used when CUBE (cube_3d_dlc.py's
        # dlc_3d_autorun_step0) launches this app for the user instead of
        # them browsing to the same folder they already picked in CUBE.
        # Only pre-fills/scans; the user still reviews the session list and
        # clicks Run themselves — this is not a headless/unattended mode.
        if len(sys.argv) > 1:
            _arg_folder = Path(sys.argv[1])
            if _arg_folder.is_dir():
                self._folder_var.set(str(_arg_folder))
                self.after(300, self._on_scan)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        C = self._C
        self.configure(fg_color=C["bg"])

        self._build_header()
        self._build_flowbar()
        self._build_footer()          # pack bottom BEFORE expanding content

        content = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        content.pack(fill="both", expand=True, padx=16, pady=(0, 4))
        content.columnconfigure(0, weight=1)

        self._build_folder_row(content)
        self._build_session_panel(content)
        self._build_settings(content)
        self._build_action_bar(content)   # prominent Start/Cancel — between settings & progress
        self._build_progress(content)
        self._build_log(content)

    def _build_header(self):
        C = self._C
        hdr = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=68)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self._logo_img = _load_logo(self._dark)
        if self._logo_img:
            ctk.CTkLabel(hdr, image=self._logo_img, text="").pack(
                side="left", padx=(16, 8), pady=8)

        title_f = ctk.CTkFrame(hdr, fg_color="transparent")
        title_f.pack(side="left", pady=8)
        ctk.CTkLabel(title_f, text="CUBE: Step 0 — Video Prep",
                     font=_font(17, True), text_color=C["accent"]).pack(anchor="w")
        ctk.CTkLabel(title_f,
                     text="Correct dropped frames · Align cameras · Export MP4 for DLC",
                     font=_font(9), text_color=C["subtext"]).pack(anchor="w")

        ctk.CTkButton(hdr, text="☽ Dark" if self._dark else "☀ Light",
                      width=90, height=30, font=_font(9),
                      fg_color=C["card2"], hover_color=C["border"],
                      text_color=C["text"],
                      command=self._toggle_theme).pack(side="right", padx=16, pady=18)

    def _build_flowbar(self):
        C = self._C
        bar = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=32)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(expand=True, pady=4)
        for i, (num, label, color) in enumerate(FLOWSTEPS):
            is_active = (num == "0")
            ctk.CTkLabel(inner, text=f"  {num} {label}  ",
                         font=_font(9, is_active),
                         fg_color=color if is_active else "transparent",
                         text_color="#000000" if is_active else C["dim"],
                         corner_radius=4).pack(side="left", padx=1)
            if i < len(FLOWSTEPS) - 1:
                ctk.CTkLabel(inner, text="▸", font=_font(9),
                             text_color=C["dim"],
                             fg_color="transparent").pack(side="left")

    def _build_folder_row(self, parent):
        C = self._C
        row = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=6)
        row.pack(fill="x", pady=(10, 4))

        ctk.CTkLabel(row, text="Input Folder", font=_font(9, True),
                     text_color=C["subtext"]).pack(side="left", padx=(12, 6), pady=8)
        self._folder_var = ctk.StringVar(value="")
        ctk.CTkEntry(row, textvariable=self._folder_var,
                     font=_font(9), height=30, width=520).pack(
            side="left", padx=4, pady=8)

        ctk.CTkButton(row, text="Browse", width=80, height=30, font=_font(9),
                      fg_color=C["card2"], hover_color=C["border"],
                      text_color=C["text"],
                      command=self._browse_folder).pack(side="left", padx=4)

        self._scan_btn = ctk.CTkButton(
            row, text="Scan Folder", width=110, height=30,
            font=_font(9, True), fg_color=C["accent"],
            hover_color="#15a882", text_color="#000000",
            command=self._on_scan)
        self._scan_btn.pack(side="left", padx=(8, 12))

    def _build_session_panel(self, parent):
        C = self._C
        card = ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=6,
                            border_width=1, border_color=C["border"])
        card.pack(fill="x", pady=4)

        ctk.CTkLabel(card, text="Discovered Sessions", font=_font(10, True),
                     text_color=C["accent"]).pack(anchor="w", padx=12, pady=(8, 2))

        self._session_frame = ctk.CTkScrollableFrame(card, fg_color="transparent",
                                                      height=80)
        self._session_frame.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(self._session_frame, text="No folder scanned yet.",
                     font=_font(9), text_color=C["dim"]).pack(anchor="w", padx=4)

    def _build_settings(self, parent):
        C = self._C
        card = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=6)
        card.pack(fill="x", pady=4)

        def _lbl(p, t):
            ctk.CTkLabel(p, text=t, font=_font(9),
                         text_color=C["subtext"]).pack(side="left", padx=(0, 4))

        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=(8, 4))

        _lbl(row1, "FPS")
        ctk.CTkLabel(row1, text="auto-detected per session", font=_font(9),
                     text_color=C["dim"]).pack(side="left", padx=(0, 16))

        _lbl(row1, "Encoder")
        self._enc_var = ctk.StringVar(value="Detecting…")
        self._enc_menu = ctk.CTkOptionMenu(
            row1, values=["Detecting…"], variable=self._enc_var,
            width=180, height=28, font=_font(9),
            fg_color=C["card2"], button_color=C["accent"],
            button_hover_color="#15a882", text_color=C["text"])
        self._enc_menu.pack(side="left", padx=(0, 16))

        _lbl(row1, "Quality CRF")
        self._crf_var = ctk.StringVar(value="23")
        ctk.CTkEntry(row1, textvariable=self._crf_var, width=50, height=28,
                     font=_font(9)).pack(side="left", padx=(0, 16))

        _lbl(row1, "Drop threshold")
        self._thresh_var = ctk.StringVar(value="50")
        ctk.CTkEntry(row1, textvariable=self._thresh_var, width=60, height=28,
                     font=_font(9)).pack(side="left", padx=(0, 4))
        ctk.CTkLabel(row1, text="ms", font=_font(9),
                     text_color=C["subtext"]).pack(side="left")

        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=(0, 4))

        _lbl(row2, "Output")
        ctk.CTkLabel(
            row2,
            text=f"{CUBE_RUN_ROOT_PREFIX} <date>_<time>/raw/ (originals)  +  "
                 f".../{CUBE0_DIRNAME}/ (corrected) — "
                 f"created inside the recording folder",
            font=_font(9), text_color=C["dim"]).pack(side="left")

        row3 = ctk.CTkFrame(card, fg_color="transparent")
        row3.pack(fill="x", padx=12, pady=(0, 8))
        self._del_mkv_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            row3,
            text="Delete original MKV files after successful conversion",
            variable=self._del_mkv_var, font=_font(9),
            text_color=C["subtext"], fg_color=C["accent"],
            hover_color="#15a882", checkmark_color="#000000",
            command=self._on_del_mkv_toggle,
        ).pack(side="left")
        self._del_warn_lbl = ctk.CTkLabel(row3, text="", font=_font(8),
                                           text_color=C["red"])
        self._del_warn_lbl.pack(side="left", padx=(10, 0))

    def _build_action_bar(self, parent):
        """Prominent Start / Cancel bar — sits between settings and progress."""
        C = self._C
        bar = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=6)
        bar.pack(fill="x", pady=(4, 4))

        self._start_btn = ctk.CTkButton(
            bar, text="▶  Start Processing",
            height=46, font=_font(13, True),
            fg_color=C["accent"], hover_color="#15a882", text_color="#000000",
            command=self._on_start,
        )
        self._start_btn.pack(side="left", expand=True, fill="x",
                             padx=(12, 6), pady=8)

        self._cancel_btn = ctk.CTkButton(
            bar, text="✕  Cancel", width=130, height=46,
            font=_font(11, True),
            fg_color=C["card2"], hover_color=C["red"],
            text_color=C["text"], state="disabled",
            command=self._on_cancel,
        )
        self._cancel_btn.pack(side="left", padx=(0, 12), pady=8)

    def _build_progress(self, parent):
        C = self._C
        card = ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=6,
                            border_width=1, border_color=C["border"])
        card.pack(fill="x", pady=4)

        ctk.CTkLabel(card, text="Progress", font=_font(10, True),
                     text_color=C["accent"]).pack(anchor="w", padx=12, pady=(8, 4))

        self._prog_frame = ctk.CTkFrame(card, fg_color="transparent")
        self._prog_frame.pack(fill="x", padx=12, pady=(0, 10))
        self._prog_frame.columnconfigure(1, weight=1)

        self._overall_lbl = ctk.CTkLabel(self._prog_frame, text="Overall",
                                         font=_font(9), text_color=C["text"],
                                         width=64, anchor="w")
        self._overall_lbl.grid(row=0, column=0, sticky="w", pady=2)
        self._overall_bar = ctk.CTkProgressBar(self._prog_frame, height=14,
                                               progress_color=C["accent"],
                                               fg_color=C["card2"])
        self._overall_bar.grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=2)
        self._overall_bar.set(0)
        self._overall_info = ctk.CTkLabel(self._prog_frame, text="—",
                                          font=_font(9), text_color=C["subtext"],
                                          width=130, anchor="w")
        self._overall_info.grid(row=0, column=2, sticky="w", pady=2)

        self._cam_bars:      dict[str, ctk.CTkProgressBar] = {}
        self._cam_info_lbls: dict[str, ctk.CTkLabel]       = {}
        self._cam_lbl_wgts:  dict[str, ctk.CTkLabel]       = {}

    def _rebuild_cam_bars(self, cam_labels: list[str]):
        C = self._C
        for w in (list(self._cam_bars.values()) +
                  list(self._cam_info_lbls.values()) +
                  list(self._cam_lbl_wgts.values())):
            w.destroy()
        self._cam_bars.clear()
        self._cam_info_lbls.clear()
        self._cam_lbl_wgts.clear()
        self._cam_progress.clear()

        for i, cl in enumerate(cam_labels, start=1):
            lbl = ctk.CTkLabel(self._prog_frame, text=cl, font=_font(9),
                               text_color=C["text"], width=64, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=1)
            bar = ctk.CTkProgressBar(self._prog_frame, height=10,
                                     progress_color=C["accent"], fg_color=C["card2"])
            bar.grid(row=i, column=1, sticky="ew", padx=(8, 8), pady=1)
            bar.set(0)
            info = ctk.CTkLabel(self._prog_frame, text="—", font=_font(9),
                                text_color=C["subtext"], width=130, anchor="w")
            info.grid(row=i, column=2, sticky="w", pady=1)
            self._cam_bars[cl]      = bar
            self._cam_info_lbls[cl] = info
            self._cam_lbl_wgts[cl]  = lbl
            self._cam_progress[cl]  = {"done": 0, "total": 1, "fills": 0}

        self._overall_bar.set(0)
        self._overall_info.configure(text="—")

    def _build_log(self, parent):
        C = self._C
        card = ctk.CTkFrame(parent, fg_color=C["log_bg"], corner_radius=6)
        card.pack(fill="both", expand=True, pady=4)

        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(hdr, text="Log", font=_font(9, True),
                     text_color=C["accent"]).pack(side="left")
        ctk.CTkButton(hdr, text="Clear", width=60, height=22, font=_font(8),
                      fg_color=C["card2"], hover_color=C["border"],
                      text_color=C["subtext"],
                      command=self._clear_log).pack(side="right")

        self._log_box = ctk.CTkTextbox(
            card, font=ctk.CTkFont(family="Consolas", size=8),
            fg_color=C["log_bg"], text_color=C["log_fg"],
            state="disabled", wrap="word")
        self._log_box.pack(fill="both", expand=True, padx=8, pady=(2, 8))

    def _build_footer(self):
        C = self._C
        foot = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=28)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        ctk.CTkLabel(foot,
                     text="Step 0 of 5  ·  P.Valiathan · 2026  ·  Karolinska Institutet",
                     font=_font(8), text_color=C["dim"]).pack(
            side="right", padx=16, pady=6)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _browse_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select recording folder")
        if d:
            self._folder_var.set(d)

    def _on_del_mkv_toggle(self):
        if self._del_mkv_var.get():
            self._del_warn_lbl.configure(
                text="⚠ MKVs will be permanently deleted after successful conversion")
        else:
            self._del_warn_lbl.configure(text="")

    def _toggle_theme(self):
        new = "light" if self._dark else "dark"
        try:
            (HERE / "theme.txt").write_text(new)
        except Exception:
            pass
        self._log("Theme change takes effect on next launch.")

    def _on_scan(self):
        folder = self._folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            self._log("Invalid folder path.")
            return
        self._scan_btn.configure(state="disabled", text="Scanning…")
        self._log(f"Scanning {folder} …")

        def _do_scan():
            sessions = scan_folder(Path(folder))
            calib, sessions = split_calibration_sessions(sessions)
            if calib:
                out_dir = _resolve_or_create_run_root(Path(folder)) / CUBE0_DIRNAME
                try:
                    dest = copy_calibration_sessions(calib, out_dir)
                    self._progress_q.put({
                        "type": "log",
                        "msg": f"Set aside {len(calib)} calibration session(s) "
                               f"({sum(len(s.cams) for s in calib)} video(s)) → {dest}"})
                except Exception as e:
                    self._progress_q.put({
                        "type": "log",
                        "msg": f"Calibration copy failed: {e}"})
            self._progress_q.put({"type": "_scan_done", "sessions": sessions})

        threading.Thread(target=_do_scan, daemon=True).start()

    def _refresh_session_list(self):
        C = self._C
        for w in self._session_frame.winfo_children():
            w.destroy()
        self._session_vars.clear()

        if not self._sessions:
            ctk.CTkLabel(self._session_frame, text="No sessions found.",
                         font=_font(9), text_color=C["dim"]).pack(anchor="w", padx=4)
            return

        all_cam_labels: list[str] = []
        for sess in self._sessions:
            var = ctk.BooleanVar(value=sess.selected)
            self._session_vars[sess.session_id] = var
            row = ctk.CTkFrame(self._session_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            if sess.valid:
                cam_labels = [c.cam_label for c in sess.cams]
                all_cam_labels = cam_labels
                cam_str = "  ".join(cam_labels)
                ctk.CTkCheckBox(
                    row,
                    text=f"{sess.session_id}   {cam_str}   "
                         f"{sess.n_frames} frames  {sess.duration_s:.1f}s  "
                         f"{sess.fps:.0f} fps",
                    variable=var, font=_font(9), text_color=C["text"],
                    fg_color=C["accent"], hover_color="#15a882",
                    checkmark_color="#000000",
                    command=lambda s=sess, v=var: self._on_session_toggle(s, v),
                ).pack(side="left", padx=4)
                self._log(f"Found: {sess.session_id} — "
                          f"{sess.n_frames} frames, {len(sess.cams)} cameras")
            else:
                ctk.CTkLabel(row,
                             text=f"  {sess.session_id}   ({sess.skip_reason})",
                             font=_font(9), text_color=C["dim"]).pack(side="left", padx=4)
                self._log(f"Skipped: {sess.session_id} — {sess.skip_reason}")

        if all_cam_labels:
            self._rebuild_cam_bars(all_cam_labels)

    def _on_session_toggle(self, sess: SessionInfo, var: ctk.BooleanVar):
        sess.selected = var.get()

    def _on_start(self):
        selected = [s for s in self._sessions if s.selected and s.valid]
        if not selected:
            self._log("No valid sessions selected.")
            return
        if self._running:
            return
        if not self._ffmpeg:
            self._log("FFmpeg not ready yet — please wait a moment.")
            return

        thresh_s = float(self._thresh_var.get()) / 1000.0
        crf_raw = self._crf_var.get()
        crf = int(crf_raw) if crf_raw.isdigit() else 23
        del_mkv = self._del_mkv_var.get()

        # Resolve the encoder the user has selected (label → codec name)
        enc = label_to_encoder(self._enc_var.get(),
                               fallback=self._encoder or "libx264")

        if del_mkv:
            from tkinter import messagebox
            if not messagebox.askyesno(
                "Confirm MKV Deletion",
                "Original MKV files will be permanently deleted after successful "
                "conversion.\n\nThis cannot be undone. Continue?",
                icon="warning",
            ):
                return

        for sess in selected:
            compute_session_mappings(sess, sess.fps, thresh_s)

        folder = Path(self._folder_var.get())
        run_root = _resolve_or_create_run_root(folder)
        out_dir = run_root / CUBE0_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        needed = sum(estimate_output_bytes(s, s.fps) for s in selected)
        if not check_disk_space(out_dir, needed):
            from tkinter import messagebox
            messagebox.showerror(
                "Disk Space",
                f"Insufficient disk space. Need ≈{needed // 1_000_000} MB free.")
            return

        # Ask once per conflicting file
        for sess in selected:
            for cam in sess.cams:
                mp4 = out_dir / f"{cam.csv_path.stem}_corrected.mp4"
                if mp4.exists():
                    from tkinter import messagebox
                    if not messagebox.askyesno("File Exists",
                                               f"{mp4.name} already exists.\nOverwrite?"):
                        self._log(f"Skipping {mp4.name}.")
                        cam.status = "skipped"

        raw_dir = run_root / RAW_DIRNAME
        self._log(f"Moving original recordings → {raw_dir} …")
        move_session_raw_files(selected, folder, raw_dir, log=self._log)

        self._running = True
        self._cancel_event.clear()
        self._start_time = time.time()
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")

        all_cams = [c for s in selected for c in s.cams if c.status != "skipped"]
        self._overall_total = sum(c.n_out for c in all_cams)
        self._overall_done = 0
        for c in all_cams:
            self._cam_progress[c.cam_label] = {"done": 0, "total": c.n_out, "fills": 0}

        def orchestrate():
            for sess in selected:
                cams_for_sess = [c for c in sess.cams if c.status != "skipped"]
                all_ok = True

                with ThreadPoolExecutor(max_workers=min(4, len(cams_for_sess))) as pool:
                    futures = {
                        pool.submit(
                            CameraWorker(
                                cam, sess.session_id, sess.common_start,
                                out_dir, sess.fps, crf, enc,
                                self._ffmpeg, self._progress_q, self._cancel_event
                            ).run
                        ): cam
                        for cam in cams_for_sess
                    }
                    for fut in as_completed(futures):
                        cam = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            cam.status = "failed"
                            self._progress_q.put({"type": "log",
                                                  "msg": f"{cam.cam_label}: exception — {e}"})
                        if cam.status != "done":
                            all_ok = False

                if del_mkv and all_ok and not self._cancel_event.is_set():
                    for cam in cams_for_sess:
                        try:
                            cam.mkv_path.unlink()
                            self._progress_q.put(
                                {"type": "log", "msg": f"Deleted {cam.mkv_path.name}"})
                        except Exception as e:
                            self._progress_q.put(
                                {"type": "log",
                                 "msg": f"Could not delete {cam.mkv_path.name}: {e}"})

                report = {
                    "session_id": sess.session_id,
                    "fps": sess.fps,
                    "drop_threshold_ms": round(thresh_s * 1000, 2),
                    "encoder": enc,
                    "common_start_posix": sess.common_start,
                    "cameras": [
                        {"cam": c.cam_label, "status": c.status,
                         "drops_detected": len(c.drops),
                         "fills_inserted": c.fill_count,
                         "n_output_frames": c.n_out}
                        for c in sess.cams
                    ],
                }
                try:
                    rpt = out_dir / f"{sess.session_id}_correction_report.json"
                    rpt.write_text(json.dumps(report, indent=2))
                except Exception:
                    pass

            # Verify outputs and build report before signalling done
            try:
                report_data = build_verification(
                    selected, out_dir, enc, crf, thresh_s)
            except Exception as _ve:
                report_data = None
                self._progress_q.put({"type": "log",
                                      "msg": f"Report error: {_ve}"})

            self._progress_q.put({"type": "done"})
            if report_data is not None:
                self._progress_q.put({"type": "report", "data": report_data})

        threading.Thread(target=orchestrate, daemon=True).start()

    def _on_cancel(self):
        self._cancel_event.set()
        self._log("Cancelling — waiting for workers to finish current frame…")
        self._cancel_btn.configure(state="disabled")

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                self._handle_msg(self._progress_q.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_msg(self, msg: dict):
        C = self._C
        t         = msg.get("type")
        cam_label = msg.get("cam", "")

        if t == "_scan_done":
            self._sessions = msg["sessions"]
            self._scan_btn.configure(state="normal", text="Scan Folder")
            self._refresh_session_list()
            return

        if t == "_enc_ready":
            enc = msg.get("enc", "libx264")
            lbl = encoder_label(enc)
            self._enc_var.set(lbl)
            self._enc_menu.configure(values=[encoder_label(e) for e in _ENC_OPTIONS])
            return

        if t == "progress":
            done  = msg["done"]
            total = msg["total"]
            fills = msg.get("fills", 0)
            if cam_label in self._cam_bars:
                pct = done / max(total, 1)
                self._cam_bars[cam_label].set(pct)
                self._cam_info_lbls[cam_label].configure(
                    text=f"{int(pct * 100)}%  {fills} fills")
                prev = self._cam_progress.get(cam_label, {}).get("done", 0)
                self._overall_done += done - prev
                self._cam_progress[cam_label]["done"]  = done
                self._cam_progress[cam_label]["fills"] = fills

                op = min(1.0, self._overall_done / max(self._overall_total, 1))
                self._overall_bar.set(op)
                eta = ""
                if self._start_time and op > 0.01:
                    elapsed   = time.time() - self._start_time
                    remaining = elapsed / op * (1 - op)
                    m2, s2 = divmod(int(remaining), 60)
                    eta = f"  ETA {m2}m {s2:02d}s"
                self._overall_info.configure(text=f"{int(op * 100)}%{eta}")

        elif t == "status":
            status = msg.get("status", "")
            fills  = msg.get("fills", 0)
            if cam_label in self._cam_info_lbls:
                label_map = {
                    "running":   "running…",
                    "done":      f"✓  {fills} fill(s)",
                    "failed":    "✗ FAILED",
                    "cancelled": "⊘ Cancelled",
                }
                self._cam_info_lbls[cam_label].configure(
                    text=label_map.get(status, status))
                if status == "done":
                    self._cam_bars[cam_label].set(1.0)
                    self._cam_bars[cam_label].configure(progress_color="#4caf50")
                elif status == "failed":
                    self._cam_bars[cam_label].configure(progress_color=C["red"])
            if msg.get("error"):
                self._log(f"{cam_label}: ERROR — {msg['error']}")

        elif t == "log":
            self._log(msg.get("msg", ""))

        elif t == "done":
            self._running = False
            self._start_btn.configure(state="normal")
            self._cancel_btn.configure(state="disabled")
            self._overall_bar.set(1.0)
            self._log("All sessions complete. Generating report…")

        elif t == "report":
            self._show_report_window(msg["data"])

    # ------------------------------------------------------------------
    # Report window
    # ------------------------------------------------------------------
    def _show_report_window(self, sessions_data: list[dict]):
        C = self._C
        win = ctk.CTkToplevel(self)
        win.title("Correction Report — CUBE Step 0")
        win.geometry("820x660")
        win.minsize(640, 480)
        win.bind("<Escape>", lambda _: win.destroy())
        ico = HERE / "assets" / "CUBE.ico"
        if ico.exists():
            try:
                win.after(150, lambda: win.iconbitmap(str(ico)))
            except Exception:
                pass
        win.after(100, win.lift)
        win.after(150, win.focus_force)

        # header
        hdr = ctk.CTkFrame(win, fg_color=C["accent"], corner_radius=0, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Correction Report",
                     font=_font(13, True), text_color="#000000").pack(
            side="left", padx=16, pady=8)
        ctk.CTkLabel(hdr, text=datetime.now().strftime("%Y-%m-%d  %H:%M"),
                     font=_font(9), text_color="#004030").pack(
            side="right", padx=16, pady=8)

        # scrollable body
        body = ctk.CTkScrollableFrame(win, fg_color=C["bg"], corner_radius=0)
        body.pack(fill="both", expand=True)

        # ── widget helpers ────────────────────────────────────────
        def _sep(pad_y=(6, 2)):
            ctk.CTkFrame(body, fg_color=C["border"], height=1,
                         corner_radius=0).pack(fill="x", padx=12, pady=pad_y)

        def _section_hdr(text: str):
            ctk.CTkLabel(body, text=text, font=_font(10, True),
                         text_color=C["accent"], anchor="w").pack(
                fill="x", padx=16, pady=(10, 2))

        def _kv(key: str, val: str, warn: bool = False):
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", padx=24, pady=1)
            ctk.CTkLabel(row, text=key, font=_font(9), text_color=C["subtext"],
                         width=200, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=val, font=_font(9),
                         text_color=C["yellow"] if warn else C["text"]).pack(
                side="left")

        def _tbl_hdr(cols: list[str], widths: list[int]):
            row = ctk.CTkFrame(body, fg_color=C["card2"], corner_radius=4)
            row.pack(fill="x", padx=16, pady=(4, 0))
            for c, w in zip(cols, widths):
                ctk.CTkLabel(row, text=c, font=_font(8, True),
                             text_color=C["subtext"], width=w, anchor="w").pack(
                    side="left", padx=(8, 0), pady=3)

        def _tbl_row(vals: list, widths: list[int], warn: bool = False):
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=0)
            clr = C["yellow"] if warn else C["text"]
            for v, w in zip(vals, widths):
                ctk.CTkLabel(row, text=str(v), font=_font(9),
                             text_color=clr, width=w, anchor="w").pack(
                    side="left", padx=(8, 0), pady=2)

        def _check(label: str, ok: bool, detail: str):
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", padx=24, pady=1)
            sym = "OK  " if ok else "FAIL"
            clr = "#4caf50" if ok else C["red"]
            ctk.CTkLabel(row, text=f"[{sym}]", font=_font(9, True),
                         text_color=clr, width=52).pack(side="left")
            ctk.CTkLabel(row, text=label, font=_font(9), text_color=C["subtext"],
                         width=190, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=detail, font=_font(9),
                         text_color=C["text"]).pack(side="left", padx=(4, 0))

        # ── render sessions ───────────────────────────────────────
        for sd in sessions_data:
            cams      = sd["cameras"]
            done_cams = [c for c in cams if c["status"] == "done"]
            al        = sd.get("alignment", {})
            outcome   = sd.get("outcome", "FAIL")
            fp        = sd["fps"]

            # session title
            sf = ctk.CTkFrame(body, fg_color=C["card"], corner_radius=6,
                              border_width=1, border_color=C["border"])
            sf.pack(fill="x", padx=12, pady=(14, 0))
            ctk.CTkLabel(sf, text=f"Session:  {sd['session_id']}",
                         font=_font(10, True), text_color=C["text"]).pack(
                side="left", padx=12, pady=6)
            ctk.CTkLabel(sf,
                         text=(f"  {sd['common_window_s']:.1f} s"
                               f"  |  {int(fp)} fps"
                               f"  |  {sd['encoder'].replace('_', ' ')}"
                               f"  |  CRF {sd['crf']}"),
                         font=_font(9), text_color=C["subtext"]).pack(
                side="left", pady=6)

            # ── input data quality ────────────────────────────────
            _section_hdr("Input Data Quality")
            _tbl_hdr(
                ["Camera", "Src frames", "Mean interval", "Jitter (1σ)", "Drops", "Max drop"],
                [70, 86, 116, 102, 54, 90])
            for cam in cams:
                if cam["status"] not in ("done", "failed"):
                    continue
                md = f"{cam['max_drop_ms']:.0f} ms" if cam["drops"] > 0 else "—"
                _tbl_row(
                    [cam["cam"], cam["src_frames"],
                     f"{cam['mean_interval_ms']:.2f} ms",
                     f"{cam['jitter_ms']:.2f} ms",
                     cam["drops"], md],
                    [70, 86, 116, 102, 54, 90],
                    warn=(cam["drops"] > 0))

            spr = sd["start_spread_ms"]
            fp_period_ms = 1000.0 / fp
            _kv("Camera start-time spread",
                f"{spr:.1f} ms  "
                f"({'within' if spr < fp_period_ms else 'exceeds'} 1 frame period"
                f" = {fp_period_ms:.1f} ms)")

            # ── correction impact ─────────────────────────────────
            _sep()
            _section_hdr("Correction Impact")
            _tbl_hdr(
                ["Camera", "Src frames", "Out frames", "Fills", "Fill %", "Status"],
                [70, 86, 86, 58, 72, 72])
            for cam in cams:
                _tbl_row(
                    [cam["cam"], cam["src_frames"], cam["out_frames"],
                     cam["fills"], f"{cam['fill_rate_pct']:.1f}%",
                     cam["status"].upper()],
                    [70, 86, 86, 58, 72, 72],
                    warn=(cam["status"] != "done"))

            fc = al.get("frame_count", sd.get("n_out_frames", "?"))
            _kv("Common output window",
                f"{sd['common_window_s']:.2f} s  ({fc} frames at {int(fp)} fps)")

            # ── verification ──────────────────────────────────────
            _sep()
            _section_hdr("Verification")

            fc_ok = al.get("frame_count_ok", False)
            _check("Frame count alignment",
                   fc_ok,
                   (f"{al.get('frame_count', '?')} frames on all {len(done_cams)} cameras"
                    if fc_ok else "MISMATCH — cameras differ"))

            fps_devs = [abs(c.get("verified_fps", 0) - fp) for c in done_cams]
            fps_ok   = all(d < 0.5 for d in fps_devs) if fps_devs else False
            _check("Container FPS",
                   fps_ok,
                   f"{int(fp)} fps constant" if fps_ok else "inconsistency detected")

            gj    = al.get("grid_jitter_ms", -1.0)
            gj_ok = 0 <= gj < 1.0
            _check("Grid uniformity",
                   gj_ok,
                   f"+/− {gj:.4f} ms (max across cameras)" if gj >= 0
                   else "corrected CSVs not found")

            cc    = al.get("cross_cam_offset_ms", -1.0)
            cc_ok = 0 <= cc < 0.1
            _check("Cross-camera alignment",
                   cc_ok,
                   f"{cc:.4f} ms max offset between cameras" if cc >= 0
                   else "corrected CSVs not found")

            # ── outcome banner ────────────────────────────────────
            _sep(pad_y=(8, 0))
            out_clr = {"OK": "#4caf50", "PARTIAL": C["yellow"],
                       "FAIL": C["red"]}.get(outcome, C["red"])
            out_txt = {
                "OK":      "READY FOR 3D DEEPLABCUT",
                "PARTIAL": "PARTIAL SUCCESS — some cameras failed, check log",
                "FAIL":    "FAILED — do not use these files for DLC",
            }.get(outcome, outcome)
            ctk.CTkLabel(body,
                         text=f"Outcome:  {out_txt}",
                         font=_font(12, True), text_color=out_clr,
                         anchor="w").pack(fill="x", padx=20, pady=(6, 2))

            # ── output files ──────────────────────────────────────
            _sep()
            _section_hdr("Output Files")
            for cam in cams:
                op = cam.get("out_path", "")
                if op:
                    ctk.CTkLabel(body, text=f"  {op}", font=_font(8),
                                 text_color=C["subtext"], anchor="w").pack(
                        fill="x", padx=24, pady=0)

        ctk.CTkLabel(body, text="").pack(pady=6)   # bottom spacer

        # ── bottom bar ────────────────────────────────────────────
        btm = ctk.CTkFrame(win, fg_color=C["panel"], corner_radius=0, height=46)
        btm.pack(fill="x", side="bottom")
        btm.pack_propagate(False)

        def _save():
            from tkinter import filedialog
            p = filedialog.asksaveasfilename(
                parent=win, title="Save Report",
                defaultextension=".txt",
                filetypes=[("Text", "*.txt"), ("All files", "*.*")],
                initialfile=(f"correction_report_"
                             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"))
            if p:
                _write_report_txt(sessions_data, Path(p))

        ctk.CTkButton(btm, text="Save as Text", width=120, height=30,
                      font=_font(9, True), fg_color=C["card2"],
                      hover_color=C["border"], text_color=C["text"],
                      command=_save).pack(side="left", padx=12, pady=8)
        ctk.CTkButton(btm, text="Close", width=100, height=30,
                      font=_font(9, True), fg_color=C["accent"],
                      hover_color="#15a882", text_color="#000000",
                      command=win.destroy).pack(side="right", padx=12, pady=8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"[{ts}] {msg}\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _detect_ffmpeg_async(self):
        def _detect():
            try:
                ffmpeg = get_ffmpeg()
                enc    = detect_encoder(ffmpeg)
                self._ffmpeg  = ffmpeg
                self._encoder = enc
                self._progress_q.put({"type": "log",
                                      "msg": f"Encoder ready: {encoder_label(enc)}"})
                self._progress_q.put({"type": "_enc_ready", "enc": enc})
            except Exception as e:
                self._progress_q.put({"type": "log",
                                      "msg": f"Encoder detection failed: {e}"})
        threading.Thread(target=_detect, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    VideoPrep().mainloop()
