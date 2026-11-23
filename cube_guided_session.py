"""Guided labelling: data loading + session controller (no Tk, so it is unit-testable and
thread-friendly). The Tk tab (cube_guided_tab.py) is a thin view over GuidedController.

View-agnostic: works on whatever feature matrix a CUBE run saved (top-view 2D, side-view 2D, 3D).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import cube_guided as cg

_VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
BACKGROUND = "Background"
NEITHER = "Neither"
CLIP_S = 1.0          # query unit = fixed 1-s clip (evidence: wave 3/4, equal-seconds comparison)


# ------------------------------------------------------------------ videos
def build_video_index(dirs, max_files: int = 300000) -> dict:
    """One os.walk per folder -> {lower file name: path}. Cheap to reuse for many sessions."""
    idx: dict = {}
    n = 0
    for d in dirs or []:
        try:
            for root, _dn, files in os.walk(str(d)):
                for fn in files:
                    if fn.lower().endswith(_VID_EXT):
                        idx.setdefault(fn.lower(), os.path.join(root, fn))
                        n += 1
                        if n >= max_files:
                            return idx
        except Exception:
            continue
    return idx


def session_stems(key: str) -> list:
    out = [key]
    k = key
    while k.startswith("_prep_"):
        k = k[len("_prep_"):]
        out.append(k)
    return out


def resolve_video(key: str, embedded, index: dict):
    """Resolution order: saved path -> '<stem>.<ext>' -> 'resized_<stem>.<ext>' -> '<stem>_labeled.mp4'.
    Returns (path or None, human-readable reason when None)."""
    if embedded and Path(str(embedded)).is_file():
        return str(embedded), ""
    for s in session_stems(key):
        for ext in _VID_EXT:
            for name in (s + ext, "resized_" + s + ext):
                hit = index.get(name.lower())
                if hit:
                    return hit, ""
        hit = index.get((s + "_labeled.mp4").lower())
        if hit:
            return hit, ""
    return None, ("Video not found. The path saved with this run no longer exists and no file named "
                  f"'{session_stems(key)[-1]}' was found in your video folders. Add the folder that "
                  "holds the original videos to the video sources, then reopen this tab.")


@dataclass
class RunLayout:
    model_dir: Path
    bout_dir: Path
    run_dir: Path        # the "CUBE 3 - Clusters" folder (labels go to <run_dir>/guided)
    video_root: Path     # the "CUBE results ..." folder (searched for the session videos)


def find_run_layout(path) -> "RunLayout | None":
    """Accepts a CUBE run root, its 'CUBE 3 - Clusters' folder, model/, example_clips/ or any folder
    inside/around them, and returns the folders the guided tab needs (None if no finished run found)."""
    p = Path(path)
    cands = [p, p / "CUBE 3 - Clusters"]
    try:                                            # the user picked a folder that CONTAINS runs: use those first
        for c in sorted(p.iterdir(), reverse=True):    # newest run first (timestamped names)
            if c.is_dir():
                cands += [c, c / "CUBE 3 - Clusters"]
    except Exception:
        pass
    cur = p                                         # only then look at the folders ABOVE (a subfolder of a run)
    for _ in range(6):
        if cur.parent == cur:
            break
        cur = cur.parent
        cands += [cur, cur / "CUBE 3 - Clusters"]
    clusters = None
    for c in cands:
        try:
            if (c / "model" / "session_bin_ranges.json").is_file():
                clusters = c
                break
            if c.name == "model" and (c / "session_bin_ranges.json").is_file():
                clusters = c.parent
                break
        except Exception:
            continue
    if clusters is None:
        return None
    root = clusters.parent
    for a in [clusters] + list(clusters.parents):
        if a.name.lower().startswith("cube results"):
            root = a
            break
    return RunLayout(clusters / "model", clusters / "bout_lengths", clusters, root)


def _probe_fps(video):
    """Frame rate read from the video itself (used only when no frame-label CSV gives one)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        f = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return f if f > 1.0 else None
    except Exception:
        return None


# -------------------------------------------------------------------- data
@dataclass
class SessionInfo:
    key: str
    s0: int
    s1: int
    fps: float
    n_frames: int
    video: str | None
    video_reason: str
    frame_clusters: np.ndarray        # per frame (final HMM cluster, -1 unknown)
    fps_from_csv: bool = True         # False: no frame-label CSV, fps was read from the video or assumed

    @property
    def n_bins(self) -> int:
        return self.s1 - self.s0

    @property
    def win100(self) -> int:
        return max(1, int(round(self.fps / 10.0)))

    @property
    def bin_s(self) -> float:
        return self.win100 / self.fps


class GuidedData:
    """Everything the guided tab needs from one finished CUBE run (read-only)."""

    def __init__(self, model_dir, bout_dir=None, video_dirs=(), default_fps: float = 30.0, progress=None):
        self._progress = progress or (lambda f, t=None: None)      # progress(fraction, text), any thread
        self.model_dir = Path(model_dir)
        self.bout_dir = Path(bout_dir) if bout_dir else self.model_dir.parent / "bout_lengths"
        self.default_fps = default_fps
        self.feats = None
        self.embedding = None
        self.sessions: list[SessionInfo] = []
        self.sess_of_bin: np.ndarray = np.array([], dtype=object)
        self.cluster_per_bin: np.ndarray = np.array([], dtype=int)
        self.problems: list[str] = []
        self._by_key: dict = {}
        self._load(video_dirs)

    def _load(self, video_dirs):
        md = self.model_dir
        sbr_p = md / "session_bin_ranges.json"
        if not sbr_p.is_file():
            raise FileNotFoundError("session_bin_ranges.json not found - re-run CUBE clustering (Step 3).")
        raw = json.loads(sbr_p.read_text())
        f_p, e_p = md / "feats_sc_cache.npz", md / "umap_embedding.npy"
        if not f_p.is_file():
            raise FileNotFoundError("feats_sc_cache.npz not found - this run did not save its feature cache, "
                                    "so guided labelling needs a re-run of Step 3.")
        self._progress(0.05, "Reading the run's saved features (the largest file) ...")
        _z = np.load(f_p)
        # Newer runs also store features for EVERY bin ("feats_all": turned-away / held-out bins included).
        # Prefer them, so labels inside turned-away stretches are usable; older runs only have the NaN-filled
        # training-row matrix ("feats_sc").
        self.feats_full = "feats_all" in _z.files
        self.feats = _z["feats_all"] if self.feats_full else _z["feats_sc"]
        _fa = md / "feats_all.npy"           # full-coverage features written next to an older run's cache
        if not self.feats_full and _fa.is_file():
            try:
                _arr = np.load(_fa, mmap_mode="r")
                if _arr.shape[0] == self.feats.shape[0]:
                    self.feats, self.feats_full = np.asarray(_arr), True
            except Exception as _e:                                # noqa: BLE001
                self.problems.append(f"feats_all.npy unreadable ({_e}); using the training-row features")
        self.embedding = np.load(e_p) if e_p.is_file() else None
        if self.embedding is not None and len(self.embedding) != len(self.feats):
            self.problems.append("embedding rows differ from feature rows; diversity ignored")
            self.embedding = None
        n = len(self.feats)
        self._progress(0.45, "Looking for the session videos ...")
        index = build_video_index(video_dirs)
        sess_of = np.full(n, "", dtype=object)
        clus = np.full(n, -1, dtype=int)
        _n_keys = max(1, len(raw) - 1)
        for _ki, (key, v) in enumerate(raw.items()):
            if key == "_total_bins" or not (isinstance(v, (list, tuple)) and len(v) >= 2):
                continue
            self._progress(0.55 + 0.4 * _ki / _n_keys, f"Reading cluster labels: video {_ki + 1} of {_n_keys}")
            s0, s1 = int(v[0]), int(v[1])
            if s1 > n or s0 < 0 or s1 <= s0:
                self.problems.append(f"{key}: bin range outside feature matrix, skipped")
                continue
            fc, fps, known = self._frame_clusters(key)
            video, reason = resolve_video(key, v[2] if len(v) >= 3 else None, index)
            if not known:
                vf = _probe_fps(video) if video else None
                if vf:
                    fps = vf
                self.problems.append(f"{key}: no frame-label CSV found; frame rate "
                                     + (f"read from the video ({fps:.2f} fps)" if vf else
                                        f"assumed {fps:.0f} fps, so label times may be off"))
            si = SessionInfo(key, s0, s1, fps, len(fc), video, reason, fc, known)
            self.sessions.append(si)
            self._by_key[key] = si
            sess_of[s0:s1] = key
            if len(fc):
                cb = fc[::si.win100][: si.n_bins]
                clus[s0:s0 + len(cb)] = cb
        self.sess_of_bin, self.cluster_per_bin = sess_of, clus

    def _frame_clusters(self, key):
        for s in session_stems(key):
            for suffix in ("_frame_labels_hmm.csv", "_frame_labels.csv"):
                p = self.bout_dir / f"{s}{suffix}"
                if p.is_file():
                    try:
                        df = pd.read_csv(p)
                        fps = self.default_fps
                        if "time_s" in df and len(df) > 2 and df["time_s"].iloc[1] > df["time_s"].iloc[0]:
                            fps = 1.0 / float(df["time_s"].iloc[1] - df["time_s"].iloc[0])
                        col = "label" if "label" in df else df.columns[2]
                        return df[col].to_numpy(int), float(fps), True
                    except Exception as e:
                        self.problems.append(f"{key}: cannot read {p.name} ({e})")
        return np.zeros(0, int), self.default_fps, False

    def session(self, key) -> SessionInfo:
        return self._by_key[key]

    @property
    def ranges(self) -> dict:
        return {s.key: (s.s0, s.s1) for s in self.sessions}

    @property
    def fps_map(self) -> dict:
        return {s.key: s.fps for s in self.sessions}

    @property
    def n_bins(self) -> int:
        return 0 if self.feats is None else len(self.feats)


# -------------------------------------------------------------- controller
@dataclass
class Query:
    row: int
    session: str
    start_s: float
    end_s: float
    top1: str
    top2: str | None
    maxprob: float
    exemplar: tuple | None = None      # (session, start_s, end_s) of a confident bin of top1


class GuidedController:
    """Labels, training, query rounds and exports for one CUBE run. Not thread-safe by itself:
    run retrain()/next_queries() in ONE worker thread and swap the results in on the UI thread."""

    PALETTE = ["#1565c0", "#2e7d32", "#e65100", "#c62828", "#6a1b9a", "#00838f", "#827717", "#4e342e",
               "#ad1457", "#37474f"]
    RESERVED = {"neither", "__neither__"}

    def __init__(self, data: GuidedData, run_dir, *, use_context: bool = True, video_centring: bool = True, log=None):
        self.data = data
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "guided"
        self.labels_path = self.dir / "user_labels.json"
        self.use_context = use_context
        self.video_centring = video_centring       # add per-video-centred features (needs use_context)
        self.log = log or (lambda m: None)
        self.load_warning = ""
        self.ls = self._load_labels()
        if not self.ls.behaviours:
            self.ls.behaviours = []
        self._snap = None                  # (model, proba): assigned together, so a reader never sees a mismatch
        self.round = 0
        self._X = None
        self._finite = None
        self.last_f1 = None
        self.dirty = False
        self.last_error = ""               # non-empty when the last save FAILED (the tab shows it)
        self._disk_sha = self.ls.sha1      # what we believe is on disk (detects a second window's writes)
        self._touched: set = set()         # sessions this window changed since its last save
        self._touch_all = False            # rename/delete rewrote labels everywhere
        self._touched_complete: set = set()   # sessions whose 'fully labelled' flag this window changed
        self.honest_history: list = []        # [(round, mean honest leakage %)] after each question round
        self.skipped: list = []            # (session, start_s, end_s) clips the user skipped: never asked again

    @property
    def model(self):
        return None if self._snap is None else self._snap[0]

    @property
    def proba(self):
        return None if self._snap is None else self._snap[1]

    def _load_labels(self):
        try:
            return cg.load_labels(self.labels_path)
        except ValueError as e:
            bak = Path(str(self.labels_path) + ".bak")
            if bak.is_file():
                ls = cg.load_labels(bak)
                self.load_warning = (f"Your labels file could not be read ({e}). Recovered {len(ls.labels)} "
                                     f"labels from the automatic backup.")
                return ls
            raise

    # ----------------------------------------------------------- behaviours
    def behaviour_names(self) -> list:
        return [b["name"] for b in self.ls.behaviours]

    def _name_ok(self, name: str, ignore: str | None = None) -> bool:
        key = name.strip().casefold()
        if not key or key in self.RESERVED:
            return False
        return all(b["name"].casefold() != key for b in self.ls.behaviours if b["name"] != ignore)

    def _next_color(self, name: str) -> str:
        if name.strip().casefold() == BACKGROUND.casefold():
            return "#757575"
        used = {b.get("color") for b in self.ls.behaviours}
        for c in self.PALETTE:
            if c not in used:
                return c
        import colorsys                                   # more behaviours than palette entries
        for i in range(1, 200):
            r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.65, 0.65)
            c = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
            if c not in used:
                return c
        return "#455a64"

    def add_behaviour(self, name: str, color: str | None = None) -> bool:
        name = (name or "").strip()
        if not self._name_ok(name):
            return False
        self.ls.behaviours.append({"name": name, "color": color or self._next_color(name)})
        self._changed()
        return True

    def rename_behaviour(self, old: str, new: str) -> bool:
        new = (new or "").strip()
        if old not in self.behaviour_names() or not self._name_ok(new, ignore=old):
            return False
        for b in self.ls.behaviours:
            if b["name"] == old:
                b["name"] = new
        for l in self.ls.labels:
            if l.get("behaviour") == old:
                l["behaviour"] = new
        self._touch_all = True
        self._changed()
        return True

    def delete_behaviour(self, name: str) -> int:
        """Removes the behaviour and all its labels; returns how many labels were removed. The previous file
        is kept as user_labels.json.bak, so the last delete can be recovered by hand."""
        n0 = len(self.ls.labels)
        self.ls.behaviours = [b for b in self.ls.behaviours if b["name"] != name]
        self.ls.labels = [l for l in self.ls.labels if l.get("behaviour") != name]
        self._touch_all = True
        self._changed()
        return n0 - len(self.ls.labels)

    # --------------------------------------------------------------- labels
    def segments_for(self, session: str) -> list:
        return [{"start_s": l["start_s"], "end_s": l["end_s"], "behaviour": l["behaviour"]}
                for l in self.ls.labels if l.get("session") == session and l.get("source", "manual") == "manual"]

    def query_segments_for(self, session: str) -> list:
        """Answers from the question card (read-only on the timeline, so they can't be painted over unseen)."""
        return [{"start_s": l["start_s"], "end_s": l["end_s"], "behaviour": l["behaviour"]}
                for l in self.ls.labels if l.get("session") == session and l.get("source") == "query"]

    def set_manual_segments(self, session: str, segs: list):
        """Replace this session's hand-marked segments (query answers are kept)."""
        keep = [l for l in self.ls.labels
                if not (l.get("session") == session and l.get("source", "manual") == "manual")]
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for s in segs:
            if s["end_s"] > s["start_s"] and s.get("behaviour"):      # unassigned (no behaviour yet) isn't saved
                keep.append({"session": session, "start_s": float(s["start_s"]), "end_s": float(s["end_s"]),
                             "behaviour": s["behaviour"], "source": "manual", "answer": None,
                             "round": 0, "ts": ts})
        self.ls.labels = keep
        self._touched.add(session)
        self._changed()

    def summary(self) -> dict:
        per: dict = {}
        sess = set()
        n_bouts = 0
        for l in self.ls.labels:
            b = l["behaviour"]
            if b == NEITHER:
                continue
            per[b] = per.get(b, 0.0) + (l["end_s"] - l["start_s"])
            sess.add(l["session"])
            n_bouts += 1
        return {"n_behaviours": len([b for b in self.behaviour_names()]),
                "n_bouts": n_bouts, "n_sessions": len(sess), "seconds": per}

    def session_status(self, key) -> str:
        """'queried' / 'labelled' / 'untouched'."""
        has_m = has_q = False
        for l in self.ls.labels:
            if l.get("session") == key:
                if l.get("source") == "query":
                    has_q = True
                else:
                    has_m = True
        return "queried" if has_q else ("labelled" if has_m else "untouched")

    def _finite_rows(self) -> np.ndarray:
        if self._finite is None:
            self._finite = np.isfinite(self.data.feats).all(axis=1)
        return self._finite

    def coverage(self) -> dict:
        """How much of what you labelled the model can actually use. A labelled bin is unusable when its feature
        row is empty (turned-away or excluded frames in runs without the full-coverage cache)."""
        y, w = self._bins()
        fin = self._finite_rows()
        per: dict = {}
        total = usable = 0
        for name in set(y[y != ""]) - {cg.NEITHER_MARK}:
            m = (y == name)
            n_lab, n_use = int(m.sum()), int((m & fin).sum())
            per[str(name)] = {"labelled": n_lab, "usable": n_use}
            total += n_lab
            usable += n_use
        known = set(self.data.ranges)
        unknown = sorted({l["session"] for l in self.ls.labels if l.get("session") not in known})
        return {"behaviours": per, "labelled": total, "usable": usable, "unusable": total - usable,
                "unknown_sessions": unknown, "full_features": bool(getattr(self.data, "feats_full", False))}

    def coverage_line(self) -> str:
        c = self.coverage()
        if c["labelled"] == 0:
            return ""
        line = f"Usable by the model: {c['usable']} of {c['labelled']} labelled bins"
        if c["unusable"]:
            line += f"  ({c['unusable']} have no features: turned-away/excluded frames)"
        return line

    def honest_report(self, min_conf: float = 0.5) -> dict:
        """True coverage / leakage over all bins of the videos marked 'fully labelled' (leave-one-video-out). {} when
        no video is marked complete or there is not enough other labelled data to train on."""
        done = sorted(set(self.ls.complete_sessions) | set(self.ls.complete_spans))
        if not done:
            return {}
        import cube_guided_audit as au
        y, w = self._bins()
        names = [b for b in self.behaviour_names() if b != cg.BACKGROUND]
        return au.complete_video_report(self.features(), y, w, self.data.sess_of_bin, done, names, min_conf=min_conf)

    def honest_check(self, min_conf: float = 0.5):
        """Record the mean honest leakage (over behaviours that were called at all) for the stop rule."""
        rep = self.honest_report(min_conf)
        vals = [d["leakage_pct"] for d in rep.values() if d.get("leakage_pct") is not None]
        if not vals:
            return None
        m = float(np.mean(vals))
        self.honest_history.append((self.round, round(m, 1)))
        return m

    def stop_advice(self, min_gain: float = 2.0, patience: int = 2) -> tuple:
        """Stop rule: stop labelling when the honest leakage on your fully labelled video(s) has stopped improving -
        less than ``min_gain`` points of gain over the last ``patience`` question rounds. (stop?, message)."""
        if not (self.ls.complete_sessions or self.ls.complete_spans):
            return False, ("Mark a video (or its first minute) as fully labelled to get a stop signal: it is the "
                           "only way to measure false positives.")
        h = [v for _r, v in self.honest_history]
        if not h:
            return False, "Honest leakage on your fully labelled video: not measured yet."
        msg = f"Honest leakage on your fully labelled video: {h[-1]:.0f}%"
        if len(h) > patience:
            gain = h[-patience - 1] - min(h[-patience:])
            msg += f" (change over the last {patience} rounds: {gain:+.1f} points)"
            if gain < min_gain:
                return True, msg + " - it has stopped improving: more of the same labelling will not help. " \
                                   "Label a different animal fully, or add examples of what is NOT your behaviours."
        return False, msg

    def label_design_warning(self) -> str:
        """Plain-language warning when the labels are confounded with video identity, which makes a classifier learn
        the animal instead of the behaviour: behaviours labelled in different videos, or 'Neither' answers only in
        videos that have no behaviour labels. Empty string = nothing to flag."""
        by_beh: dict = {}
        neither_v: set = set()
        for l in self.ls.labels:
            b = str(l.get("behaviour", ""))
            if b in ("", cg.BACKGROUND):
                continue
            if b == cg.NEITHER or l.get("answer") == "neither":
                neither_v.add(l.get("session"))
            else:
                by_beh.setdefault(b, set()).add(l.get("session"))
        msgs = []
        names = sorted(by_beh)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both, either = by_beh[a] & by_beh[b], by_beh[a] | by_beh[b]
                if len(either) >= 3 and len(both) < 0.5 * min(len(by_beh[a]), len(by_beh[b])) + 0.01:
                    msgs.append(f"{a} and {b} are labelled in different videos (only {len(both)} of {len(either)} "
                                f"contain both): label them in the SAME videos so CUBE can tell them apart.")
        pos_v = set().union(*by_beh.values()) if by_beh else set()
        lone = neither_v - pos_v
        if len(neither_v) >= 3 and len(lone) > 0.5 * len(neither_v):
            msgs.append(f"{len(lone)} of {len(neither_v)} videos with 'Neither' answers have no behaviour labels: "
                        f"also label real examples in those videos.")
        return "  ".join(msgs[:2])

    def can_train(self) -> tuple[bool, str]:
        c = self.coverage()
        usable_beh = [b for b, v in c["behaviours"].items() if v["usable"] > 0]
        if len(usable_beh) >= 2:
            return True, ""
        if c["unusable"] and len([b for b, v in c["behaviours"].items() if v["labelled"] > 0]) >= 2:
            return False, (f"Only {c['usable']} of your {c['labelled']} labelled bins have usable features "
                           f"(the rest are turned-away or excluded frames), which covers "
                           f"{len(usable_beh)} behaviour(s). Label stretches where the animal is clearly "
                           f"visible, or re-run CUBE so these frames keep their features.")
        return False, "Add at least 2 labelled examples of 2 different behaviours to start questions."

    # ------------------------------------------------- 'this video is fully labelled'
    def is_complete(self, session) -> bool:
        return session in self.ls.complete_sessions or bool(self.ls.complete_spans.get(session))

    def complete_extent(self, session) -> str:
        """'whole video', 'first 60 s', '' - for the tab."""
        if session in self.ls.complete_sessions:
            return "whole video"
        sp = self.ls.complete_spans.get(session) or []
        return ", ".join(f"{a:g}-{b:g} s" for a, b in sp)

    def set_complete_span(self, session, start_s: float, end_s: float):
        """Only [start_s, end_s] of this video is fully labelled (e.g. its first minute)."""
        self.ls.complete_sessions = [s for s in self.ls.complete_sessions if s != session]
        self.ls.complete_spans[session] = [[float(start_s), float(end_s)]]
        self._touched_complete.add(session)
        self._touched.add(session)
        self._snap = None
        self._changed()

    def set_complete(self, session, flag: bool):
        """Declare that EVERY instance of your behaviours in this video is marked, so what is not marked is a real
        negative. This is the strongest labelling you can give: it supplies within-video negatives (no 'which animal'
        shortcut), the true share of each behaviour, and an honest leakage measure over all of the video's bins."""
        cur = set(self.ls.complete_sessions)
        (cur.add if flag else cur.discard)(session)
        self.ls.complete_sessions = sorted(cur)
        self.ls.complete_spans.pop(session, None)
        self._touched_complete.add(session)
        self._touched.add(session)
        self._snap = None                                          # the training set changed
        self._changed()

    # ---------------------------------------------------------------- saving
    def _changed(self):
        self.dirty = True
        self.save()

    def _merge_with_disk(self):
        """If another window changed the file since we last saved, keep ITS labels for the sessions this window
        has not touched, so two windows never silently overwrite each other (last-save-wins)."""
        path = self.labels_path
        if self._touch_all or not path.is_file():
            return
        try:
            disk = cg.load_labels(path)
        except ValueError:
            return
        if disk.sha1 == self._disk_sha:
            return                                                # nobody else wrote: nothing to merge
        mine = [l for l in self.ls.labels if l.get("session") in self._touched]
        theirs = [l for l in disk.labels if l.get("session") not in self._touched]
        comp = {s for s in disk.complete_sessions if s not in self._touched_complete}
        comp |= {s for s in self.ls.complete_sessions if s in self._touched_complete}
        self.ls.complete_sessions = sorted(comp)
        spans = {k: v for k, v in disk.complete_spans.items() if k not in self._touched_complete}
        spans.update({k: v for k, v in self.ls.complete_spans.items() if k in self._touched_complete})
        self.ls.complete_spans = spans
        names = {b["name"].casefold() for b in self.ls.behaviours}
        for b in disk.behaviours:
            if b["name"].casefold() not in names:
                self.ls.behaviours.append(b)
        self.ls.labels = theirs + mine
        self.log("[guided] another window had saved labels: merged them with this window's changes")

    def save(self) -> bool:
        try:
            self._merge_with_disk()
            if self.labels_path.is_file():                        # keep the previous version (recover a delete)
                import shutil
                shutil.copy2(self.labels_path, str(self.labels_path) + ".bak")
            cg.save_labels(self.ls, self.labels_path)
            self._disk_sha = self.ls.sha1
            self._touched.clear()
            self._touched_complete.clear()
            self._touch_all = False
            self.dirty = False
            self.last_error = ""
            return True
        except Exception as e:                                    # never lose the session over a disk hiccup
            self.last_error = str(e)
            self.dirty = True
            self.log(f"[guided] autosave failed: {e}")
            return False

    # ------------------------------------------------------------- learning
    def features(self):
        if self._X is None:
            self._X = cg.build_context_features(self.data.feats, self.data.ranges, self.data.fps_map,
                                                enabled=self.use_context)
            if self.use_context and self.video_centring:
                self._X = cg.hybrid_features(self._X, self.data.ranges)
        return self._X

    def _bins(self):
        return cg.labels_to_bins(self.ls, self.data.ranges, self.data.fps_map, self.data.n_bins)

    progress = None                      # optional progress(fraction, text); may be called from a worker thread

    def _p(self, fraction, text):
        try:
            if self.progress is not None:
                self.progress(fraction, text)
        except Exception:                                         # noqa: BLE001 - progress must never break work
            pass

    def retrain(self) -> dict:
        ok, why = self.can_train()
        if not ok:
            raise ValueError(why)
        y, w = self._bins()                                       # ONE snapshot of the labels for this job
        self._p(0.03, "Preparing the features (the first round also looks 0.5-4 s around every bin) ...")
        X = self.features()
        self._p(0.35, "Training the classifier on your labels ...")
        model = cg.train_guided(X, y, w)
        self._p(0.6, "Scoring every time bin ...")
        proba = model.predict_proba(X)
        self._p(0.75, "Checking accuracy on videos it was not trained on ...")
        self.last_f1 = self.estimate_f1(X, y, w)
        self._p(1.0, "Model ready")
        self._snap = (model, proba)                               # published together
        return {"classes": list(model.classes_), "cv_f1": self.last_f1}

    def estimate_f1(self, X, y, w):
        """Leave-sessions-out macro-F1 over labelled sessions (None if < 2 labelled sessions)."""
        try:
            from sklearn.metrics import f1_score
            fin = self._finite_rows()
            lab = np.array([(v not in ("", "__neither__")) and (ww > 0) for v, ww in zip(y, w)]) & fin
            sess = self.data.sess_of_bin
            ls_sess = sorted({s for s in sess[lab]})
            if len(ls_sess) < 2:
                return None
            folds = [ls_sess[i::min(3, len(ls_sess))] for i in range(min(3, len(ls_sess)))]
            yt, yp = [], []
            for f in folds:
                te = lab & np.isin(sess, f)
                tr_mask = lab & ~np.isin(sess, f)
                if len(set(y[tr_mask])) < 2 or te.sum() == 0:
                    continue
                yy = np.where(tr_mask, y, "")
                m = cg.train_guided(X, yy, np.where(tr_mask, w, 0.0), n_estimators=60)
                p = m.predict_proba(X[te])
                yp += list(np.array(m.classes_)[np.nanargmax(np.nan_to_num(p, nan=-1), axis=1)])
                yt += list(y[te])
            return float(f1_score(yt, yp, average="macro")) if yt else None
        except Exception as e:
            self.log(f"[guided] f1 estimate skipped: {e}")
            return None

    def skip(self, q: Query):
        """Remember a skipped clip so the same one is never asked again."""
        self.skipped.append((q.session, float(q.start_s), float(q.end_s)))

    def _skipped_rows(self) -> set:
        rows = set()
        for sess, a, b in self.skipped:
            si = self.data._by_key.get(sess)
            if si is None:
                continue
            lo = si.s0 + int(a / si.bin_s)
            hi = min(si.s1, si.s0 + int(math.ceil(b / si.bin_s)))
            rows.update(range(max(si.s0, lo), hi))
        return rows

    focus: "str | None" = None           # ask about this behaviour (None / "" = whatever CUBE finds most useful)

    def next_queries(self, n: int = 10, strategy: str = "ours") -> list:
        if self._snap is None:
            self.retrain()
        model, proba = self._snap                                 # one consistent (model, proba) pair
        self._p(0.1, "Choosing the most useful clips to ask you about ...")
        y, _w = self._bins()                                      # ONE label snapshot for the whole selection
        labelled_idx = np.where(y != "")[0]
        scores = cg.score_queries(model, self.features(), self.data.embedding, proba=proba)
        sess = self.data.sess_of_bin
        lab_sessions = set(sess[labelled_idx]) if len(labelled_idx) else set()
        exclude = set(labelled_idx.tolist())
        # also exclude the whole 1-s neighbourhood of anything labelled, so clips never overlap labels
        for r in list(exclude)[:200000]:
            si = self._session_of(r)
            if si is None:
                continue
            h = int(round(CLIP_S / si.bin_s / 2))
            exclude.update(range(max(si.s0, r - h), min(si.s1, r + h + 1)))
        exclude |= self._skipped_rows()
        for si in self.data.sessions:                             # never ask about a video that can't be shown
            if not si.video:
                exclude.update(range(si.s0, si.s1))
        cls0 = list(model.classes_)
        focus_p = (proba[:, cls0.index(self.focus)] if (self.focus and self.focus in cls0) else None)
        # with a focus behaviour, stay mostly in videos that already have labels: contrasts INSIDE a video teach the
        # behaviour, contrasts between different animals teach the animal
        rows = cg.select_queries(scores, sess, lab_sessions, n * 3, strategy,
                                 quota_unlabelled_sessions=(0.2 if focus_p is not None else 0.5),
                                 embedding=self.data.embedding, exclude=exclude, seed=self.round, focus_p=focus_p)
        # the selector fills its cross-session quota FIRST, so taking the first n would be 100% unlabelled
        # sessions: interleave unlabelled-session and labelled-session picks (about half each)
        u = [r for r in rows if sess[r] not in lab_sessions]
        l = [r for r in rows if sess[r] in lab_sessions]
        order, iu, il = [], 0, 0
        while iu < len(u) or il < len(l):
            take_u = (len(order) % 2 == 0 and iu < len(u)) or il >= len(l)
            if take_u:
                order.append(u[iu]); iu += 1
            else:
                order.append(l[il]); il += 1
        out, taken = [], {}
        cls = list(model.classes_)
        for r in order:
            si = self._session_of(r)
            if si is None:
                continue
            h = int(round(CLIP_S / si.bin_s / 2))
            a, b = max(si.s0, r - h), min(si.s1, r + h + 1)
            if any(not (b <= x[0] or a >= x[1]) for x in taken.get(si.key, [])):
                continue                                   # keep a batch's clips non-overlapping
            taken.setdefault(si.key, []).append((a, b))
            t1, t2 = scores.iloc[r]["top1"], scores.iloc[r]["top2"]
            out.append(Query(int(r), si.key, (a - si.s0) * si.bin_s, (b - si.s0) * si.bin_s,
                             str(t1), None if (t2 is None or str(t2) == "nan") else str(t2),
                             float(scores.iloc[r]["maxprob"]), self._exemplar(str(t1), cls, y, proba)))
            if len(out) >= n:
                break
        return out

    def _session_of(self, row):
        key = self.data.sess_of_bin[row]
        return self.data._by_key.get(key) if key else None

    def _exemplar(self, behaviour: str, classes, y, proba):
        """A confidently-predicted bin of `behaviour` (prefers one the user labelled)."""
        if behaviour not in classes or proba is None:
            return None
        j = classes.index(behaviour)
        col = np.nan_to_num(proba[:, j], nan=-1)
        pool = np.where((y == behaviour))[0]
        r = int(pool[np.argmax(col[pool])]) if len(pool) else int(np.argmax(col))
        si = self._session_of(r)
        if si is None or not si.video:
            return None
        h = int(round(CLIP_S / si.bin_s / 2))
        a, b = max(si.s0, r - h), min(si.s1, r + h + 1)
        return (si.key, (a - si.s0) * si.bin_s, (b - si.s0) * si.bin_s)

    def answer(self, q: Query, answer: str):
        """answer: any behaviour name the user defined (incl. Background), or 'neither'.
        'A' / 'B' are still accepted as shorthand for CUBE's first / second guess. The stored
        ``answer`` tag is 'A' (= first guess), 'B' (= second guess), 'other' (any other behaviour)
        or 'neither'."""
        if answer == "A":
            beh, tag = q.top1, "A"
        elif answer == "B":
            beh, tag = q.top2, "B"
        elif answer == "neither" or answer == NEITHER:
            beh, tag = NEITHER, "neither"
        else:
            beh = answer
            tag = "A" if answer == q.top1 else ("B" if answer == q.top2 else "other")
        if beh is None:
            beh, tag = NEITHER, "neither"
        answer = tag
        self.ls.labels.append({"session": q.session, "start_s": q.start_s, "end_s": q.end_s,
                               "behaviour": beh, "source": "query", "answer": answer,
                               "round": self.round, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._touched.add(q.session)
        self._changed()

    def undo_last_answer(self):
        """Remove the most recent question answer (returns that label, or None)."""
        for i in range(len(self.ls.labels) - 1, -1, -1):
            if self.ls.labels[i].get("source") == "query":
                lab = self.ls.labels.pop(i)
                self._touched.add(lab.get("session"))
                self._changed()
                return lab
        return None

    def end_round(self):
        self.round += 1

    # -------------------------------------------------------------- exports
    def votes(self) -> pd.DataFrame:
        y, _ = self._bins()
        yy = np.where(np.isin(y, ["", "__neither__", BACKGROUND]), "", y)
        return cg.cluster_label_votes(self.data.cluster_per_bin, yy)

    def export_all(self, out_dir=None) -> dict:
        out = Path(out_dir) if out_dir else self.dir
        out.mkdir(parents=True, exist_ok=True)
        res = {}
        if self._snap is None:
            self._p(0.05, "Training on your labels first ...")
            self.retrain()
        model, proba = self._snap
        self._p(0.25, "Smoothing the behaviour track over time (HMM) ...")
        track = cg.smooth_guided_track(proba, self.data.ranges, list(model.classes_))
        p = out / "guided_track.csv"
        cg.export_guided_track_csv(track, list(model.classes_), self.data.ranges, p,
                                   proba=proba, fps=self.data.fps_map)
        res["track_csv"] = str(p)
        self._p(0.4, "Writing the cluster-to-behaviour mapping ...")
        v = self.votes()
        t = out / "cluster_behaviour_mapping_guided.tsv"
        cg.export_mapping_tsv(v, t)
        res["mapping_tsv"] = str(t)
        res["mixed_clusters"] = [int(c) for c in v[v["is_mixed"]]["cluster"]] if len(v) else []
        try:                                                      # guided behaviours + sub-types for the Analyser
            res.update(self.export_behaviours())
        except Exception as e:                                    # noqa: BLE001 - never lose the other exports
            self.log(f"[guided] guided-behaviour export failed: {e}")
            res["guided_behaviours_error"] = str(e)
        return res

    def export_behaviours(self, out_dir=None) -> dict:
        """Write the guided behaviours (+ sub-type clusters per behaviour) as an Analyser-ready folder,
        'CUBE 3b - Guided behaviours', next to this run's 'CUBE 3 - Clusters'. Reuses the trained model."""
        import cube_guided_behaviour as gb
        if self._snap is None:
            self.retrain()
        model, proba = self._snap
        run_dir = Path(self.data.model_dir).parent
        out = Path(out_dir) if out_dir else run_dir.parent / "CUBE 3b - Guided behaviours"
        reserved = []
        try:
            summ = json.loads((run_dir / "bsoid_run_summary.json").read_text())
            reserved = [int(v) for v in (summ.get("turned_away_cluster_id"), summ.get("impure_cluster_display_id"))
                        if v is not None]
        except Exception:                                         # noqa: BLE001
            pass
        keys = [s.key for s in self.data.sessions]
        r = gb.build_guided_behaviours(
            self.data.feats, self.data.ranges, self.data.fps_map, self.ls,
            {s.key: s.frame_clusters for s in self.data.sessions}, {s.key: s.n_frames for s in self.data.sessions},
            out, stems=gb.stems_for(self.data.bout_dir, keys), proba=proba, classes=list(model.classes_),
            embedding=self.data.embedding, reserved_ids=reserved, source_run=run_dir,
            progress=lambda f, t=None: self._p(0.45 + 0.55 * f, t), log=self.log,
            cluster_gate=bool(getattr(self, "cluster_gate", False)))
        return {"guided_behaviours_dir": str(out), "guided_behaviour_groups": r["groups"]}

    def provenance(self) -> dict:
        s = self.summary()
        return {"guided_labels_path": str(self.labels_path), "labels_sha1": self.ls.sha1,
                "n_bouts": s["n_bouts"], "n_behaviours": s["n_behaviours"],
                "seconds_per_behaviour": s["seconds"]}
