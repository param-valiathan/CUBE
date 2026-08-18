"""
cube_3d_dlc.py — 3D DLC + Anipose pipeline for CUBE

Orchestrates per-camera DeepLabCut Zoo adaptation, fill-frame masking,
aniposelib triangulation, quad-composite video export, and optional 3D
skeleton visualization for Step 0 synchronised recordings (4 cameras).

Called from cube.py via _launch_3d_dlc → _run_step("dlc_3d", _run_3d_dlc_step).
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Session discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_step0_sessions(root: Path) -> list[dict]:
    """
    Discover completed Step 0 sessions under *root* (recursive).

    Returns a list of session dicts sorted by session_id:
        {
          "session_id": str,
          "cameras":    {cam_label: Path(mp4)},
          "timestamps": {cam_label: Path(csv)},
          "report":     dict,            # parsed JSON; {} if missing
          "fps":        float,           # from report or 30.0 default
        }

    Only sessions where every camera in the report has status=="done" are
    returned.  Sessions with a missing report are included only if at least
    2 camera MP4s are found (partial-completion guard is then skipped).
    """
    sessions: dict[str, dict] = {}

    for mp4 in sorted(root.rglob("*_corrected.mp4")):
        stem = mp4.stem.replace("_corrected", "")
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue
        cam_label, session_id = parts[0], parts[1]
        if session_id not in sessions:
            sessions[session_id] = {
                "session_id": session_id,
                "cameras": {},
                "timestamps": {},
                "report": {},
                "fps": 30.0,
            }
        sessions[session_id]["cameras"][cam_label] = mp4

        ts_csv = mp4.parent / f"{cam_label}_{session_id}_corrected_timestamps.csv"
        if ts_csv.exists():
            sessions[session_id]["timestamps"][cam_label] = ts_csv

    for sid, sess in sessions.items():
        report_candidates = list(root.rglob(f"{sid}_correction_report.json"))
        if report_candidates:
            try:
                sess["report"] = json.loads(report_candidates[0].read_text())
                sess["fps"] = float(sess["report"].get("fps", 30.0))
            except Exception:
                pass

    complete = []
    for sess in sessions.values():
        report = sess["report"]
        if report and "cameras" in report:
            if not all(c["status"] == "done" for c in report["cameras"]):
                continue
        elif len(sess["cameras"]) < 2:
            continue
        complete.append(sess)

    complete.sort(key=lambda s: s["session_id"])
    return complete


def select_representative(session_dicts: list[dict], cam_label: str) -> "Path | None":
    """
    Return the corrected MP4 for *cam_label* that has the lowest fill rate
    (most genuine frames → best for Zoo adaptation).
    """
    candidates = []
    for sess in session_dicts:
        mp4 = sess["cameras"].get(cam_label)
        if mp4 is None:
            continue
        report = sess["report"]
        fill_rate = 1.0
        if report and "cameras" in report:
            for c in report["cameras"]:
                if c["cam"] == cam_label and c.get("n_output_frames", 0) > 0:
                    fill_rate = c.get("fills_inserted", 0) / c["n_output_frames"]
                    break
        candidates.append((fill_rate, mp4))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ─────────────────────────────────────────────────────────────────────────────
#  Fill-frame masking
# ─────────────────────────────────────────────────────────────────────────────

def mask_fill_frames_in_h5(h5_path: Path, timestamps_csv: Path,
                            out_path: Path, log_fn=print) -> None:
    """
    Zero-out the likelihood of fill frames in a DLC H5 so that
    triangulation discards them when choosing the best camera.

    Atomic: writes to a .tmp.h5 then renames (safe if h5_path == out_path).
    """
    import pandas as pd

    data = np.genfromtxt(str(timestamps_csv), delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    fill_mask = data[:, 3].astype(bool)

    df = pd.read_hdf(str(h5_path))
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df[scorer].columns.get_level_values(0).unique().tolist()

    # Guard: reconcile frame counts if DLC inference dropped or added frames
    n_h5  = len(df)
    n_csv = len(fill_mask)
    if n_h5 != n_csv:
        log_fn(f"  [WARN] mask_fill_frames: H5 has {n_h5} frames but CSV has {n_csv} "
               f"timestamps — reconciling (delta={n_h5 - n_csv:+d})")
        if n_csv > n_h5:
            fill_mask = fill_mask[:n_h5]
        else:
            fill_mask = np.pad(fill_mask, (0, n_h5 - n_csv), constant_values=False)

    for bp in bodyparts:
        try:
            df.loc[fill_mask, (scorer, bp, "likelihood")] = 0.0
        except Exception:
            pass

    tmp = out_path.with_suffix(".tmp.h5")
    df.to_hdf(str(tmp), key="df_with_missing", mode="w", format="fixed")
    tmp.replace(out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Quad composite video
# ─────────────────────────────────────────────────────────────────────────────

def _probe_video_dimensions(ffmpeg_exe: str, video_path: Path) -> tuple | None:
    """Return (width, height) by parsing FFmpeg's -i stderr output."""
    import re
    try:
        r = subprocess.run(
            [ffmpeg_exe, "-i", str(video_path)],
            capture_output=True, timeout=15)
        text = r.stderr.decode(errors="replace")
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def make_quad_video(cam_mp4s: dict, out_path: Path,
                    ffmpeg_exe: str, logger) -> None:
    """
    Create a 2×2 composite MP4 from up to four camera streams.

    All input streams are scaled to a common resolution (most frequent W×H
    across cameras) before compositing, so mismatched sensors produce a
    correctly-dimensioned output.  Encoding prefers NVENC (GPU) and falls
    back to multi-threaded libx264 (CPU).

    cam_mp4s: {cam_label: Path}  (cam0 TL, cam1 TR, cam2 BL, cam3 BR)
    """
    sorted_cams = sorted(cam_mp4s.items())
    n = len(sorted_cams)
    if n < 2:
        raise ValueError(f"Need at least 2 cameras for quad video, got {n}")

    # ── Probe each stream's dimensions ───────────────────────────────────────
    dims: list[tuple[int, int] | None] = [
        _probe_video_dimensions(ffmpeg_exe, p) for _, p in sorted_cams
    ]
    valid_dims = [d for d in dims if d is not None]

    if valid_dims:
        from collections import Counter
        target_w = Counter(w for w, _ in valid_dims).most_common(1)[0][0]
        target_h = Counter(h for _, h in valid_dims).most_common(1)[0][0]
        logger.info(f"  Quad video: normalising streams to {target_w}x{target_h}")
    else:
        target_w = target_h = None

    # ── Build filter_complex ──────────────────────────────────────────────────
    inputs: list[str] = []
    for _, p in sorted_cams:
        inputs += ["-i", str(p)]

    if target_w and target_h:
        scale_part = "".join(
            f"[{i}:v]scale={target_w}:{target_h}[s{i}];" for i in range(n))
        ins = "".join(f"[s{i}]" for i in range(n))
    else:
        scale_part = ""
        ins = "".join(f"[{i}:v]" for i in range(n))

    if n == 4:
        xstack = f"{ins}xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]"
    elif n == 2:
        xstack = f"{ins}hstack=inputs=2[v]"
    else:
        xstack = f"{ins}xstack=inputs=3:layout=0_0|w0_0|0_h0[v]"

    filter_graph = scale_part + xstack

    base_cmd = [ffmpeg_exe, "-y"] + inputs + [
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]

    # ── Try GPU (NVENC) then fall back to multi-threaded CPU ──────────────────
    encoders = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]),
        ("libx264",    ["-c:v", "libx264", "-threads", "0", "-crf", "23", "-preset", "fast"]),
    ]
    for enc_name, enc_args in encoders:
        cmd = base_cmd + enc_args + [str(out_path)]
        logger.info(f"  Creating quad video [{enc_name}]: {out_path.name}")
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            logger.info(f"  Quad video written: {out_path.name}")
            return
        if enc_name == "h264_nvenc":
            logger.info("  NVENC unavailable, retrying with libx264")
        else:
            raise RuntimeError(
                f"FFmpeg quad composite failed:\n{result.stderr.decode()}")


# ─────────────────────────────────────────────────────────────────────────────
#  3D skeleton visualization video
# ─────────────────────────────────────────────────────────────────────────────

def _infer_skeleton(bps: list[str]) -> list[tuple[str, str]]:
    """
    Build skeleton connections from DLC-style body-part names.
    Falls back to sequential nearest-neighbour if names are not recognised.
    """
    KNOWN_CHAINS = [
        ["nose", "neck", "spine1", "spine2", "spine3", "tailbase", "tail1", "tail2"],
        ["left_ear", "neck"],
        ["right_ear", "neck"],
        ["left_forelegroot", "left_foreleg1", "left_foreleg2", "left_foreleg3"],
        ["right_forelegroot", "right_foreleg1", "right_foreleg2", "right_foreleg3"],
        ["left_hindlegroot", "left_hindleg1", "left_hindleg2", "left_hindleg3"],
        ["right_hindlegroot", "right_hindleg1", "right_hindleg2", "right_hindleg3"],
        ["neck", "left_forelegroot"],
        ["neck", "right_forelegroot"],
        ["tailbase", "left_hindlegroot"],
        ["tailbase", "right_hindlegroot"],
    ]
    bp_set = set(b.lower() for b in bps)
    bp_map = {b.lower(): b for b in bps}

    edges = []
    for chain in KNOWN_CHAINS:
        present = [b for b in chain if b in bp_set]
        for a, b in zip(present, present[1:]):
            edges.append((bp_map[a], bp_map[b]))

    if not edges:
        for a, b in zip(bps, bps[1:]):
            edges.append((a, b))

    return edges


def make_skeleton_video(h5_3d: Path, out_path: Path, fps: float,
                         ffmpeg_exe: str, logger,
                         subsample: int = 2) -> None:
    """
    Render a 3D skeleton animation from a triangulated 3D H5 file and pipe
    frames to FFmpeg.  Uses matplotlib Agg backend (no display needed).

    subsample: render every Nth frame (default 2 → output at fps/2).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import pandas as pd

    df = pd.read_hdf(str(h5_3d))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()

    try:
        X  = df[scorer].xs("x",          level="coords", axis=1).values
        Y  = df[scorer].xs("y",          level="coords", axis=1).values
        Z  = df[scorer].xs("z",          level="coords", axis=1).values
        LL = df[scorer].xs("likelihood", level="coords", axis=1).values
    except KeyError as e:
        raise ValueError(f"H5 file lacks required coord column: {e}")

    XYZ = np.stack([X, Y, Z], axis=-1)   # (N_frames, N_bp, 3)
    SKELETON = _infer_skeleton(bps)

    mask = LL > 0.3
    valid_pts = XYZ[mask.any(axis=1)].reshape(-1, 3)
    lo = np.nanpercentile(valid_pts, 2.5, axis=0)
    hi = np.nanpercentile(valid_pts, 97.5, axis=0)
    padding = (hi - lo) * 0.1 + 1e-6
    lo -= padding
    hi += padding

    out_fps = fps / subsample
    fig = plt.figure(figsize=(6, 6), dpi=120, facecolor="#0d0d1a")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d1a")
    ax.view_init(elev=25, azim=45)

    cmd = [ffmpeg_exe, "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "720x720", "-r", str(out_fps), "-i", "pipe:0",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        n_total = len(XYZ)
        for k in range(0, n_total, subsample):
            pts = XYZ[k]
            ll  = LL[k]

            ax.cla()
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#555577")
            ax.xaxis.label.set_color("#8888aa")
            ax.yaxis.label.set_color("#8888aa")
            ax.zaxis.label.set_color("#8888aa")

            for bp_a, bp_b in SKELETON:
                if bp_a not in bps or bp_b not in bps:
                    continue
                ia, ib = bps.index(bp_a), bps.index(bp_b)
                conf = float(min(ll[ia], ll[ib]))
                color = plt.cm.plasma(conf)
                ax.plot([pts[ia, 0], pts[ib, 0]],
                        [pts[ia, 1], pts[ib, 1]],
                        [pts[ia, 2], pts[ib, 2]],
                        color=color, linewidth=2)

            colors = plt.cm.plasma(ll)
            ax.scatter(pts[:, 0], pts[:, 1], zs=pts[:, 2],
                       s=30, c=colors, depthshade=True)

            ax.set_title(f"Frame {k}/{n_total}", color="#ccccee", fontsize=9)

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            proc.stdin.write(buf.tobytes())

        proc.stdin.close()
        proc.wait()
    except Exception:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
        raise
    finally:
        plt.close(fig)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Skeleton video FFmpeg failed:\n{proc.stderr.read().decode()}")
    logger.info(f"  Skeleton video written: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-camera Zoo adaptation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_batch_size() -> int:
    """Return a conservative GPU batch size (85 % of free VRAM)."""
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            usable  = free_gb * 0.85
            return 32 if usable >= 10 else (16 if usable >= 5 else 8)
    except Exception:
        pass
    return 8


def _adapt_zoo_for_camera(rep_video: Path, cam_label: str,
                           adapt_work: Path, logger, n_epochs: int,
                           sa_name: str, model_name: str, detector_name: str,
                           pcutoff: float, bbox_thr: float,
                           max_ind: int, inf_batch: int):
    """
    Run DLC Zoo video adaptation on *rep_video*.
    Returns (adapted_pose_ckpt, adapted_detector_ckpt, snapshot_base, adapt_project_dir)
    where the first available pair should be used for batch inference.
    """
    import deeplabcut as dlc

    cam_adapt_dir = adapt_work / f"adapt_{cam_label}"
    cam_adapt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [{cam_label}] Zoo adaptation on: {rep_video.name}")

    adapt_config_path  = None
    adapt_project_dir  = None
    adapted_pose_ckpt  = None
    adapted_det_ckpt   = None
    snapshot_base      = None

    try:
        if hasattr(dlc, "create_video_adaptation_project"):
            kw = dict(
                videos            = [str(rep_video)],
                working_directory = str(cam_adapt_dir),
                superanimal_name  = sa_name,
                num_epochs        = n_epochs,
                batch_size        = inf_batch,
            )
            try:
                import inspect as _insp
                sig = _insp.signature(dlc.create_video_adaptation_project)
                if "display_iters" in sig.parameters:
                    kw["display_iters"] = 100
                elif "displayiters" in sig.parameters:
                    kw["displayiters"] = 100
            except Exception:
                pass

            result = dlc.create_video_adaptation_project(**kw)
            if result and Path(str(result)).is_file():
                adapt_config_path = str(result)
                adapt_project_dir = Path(result).parent

        if adapt_config_path is None:
            # Fallback: video_inference_superanimal with video_adapt=True
            rep_dest = cam_adapt_dir / f"{rep_video.stem}_adapt_results"
            rep_dest.mkdir(parents=True, exist_ok=True)
            rep_inf = rep_dest / rep_video.name
            if not rep_inf.exists():
                shutil.copy2(str(rep_video), str(rep_inf))

            sa_kw = dict(
                superanimal_name     = sa_name,
                model_name           = model_name,
                detector_name        = detector_name,
                scale_list           = list(range(100, 650, 50)),
                pcutoff              = pcutoff,
                bbox_threshold       = bbox_thr,
                max_individuals      = max_ind,
                batch_size           = inf_batch,
                create_labeled_video = False,
                video_adapt          = True,
                pseudo_threshold     = 0.5,
                detector_epochs      = n_epochs,
                pose_epochs          = n_epochs,
                device               = "auto",
            )
            try:
                import inspect as _insp
                sig = _insp.signature(dlc.video_inference_superanimal).parameters
                if "superanimal_transfer_learning" in sig:
                    sa_kw["superanimal_transfer_learning"] = True
                if "video_adapt_batch_size" in sig:
                    sa_kw["video_adapt_batch_size"] = inf_batch
            except Exception:
                pass

            dlc.video_inference_superanimal([str(rep_inf)], **sa_kw)
            adapt_project_dir = cam_adapt_dir

        # Find adapted checkpoints
        adapted_pose_ckpt, adapted_det_ckpt = _find_adapted_pt_checkpoints(
            adapt_project_dir, model_name, detector_name, logger)
        if adapted_pose_ckpt is None:
            snapshot_base = _find_highest_snapshot_path(adapt_project_dir, logger)

    except Exception:
        logger.error(
            f"  [{cam_label}] Adaptation failed:\n{traceback.format_exc()}")

    return adapted_pose_ckpt, adapted_det_ckpt, snapshot_base, adapt_project_dir


def _find_adapted_pt_checkpoints(search_root: Path, model_name: str,
                                  detector_name: str, logger
                                  ) -> tuple[str | None, str | None]:
    """Locate DLC 3.x PyTorch .pt checkpoints in adaptation workspace."""
    pose_ckpt = None
    det_ckpt  = None
    if search_root is None or not search_root.exists():
        return None, None
    all_pts = sorted(search_root.rglob("*.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for pt in all_pts:
        nm = pt.name.lower()
        if "pose" in nm or model_name.lower() in nm:
            if pose_ckpt is None:
                pose_ckpt = str(pt)
        elif "detect" in nm or detector_name.lower() in nm:
            if det_ckpt is None:
                det_ckpt = str(pt)
        if pose_ckpt and det_ckpt:
            break
    if pose_ckpt:
        logger.info(f"  Found adapted pose checkpoint: {Path(pose_ckpt).name}")
    if det_ckpt:
        logger.info(f"  Found adapted detector checkpoint: {Path(det_ckpt).name}")
    return pose_ckpt, det_ckpt


def _find_highest_snapshot_path(search_root: Path, logger) -> str | None:
    """Locate the highest-numbered TF snapshot-N.index file."""
    if search_root is None or not search_root.exists():
        return None
    import re as _re
    best_num  = -1
    best_path = None
    for p in search_root.rglob("snapshot-*.index"):
        m = _re.search(r"snapshot-(\d+)\.index$", p.name)
        if m:
            n = int(m.group(1))
            if n > best_num:
                best_num  = n
                best_path = str(p.with_suffix(""))   # strip .index
    if best_path:
        logger.info(f"  Found TF snapshot: snapshot-{best_num}")
    return best_path


def _infer_camera_videos(dlc, videos: list[Path], cam_label: str,
                          adapted_pose_ckpt: str | None,
                          adapted_det_ckpt: str | None,
                          _snapshot_base: str | None,
                          inference_config: str | None,
                          sa_name: str, model_name: str, detector_name: str,
                          pcutoff: float, bbox_thr: float, max_ind: int,
                          filter_types: list,
                          fps: float,
                          logger, inf_batch: int) -> None:
    """
    Run batch inference for *cam_label* on all *videos*, applying OOM-retry.
    Writes {stem}_filtered.h5 alongside each input video.
    """
    from cube_core import filter_dlc_h5

    current_batch = inf_batch
    for idx, vpath in enumerate(videos, 1):
        vname       = vpath.name
        base_noext  = vpath.stem
        dest_folder = vpath.parent / f"{base_noext}_results"
        dest_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"  [{cam_label}] [{idx}/{len(videos)}] {vname}")

        h5_before = set(dest_folder.glob("*.h5"))
        succeeded = False
        while not succeeded:
            try:
                if adapted_pose_ckpt:
                    sa_kw = dict(
                        superanimal_name               = sa_name,
                        model_name                     = model_name,
                        detector_name                  = detector_name,
                        scale_list                     = list(range(100, 650, 50)),
                        pcutoff                        = pcutoff,
                        bbox_threshold                 = bbox_thr,
                        max_individuals                = max_ind,
                        batch_size                     = current_batch,
                        create_labeled_video           = False,
                        video_adapt                    = False,
                        dest_folder                    = str(dest_folder),
                        device                         = "auto",
                        customized_pose_checkpoint     = adapted_pose_ckpt,
                    )
                    if adapted_det_ckpt:
                        sa_kw["customized_detector_checkpoint"] = adapted_det_ckpt
                    dlc.video_inference_superanimal([str(vpath)], **sa_kw)
                elif inference_config:
                    dlc.analyze_videos(
                        inference_config,
                        [str(vpath)],
                        save_as_csv    = True,
                        destfolder     = str(dest_folder),
                        batchsize      = current_batch,
                        allow_growth   = True,
                        robust_nframes = True,
                    )
                else:
                    sa_kw = dict(
                        superanimal_name     = sa_name,
                        model_name           = model_name,
                        detector_name        = detector_name,
                        scale_list           = list(range(100, 650, 50)),
                        pcutoff              = pcutoff,
                        bbox_threshold       = bbox_thr,
                        max_individuals      = max_ind,
                        batch_size           = current_batch,
                        create_labeled_video = False,
                        video_adapt          = False,
                        dest_folder          = str(dest_folder),
                        device               = "auto",
                    )
                    dlc.video_inference_superanimal([str(vpath)], **sa_kw)
                succeeded = True
            except RuntimeError as oom_exc:
                exc_l = str(oom_exc).lower()
                if "cuda" in exc_l and ("memory" in exc_l or "oom" in exc_l
                                         or "alloc" in exc_l):
                    if current_batch <= 1:
                        logger.error(
                            f"  [{cam_label}] OOM at batchsize=1 — skipping {vname}")
                        succeeded = True
                    else:
                        current_batch = max(1, current_batch // 2)
                        logger.warn(
                            f"  [{cam_label}] OOM → batchsize={current_batch}, retrying")
                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    logger.error(f"  [{cam_label}] RuntimeError: {oom_exc}")
                    succeeded = True
            except Exception:
                logger.error(
                    f"  [{cam_label}] Inference error on {vname}:\n{traceback.format_exc()}")
                succeeded = True

        # Post-inference: filter H5
        h5_new = [p for p in dest_folder.glob("*.h5")
                  if p not in h5_before
                  and not p.name.startswith("BSOID_")
                  and not p.stem.endswith("_filtered")]
        if h5_new:
            final_h5       = h5_new[0]
            clean_filtered = dest_folder / f"{base_noext}_filtered.h5"
            if filter_types:
                try:
                    filter_dlc_h5(final_h5, filter_types, log_fn=logger,
                                  out_path=clean_filtered, fps=fps,
                                  likelihood_thresh=pcutoff)
                except Exception as fe:
                    logger.warn(f"  filter_dlc_h5 failed: {fe} — copying raw")
                    shutil.copy2(str(final_h5), str(clean_filtered))
            else:
                shutil.copy2(str(final_h5), str(clean_filtered))
            try:
                final_h5.rename(dest_folder / f"{base_noext}.h5")
            except Exception:
                pass

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _find_camera_h5(mp4: Path) -> Path | None:
    """Locate the *_filtered.h5 written next to a corrected MP4 by _infer_camera_videos."""
    stem = mp4.stem  # e.g. cam0_30fps2026-06-23T16_46_24_corrected
    results_dir = mp4.parent / f"{stem}_results"
    base = stem
    filtered = results_dir / f"{base}_filtered.h5"
    if filtered.exists():
        return filtered
    # Fallback: any *_filtered.h5 in the results dir
    cands = list(results_dir.glob("*_filtered.h5"))
    return cands[0] if cands else None


# ─────────────────────────────────────────────────────────────────────────────
#  Triangulation (per session)
# ─────────────────────────────────────────────────────────────────────────────

def _triangulate_session(sess: dict, cam_labels: list[str],
                          calib_toml: Path, out_folder: Path,
                          ll_agg: str, logger,
                          use_ransac: bool = False,
                          ransac_threshold: float = 0.5,
                          ll_threshold: float = 0.0,
                          ll_gate: float = 0.6,
                          median_window: int = 3) -> Path:
    """
    Triangulate all cameras for one session.

    Writes {out_folder}/{session_id}_3d_filtered.h5 (atomic).
    Returns the output H5 path.
    """
    import pandas as pd
    from cube_core import triangulate_cameras

    sid = sess["session_id"]
    session_dir = out_folder / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    out_h5 = session_dir / f"{sid}_3d_filtered.h5"
    if out_h5.exists():
        logger.info(f"  [{sid}] Skipping (already exists): {out_h5.name}")
        return out_h5

    # Collect H5s per camera (only cameras that have a valid H5)
    cam_h5s: dict[str, Path] = {}
    for cam in cam_labels:
        mp4 = sess["cameras"].get(cam)
        if mp4 is None:
            continue
        h5 = _find_camera_h5(mp4)
        if h5 and h5.exists():
            cam_h5s[cam] = h5
        else:
            logger.warn(f"  [{sid}] No H5 for {cam} — will be skipped in triangulation")

    if len(cam_h5s) < 2:
        raise RuntimeError(
            f"[{sid}] Need at least 2 cameras with H5 files; found {len(cam_h5s)}")

    present_cams = sorted(cam_h5s.keys())
    logger.info(f"  [{sid}] Triangulating cameras: {present_cams}")

    # Load all DLC DataFrames; align bodyparts
    dfs: list = []
    bps_per_cam: list[list[str]] = []
    for cam in present_cams:
        df = pd.read_hdf(str(cam_h5s[cam]))
        scorer = df.columns.get_level_values(0)[0]
        bps = df[scorer].columns.get_level_values(0).unique().tolist()
        bps_per_cam.append(bps)
        dfs.append((scorer, df))

    # Intersect bodyparts across cameras for a consistent output
    common_bps = list(bps_per_cam[0])
    for bps_k in bps_per_cam[1:]:
        common_bps = [b for b in common_bps if b in bps_k]
    if not common_bps:
        raise RuntimeError(f"[{sid}] No common bodyparts across cameras")

    # Determine frame count (guaranteed identical by Step 0; take minimum as safety)
    n_frames = min(len(df) for _, df in dfs)
    n_bp     = len(common_bps)
    n_cams   = len(present_cams)

    pts2d  = np.zeros((n_cams, n_frames, n_bp, 2), dtype=np.float64)
    scores = np.zeros((n_cams, n_frames, n_bp),    dtype=np.float64)

    for ci, (cam, (scorer, df)) in enumerate(zip(present_cams, dfs)):
        for bi, bp in enumerate(common_bps):
            try:
                x  = df[scorer, bp, "x"].values[:n_frames].astype(float)
                y  = df[scorer, bp, "y"].values[:n_frames].astype(float)
                ll = df[scorer, bp, "likelihood"].values[:n_frames].astype(float)
            except Exception:
                x = y = ll = np.zeros(n_frames)
            pts2d[ci, :, bi, 0] = x
            pts2d[ci, :, bi, 1] = y
            scores[ci, :, bi]   = ll

    # Run triangulation
    pts3d = triangulate_cameras(pts2d, scores, calib_toml,
                                use_ransac=use_ransac,
                                ransac_threshold=ransac_threshold,
                                ll_gate=ll_gate,
                                median_window=median_window,
                                log_fn=logger.info)
    # pts3d: (n_frames, n_bp, 3)

    # Combine likelihood across cameras
    if ll_agg == "mean":
        combined_ll = scores.mean(axis=0)   # (n_frames, n_bp)
    elif ll_agg == "max":
        combined_ll = scores.max(axis=0)
    else:
        combined_ll = scores.min(axis=0)    # default "min" — most conservative

    # Per-point confidence gate: zero likelihood below threshold so BSoid prep
    # excludes these bodypart slots during bodypart conservation scoring.
    if ll_threshold > 0.0:
        combined_ll = combined_ll.copy()
        combined_ll[combined_ll < ll_threshold] = 0.0

    # Build 4-coord H5: (scorer, bodyparts, ['x','y','z','likelihood'])
    scorer_name = "cube_3d"
    arrays = [
        [scorer_name] * (4 * n_bp),
        np.repeat(common_bps, 4).tolist(),
        ["x", "y", "z", "likelihood"] * n_bp,
    ]
    midx = pd.MultiIndex.from_arrays(arrays, names=["scorer", "bodyparts", "coords"])

    data = np.zeros((n_frames, 4 * n_bp), dtype=np.float64)
    for bi in range(n_bp):
        col = bi * 4
        data[:, col]     = pts3d[:, bi, 0]
        data[:, col + 1] = pts3d[:, bi, 1]
        data[:, col + 2] = pts3d[:, bi, 2]
        data[:, col + 3] = combined_ll[:, bi]

    df_out = pd.DataFrame(data, columns=midx)

    tmp = out_h5.with_suffix(".tmp.h5")
    df_out.to_hdf(str(tmp), key="df_with_missing", mode="w", format="fixed")
    tmp.replace(out_h5)
    logger.info(f"  [{sid}] 3D H5 written: {out_h5.name}")

    return out_h5


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _run_3d_dlc_step(session: dict, settings, logger, pb, after_fn: Callable):
    """
    4-phase 3D DLC + Anipose pipeline.

    Phase 1: Validate inputs, discover Step 0 sessions.
    Phase 2: Per-camera Zoo adaptation + inference + fill masking.
    Phase 3: Parallel triangulation → 3D H5 + quad composite video.
    Phase 4: BSOID prep (inline, if auto_bsoid disabled or dlc_run_prep=True).

    ntfy notifications are sent at each major milestone.
    """
    # Import send_push_notification lazily to avoid circular import
    # (cube.py imports cube_3d_dlc at step launch time, not at module load)
    import importlib
    _cube_mod = importlib.import_module("cube")
    _notify = getattr(_cube_mod, "send_push_notification", lambda *a, **k: None)

    os.environ["DL_LIGHT"] = "True"

    try:
        import deeplabcut as dlc
    except ImportError:
        raise ImportError(
            "DeepLabCut is not installed. "
            "Activate your DLC conda environment and relaunch.")

    try:
        from aniposelib.cameras import CameraGroup  # noqa: F401 — import test
    except ImportError:
        raise ImportError(
            "aniposelib is not installed.  "
            "pip install aniposelib  (inside your cube conda environment)")

    # =========================================================================
    #  Phase 1 — Validate & Discover
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "3D DLC Phase 1/4: Validation & Session Discovery")
    pb.step_indeterminate("3D DLC: Discovering sessions …")

    calib_folder = Path(session.get("dlc_3d_calib_folder", ""))
    calib_toml   = calib_folder / "calibration.toml"
    if not calib_toml.exists():
        raise FileNotFoundError(
            f"calibration.toml not found in: {calib_folder}\n"
            "Please set the calibration folder in the 3D DLC tab.")

    # On re-runs video_folders contains session subdirs, not source folders.
    # Use dlc_3d_source_folders when populated (set at end of first run), else video_folders.
    _raw_src = (session.get("dlc_3d_source_folders") or
                session.get("video_folders", []))
    source_folders = [Path(f) for f in _raw_src if Path(f).is_dir()]
    if not source_folders:
        raise FileNotFoundError(
            "No video source folders found.  "
            "Add Step 0 output folders in the main panel before running 3D DLC.")

    video_groups: dict = dict(session.get("video_groups", {}))
    explicit_out: str  = session.get("dlc_3d_output_folder", "").strip()

    cam_labels_raw = session.get("dlc_3d_cam_labels", ["cam0", "cam1", "cam2", "cam3"])
    if isinstance(cam_labels_raw, str):
        cam_labels = [c.strip() for c in cam_labels_raw.split(",") if c.strip()]
    else:
        cam_labels = list(cam_labels_raw)

    # Discover Step 0 sessions across all source folders; first occurrence wins
    sessions = []
    session_src: dict[str, Path] = {}   # session_id → source folder Path
    for vf in source_folders:
        for sess_dict in find_step0_sessions(vf):
            sid = sess_dict["session_id"]
            if sid not in session_src:
                session_src[sid] = vf
                sessions.append(sess_dict)

    if not sessions:
        raise RuntimeError(
            f"No complete Step 0 sessions found in: "
            f"{[str(f) for f in source_folders]}")

    logger.success(
        f"  Found {len(sessions)} session(s) across {len(source_folders)} folder(s)")

    def _out_root(sid: str) -> Path:
        """Return the output root for a session (explicit folder or its source folder)."""
        root = Path(explicit_out) if explicit_out else session_src[sid]
        root.mkdir(parents=True, exist_ok=True)
        return root

    _adv           = session.get("dlc_advanced_cfg", {})
    sa_name        = str(_adv.get("dlc_superanimal_name", "superanimal_quadruped"))
    model_name     = str(_adv.get("dlc_architecture", "hrnet_w32"))
    detector_name  = str(_adv.get("dlc_detector",
                                   "fasterrcnn_mobilenet_v3_large_fpn"))
    pcutoff        = float(_adv.get("dlc_pcutoff",        0.6))
    bbox_thr       = float(_adv.get("dlc_bbox_threshold", 0.6))
    max_ind        = int(_adv.get("dlc_max_individuals",  1))
    n_epochs       = int(settings.get("dlc_epochs", 15))

    filter_key   = settings.get("dlc_filter", "Sequential  Median  Gaussian")
    FILTER_OPTIONS = {
        "Sequential  Median  Gaussian": ["median", "gaussian"],
        "Median only":                  ["median"],
        "None":                         [],
    }
    filter_types = FILTER_OPTIONS.get(filter_key, ["median", "gaussian"])

    fps              = float(sessions[0]["fps"]) if sessions else 30.0
    ll_agg           = str(session.get("dlc_3d_ll_agg", "min"))
    use_ransac       = bool(session.get("dlc_3d_use_ransac", True))
    ransac_threshold = float(session.get("dlc_3d_ransac_threshold", 0.5))
    ll_threshold     = float(session.get("dlc_3d_ll_threshold", 0.0))
    ll_gate          = float(session.get("dlc_3d_ll_gate", 0.6))
    median_window    = int(session.get("dlc_3d_median_window", 3))
    export_skel      = bool(session.get("dlc_3d_export_skeleton_video", False))
    del_orig_vid     = bool(session.get("dlc_3d_delete_orig_videos",  False))
    del_cam_h5s      = bool(session.get("dlc_3d_delete_cam_h5s",      False))

    inf_batch = _gpu_batch_size()

    # Adaptation workspace: use explicit output folder if set, otherwise first source folder.
    adapt_work = (Path(explicit_out) if explicit_out else source_folders[0]) / "3d_adapt_workspace"
    adapt_work.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    #  Phase 2 — Per-camera Zoo adaptation + inference + fill masking
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "3D DLC Phase 2/4: Per-camera DLC inference")

    models: dict = dict(session.get("dlc_3d_models", {}))

    # Import monkeypatch helper from cube.py if available
    try:
        _cube_mod_ref = importlib.import_module("cube")
        _monkeypatch = getattr(_cube_mod_ref, "_apply_dlc_monkeypatch",
                                lambda lg: None)
        _monkeypatch(logger)
    except Exception:
        pass

    for cam_label in cam_labels:
        logger.step(f"  Camera: {cam_label}")
        pb.step_start(f"3D DLC — {cam_label}: adaptation", 1)

        if cam_label not in models:
            rep_video = select_representative(sessions, cam_label)
            if rep_video is None:
                logger.warn(f"  No video found for {cam_label} — skipping")
                after_fn(lambda: pb.step_done())
                continue

            pose_ckpt, det_ckpt, snap_base, adapt_dir = _adapt_zoo_for_camera(
                rep_video, cam_label, adapt_work, logger,
                n_epochs, sa_name, model_name, detector_name,
                pcutoff, bbox_thr, max_ind, inf_batch)

            models[cam_label] = {
                "pose_ckpt": pose_ckpt,
                "det_ckpt":  det_ckpt,
                "snap_base": snap_base,
                "adapt_dir": str(adapt_dir) if adapt_dir else None,
            }
            session["dlc_3d_models"] = models
            try:
                session.save()
            except Exception:
                pass

            _notify(session,
                    f"Camera {cam_label} adaptation complete on: {rep_video.name}",
                    title=f"CUBE 3D — {cam_label} Adapted", logger=logger)
        else:
            logger.info(f"  [{cam_label}] Using cached model")

        after_fn(lambda: pb.step_done())

        # ── Batch inference ──────────────────────────────────────────────────
        cam_info = models[cam_label]
        pose_ckpt    = cam_info.get("pose_ckpt")
        det_ckpt     = cam_info.get("det_ckpt")
        snap_base    = cam_info.get("snap_base")
        adapt_dir    = Path(cam_info["adapt_dir"]) if cam_info.get("adapt_dir") else None
        inf_cfg      = None
        if adapt_dir and adapt_dir.exists():
            cfg_cands = list(adapt_dir.rglob("config.yaml"))
            if cfg_cands:
                inf_cfg = str(cfg_cands[0])

        all_cam_videos = [s["cameras"][cam_label]
                          for s in sessions if cam_label in s["cameras"]]
        pb.step_start(f"3D DLC — {cam_label}: inference", len(all_cam_videos))

        _infer_camera_videos(
            dlc, all_cam_videos, cam_label,
            pose_ckpt, det_ckpt, snap_base, inf_cfg,
            sa_name, model_name, detector_name,
            pcutoff, bbox_thr, max_ind, filter_types, fps,
            logger, inf_batch)

        after_fn(lambda: pb.step_done())

        # ── Fill-frame masking ───────────────────────────────────────────────
        for sess in sessions:
            mp4 = sess["cameras"].get(cam_label)
            ts_csv = sess["timestamps"].get(cam_label)
            if mp4 is None or ts_csv is None:
                continue
            h5 = _find_camera_h5(mp4)
            if h5 and h5.exists():
                try:
                    mask_fill_frames_in_h5(h5, ts_csv, h5, log_fn=logger.info)
                except Exception as fe:
                    logger.warn(f"  Fill-mask failed for {h5.name}: {fe}")

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    _notify(session,
            f"DLC inference complete for {len(cam_labels)} camera(s), "
            f"{len(sessions)} session(s).",
            title="CUBE 3D — Inference Complete", logger=logger)

    # =========================================================================
    #  Phase 3 — Triangulation + quad composite video
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "3D DLC Phase 3/4: Triangulation & composite video")

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    completed_sessions: list[dict] = []
    failed_sessions:    list[str]  = []

    pb.step_start("3D DLC: Triangulation", len(sessions))

    def _triang_worker(sess):
        return _triangulate_session(
            sess, cam_labels, calib_toml, _out_root(sess["session_id"]), ll_agg, logger,
            use_ransac=use_ransac,
            ransac_threshold=ransac_threshold,
            ll_threshold=ll_threshold,
            ll_gate=ll_gate,
            median_window=median_window)

    with ThreadPoolExecutor(max_workers=min(4, len(sessions))) as pool:
        future_to_sid = {pool.submit(_triang_worker, s): s["session_id"]
                         for s in sessions}
        for fut in as_completed(future_to_sid):
            sid = future_to_sid[fut]
            try:
                out_h5 = fut.result()
                completed_sessions.append(
                    {"session_id": sid, "h5": out_h5,
                     "session": next(s for s in sessions if s["session_id"] == sid)})
                logger.success(f"  Triangulated: {sid}")
            except Exception as exc:
                logger.error(f"  Triangulation FAILED for {sid}: {exc}")
                failed_sessions.append(sid)
            after_fn(lambda: pb.step_tick(1, len(sessions)))

    pb.step_done()

    _notify(session,
            f"Triangulation complete: {len(completed_sessions)} OK, "
            f"{len(failed_sessions)} failed.",
            title="CUBE 3D — Triangulation Complete", logger=logger)

    # ── Quad composite video per session ─────────────────────────────────────
    pb.step_start("3D DLC: Quad videos", len(completed_sessions))

    for item in completed_sessions:
        sid  = item["session_id"]
        sess = item["session"]
        session_dir = _out_root(sid) / sid
        quad_path   = session_dir / f"{sid}_quad.mp4"

        cam_mp4s = {cam: sess["cameras"][cam]
                    for cam in cam_labels if cam in sess["cameras"]}

        if len(cam_mp4s) >= 2 and not quad_path.exists():
            try:
                make_quad_video(cam_mp4s, quad_path, ffmpeg_exe, logger)
                if del_orig_vid and quad_path.exists() and quad_path.stat().st_size > 0:
                    for mp4 in cam_mp4s.values():
                        try:
                            mp4.unlink()
                            logger.info(f"  Deleted: {mp4.name}")
                        except Exception as de:
                            logger.warn(f"  Could not delete {mp4.name}: {de}")
            except Exception as qe:
                logger.error(f"  Quad video failed for {sid}: {qe}")
        else:
            if quad_path.exists():
                logger.info(f"  [{sid}] Quad video already exists")
            elif len(cam_mp4s) < 2:
                logger.warn(f"  [{sid}] Not enough camera videos for quad composite")

        if del_cam_h5s:
            out_h5 = item["h5"]
            if out_h5.exists():
                for cam in cam_labels:
                    mp4 = sess["cameras"].get(cam)
                    if mp4 is None:
                        continue
                    h5 = _find_camera_h5(mp4)
                    if h5 and h5.exists():
                        try:
                            h5.unlink()
                            logger.info(f"  Deleted cam H5: {h5.name}")
                        except Exception:
                            pass

        if export_skel:
            skel_path = session_dir / f"{sid}_skeleton3d.mp4"
            if not skel_path.exists():
                try:
                    make_skeleton_video(
                        item["h5"], skel_path, fps, ffmpeg_exe, logger)
                except Exception as se:
                    logger.error(f"  Skeleton video failed for {sid}: {se}")

        after_fn(lambda: pb.step_tick(1, len(completed_sessions)))

    pb.step_done()

    _notify(session,
            f"Quad composite videos created for {len(completed_sessions)} session(s).",
            title="CUBE 3D — Quad Videos Created", logger=logger)

    # Persist source folders so a re-run can rediscover sessions correctly
    session["dlc_3d_source_folders"] = [str(vf) for vf in source_folders]

    # ── Replace video_folders with session output dirs; carry group labels ────
    new_session_dirs: list[str] = []
    new_video_groups: dict = dict(video_groups)  # preserve any pre-existing mappings
    for item in completed_sessions:
        sid  = item["session_id"]
        sdir = str(_out_root(sid) / sid)
        new_session_dirs.append(sdir)
        src = session_src.get(sid)
        if src:
            grp = video_groups.get(str(src), "")
            if grp:
                new_video_groups[sdir] = grp
    session["video_folders"] = new_session_dirs
    session["video_groups"]  = new_video_groups
    try:
        session.save()
    except Exception:
        pass

    # =========================================================================
    #  Phase 4 — BSOID Prep (inline when auto_bsoid is off or dlc_run_prep=True)
    # =========================================================================
    auto_bsoid = bool(session.get("auto_bsoid", True))
    run_prep   = bool(settings.get("dlc_run_prep", True))

    if run_prep and not auto_bsoid:
        logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "3D DLC Phase 4/4: BSOID Pre-processing")
        from cube_core import run_bsoid_prep
        bsoid_roots = []
        for item in completed_sessions:
            sid = item["session_id"]
            sdir = _out_root(sid) / sid
            try:
                root = run_bsoid_prep(
                    str(sdir), log_fn=logger,
                    min_confidence=float(session.get("bsoid_min_conf", 0.30)),
                    conf_metric=str(session.get("bsoid_conf_metric", "median")),
                    min_session_frac=float(session.get("bsoid_min_sess_frac", 0.6)),
                    min_keep=int(session.get("bsoid_min_keep", 6)))
                if root:
                    bsoid_roots.append(str(root))
                    logger.success(f"  BSOID prep done: {sid}")
            except Exception as be:
                logger.error(f"  BSOID prep failed for {sid}: {be}")
        session["bsoid_ready_dirs"] = bsoid_roots
        try:
            session.save()
        except Exception:
            pass

        _notify(session,
                f"BSOID pre-processing done for {len(bsoid_roots)} session(s).",
                title="CUBE 3D — Pre-processing Complete", logger=logger)

    if failed_sessions:
        logger.warn(
            f"3D DLC finished with {len(failed_sessions)} triangulation failure(s): "
            f"{failed_sessions}")
    else:
        logger.success(
            f"3D DLC complete: {len(completed_sessions)} session(s) processed  "
            f"[{datetime.now().strftime('%H:%M:%S')}]")

    _notify(session,
            f"3D DLC complete: {len(completed_sessions)} OK"
            + (f", {len(failed_sessions)} failed" if failed_sessions else "") + ".",
            title="CUBE 3D — Complete", logger=logger)
