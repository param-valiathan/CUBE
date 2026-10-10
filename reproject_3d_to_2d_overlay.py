"""
3D->2D reprojection overlay tool.

Projects the already-triangulated {session_id}_3d_filtered.h5 back through
each camera's own calibration.toml intrinsics/extrinsics into that camera's
own pixel space, and draws it (hollow circle) next to the camera's raw 2D
DLC detection (solid dot) on that camera's own cropped corrected video --
one output video per camera, restricted to a small set of "main" bodyparts
(face, neck, back, paws) per user request, since drawing all ~24
SuperAnimal-Quadruped bodyparts here would be unreadable clutter.

This is a read-only QC tool: it never touches any pipeline H5, calibration,
or video. Use it to spot which camera/frame drove a bad triangulation --
a large, sustained gap between the reprojected (hollow) and raw-detected
(solid) point for one bodypart on one camera, while the other cameras'
reprojections track their own raw detections tightly, points at that
camera's detection as the likely bad input for that stretch (the same
diagnostic logic TMZ7_TRIANGULATION_QUALITY_HANDOFF_2026-09-09.md and the
double-sided-calibration-board investigation both used, just automated
and visual instead of manual per-frame H5 inspection).

Usage:
    "C:\\Users\\param\\anaconda3\\envs\\cube\\python.exe" reproject_3d_to_2d_overlay.py ^
        --session-dir "F:\\TMZ7\\CUBE results 2026-09-04_15-41-51\\CUBE 0 - Corrected 3D vids" ^
        --session-id "MOUSE 1_0000" ^
        --calib "F:\\TMZ7\\...\\CALIBRATION_0002\\calibration_auto\\calibration_refined.toml" ^
        --prep-dir "F:\\TMZ7\\CUBE results 2026-09-04_15-41-51\\CUBE 1b - 3D DLC\\_prep" ^
        --out-dir "F:\\TMZ7\\reprojection_qc\\MOUSE 1_0000"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

# Small, fixed "main bodypart" set per user request -- face, neck, back, paws.
# Matches SuperAnimal-Quadruped naming used throughout cube_3d_dlc.py
# (MOUSE_OUTLINE_BODYPARTS is the closest existing precedent, minus the
# knee joints -- not requested here).
MAIN_BODYPARTS = [
    "nose", "left_eye", "right_eye", "left_earbase", "right_earbase",
    "neck_base", "back_end",
    "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
]

BGR_COLORS = {
    "nose":            (60, 60, 230),
    "left_eye":        (60, 160, 230),
    "right_eye":       (60, 220, 230),
    "left_earbase":    (60, 230, 130),
    "right_earbase":   (60, 230, 60),
    "neck_base":       (200, 200, 60),
    "back_end":        (230, 160, 60),
    "front_left_paw":  (230, 60, 200),
    "front_right_paw": (180, 60, 230),
    "back_left_paw":   (130, 60, 230),
    "back_right_paw":  (90, 60, 230),
}


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


def _nvenc_available(ffmpeg_exe: str) -> bool:
    try:
        probe = subprocess.run(
            [ffmpeg_exe, "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=15)
        return probe.returncode == 0
    except Exception:
        return False


def _drain_stream(stream, sink: list) -> None:
    try:
        sink.append(stream.read())
    except Exception:
        pass


def load_crop_rect(prep_dir: Path, cam: str, session_id: str) -> dict:
    """Read {cam}_{session_id}_corrected_cropped.crop.json; {x:0,y:0} if absent."""
    sidecar = prep_dir / f"{cam}_{session_id}_corrected_cropped.crop.json"
    if sidecar.exists():
        try:
            rect = json.loads(sidecar.read_text())
            return {"x": float(rect.get("x", 0)), "y": float(rect.get("y", 0))}
        except Exception:
            pass
    return {"x": 0.0, "y": 0.0}


def load_3d_h5(h5_path: Path):
    df = pd.read_hdf(str(h5_path))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()
    X = df[scorer].xs("x", level="coords", axis=1).values
    Y = df[scorer].xs("y", level="coords", axis=1).values
    Z = df[scorer].xs("z", level="coords", axis=1).values
    LL = df[scorer].xs("likelihood", level="coords", axis=1).values
    return bps, X, Y, Z, LL


def load_2d_h5(h5_path: Path):
    df = pd.read_hdf(str(h5_path))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()
    X = df[scorer].xs("x", level="coords", axis=1).values
    Y = df[scorer].xs("y", level="coords", axis=1).values
    LL = df[scorer].xs("likelihood", level="coords", axis=1).values
    return bps, X, Y, LL


def compute_turned_away_frame_mask(ll_full: np.ndarray, bps_full: list, fps: float) -> "np.ndarray | None":
    """
    Per-frame boolean mask, True where THIS camera's own 2D view judges the
    animal turned away from it -- reuses cube_core.detect_turned_away_bins
    (the validated single-2D-camera algorithm; see CLAUDE.md's 2026-09-09
    entry on why it's disabled for post-triangulation 3D combined
    likelihood but stays correct for a genuine single 2D camera view, which
    is exactly what each per-camera QC panel here is) on the FULL bodypart
    set from that camera's own H5 (not the MAIN_BODYPARTS QC subset -- the
    algorithm needs the real Head/Mouth region grouping), then upsamples
    the per-100ms-bin result back to per-frame via np.repeat, matching the
    same pattern BSoidEngine.run() uses for this exact upsampling.
    Returns None if the algorithm can't run (e.g. no 'nose' bodypart).
    """
    import cube_core
    try:
        bin_mask = cube_core.detect_turned_away_bins(
            ll_full, bps_full, fps,
            likelihood_thresh=0.3, adaptive_pct=10,
            head_frac_on=0.30, min_window_s=0.4, merge_gap_s=0.5)
    except Exception:
        return None
    if bin_mask.size == 0:
        return None
    win = max(1, int(round(fps / 10)))
    frame_mask = np.repeat(bin_mask, win)
    n = ll_full.shape[0]
    if frame_mask.shape[0] < n:
        frame_mask = np.concatenate([frame_mask, np.zeros(n - frame_mask.shape[0], dtype=bool)])
    else:
        frame_mask = frame_mask[:n]
    return frame_mask


def find_2d_h5(prep_dir: Path, cam: str, session_id: str) -> Path:
    results_dir = prep_dir / f"{cam}_{session_id}_corrected_cropped_results"
    filtered = results_dir / f"{cam}_{session_id}_corrected_cropped_filtered.h5"
    if filtered.exists():
        return filtered
    cands = sorted(results_dir.glob("*_filtered.h5"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No *_filtered.h5 in {results_dir}")


def compute_raw_pts3d(prep_dir: Path, calib_toml: Path, session_id: str,
                       cam_labels: list, common_bps: list,
                       ll_gate: float = 0.4, median_window: int = 3) -> np.ndarray:
    """
    RAW (pre-bone_constraints/pre-geometric-priors) triangulation, for QC
    comparison against the final, fully-refined 3D H5 -- rebuilds pts2d/
    scores from each camera's own refined 2D H5 with the same crop-offset
    convention _triangulate_session() uses, then calls
    cube_core.triangulate_cameras() with ONLY ll_gate/median_window (no
    RANSAC/confidence-weighting/bone_constraints/hard-bone-length-gate/
    displacement-clamp/nose-multipoint-prior/ray-temporal-fallback/hindpaw-
    floor-pin). This IS the same intermediate result _triangulate_session()
    itself computes internally before any of its own refinement passes --
    see cube_3d_dlc.py's own "Baseline (plain-method) triangulation" comment.

    Built 2026-09-17 after a real-data investigation found bone_constraints
    (not calibration quality) is what makes one camera's raw 2D detection
    look visually more trustworthy than the final reprojected 3D point for
    well-tracked bodyparts -- this lets a QC video show BOTH the raw
    geometric fit and the final refined point side by side, so a reviewer
    can tell "smoothing artifact" from "real triangulation disagreement" at
    a glance instead of having to re-run a standalone diagnostic.
    """
    import pandas as pd
    from cube_core import triangulate_cameras

    dfs = {}
    for cam in cam_labels:
        h5 = find_2d_h5(prep_dir, cam, session_id)
        df = pd.read_hdf(str(h5))
        scorer = df.columns.get_level_values(0)[0]
        dfs[cam] = (scorer, df)

    n_frames = min(len(df) for _, df in dfs.values())
    n_bp = len(common_bps)
    n_cams = len(cam_labels)
    pts2d = np.zeros((n_cams, n_frames, n_bp, 2), dtype=np.float64)
    scores = np.zeros((n_cams, n_frames, n_bp), dtype=np.float64)

    for ci, cam in enumerate(cam_labels):
        scorer, df = dfs[cam]
        rect = load_crop_rect(prep_dir, cam, session_id)
        for bi, bp in enumerate(common_bps):
            try:
                x = df[scorer, bp, "x"].values[:n_frames].astype(float)
                y = df[scorer, bp, "y"].values[:n_frames].astype(float)
                ll = df[scorer, bp, "likelihood"].values[:n_frames].astype(float)
            except Exception:
                x = y = ll = np.zeros(n_frames)
            pts2d[ci, :, bi, 0] = x + rect["x"]
            pts2d[ci, :, bi, 1] = y + rect["y"]
            scores[ci, :, bi] = ll

    return triangulate_cameras(pts2d, scores, calib_toml, ll_gate=ll_gate,
                                median_window=median_window, log_fn=lambda *a, **k: None)


def render_camera_overlay(cam: str, session_id: str, mp4_path: Path,
                           bps3d: list, XYZ3D: np.ndarray, LL3D: np.ndarray,
                           bps2d: list, X2D: np.ndarray, Y2D: np.ndarray, LL2D: np.ndarray,
                           cam_obj, x_off: float, y_off: float,
                           out_path: Path, ll_gate: float,
                           turned_away_mask: "np.ndarray | None" = None,
                           dot_radius: int = 4, ring_radius: int = 7,
                           XYZ3D_raw: "np.ndarray | None" = None,
                           raw_ring_radius: int = 12) -> None:
    """
    Draws, per main bodypart: a solid dot at the camera's own raw 2D
    detection, a hollow ring at the 3D point reprojected into this camera's
    pixel space, and a thin connecting line between them (visible only when
    both are confidently tracked that frame) -- the gap length IS the
    reprojection error for that bodypart/frame/camera.

    XYZ3D_raw (optional, same shape/bodypart-order as XYZ3D): when given,
    also draws a hollow SQUARE (in the bodypart's own color) at the RAW
    (pre-bone_constraints) triangulation's reprojection -- see
    compute_raw_pts3d(). Lets a reviewer see, per frame, whether the gap
    between dot and (final) ring is a real geometric triangulation
    disagreement (raw square ALSO far from the dot) or a smoothing
    artifact introduced by bone_constraints/the geometric-prior refinement
    stack (raw square close to the dot, final ring pulled away from both).
    """
    if out_path.exists():
        return

    present_bps = [bp for bp in MAIN_BODYPARTS if bp in bps3d and bp in bps2d]
    if not present_bps:
        raise RuntimeError(f"None of MAIN_BODYPARTS present in both H5s for {cam}")

    idx3d = {bp: bps3d.index(bp) for bp in present_bps}
    idx2d = {bp: bps2d.index(bp) for bp in present_bps}

    n_frames = min(XYZ3D.shape[0], X2D.shape[0])

    # Reproject every present bodypart's 3D trajectory into this camera's
    # pixel space in one batched call, then subtract the crop offset so it
    # lines up with the (cropped) video / raw 2D detections' own coordinate
    # frame -- mirrors _triangulate_session()'s ADD of this same offset when
    # building pts2d from the cropped H5s in the other direction.
    pts3d_flat = np.stack([XYZ3D[:n_frames, idx3d[bp], :] for bp in present_bps], axis=1)
    valid3d = ~np.any(np.isnan(pts3d_flat), axis=-1) & ~np.all(np.abs(pts3d_flat) < 1e-9, axis=-1)
    safe_pts = np.where(valid3d[..., None], pts3d_flat, 0.0).reshape(-1, 3)
    proj = cam_obj.project(safe_pts).reshape(n_frames, len(present_bps), 2)
    proj[..., 0] -= x_off
    proj[..., 1] -= y_off

    proj_raw = None
    valid_raw = None
    if XYZ3D_raw is not None:
        pts3d_raw_flat = np.stack([XYZ3D_raw[:n_frames, idx3d[bp], :] for bp in present_bps], axis=1)
        valid_raw = (~np.any(np.isnan(pts3d_raw_flat), axis=-1)
                     & ~np.all(np.abs(pts3d_raw_flat) < 1e-9, axis=-1))
        safe_raw_pts = np.where(valid_raw[..., None], pts3d_raw_flat, 0.0).reshape(-1, 3)
        proj_raw = cam_obj.project(safe_raw_pts).reshape(n_frames, len(present_bps), 2)
        proj_raw[..., 0] -= x_off
        proj_raw[..., 1] -= y_off

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(n_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    ffmpeg_exe = get_ffmpeg()
    encoders = [("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]),
                ("libx264",    ["-c:v", "libx264", "-threads", "0", "-crf", "20", "-preset", "fast"])]
    if not _nvenc_available(ffmpeg_exe):
        encoders = encoders[1:]

    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    last_err = None

    for enc_name, enc_args in encoders:
        cap = cv2.VideoCapture(str(mp4_path))
        cmd = [ffmpeg_exe, "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
               *enc_args, "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(tmp_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        stderr_sink: list = []
        t = threading.Thread(target=_drain_stream, args=(proc.stderr, stderr_sink), daemon=True)
        t.start()
        try:
            try:
                for k in range(n_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break

                    for bi, bp in enumerate(present_bps):
                        color = BGR_COLORS.get(bp, (255, 255, 255))

                        ll2 = LL2D[k, idx2d[bp]]
                        if ll2 >= ll_gate:
                            p2 = (int(round(X2D[k, idx2d[bp]])), int(round(Y2D[k, idx2d[bp]])))
                            cv2.circle(frame, p2, dot_radius, color, -1, cv2.LINE_AA)

                        ll3 = LL3D[k, idx3d[bp]]
                        if valid3d[k, bi] and ll3 > 0.0:
                            p3 = (int(round(proj[k, bi, 0])), int(round(proj[k, bi, 1])))
                            cv2.circle(frame, p3, ring_radius, color, 2, cv2.LINE_AA)

                            if ll2 >= ll_gate:
                                cv2.line(frame, p2, p3, color, 1, cv2.LINE_AA)

                        # RAW (pre-bone_constraints) reprojection -- a hollow
                        # SQUARE in the bodypart's own color (a thin neutral-
                        # gray ring proved nearly invisible against this
                        # rig's washed-out enclosure footage; a colored
                        # square is distinguished from the final position's
                        # round ring by SHAPE, not by a hard-to-see color).
                        # Square close to the dot -> smoothing moved the
                        # point away from real geometric agreement; square
                        # close to the round ring -> the gap to the dot is a
                        # genuine multi-view disagreement, not a smoothing
                        # artifact.
                        if proj_raw is not None and valid_raw[k, bi]:
                            pr = (int(round(proj_raw[k, bi, 0])), int(round(proj_raw[k, bi, 1])))
                            # Moderately larger than ring_radius (12 vs 7)
                            # so the square is distinguishable by shape when
                            # nearly concentric with the final ring, without
                            # dominating the frame the way a 20px square did.
                            cv2.rectangle(frame,
                                          (pr[0] - raw_ring_radius, pr[1] - raw_ring_radius),
                                          (pr[0] + raw_ring_radius, pr[1] + raw_ring_radius),
                                          color, 2, cv2.LINE_AA)

                    label = f"{cam}  frame {k}"
                    if proj_raw is not None:
                        label += "   dot=raw2D  ring=final3D  square=raw3D(pre-smoothing)"
                    # White-on-bright-background is illegible (a real problem
                    # on this rig's washed-out enclosure walls) -- black
                    # outline drawn first (thicker), white fill on top, the
                    # standard cv2 legible-caption technique.
                    cv2.putText(frame, label, (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, label, (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    if turned_away_mask is not None and k < len(turned_away_mask) and turned_away_mask[k]:
                        bar_h = max(24, h // 18)
                        cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 200), -1)
                        cv2.putText(frame, "TURNED AWAY", (8, h - bar_h // 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, bar_h / 40.0,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                    try:
                        proc.stdin.write(frame.tobytes())
                    except (BrokenPipeError, OSError) as e:
                        t.join(timeout=5)
                        err = stderr_sink[0].decode(errors="replace") if stderr_sink else str(e)
                        raise RuntimeError(f"FFmpeg exited early:\n{err}") from e

                proc.stdin.close()
                proc.wait(timeout=300)
            finally:
                cap.release()

            t.join(timeout=5)
            if proc.returncode != 0:
                err = stderr_sink[0].decode(errors="replace") if stderr_sink else ""
                raise RuntimeError(f"FFmpeg failed:\n{err}")
            tmp_path.replace(out_path)
            print(f"  [OK] {enc_name}: {out_path}")
            return
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            last_err = e
            if enc_name == "h264_nvenc":
                print(f"  NVENC failed for {cam}, retrying with libx264")

    raise RuntimeError(f"Overlay failed for {cam}: {last_err}")


def _probe_dims(ffmpeg_exe: str, path: Path) -> "tuple[int, int] | None":
    try:
        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (w, h) if w and h else None
    except Exception:
        return None


def make_quad_reprojection_video(cam_mp4s: dict, out_path: Path, ffmpeg_exe: str,
                                  border_px: int = 12,
                                  border_color: str = "white") -> None:
    """
    2x2 composite of the 4 per-camera reprojection-QC videos (cam0 TL, cam1
    TR, cam2 BL, cam3 BR -- same layout convention as cube_3d_dlc.py's own
    make_quad_video), each pane scaled to a common resolution, with a solid
    border frame drawn around the OUTSIDE of the whole composite (not
    between panes) so the quad grid reads as one bordered panel rather than
    floating edge-to-edge against the player background.
    """
    if out_path.exists():
        return

    sorted_cams = sorted(cam_mp4s.items())
    n = len(sorted_cams)
    if n < 2:
        raise ValueError(f"Need at least 2 cameras for quad video, got {n}")

    dims = [_probe_dims(ffmpeg_exe, p) for _, p in sorted_cams]
    valid_dims = [d for d in dims if d is not None]
    if valid_dims:
        from collections import Counter
        target_w = Counter(w for w, _ in valid_dims).most_common(1)[0][0]
        target_h = Counter(h for _, h in valid_dims).most_common(1)[0][0]
    else:
        target_w = target_h = None

    inputs: list[str] = []
    for _, p in sorted_cams:
        inputs += ["-i", str(p)]

    if target_w and target_h:
        scale_part = "".join(
            f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2[s{i}];"
            for i in range(n))
        ins = "".join(f"[s{i}]" for i in range(n))
    else:
        scale_part = ""
        ins = "".join(f"[{i}:v]" for i in range(n))

    if n == 4:
        xstack = f"{ins}xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[grid]"
    elif n == 2:
        xstack = f"{ins}hstack=inputs=2[grid]"
    else:
        xstack = f"{ins}xstack=inputs=3:layout=0_0|w0_0|0_h0[grid]"

    # Outer border only -- pad the finished grid on all four sides, not the
    # individual panes (that would put a border between quadrants too).
    border_part = (f"[grid]pad=iw+{2*border_px}:ih+{2*border_px}:"
                   f"{border_px}:{border_px}:color={border_color}[v]")

    filter_graph = scale_part + xstack + ";" + border_part

    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    base_cmd = [ffmpeg_exe, "-y"] + inputs + [
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    encoders = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]),
        ("libx264",    ["-c:v", "libx264", "-threads", "0", "-crf", "20", "-preset", "fast"]),
    ]
    for enc_name, enc_args in encoders:
        cmd = base_cmd + enc_args + [str(tmp_path)]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            tmp_path.replace(out_path)
            print(f"  [OK] quad {enc_name}: {out_path}")
            return
        if enc_name == "h264_nvenc":
            print("  NVENC unavailable for quad, retrying with libx264")
        else:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Quad composite failed:\n{result.stderr.decode(errors='replace')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", required=True, help="CUBE 0 - Corrected 3D vids folder (has {sid}_3d_filtered.h5)")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--calib", required=True, help="calibration.toml or calibration_refined.toml")
    ap.add_argument("--prep-dir", required=True, help="CUBE 1b - 3D DLC/_prep folder (per-camera cropped mp4/H5 + crop.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ll-gate", type=float, default=0.4)
    ap.add_argument("--cams", default="", help="comma-separated cam labels to restrict to (default: all found)")
    ap.add_argument("--show-raw", action="store_true",
                     help="Also draw the RAW (pre-bone_constraints) triangulation's reprojection "
                          "as a second gray ring, to distinguish real triangulation disagreement "
                          "from a bone_constraints/geometric-prior smoothing artifact.")
    args = ap.parse_args()

    from aniposelib.cameras import CameraGroup

    session_dir = Path(args.session_dir)
    prep_dir = Path(args.prep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = args.session_id

    h5_3d = session_dir / f"{sid}_3d_filtered.h5"
    bps3d, X3D, Y3D, Z3D, LL3D = load_3d_h5(h5_3d)
    XYZ3D = np.stack([X3D, Y3D, Z3D], axis=-1)
    print(f"Loaded 3D H5: {h5_3d} ({XYZ3D.shape[0]} frames, {len(bps3d)} bodyparts)")

    cgroup = CameraGroup.load(str(args.calib))
    cams_by_name = {c.get_name(): c for c in cgroup.cameras}

    wanted_cams = [c.strip() for c in args.cams.split(",") if c.strip()] or sorted(cams_by_name.keys())
    rendered: dict = {}

    XYZ3D_raw = None
    if args.show_raw:
        # Triangulation needs >=2 cameras -- always use EVERY camera the
        # calibration knows about here, never wanted_cams (the --cams
        # RENDER filter), which can legitimately be a single camera and
        # would otherwise silently produce all-degenerate (zero/NaN) raw
        # points, making every raw square invisible regardless of size.
        all_cams_for_raw = sorted(cams_by_name.keys())
        print(f"Computing RAW (pre-bone_constraints) triangulation for comparison "
              f"rings, using all {len(all_cams_for_raw)} calibrated cameras "
              f"{all_cams_for_raw} (independent of --cams render filter)...")
        pts3d_raw = compute_raw_pts3d(prep_dir, args.calib, sid, all_cams_for_raw, bps3d,
                                       ll_gate=args.ll_gate)
        XYZ3D_raw = pts3d_raw  # (n_frames, n_bp, 3), same bodypart order as bps3d

    for cam in wanted_cams:
        cam_obj = cams_by_name.get(cam)
        if cam_obj is None:
            print(f"  [skip] {cam}: not in calibration {list(cams_by_name)}")
            continue

        mp4_path = prep_dir / f"{cam}_{sid}_corrected_cropped.mp4"
        if not mp4_path.exists():
            print(f"  [skip] {cam}: no cropped video at {mp4_path}")
            continue

        try:
            h5_2d = find_2d_h5(prep_dir, cam, sid)
        except FileNotFoundError as e:
            print(f"  [skip] {cam}: {e}")
            continue

        bps2d, X2D, Y2D, LL2D = load_2d_h5(h5_2d)
        rect = load_crop_rect(prep_dir, cam, sid)

        cap = cv2.VideoCapture(str(mp4_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        ta_mask = compute_turned_away_frame_mask(LL2D, bps2d, fps)
        if ta_mask is not None:
            print(f"  {cam}: turned-away in {int(ta_mask.sum())}/{len(ta_mask)} frames "
                  f"({100.0 * ta_mask.mean():.1f}%)")

        out_path = out_dir / f"{cam}_{sid}_reprojection_qc.mp4"
        print(f"Rendering {cam} -> {out_path.name} (crop offset x={rect['x']}, y={rect['y']})")
        render_camera_overlay(
            cam, sid, mp4_path,
            bps3d, XYZ3D, LL3D,
            bps2d, X2D, Y2D, LL2D,
            cam_obj, rect["x"], rect["y"],
            out_path, args.ll_gate,
            turned_away_mask=ta_mask,
            XYZ3D_raw=XYZ3D_raw)
        rendered[cam] = out_path

    if len(rendered) >= 2:
        quad_path = out_dir / f"{sid}_reprojection_qc_quad.mp4"
        print(f"Building bordered quad composite -> {quad_path.name}")
        make_quad_reprojection_video(rendered, quad_path, get_ffmpeg())


if __name__ == "__main__":
    main()
