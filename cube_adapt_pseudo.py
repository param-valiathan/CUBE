"""
Smart Adapt v4 -- pooled, filtered pseudo-labels (the "Z1" recipe).

Standard Smart Adapt fine-tunes the SuperAnimal weights on ONE representative video using every confident prediction as a label.
This module builds a better pseudo-label set, unsupervised (no hand labels), from several videos of the SAME camera:

  A1  precise-flag filter : a pseudo-label keypoint is dropped if the base model's track is implausible there --
                            out-and-back spike, Hampel outlier, optical-flow odd-one-out (the label does not follow the pixels
                            between neighbouring frames) or a left/right paw swap -- each with +-1 frame margin.
  A2  pose-balanced frames: all videos pooled, frames clustered by pose (k-means, 20 clusters) and sampled evenly per cluster, so
                            the training set is not dominated by the animal's most common posture.
  A3  per-body-part gates : each of the 24 mouse body parts is kept above its OWN threshold, clip(p40 of its confidence, 0.25, 0.6),
                            instead of one global 0.4. The 15 SuperAnimal keypoints with no mouse anatomy keep 0.4.

Output mirrors deeplabcut.utils.pseudo_label's COCO layout (<root>/pseudo_<stem>/annotations + images) so DeepLabCut's video-adapt
training uses it instead of building its own. Pure numpy/pandas/cv2/scipy/sklearn; no DeepLabCut import here.
Validated in the standalone testbed D:\\CUBE\\ADAPT_LABEL_TESTS (see DLC_ADAPTATION_IMPROVEMENT_PLAN.md).
"""
from __future__ import annotations
import os, json, glob
import numpy as np
import pandas as pd

# SuperAnimal-Quadruped keypoints with no mouse anatomy (same list as SUPERANIMAL_QUADRUPED_EXCLUDED_BODYPARTS in cube_3d_dlc.py)
EXCLUDED = {
    "mouth_end_right", "mouth_end_left", "right_antler_base", "right_antler_end", "left_antler_base", "left_antler_end",
    "throat_base", "throat_end", "tail_end", "belly_bottom", "body_middle_right", "body_middle_left", "neck_end", "back_base", "back_middle",
}
INFO_BPS = ["nose", "upper_jaw", "lower_jaw", "left_eye", "right_eye", "neck_base", "back_end", "tail_base",
            "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"]
LL_GATE = 0.4
FLOW_PX_FRAC = 0.07       # optical-flow residual limit as a fraction of the median torso length (30 px on a 430 px torso)


def schema_supported(bodyparts) -> bool:
    """The recipe needs the SuperAnimal-Quadruped bodypart names; other schemas fall back to standard Smart Adapt."""
    s = set(bodyparts)
    return all(b in s for b in INFO_BPS) and "left_earbase" in s


# ----------------------------------------------------------------------------- loading
def find_prediction_files(dest_dir, video_path):
    """(json, h5) written by video_inference_superanimal for `video_path` inside dest_dir (None if missing)."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    js = [p for p in glob.glob(os.path.join(dest_dir, stem + "*.json"))]
    js_b = [p for p in js if p.endswith("_before_adapt.json")]
    h5 = [p for p in glob.glob(os.path.join(dest_dir, stem + "*.h5")) if "_labeled" not in p and not p.endswith("_filtered.h5")]
    return (js_b or js or [None])[0], (h5 or [None])[0]


def load_prediction(video_path, h5_path, json_path):
    df = pd.read_hdf(str(h5_path))
    lv = list(df.columns.names); bl, cl = lv.index("bodyparts"), lv.index("coords")
    names39 = list(dict.fromkeys(df.columns.get_level_values(bl)))
    bps = [b for b in names39 if b not in EXCLUDED]
    cols = df.columns

    def get(bp, coord):
        m = (cols.get_level_values(bl) == bp) & (cols.get_level_values(cl) == coord)
        return df.loc[:, m].iloc[:, 0].to_numpy(float)
    P = dict(bps=bps, X=np.stack([get(b, "x") for b in bps], 1), Y=np.stack([get(b, "y") for b in bps], 1),
             L=np.stack([get(b, "likelihood") for b in bps], 1), bi={b: i for i, b in enumerate(bps)}, ll_gate=LL_GATE)
    P["torso"] = torso_len(P)
    P["torso_med"] = float(np.nanmedian(P["torso"]))
    return dict(video=video_path, key=os.path.splitext(os.path.basename(video_path))[0], P=P, names39=names39,
                json=json.load(open(json_path)))


def torso_len(P):
    a, b = P["bi"]["neck_base"], P["bi"]["back_end"]
    ok = (P["L"][:, a] >= LL_GATE) & (P["L"][:, b] >= LL_GATE)
    d = np.hypot(P["X"][:, a] - P["X"][:, b], P["Y"][:, a] - P["Y"][:, b]); d[~ok] = np.nan
    return d


# ----------------------------------------------------------------------------- A1 flags
def _series(P, k):
    ok = (P["L"][:, k] >= LL_GATE) & ~np.isnan(P["X"][:, k])
    return P["X"][:, k], P["Y"][:, k], ok


def spike_flags(P, ratio=3.0):
    T, K = P["X"].shape; out = np.zeros((T, K), bool)
    for k in range(K):
        x, y, ok = _series(P, k)
        v = ok[1:-1] & ok[:-2] & ok[2:]
        if v.sum() < 50:
            continue
        a = np.hypot(x[1:-1] - x[:-2], y[1:-1] - y[:-2]); b = np.hypot(x[1:-1] - x[2:], y[1:-1] - y[2:]); c = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2])
        floor = max(np.median(np.minimum(a, b)[v]) * 6.0, 3.0)
        out[1:-1][v & (np.minimum(a, b) / (c + 1e-3) > ratio) & (np.minimum(a, b) > floor), k] = True
    return out


def hampel_flags(P, win=7, z=8.0):
    T, K = P["X"].shape; out = np.zeros((T, K), bool); half = win // 2
    for k in range(K):
        x, y, ok = _series(P, k)
        xs, ys = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
        sx = pd.Series(xs).rolling(win, center=True, min_periods=half + 1).median().to_numpy()
        sy = pd.Series(ys).rolling(win, center=True, min_periods=half + 1).median().to_numpy()
        dev = np.hypot(xs - sx, ys - sy); v = ok & ~np.isnan(dev)
        if v.sum() < 50:
            continue
        med = np.median(dev[v]); mad = 1.4826 * np.median(np.abs(dev[v] - med)) + 1e-6
        out[v & (dev > med + z * mad), k] = True
    return out


def swap_flags(P, A, B, ratio=0.6, floor_frac=0.12):
    a, b = P["bi"][A], P["bi"][B]; T = P["X"].shape[0]
    ok = (P["L"][:, a] >= LL_GATE) & (P["L"][:, b] >= LL_GATE); ok = ok[1:] & ok[:-1]
    d = lambda i, j, s, e: np.hypot(P["X"][s, i] - P["X"][e, j], P["Y"][s, i] - P["Y"][e, j])
    keep = d(a, a, slice(1, None), slice(None, -1)) + d(b, b, slice(1, None), slice(None, -1))
    swp = d(a, b, slice(1, None), slice(None, -1)) + d(b, a, slice(1, None), slice(None, -1))
    o = np.zeros(T, bool); o[1:] = ok & (swp < ratio * keep) & (keep > floor_frac * P["torso_med"])
    return o


def flow_odd_flags(P, video_path, cache=None):
    """Lucas-Kanade: the pixel under the label at t-1 is tracked into t (and the label at t back into t-1). A label that disagrees with
    BOTH tracks by more than FLOW_PX_FRAC x torso is the odd one out. Only the 12 informative body parts are checked."""
    import cv2
    T, K = P["X"].shape; kidx = [P["bi"][b] for b in INFO_BPS]
    if cache and os.path.exists(cache):
        d = np.load(cache); fwd, back = d["fwd"], d["back"]
    else:
        fwd = np.full((T, K), np.nan); back = np.full((T, K), np.nan)
        conf = (P["L"] >= LL_GATE) & ~np.isnan(P["X"])
        cap = cv2.VideoCapture(video_path); ok, fr = cap.read()
        if not ok:
            return np.zeros((T, K), bool)
        prev = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        lk = dict(winSize=(41, 41), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        for t in range(1, T):
            ok, fr = cap.read()
            if not ok:
                break
            cur = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ks = [k for k in kidx if conf[t - 1, k] and conf[t, k]]
            if ks:
                p0 = np.array([[P["X"][t - 1, k], P["Y"][t - 1, k]] for k in ks], np.float32).reshape(-1, 1, 2)
                q0 = np.array([[P["X"][t, k], P["Y"][t, k]] for k in ks], np.float32).reshape(-1, 1, 2)
                p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
                p0r, s2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
                pb, s3, _ = cv2.calcOpticalFlowPyrLK(cur, prev, q0, None, **lk)
                fb = np.linalg.norm((p0r - p0).reshape(-1, 2), axis=1)
                for j, k in enumerate(ks):
                    if s1[j, 0] and s2[j, 0] and fb[j] < 3.0:
                        fwd[t, k] = np.hypot(p1[j, 0, 0] - P["X"][t, k], p1[j, 0, 1] - P["Y"][t, k])
                    if s3[j, 0]:
                        back[t - 1, k] = np.hypot(pb[j, 0, 0] - P["X"][t - 1, k], pb[j, 0, 1] - P["Y"][t - 1, k])
            prev = cur
        cap.release()
        if cache:
            np.savez(cache, fwd=fwd, back=back)
    lim = FLOW_PX_FRAC * P["torso_med"]
    odd = (np.nan_to_num(fwd, nan=-1) > lim) & (np.nan_to_num(back, nan=-1) > lim)
    mask = np.zeros((T, K), bool); mask[:, kidx] = True
    return odd & mask


def a1_flags(pred, cache_dir=None, margin=1):
    """(T,K24) bool: True = zero this keypoint in the pseudo-label."""
    from scipy.ndimage import maximum_filter1d
    P = pred["P"]; T, K = P["X"].shape
    cache = os.path.join(cache_dir, f"flow_{pred['key']}.npz") if cache_dir else None
    # spike / Hampel / swap run over all 24 mouse body parts; optical flow over the 12 informative ones (as validated in the testbed)
    U = spike_flags(P) | hampel_flags(P) | flow_odd_flags(P, pred["video"], cache)
    for A, B in [("front_left_paw", "front_right_paw"), ("back_left_paw", "back_right_paw")]:
        f = swap_flags(P, A, B); U[:, P["bi"][A]] |= f; U[:, P["bi"][B]] |= f
    if margin > 0:
        U = maximum_filter1d(U.astype(np.uint8), size=2 * margin + 1, axis=0, mode="constant").astype(bool)
    return U


# ----------------------------------------------------------------------------- A3 thresholds, A2 selection
def per_bp_thresholds(preds, lo=0.25, hi=0.6, default=0.4):
    names39 = preds[0]["names39"]; mouse = set(preds[0]["P"]["bps"])
    conf = np.concatenate([np.array([np.array(f["bodyparts"][0], float)[:, 2] for f in p["json"]]) for p in preds], 0)
    thr = np.full(len(names39), default)
    for k, n in enumerate(names39):
        if n in mouse:
            thr[k] = float(np.clip(np.percentile(conf[:, k], 40), lo, hi))
    return thr


def select_frames(preds, budget=360, k_clusters=20, stride=3, min_conf_bps=18, bbox_thr=0.75, seed=0, frame_masks=None):
    """frame_masks: optional {video key: bool array} of ALLOWED frames (e.g. False on the duplicated 'fill' frames of a corrected video)."""
    from sklearn.cluster import KMeans
    rows, feats = [], []
    for p in preds:
        P = p["P"]; bi = P["bi"]; T = P["X"].shape[0]
        allowed = (frame_masks or {}).get(p["key"])
        for t in range(0, min(T, len(p["json"])), stride):
            if allowed is not None and (t >= len(allowed) or not allowed[t]):
                continue
            if p["json"][t]["bbox_scores"][0] < bbox_thr:
                continue
            conf = P["L"][t] >= LL_GATE
            if conf.sum() < min_conf_bps:
                continue
            c = np.array([P["X"][t, bi["neck_base"]] + P["X"][t, bi["back_end"]], P["Y"][t, bi["neck_base"]] + P["Y"][t, bi["back_end"]]]) / 2.0
            sc = P["torso"][t] if not np.isnan(P["torso"][t]) else P["torso_med"]
            v = np.concatenate([(P["X"][t] - c[0]) / sc, (P["Y"][t] - c[1]) / sc]); v[np.concatenate([~conf, ~conf])] = np.nan
            rows.append((p["key"], t)); feats.append(v)
    if len(rows) < 10:
        return [], dict(candidates=len(rows), selected=0)
    F = np.array(feats); colmean = np.nanmean(F, 0); idx = np.where(np.isnan(F)); F[idx] = np.take(colmean, idx[1])
    k_clusters = max(2, min(k_clusters, len(rows) // 5))
    km = KMeans(n_clusters=k_clusters, n_init=3, random_state=seed).fit(F)
    rng = np.random.default_rng(seed)
    clusters = [list(np.where(km.labels_ == c)[0]) for c in range(k_clusters)]
    for c in clusters:
        rng.shuffle(c)
    quota = budget // k_clusters; chosen = []
    for c in clusters:
        chosen += c[:quota]
    pool = [i for c in clusters for i in c[quota:]]; rng.shuffle(pool); chosen += pool[:max(0, budget - len(chosen))]
    sel = sorted(set(rows[i] for i in chosen))
    return sel, dict(candidates=len(rows), selected=len(sel), per_video={k: sum(1 for x in sel if x[0] == k) for k in {r[0] for r in rows}})


# ----------------------------------------------------------------------------- COCO writer
def write_coco(out_dir, selected, preds_by_key, thr_vec, flags, bbox_thr=0.75):
    import cv2
    img_dir = os.path.join(out_dir, "images"); ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(ann_dir, exist_ok=True)
    images, annotations, caps = [], [], {}
    st = dict(kp_total=0, kp_below_thr=0, kp_flagged=0, kp_kept=0, frames=0)
    names39 = next(iter(preds_by_key.values()))["names39"]
    for key, t in selected:
        pr = preds_by_key[key]; f = pr["json"][t]
        pose = np.array(f["bodyparts"][0], float).copy(); bbox = np.array(f["bboxes"][0], float)
        if float(f["bbox_scores"][0]) < bbox_thr:
            continue
        st["kp_total"] += len(pose)
        zero = pose[:, 2] < thr_vec; st["kp_below_thr"] += int(zero.sum())
        for name, k24 in pr["P"]["bi"].items():
            k39 = names39.index(name)
            if flags[key][t, k24] and not zero[k39]:
                zero[k39] = True; st["kp_flagged"] += 1
        pose[zero, :2] = 0.0; st["kp_kept"] += int((~zero).sum()); pose[:, 2] = 2
        if key not in caps:
            caps[key] = cv2.VideoCapture(pr["video"])
        caps[key].set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, fr = caps[key].read()
        if not ok:
            continue
        fn = f"{key.replace(' ', '')}_f{t:05d}.png"
        cv2.imwrite(os.path.join(img_dir, fn), fr)
        h, w = fr.shape[:2]
        images.append(dict(id=len(images), file_name=fn, width=w, height=h))
        annotations.append(dict(category_id=1, keypoints=[float(v) for v in pose.reshape(-1)], num_keypoints=len(pose), image_id=len(images) - 1,
                                bbox=[float(v) for v in bbox[:4]], area=float(bbox[2] * bbox[3]), iscrowd=0, id=len(annotations)))
        st["frames"] += 1
    for c in caps.values():
        c.release()
    cats = [dict(name="quadruped", id=1, supercategory="animal", keypoints=list(names39))]
    json.dump(dict(images=images[:10], annotations=annotations[:10], categories=cats), open(os.path.join(ann_dir, "test.json"), "w"), indent=1)
    json.dump(dict(images=images, annotations=annotations, categories=cats), open(os.path.join(ann_dir, "train.json"), "w"), indent=1)
    return st


def make_seed_clip(path, src, n=30):
    import cv2
    if os.path.exists(path):
        return
    cap = cv2.VideoCapture(src); w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
    vw.release(); cap.release()


def load_fill_mask(csv_path, n_frames):
    """Allowed-frame mask for a corrected video: False where the timestamps CSV marks a duplicated 'fill' frame (4th column).
    A missing/unreadable CSV allows every frame; the length is reconciled to n_frames (pad allowed / truncate)."""
    allowed = np.ones(int(n_frames), bool)
    try:
        data = np.genfromtxt(str(csv_path), delimiter=",", skip_header=1)
        fill = np.atleast_2d(data)[:, 3].astype(bool)
    except Exception:
        return allowed
    n = min(len(fill), len(allowed))
    allowed[:n] = ~fill[:n]
    return allowed


def build_pooled_pseudo_dataset(base_predictions, root, bbox_thr=0.75, budget=360, log=print, frame_masks=None):
    """
    base_predictions: list of (video_path, h5_path, json_path) from base-model inference on same-camera videos.
    frame_masks: optional {video_path: bool array of allowed frames}.
    Writes <root>/seed.mp4 and <root>/pseudo_seed/{annotations,images} (DeepLabCut then skips building its own).
    Returns a stats dict, or None when the recipe does not apply (unsupported schema / too few usable frames).
    """
    keyed_masks = {os.path.splitext(os.path.basename(v))[0]: m for v, m in (frame_masks or {}).items()}
    os.makedirs(root, exist_ok=True)
    preds = [load_prediction(v, h, j) for v, h, j in base_predictions if h and j]
    if not preds or not schema_supported(preds[0]["P"]["bps"]):
        log("  pooled pseudo-labels: bodypart schema is not SuperAnimal-Quadruped - using standard adaptation")
        return None
    cache_dir = os.path.join(root, "_cache"); os.makedirs(cache_dir, exist_ok=True)
    log(f"  A1: scoring {len(preds)} video(s) for implausible base-model labels (spike / Hampel / optical flow / paw swap)")
    flags = {p["key"]: a1_flags(p, cache_dir) for p in preds}
    thr = per_bp_thresholds(preds)
    sel, info = select_frames(preds, budget=budget, bbox_thr=bbox_thr, frame_masks=keyed_masks)
    if len(sel) < 20:
        log(f"  pooled pseudo-labels: only {len(sel)} usable frames - using standard adaptation")
        return None
    st = write_coco(os.path.join(root, "pseudo_seed"), sel, {p["key"]: p for p in preds}, thr, flags, bbox_thr=bbox_thr)
    make_seed_clip(os.path.join(root, "seed.mp4"), preds[0]["video"])
    m12 = [preds[0]["P"]["bi"][b] for b in INFO_BPS]
    info.update(st); info["flag_rate"] = {k: round(float(v[:, m12].mean()), 4) for k, v in flags.items()}
    info["thresholds"] = {n: round(float(t), 3) for n, t in zip(preds[0]["names39"], thr)}
    json.dump(info, open(os.path.join(root, "pooled_pseudo_report.json"), "w"), indent=1)
    log(f"  A2/A3: {info['selected']} frames from {len(preds)} video(s); {st['kp_kept']}/{st['kp_total']} keypoints kept "
        f"({st['kp_flagged']} removed by the plausibility filter, {st['kp_below_thr']} below their per-body-part threshold)")
    return info


# ----------------------------------------------------------------------------- orchestration (shared by 2D Smart Adapt and the 3D per-camera adaptation)
def _video_info(path):
    import cv2
    c = cv2.VideoCapture(str(path))
    size = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return size, n


def prediction_cache_valid(video_path, h5_path, json_path):
    """A cached base-model prediction is reused only if its frame count matches the video (a stale or truncated file is recomputed)."""
    try:
        if not (h5_path and json_path and os.path.isfile(h5_path) and os.path.isfile(json_path)):
            return False
        _, n = _video_info(video_path)
        return abs(len(pd.read_hdf(str(h5_path))) - n) <= 1
    except Exception:
        return False


def run_pooled_adaptation(dlc, videos, root, *, sa_name, model_name, detector_name, bbox_thr, bbox_thr_adapt, max_ind, inf_batch, det_batch,
                          det_epochs, pose_epochs, pseudo_thr, transfer, logger, frame_masks=None, pcutoff_base=0.1):
    """
    Pooled, filtered adaptation for ONE camera. `videos` are same-camera videos with the preferred one first; a video whose frame size
    differs from the first is dropped. Base-model predictions are cached under <root>/base and reused (any bbox_thr_adapt change only
    redoes the frame selection, never the inference). Returns the folder holding the adapted checkpoints (<root>), or None when the recipe
    does not apply or fails -- the caller then runs its standard single-video adaptation. Never raises.
    """
    import traceback
    try:
        root = os.fspath(root); os.makedirs(root, exist_ok=True)
        videos = [os.fspath(v) for v in videos if os.path.isfile(os.fspath(v))]
        if not videos:
            return None
        size0, _ = _video_info(videos[0]); pool = [videos[0]]
        for v in videos[1:]:
            if _video_info(v)[0] == size0:
                pool.append(v)
            else:
                logger.warn(f"  Pooled adaptation: skipping {os.path.basename(v)} (frame size differs from {os.path.basename(videos[0])})")
        logger.info(f"  Pooled adaptation: {len(pool)} same-camera video(s) ({size0[0]}x{size0[1]}) -> {[os.path.basename(v) for v in pool]}")
        base_dir = os.path.join(root, "base"); os.makedirs(base_dir, exist_ok=True)
        preds = []
        for vp in pool:
            js, h5 = find_prediction_files(base_dir, vp)
            if not prediction_cache_valid(vp, h5, js):
                logger.info(f"  Base-model inference on {os.path.basename(vp)} (labels to be filtered) ...")
                dlc.video_inference_superanimal(
                    [vp], superanimal_name=sa_name, model_name=model_name, detector_name=detector_name, video_adapt=False,
                    pcutoff=pcutoff_base, bbox_threshold=bbox_thr, max_individuals=max_ind, batch_size=inf_batch, detector_batch_size=det_batch,
                    dest_folder=base_dir, create_labeled_video=False, plot_bboxes=False, device="auto")
                js, h5 = find_prediction_files(base_dir, vp)
            preds.append((vp, h5, js))
        info = build_pooled_pseudo_dataset(preds, root, bbox_thr=bbox_thr_adapt, log=logger.info, frame_masks=frame_masks)
        if info is None:
            return None
        sa_kw = dict(superanimal_name=sa_name, model_name=model_name, detector_name=detector_name, video_adapt=True, pcutoff=pcutoff_base,
                     pseudo_threshold=pseudo_thr, bbox_threshold=bbox_thr_adapt, max_individuals=max_ind, batch_size=inf_batch,
                     detector_batch_size=det_batch, detector_epochs=det_epochs, pose_epochs=pose_epochs, dest_folder=os.path.join(root, "adapt_out"),
                     create_labeled_video=False, plot_bboxes=False, device="auto")
        try:
            import inspect
            sig = inspect.signature(dlc.video_inference_superanimal).parameters
            if "superanimal_transfer_learning" in sig:
                sa_kw["superanimal_transfer_learning"] = transfer
            if "video_adapt_batch_size" in sig:
                sa_kw["video_adapt_batch_size"] = inf_batch
        except Exception:
            pass
        logger.info(f"  Adapting on {info['selected']} pooled, filtered pseudo-label frames ({pose_epochs} pose / {det_epochs} detector epochs) ...")
        dlc.video_inference_superanimal([os.path.join(root, "seed.mp4")], **sa_kw)
        logger.success("  Pooled adaptation complete.")
        from pathlib import Path
        return Path(root)
    except Exception as exc:
        logger.warn(f"  Pooled adaptation unavailable ({exc}); falling back to standard single-video adaptation.")
        logger.info(traceback.format_exc())
        return None
