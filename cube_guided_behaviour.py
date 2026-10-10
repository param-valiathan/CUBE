"""Guided behaviours: classify the user's behaviours directly from their labels, then find sub-types inside each.

Why this exists (Xenium test, 2026-10-08): routing user labels through CUBE's clusters (refine -> classifier -> HMM) gave
held-out Rearing 66-73 % coverage and Guarding 0-19 %, while a classifier trained straight on the labels (A-SOiD style:
random forest on windowed pose features, class-balanced, labels kept for turned-away bins) reached 95 % for Rearing and
88 % for Licking inside turned-away stretches. So for the behaviours the user DEFINED the supervised result leads and
CUBE's unsupervised machinery works around it:

  1. per time bin: the user's behaviour (or "Other") from the guided learner (+ optional HMM smoothing);
  2. inside every behaviour: UMAP + HDBSCAN on that behaviour's bins -> sub-type clusters (A-SOiD's "directed
     unsupervised" step); a behaviour is therefore a GROUP of one or more clusters;
  3. every "Other" bin keeps the cluster CUBE found, so unsupervised discovery of new behaviours continues.

Everything is written in the format the Behaviour Analyser already reads (bout_lengths CSVs + cluster_behaviour_mapping.tsv,
several sub-type clusters sharing a behaviour_group), into a separate folder so it never mixes with the blind run's files.

Cluster ids in the guided folder: sub-types 0..S-1 first (behaviour by behaviour), then every CUBE cluster as S + its id.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cube_guided as cg

OTHER = "Other"
GUIDED_SOURCE = "guided_behaviour"
NON_BEHAVIOUR = ("", "Background", "Neither", "__neither__")


def stems_for(bout_dir, keys) -> dict:
    """Session key -> the file stem its CUBE bout/frame CSVs already use (falls back to the key)."""
    from cube_guided_session import session_stems
    out = {}
    bd = Path(bout_dir) if bout_dir else None
    for k in keys:
        cand = session_stems(k)
        hit = None
        if bd is not None:
            for st in cand:
                if (bd / f"{st}_frame_labels_hmm.csv").is_file() or (bd / f"{st}_frame_labels.csv").is_file():
                    hit = st
                    break
        out[k] = hit or cand[0]
    return out


# ------------------------------------------------------------------ assignment
def smooth_proba(proba, ranges, width=5):
    """Moving average of the class probabilities over ``width`` bins (0.5 s) inside each session. Robust: unlike an
    HMM fitted to probabilities of RARE classes it cannot collapse onto one class (observed on real data: the HMM
    track called Guarding on 42 % of bins where the raw argmax said 4 %)."""
    from scipy.ndimage import uniform_filter1d
    P = np.array(proba, dtype=float, copy=True)
    for _k, (s0, s1) in ranges.items():
        seg = P[s0:s1]
        if len(seg) >= width and np.isfinite(seg).all():
            P[s0:s1] = uniform_filter1d(seg, size=width, axis=0, mode="nearest")
    return P


def assign_behaviours(proba, classes, ranges, behaviours, smooth="proba", min_conf=0.5):
    """Per-bin (behaviour name or OTHER, confidence). ``classes`` = the learner's class names (may include
    Background); only names in ``behaviours`` can be assigned, everything else is OTHER. ``smooth``: "proba"
    (default, moving average), "hmm" (cg.smooth_guided_track - can collapse for rare classes) or False/None."""
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    if smooth is True:
        smooth = "proba"
    if smooth == "proba":
        proba = smooth_proba(proba, ranges)
    n = len(proba)
    ok = np.isfinite(proba).all(axis=1)
    idx = np.full(n, -1, dtype=int)
    if ok.any():
        idx[ok] = np.argmax(proba[ok], axis=1)
    conf = np.where(ok, np.nanmax(np.where(ok[:, None], proba, 0.0), axis=1), 0.0)
    if smooth == "hmm":
        try:
            idx = cg.smooth_guided_track(proba, ranges, classes)
        except Exception:                                           # noqa: BLE001 - fall back to the raw argmax
            pass
    names = np.array([classes[i] if 0 <= i < len(classes) else OTHER for i in idx], dtype=object)
    keep = np.isin(names, list(behaviours)) & (conf >= float(min_conf))
    return np.where(keep, names, OTHER), conf


# -------------------------------------------------------------------- sub-types
def find_subtypes(X, seed=0, min_cluster_frac=0.05, min_cluster=25, max_subtypes=6):
    """Cluster one behaviour's bins into sub-types. Returns (ids 0..k-1 sorted by size, k, info).
    PCA -> UMAP(3) -> HDBSCAN; noise joins the nearest sub-type (every bin of a behaviour has a sub-type)."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    n = len(X)
    info = {"n": int(n), "method": "umap+hdbscan"}
    if n < 2 * min_cluster:
        info["method"] = "single (too few bins)"
        return np.zeros(n, dtype=int), 1, info
    from sklearn.decomposition import PCA
    k = int(min(30, X.shape[1], n - 1))
    Z = PCA(n_components=k, random_state=seed).fit_transform(X) if X.shape[1] > k else X
    try:
        import umap
        Z = umap.UMAP(n_components=3, n_neighbors=int(min(30, max(2, n - 1))), min_dist=0.0, random_state=seed,
                      n_jobs=1).fit_transform(Z)
    except Exception:                                               # noqa: BLE001 - PCA space is a fine fallback
        info["method"] = "pca+hdbscan"
    import hdbscan
    mcs = int(max(min_cluster, round(min_cluster_frac * n)))
    lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit_predict(Z)
    ids = sorted({int(v) for v in lab if v >= 0}, key=lambda c: -int((lab == c).sum()))
    ids = ids[:max_subtypes]
    if len(ids) < 2:
        info["method"] += " -> one sub-type"
        return np.zeros(n, dtype=int), 1, info
    remap = {c: i for i, c in enumerate(ids)}
    out = np.array([remap.get(int(v), -1) for v in lab])
    if (out < 0).any():                                             # noise / dropped small clusters -> nearest sub-type
        from sklearn.neighbors import KNeighborsClassifier
        good = out >= 0
        knn = KNeighborsClassifier(n_neighbors=min(15, int(good.sum())), n_jobs=1).fit(Z[good], out[good])
        out[~good] = knn.predict(Z[~good])
    info["noise_frac"] = round(float((lab < 0).mean()), 3)
    return out, len(ids), info


def sustained_subtypes(rows, ids, min_bouts=3, min_median_bins=3):
    """True when every sub-type recurs as >= ``min_bouts`` temporal bouts whose median length is >=
    ``min_median_bins`` bins. ``rows`` = the (sorted) global bin indices of one behaviour, ``ids`` its sub-type per row.
    A split of structureless data (UMAP clumps) is scattered in time as 1-2-bin blips and fails this; a real
    sub-type (a posture the animal holds / repeats) passes. Bout recurrence, not silhouette, is the evidence."""
    rows = np.asarray(rows)
    ids = np.asarray(ids)
    brk = np.r_[True, (np.diff(rows) != 1) | (np.diff(ids) != 0)]
    start = np.flatnonzero(brk)
    lens = np.diff(np.r_[start, len(rows)])
    lab = ids[start]
    for k in np.unique(ids):
        ln = lens[lab == k]
        if len(ln) < min_bouts or float(np.median(ln)) < min_median_bins:
            return False
    return True


# ---------------------------------------------------------------------- writers
def _runs(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    ch = np.flatnonzero(np.diff(arr) != 0)
    st = np.concatenate(([0], ch + 1))
    en = np.concatenate((ch, [arr.size - 1]))
    return [(int(arr[s]), int(s), int(e - s + 1)) for s, e in zip(st, en)]


def _bin_to_frames(bin_vals, win, n_frames):
    f = np.repeat(np.asarray(bin_vals), win)
    if len(f) < n_frames:
        f = np.pad(f, (0, n_frames - len(f)), mode="edge")
    return f[:n_frames]


def build_guided_behaviours(feats, ranges, fps_map, ls, cluster_frame_labels, n_frames, out_dir, *,
                            stems=None, proba=None, classes=None, X_ctx=None, embedding=None, reserved_ids=(),
                            smooth="proba", min_conf=0.5, subtypes=True, min_subtype_bins=200, max_subtypes=6,
                            video_centring=True, cluster_gate=False, gate_strength=1.0, gate_prior=5.0,
                            seed=0, progress=None, log=None, source_run=None) -> dict:
    """See the module docstring. ``feats`` (n_bins, n_feat) standardised features of EVERY bin (turned-away bins
    included); ``cluster_frame_labels`` {session: per-frame CUBE cluster ids}; ``n_frames`` {session: frame count};
    ``reserved_ids`` CUBE ids that mean "turned away / low quality" (kept reserved so the analyser drops them)."""
    prog = progress or (lambda f, t=None: None)
    log = log or (lambda m: None)
    out = Path(out_dir)
    (out / "bout_lengths").mkdir(parents=True, exist_ok=True)
    res: dict = {"out_dir": str(out)}
    behaviours = [b["name"] for b in ls.behaviours if b["name"] not in NON_BEHAVIOUR]
    if len(behaviours) == 0:
        raise ValueError("no behaviours defined")

    if proba is None:
        prog(0.05, "Building context features for every time bin ...")
        if X_ctx is None:                 # built here: raw + per-video-centred (a caller-supplied X_ctx is used as is)
            X_ctx = cg.build_context_features(feats, ranges, fps_map)
            if video_centring:
                X_ctx = cg.hybrid_features(X_ctx, ranges)
        y, w = cg.labels_to_bins(ls, ranges, fps_map, len(feats))
        y = np.array([("" if v == "__neither__" else v) for v in y], dtype=object)
        prog(0.25, "Training the classifier on your labels ...")
        model = cg.train_guided(X_ctx, y, w)
        prog(0.45, "Scoring every time bin ...")
        proba, classes = model.predict_proba(X_ctx), list(model.classes_)
    gate_info = None
    if cluster_gate:
        # cluster-evidence gate (cube_guided_gate): the blind CUBE clusters veto a behaviour in clusters whose labels
        # say it is something else; opt-in, inert without labels
        import cube_guided_gate as _gg
        y_g, _w_g = cg.labels_to_bins(ls, ranges, fps_map, len(feats))
        y_g = np.array([("" if v == "__neither__" else v) for v in y_g], dtype=object)
        cl_bins = _gg.bins_from_frame_clusters(cluster_frame_labels, ranges, fps_map, len(feats))
        fac, gate_info = _gg.cluster_evidence_factors(y_g, cl_bins, behaviours, prior_strength=gate_prior,
                                                      ignore_clusters=reserved_ids)
        proba = _gg.apply_cluster_gate(proba, classes, behaviours, cl_bins, fac, gamma=gate_strength)
        gate_info = {**gate_info, "strength": float(gate_strength)}
        log(f"[guided] cluster gate on: gated {gate_info['gated']}, not gated (too few labels) {gate_info['skipped_rare']}")
    prog(0.55, "Smoothing the behaviour track over time ...")
    beh, conf = assign_behaviours(proba, classes, ranges, behaviours, smooth=smooth, min_conf=min_conf)
    present = [b for b in sorted(behaviours) if (beh == b).any()]

    # ---- sub-types inside each behaviour
    sub = np.full(len(beh), -1, dtype=int)
    n_sub: dict = {}
    sub_info: dict = {}
    Xf = np.nan_to_num(np.asarray(feats, dtype=np.float32))
    for bi, b in enumerate(present):
        prog(0.6 + 0.2 * bi / max(1, len(present)), f"Finding sub-types inside {b} ...")
        rows = np.flatnonzero(beh == b)
        if subtypes and len(rows) >= min_subtype_bins:
            ids, k, info = find_subtypes(Xf[rows], seed=seed, max_subtypes=max_subtypes)
            if k > 1 and not sustained_subtypes(rows, ids):
                info = {**info, "method": info["method"] + " -> rejected: sub-types not sustained in time"}
                ids, k = np.zeros(len(rows), dtype=int), 1
        else:
            ids, k, info = np.zeros(len(rows), dtype=int), 1, {"n": int(len(rows)), "method": "single"}
        sub[rows] = ids
        n_sub[b], sub_info[b] = k, info
    base = {}
    c = 0
    for b in present:
        base[b] = c
        c += n_sub[b]
    n_guided = c                                                    # ids 0..n_guided-1 are guided sub-types

    # ---- per-session frame labels + bout files
    prog(0.85, "Writing the Analyser files ...")
    stems = stems or {k: k for k in ranges}
    other_ids = set()
    track_rows = []
    unk_id = n_guided + int(max(list(cluster_frame_labels_max(cluster_frame_labels)) + [0])) + 1
    for key, (s0, s1) in ranges.items():
        win = max(1, int(round(float(fps_map.get(key, 30.0)) / 10.0)))
        nfr = int(n_frames.get(key, (s1 - s0) * win))
        cl = np.asarray(cluster_frame_labels.get(key, np.full(nfr, -1)), dtype=int)
        if len(cl) < nfr:
            cl = np.pad(cl, (0, nfr - len(cl)), constant_values=-1)
        cl = cl[:nfr]
        bb = beh[s0:s1]
        bid = np.array([base[b] + sub[s0 + i] if b != OTHER else -1 for i, b in enumerate(bb)], dtype=int)
        fb = _bin_to_frames(bid, win, nfr)
        new = np.where(fb >= 0, fb, np.where(cl >= 0, n_guided + cl, -1))
        new = np.where(new < 0, unk_id, new)           # frames CUBE itself left unlabelled share one extra id
        other_ids.update(int(v) for v in np.unique(new[fb < 0]))
        stem = stems.get(key, key)
        fps = float(fps_map.get(key, 30.0))
        pd.DataFrame({"frame": np.arange(nfr), "time_s": np.arange(nfr) / fps, "label": new, "raw_label": new,
                      "turned_away_reason": ""}).to_csv(out / "bout_lengths" / f"{stem}_frame_labels_hmm.csv",
                                                         index=False)
        rows_b = [(lab, st, ln) for lab, st, ln in _runs(new)]
        bdf = pd.DataFrame(rows_b, columns=["B-SOiD labels", "Start time (frames)", "Run lengths"])
        bdf.to_csv(out / "bout_lengths" / f"{stem}_bout_lengths_hmm.csv", index=False)
        bdf.to_csv(out / "bout_lengths" / f"{stem}_bout_lengths.csv", index=False)
        for i in range(s1 - s0):
            track_rows.append((key, i, bb[i], round(float(conf[s0 + i]), 4), int(bid[i])))
    pd.DataFrame(track_rows, columns=["session", "bin", "behaviour", "confidence", "guided_cluster_id"]).to_csv(
        out / "guided_behaviour_track.csv", index=False)

    # ---- mapping + groups for the Behaviour Analyser
    map_rows, groups = [], {}
    for b in present:
        for j in range(n_sub[b]):
            cid = base[b] + j
            groups.setdefault(b, []).append(cid)
            map_rows.append({"bsoid_group_id": cid, "behaviour_group": b, "status": "assigned", "example_count": 0,
                             "gif_path": "", "mp4_paths": "", "n_bins": int(((beh == b) & (sub == j)).sum()),
                             "clusters_in_behaviour": n_sub[b], "part": j + 1,
                             "subtype_name": f"{b} · sub-type {j + 1}", "origin_cluster": "",
                             "guided_source": GUIDED_SOURCE})
    for cid in sorted(other_ids):
        map_rows.append({"bsoid_group_id": cid, "behaviour_group": "", "status": "pending", "example_count": 0,
                         "gif_path": "", "mp4_paths": "", "n_bins": 0, "clusters_in_behaviour": 0, "part": 0,
                         "subtype_name": "", "origin_cluster": int(cid - n_guided), "guided_source": "cube_cluster"})
    pd.DataFrame(map_rows).to_csv(out / "cluster_behaviour_mapping.tsv", sep="\t", index=False)
    families = behaviour_families(ls)
    fam_json = {}
    if families:
        fam_rows = []
        for r in map_rows:
            r2 = dict(r)
            for fam in families:
                if r["behaviour_group"] in fam["members"]:
                    r2["behaviour_group"] = fam["name"]
            fam_rows.append(r2)
        pd.DataFrame(fam_rows).to_csv(out / "cluster_behaviour_mapping_families.tsv", sep="\t", index=False)
        fam_json = {f["name"]: {"members": f["members"], "adjacent_label_pairs": f["pairs"], "order": f["order"]}
                    for f in families}
        (out / "behaviour_families.json").write_text(json.dumps(fam_json, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    (out / "behaviour_groups.json").write_text(json.dumps(
        {"behaviour_groups": {b: {"clusters": cl, "n_clusters": len(cl)} for b, cl in groups.items()},
         "multi_cluster_behaviours": sorted(b for b, cl in groups.items() if len(cl) > 1),
         "source": "guided classifier on user labels + sub-type clustering", "labels_sha1": ls.sha1,
         "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2, ensure_ascii=False), encoding="utf-8")
    reserved_new = sorted(int(n_guided + int(r)) for r in reserved_ids)
    (out / "bsoid_run_summary.json").write_text(json.dumps(
        {"pipeline": "CUBE guided behaviours", "turned_away_cluster_id": (reserved_new[0] if reserved_new else None),
         "impure_cluster_display_id": (reserved_new[1] if len(reserved_new) > 1 else None),
         "source_run": str(source_run or ""), "n_guided_subtype_clusters": n_guided}, indent=2), encoding="utf-8")
    secs = {}
    pool_share = _cluster_pool_share(ls, ranges, fps_map, cluster_frame_labels, len(beh))
    for b in present:
        nb = int((beh == b).sum())
        share = 100.0 * nb / max(1, len(beh))
        ref = pool_share.get(b)
        secs[b] = {"bins": nb, "approx_seconds": round(nb * 0.1, 1), "share_of_all_bins_pct": round(share, 1),
                   "cluster_pool_share_pct": (round(ref, 1) if ref is not None else None),
                   "sub_types": n_sub[b], "sub_type_method": sub_info[b],
                   "mean_confidence": round(float(conf[beh == b].mean()), 3)}
        if ref is not None and ref > 0 and share > 2.0 * ref:
            secs[b]["warning"] = (f"called on {share:.0f}% of all bins but your labelled-pure clusters cover only "
                                  f"{ref:.0f}%: probably over-called. Label some 'Background' / 'Neither' clips of other "
                                  f"behaviours (and answer more questions) so it learns what is NOT {b}.")
            log(f"[guided] WARNING {b}: {secs[b]['warning']}")
    audit = {}
    try:
        import cube_guided_audit as _au
        audit = _au.audit_track(beh, ranges, fps_map, ls, present)
        for fl in audit.get("flags", []):
            log(f"[guided] AUDIT {fl}")
    except Exception as e:                                          # noqa: BLE001 - diagnostics only
        log(f"[guided] audit skipped: {e}")
    summary = {"behaviours": secs, "audit": audit, "n_other_bins": int((beh == OTHER).sum()), "n_bins": int(len(beh)),
               "classes": list(classes) if classes is not None else None, "smoothed": bool(smooth),
               "min_confidence": float(min_conf), "labels_sha1": ls.sha1, "n_guided_subtype_clusters": n_guided,
               "cluster_gate": gate_info}
    (out / "guided_behaviours_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["families"] = fam_json
    (out / "guided_behaviours_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    res.update(groups=groups, summary=summary, mapping=str(out / "cluster_behaviour_mapping.tsv"),
               families=fam_json)
    if embedding is not None:
        try:
            res["plot"] = _plot(embedding, beh, sub, base, present, out / "guided_behaviours_umap.png")
        except Exception as e:                                      # noqa: BLE001
            log(f"[guided] behaviour plot skipped: {e}")
    prog(1.0, "Done")
    return res


def _cluster_pool_share(ls, ranges, fps_map, cluster_frame_labels, n_bins):
    """% of all bins inside the clusters the labels prove (purity-gated) to be each behaviour: an independent estimate
    of how common the behaviour is, used to flag a classifier that calls it far too often."""
    try:
        import cube_guided_report as rep
        y, _w = cg.labels_to_bins(ls, ranges, fps_map, n_bins)
        truth = np.array([None if v in ("", "__neither__") else str(v) for v in y], dtype=object)
        cb = np.full(n_bins, -1, dtype=int)
        for key, (s0, s1) in ranges.items():
            cl = np.asarray(cluster_frame_labels.get(key, []), dtype=int)
            win = max(1, int(round(float(fps_map.get(key, 30.0)) / 10.0)))
            sub = cl[::win][: s1 - s0]
            cb[s0:s0 + len(sub)] = sub
        pools = rep.pools_by_purity(cb, truth, min_bins=3, min_purity=0.7)
        return {b: 100.0 * float(np.isin(cb, p).mean()) for b, p in pools.items() if p}
    except Exception:                                               # noqa: BLE001 - diagnostics only
        return {}


def behaviour_families(ls, gap_s=3.0, min_pairs=3, min_share=0.4):
    """Behaviours that your labels place right next to each other (<= gap_s, same video) at least ``min_pairs`` times
    form a FAMILY (and the pairs are >= 40 % of the rarer behaviour's labels) - e.g. Guarding then Licking. The Analyser can then work on the family as one unit
    (cluster_behaviour_mapping_families.tsv) while the per-behaviour mapping keeps them apart. A family is detected
    first and split second: the whole activity is far easier to find than its parts."""
    segs = sorted((l["session"], float(l["start_s"]), float(l["end_s"]), str(l["behaviour"])) for l in ls.labels
                  if str(l.get("behaviour", "")) not in NON_BEHAVIOUR and l.get("answer") != "neither")
    pairs: dict = {}
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if b[0] != a[0] or b[1] - a[2] > gap_s:
                if b[0] != a[0]:
                    break
                continue
            if a[3] != b[3] and b[1] >= a[2]:
                k = (a[3], b[3])
                pairs[k] = pairs.get(k, 0) + 1
    und: dict = {}
    for (a, b), c in pairs.items():
        key = tuple(sorted((a, b)))
        und[key] = und.get(key, 0) + c
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    n_lab: dict = {}
    for sg in segs:
        n_lab[sg[3]] = n_lab.get(sg[3], 0) + 1
    # adjacency must be a substantial share of the SMALLER behaviour's labels, or two behaviours that merely touch a
    # few times (e.g. Rearing now and then next to Licking) would be chained into one family
    strong = {k: c for k, c in und.items()
              if c >= min_pairs and c >= min_share * min(n_lab.get(k[0], 1), n_lab.get(k[1], 1))}
    for (a, b) in strong:
        parent[find(a)] = find(b)
    comps: dict = {}
    for x in list(parent):
        comps.setdefault(find(x), set()).add(x)
    out = []
    for members in comps.values():
        m = sorted(members)
        fwd = {a: sum(c for (p, q), c in pairs.items() if p == a and q in members) -
               sum(c for (p, q), c in pairs.items() if q == a and p in members) for a in m}
        order = sorted(m, key=lambda a: -fwd[a])                      # most 'precedes others' first
        out.append({"name": " + ".join(order) + " complex", "members": order, "order": order,
                    "pairs": int(sum(c for k, c in strong.items() if set(k) <= members))})
    return out


def cluster_frame_labels_max(cfl: dict):
    for v in cfl.values():
        if len(v):
            yield int(np.max(v))


def _plot(emb, beh, sub, base, present, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    emb = np.asarray(emb, dtype=float)
    cols = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for a in ax:
        a.scatter(emb[:, 0], emb[:, 1], s=1, color="#bdbdbd", alpha=0.25, linewidths=0, rasterized=True)
        a.set_xticks([])
        a.set_yticks([])
    for i, b in enumerate(present):
        m = beh == b
        ax[0].scatter(emb[m, 0], emb[m, 1], s=3, color=cols(i % 10), alpha=0.7, linewidths=0, label=b, rasterized=True)
        ks = sorted(set(sub[m]))
        for j in ks:
            mm = m & (sub == j)
            shade = 0.45 + 0.55 * (j / max(1, len(ks) - 1)) if len(ks) > 1 else 1.0
            ax[1].scatter(emb[mm, 0], emb[mm, 1], s=3, color=cols(i % 10), alpha=0.25 + 0.6 * shade, linewidths=0,
                          label=f"{b} sub-type {j + 1}", rasterized=True)
    ax[0].set_title("Your behaviours (guided classifier)", fontweight="bold")
    ax[1].set_title("Sub-type clusters inside each behaviour", fontweight="bold")
    for a in ax:
        a.legend(markerscale=4, fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
