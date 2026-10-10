"""Poorly-tracked / turned-away bins: recluster them among themselves, guided by the user's labels.

A user can name a behaviour when the animal is turned away or partly hidden far better than the code can. Those
bins often share too little POSTURE with the well-tracked examples of the same behaviour to belong in the same
cluster, so forcing them together dilutes both. Instead:

  1. ``poor_tracking_mask`` flags poorly tracked bins (originally turned-away, or in the lowest
     ``quantile`` of per-bin mean likelihood).
  2. ``recluster_poor`` clusters those bins ALONE (PCA -> UMAP -> HDBSCAN on the engine's own training
     features). A sub-cluster becomes a real cluster only when the user's labels PROVE it pure: enough
     labelled bins (the same evidence rule as the rest of the guided engine) and >= ``pure_thresh`` of them one
     behaviour. Its members leave whatever cluster / noise they were in. A behaviour may therefore be made of
     several clusters (well-tracked ones and poorly-tracked ones) as long as each is pure.
  3. Everything else is untouched; a parent cluster that is already user-pure for the same behaviour keeps its bins.

Conservative by construction: only moves bins into clusters the labels vouch for, never merges, never forces.
Inert without labels (the engine does not call this) and when the poor stratum or its labels are too small.
"""
from __future__ import annotations

import numpy as np


def poor_tracking_mask(all_ll: list, nbins_per_session: list, win_per_session: list, orig_turned_away: np.ndarray,
                       quantile: float = 0.2) -> np.ndarray:
    """Global per-bin boolean: originally turned-away OR mean likelihood in the lowest ``quantile`` of all bins."""
    n = int(sum(nbins_per_session))
    q = np.full(n, np.nan)
    off = 0
    for ll, nb, win in zip(all_ll, nbins_per_session, win_per_session):
        try:
            if ll is not None and getattr(ll, "size", 0):
                fm = np.nanmean(np.asarray(ll, dtype=float), axis=1)
                need = nb * int(win)
                if fm.size < need:
                    fm = np.pad(fm, (0, need - fm.size), mode="edge")
                q[off:off + nb] = fm[:need].reshape(nb, int(win)).mean(axis=1)
        except Exception:                                           # noqa: BLE001 - quality is best-effort
            pass
        off += nb
    poor = np.asarray(orig_turned_away, dtype=bool).copy()
    if poor.shape[0] != n:
        poor = np.zeros(n, dtype=bool)
    ok = np.isfinite(q)
    if ok.sum() > 50 and 0.0 < quantile < 1.0:
        poor |= ok & (q <= np.nanquantile(q[ok], quantile))
    return poor


def recluster_poor(labels: np.ndarray, feats_rows: np.ndarray, poor_rows: np.ndarray, ctx, *, min_poor: int = 150,
                   seed: int = 42, log=None):
    """Returns (new_labels, info). ``labels`` (n_rows,) cluster ids (-1 noise), ``feats_rows`` (n_rows, n_feat),
    ``poor_rows`` (n_rows,) bool, ``ctx`` the engine's GuidedContext (rows aligned with ``labels``)."""
    log = log or (lambda m: None)
    labels = np.asarray(labels).astype(int)
    info = {"ran": False, "reason": "", "subclusters": [], "moved": 0}
    n_rows = len(labels)
    if (ctx is None or not ctx.aligned(labels) or feats_rows.shape[0] != n_rows
            or np.asarray(poor_rows).shape[0] != n_rows):
        info["reason"] = "not aligned"
        return labels, info
    pidx = np.flatnonzero(np.asarray(poor_rows, dtype=bool))
    wfull = np.zeros(n_rows)
    wfull[ctx.lab_idx] = ctx.lab_w
    n_lab_poor = int((wfull[pidx] > 0).sum())
    if pidx.size < min_poor or n_lab_poor < 2 * ctx.min_bins:
        info["reason"] = f"poor stratum too small ({pidx.size} bins, {n_lab_poor} labelled)"
        return labels, info
    try:
        import hdbscan
        from sklearn.decomposition import PCA
        X = np.nan_to_num(np.asarray(feats_rows[pidx], dtype=np.float32))
        k = int(min(20, X.shape[1], X.shape[0] - 1))
        Z = PCA(n_components=k, random_state=seed).fit_transform(X) if X.shape[1] > k else X
        try:
            import umap
            Z = umap.UMAP(n_components=3, n_neighbors=int(min(15, max(2, len(pidx) - 1))), min_dist=0.0,
                          random_state=seed, n_jobs=1).fit_transform(Z)
        except Exception:                                          # noqa: BLE001 - PCA space is a fine fallback
            pass
        mcs = int(max(10, round(0.02 * len(pidx))))
        sub = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit_predict(Z)
    except Exception as e:                                         # noqa: BLE001
        info["reason"] = f"sub-clustering failed ({e})"
        return labels, info
    info["ran"] = True
    out = labels.copy()
    next_id = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    dom = ctx.dominant(labels)
    pure_of = {c: b for c, b in dom.items() if ctx.is_user_pure(labels, c)}      # computed once, not per bin
    for s in sorted(set(int(v) for v in sub if v >= 0)):
        mem = pidx[sub == s]
        w = {}
        for r in mem:
            if wfull[r] > 0:
                w[ctx.rows[r]] = w.get(ctx.rows[r], 0.0) + float(wfull[r])
        tot = float(sum(w.values()))
        rec = {"size": int(mem.size), "labelled_weight": round(tot, 2),
               "behaviours": {k: round(v, 2) for k, v in sorted(w.items())}}
        if tot <= 0:
            rec["verdict"] = "no_labels_left_as_is"
        else:
            b, bw = max(w.items(), key=lambda kv: kv[1])
            pur = bw / tot
            need = float(ctx._threshold(mem.size))
            if bw >= need and pur >= ctx.pure_thresh:
                keep = np.ones(mem.size, dtype=bool)
                for j, r in enumerate(mem):                         # parent already user-pure for b: leave it
                    if pure_of.get(int(labels[r])) == b:
                        keep[j] = False
                mv = mem[keep]
                if mv.size:
                    out[mv] = next_id
                    rec.update(verdict="accepted", behaviour=b, new_id=next_id, moved=int(mv.size),
                               purity=round(pur, 3))
                    info["moved"] += int(mv.size)
                    ctx.note("poor_tracking_cluster", cluster=next_id, behaviour=b, size=int(mv.size),
                             purity=round(pur, 3))
                    next_id += 1
                else:
                    rec["verdict"] = "parents_already_pure"
            else:
                rec["verdict"] = f"not_pure_enough(purity {pur:.2f}, evidence {bw:.1f}/{need:.0f})"
        info["subclusters"].append(rec)
    if info["moved"]:
        ids = np.unique(out[out >= 0])
        remap = {int(o): i for i, o in enumerate(ids)}
        out = np.array([remap[int(v)] if v >= 0 else -1 for v in out], dtype=int)
    acc = sum(1 for r in info["subclusters"] if r.get("verdict") == "accepted")
    log(f"  [GUIDED] poor-tracking recluster: {pidx.size} poorly tracked bins ({n_lab_poor} labelled) -> "
        f"{len(info['subclusters'])} sub-cluster(s), {acc} accepted as pure, {info['moved']} bin(s) moved")
    return out, info
