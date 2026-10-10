"""Cluster-evidence gate: a second, independent opinion on the guided classifier's calls.

The guided classifier looks at one time bin at a time and always names one of YOUR behaviours. The blind CUBE clusters
are an independent view: if the training labels say a cluster is mostly something else (another of your behaviours, or
"none of my behaviours"), a bin in it should need much stronger classifier evidence before it counts as behaviour B.

For behaviour B and cluster C (training labels only, never the video being scored):

    e_B(C) = (n_B(C) + a * pi_B) / (n_lab(C) + a)             pi_B = share of B among all labelled bins
    f_B(C) = min(1, e_B(C) / pi_B) ** gamma                    in (0, 1]

n_B = labelled bins of B in C, n_lab = labelled bins of ANY behaviour or of "Background" (fully labelled negatives) in C,
a = prior strength (shrinks clusters with few labels towards "no information"). A cluster with no labels gets f = 1
(no veto); a cluster with many labelled bins and none of B gets f -> 0. The classifier's probability for B is multiplied
by f_B(C) and the usual confidence floor is applied afterwards, so the gate acts as a soft veto that needs no hand-made
allow-list and copes with a behaviour that lives in several clusters.

Behaviours with fewer than ``min_labelled`` labelled bins are not gated (too little evidence to veto anything).
Pure numpy; inert unless explicitly switched on.
"""
from __future__ import annotations

import numpy as np

NEGATIVES = ("Background",)


def cluster_evidence_factors(y, clusters, behaviours, negatives=NEGATIVES, prior_strength=5.0, min_labelled=30,
                             ignore_clusters=(), target_share=None):
    """-> ({behaviour: {cluster_id: factor}}, info). ``y`` per-bin label ("" = unlabelled), ``clusters`` per-bin
    cluster id (<0 = none). Only bins with a behaviour/negative label are used.
    ``target_share`` (default None = the behaviour's overall label share pi_B): a cluster is only trusted fully when its
    labelled share of B reaches max(pi_B, target_share); e.g. 0.5 = 'majority B'. Stricter, vetoes more."""
    y = np.asarray(y, dtype=object)
    cl = np.asarray(clusters).astype(int)
    labelled = np.isin(y, list(behaviours) + list(negatives)) & (cl >= 0)
    if ignore_clusters:
        labelled &= ~np.isin(cl, list(ignore_clusters))
    n_lab_total = int(labelled.sum())
    out: dict = {}
    info: dict = {"n_labelled": n_lab_total, "gated": [], "skipped_rare": [], "prior_strength": float(prior_strength)}
    if n_lab_total == 0:
        return {b: {} for b in behaviours}, info
    ids = np.unique(cl[labelled])
    n_all = {int(c): int((labelled & (cl == c)).sum()) for c in ids}
    for b in behaviours:
        nb_total = int((labelled & (y == b)).sum())
        if nb_total < min_labelled:
            out[b] = {}
            info["skipped_rare"].append(b)
            continue
        pi = nb_total / n_lab_total
        f = {}
        for c in ids:
            nb = int((labelled & (cl == c) & (y == b)).sum())
            e = (nb + prior_strength * pi) / (n_all[int(c)] + prior_strength)
            f[int(c)] = float(min(1.0, e / max(pi, float(target_share or 0.0))))
        out[b] = f
        info["gated"].append(b)
    return out, info


def apply_cluster_gate(proba, classes, behaviours, clusters, factors, gamma=1.0):
    """Multiply each behaviour's probability by its cluster factor ** gamma. Bins with no cluster (< 0) or a cluster
    that carries no labels are untouched. Returns a new array (not renormalised: a lowered probability simply fails
    the confidence floor later)."""
    P = np.array(proba, dtype=float, copy=True)
    cl = np.asarray(clusters).astype(int)
    classes = list(classes)
    for b in behaviours:
        if b not in classes or not factors.get(b):
            continue
        j = classes.index(b)
        fb = factors[b]
        lut_ids = np.array(sorted(fb))
        lut = np.array([fb[c] for c in lut_ids])
        pos = np.searchsorted(lut_ids, cl)
        pos = np.clip(pos, 0, len(lut_ids) - 1)
        hit = (cl >= 0) & (lut_ids[pos] == cl)
        fac = np.where(hit, lut[pos], 1.0) ** float(gamma)
        P[:, j] = P[:, j] * fac
    return P


def bins_from_frame_clusters(cluster_frame_labels, ranges, fps_map, n_bins):
    """Per-bin cluster id from {session: per-frame ids} (the label at the first frame of each 100 ms bin)."""
    out = np.full(n_bins, -1, dtype=int)
    for key, (s0, s1) in ranges.items():
        fl = cluster_frame_labels.get(key)
        if fl is None:
            continue
        fl = np.asarray(fl).astype(int)
        win = max(1, int(round(float(fps_map.get(key, 30.0)) / 10.0)))
        f = np.arange(s1 - s0) * win
        ok = f < len(fl)
        seg = out[s0:s1]
        seg[ok] = fl[f[ok]]
    return out
