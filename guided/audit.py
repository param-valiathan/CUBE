"""Does what the classifier finds in UNLABELLED videos look like what the user labelled?

The user's labels are sparse, so the usual accuracy numbers only see the labelled bins and are blind to the main way a
classifier fails: over-calling a behaviour on the (large) unlabelled part. But the labels also carry the behaviour's
SIGNATURE - how long a bout lasts, how often it recurs, what it follows - and that can be checked everywhere, with no
extra labelling:

  * bout length   - median predicted bout vs the median labelled bout (a classifier that flickers produces 0.1-0.2 s
                    "bouts" of a behaviour the user marked as lasting ~1 s; one that merges produces minutes);
  * flicker       - share of predicted bouts shorter than 0.3 s;
  * recurrence    - bouts per minute in videos the user did not label vs videos they did;
  * sequence      - for behaviours that your labels place next to each other (Guarding then Licking), how often the
                    prediction has the same successor.

``audit_track`` returns plain numbers + human-readable flags; it is cheap enough to use as a SELECTION criterion (choose
the confidence threshold / smoothing / sub-type merge whose unlabelled-video bouts best match the labelled ones).
"""
from __future__ import annotations

import numpy as np


def _runs(vals):
    vals = np.asarray(vals, dtype=object)
    if len(vals) == 0:
        return []
    ch = np.flatnonzero(vals[1:] != vals[:-1])
    st = np.concatenate(([0], ch + 1))
    en = np.concatenate((ch, [len(vals) - 1]))
    return [(vals[s], int(s), int(e - s + 1)) for s, e in zip(st, en)]


def labelled_bout_stats(ls, behaviours, merge_gap_s=0.25):
    """{behaviour: {median_s, n, videos:set}} from the user's labels (adjacent same-behaviour segments merged)."""
    by: dict = {}
    for l in ls.labels:
        b = str(l.get("behaviour", ""))
        # active-learning clips are fixed ~1 s windows: their length says nothing about how long the behaviour lasts
        if b in behaviours and l.get("answer") != "neither" and l.get("source") != "query":
            by.setdefault((l["session"], b), []).append((float(l["start_s"]), float(l["end_s"])))
    out: dict = {}
    for (sess, b), segs in by.items():
        segs.sort()
        merged = [list(segs[0])]
        for a, z in segs[1:]:
            if a - merged[-1][1] <= merge_gap_s:
                merged[-1][1] = max(merged[-1][1], z)
            else:
                merged.append([a, z])
        d = out.setdefault(b, {"durs": [], "videos": set()})
        d["durs"] += [z - a for a, z in merged]
        d["videos"].add(sess)
    return {b: {"median_s": float(np.median(v["durs"])), "n": len(v["durs"]), "videos": v["videos"]}
            for b, v in out.items()}


def audit_track(beh, ranges, fps_map, ls, behaviours, bin_s=0.1, min_bouts=5, flicker_s=0.3):
    """``beh`` per-bin behaviour name (or 'Other') over all bins; ``ranges`` {session: (s0, s1)}.
    Returns {"behaviours": {b: {...}}, "flags": [str], "sequence": {...}}."""
    lab = labelled_bout_stats(ls, set(behaviours))
    res: dict = {"behaviours": {}, "flags": [], "sequence": {}}
    track = {k: np.asarray(beh[s0:s1], dtype=object) for k, (s0, s1) in ranges.items()}
    for b in behaviours:
        lv = lab.get(b, {}).get("videos", set())
        unl_runs, lab_runs, unl_min, lab_min = [], [], 0.0, 0.0
        for k, seq in track.items():
            bs = bin_s
            dur_min = len(seq) * bs / 60.0
            rr = [ln * bs for v, _s, ln in _runs(seq) if v == b]
            if k in lv:
                lab_runs += rr
                lab_min += dur_min
            else:
                unl_runs += rr
                unl_min += dur_min
        row = {"labelled_median_s": (round(lab[b]["median_s"], 2) if b in lab else None),
               "labelled_bouts": lab.get(b, {}).get("n", 0)}
        if len(unl_runs) >= 1:
            arr = np.array(unl_runs)
            row.update(unlabelled_videos=int(len([k for k in track if k not in lv])),
                       predicted_bouts_unlabelled=int(len(arr)),
                       predicted_median_s_unlabelled=round(float(np.median(arr)), 2),
                       bouts_per_min_unlabelled=round(len(arr) / max(unl_min, 1e-6), 2),
                       flicker_frac_unlabelled=round(float((arr < flicker_s).mean()), 2),
                       share_of_time_unlabelled_pct=round(100 * arr.sum() / max(unl_min * 60.0, 1e-6), 1))
        if len(lab_runs) >= 1:
            arr = np.array(lab_runs)
            row.update(predicted_median_s_labelled_videos=round(float(np.median(arr)), 2),
                       bouts_per_min_labelled_videos=round(len(arr) / max(lab_min, 1e-6), 2),
                       share_of_time_labelled_pct=round(100 * arr.sum() / max(lab_min * 60.0, 1e-6), 1))
        res["behaviours"][b] = row
        if b in lab and lab[b]["n"] >= min_bouts and "predicted_median_s_unlabelled" in row:
            r = row["predicted_median_s_unlabelled"] / max(lab[b]["median_s"], 1e-6)
            if r < 0.5:
                res["flags"].append(f"{b}: predicted bouts in unlabelled videos are {r:.1f}x as long as your labelled "
                                    f"bouts - fragmented (flickering). Raise the smoothing / confidence.")
            elif r > 2.5:
                res["flags"].append(f"{b}: predicted bouts are {r:.1f}x as long as your labelled bouts - merged "
                                    f"together or over-called.")
        if row.get("flicker_frac_unlabelled", 0) > 0.4:
            res["flags"].append(f"{b}: {100 * row['flicker_frac_unlabelled']:.0f}% of predicted bouts in unlabelled "
                                f"videos are shorter than {flicker_s}s - mostly noise.")
        if "share_of_time_unlabelled_pct" in row and "share_of_time_labelled_pct" in row:
            if row["share_of_time_unlabelled_pct"] > 3 * max(row["share_of_time_labelled_pct"], 1.0):
                res["flags"].append(f"{b}: takes {row['share_of_time_unlabelled_pct']:.0f}% of the time in videos you "
                                    f"did not label vs {row['share_of_time_labelled_pct']:.0f}% in the ones you did: "
                                    f"likely over-called there.")
    # sequence: what follows each behaviour within 3 s (predicted, all videos)
    succ: dict = {}
    for seq in track.values():
        rr = _runs(seq)
        for i, (v, s, ln) in enumerate(rr):
            if v not in behaviours:
                continue
            end = s + ln
            for v2, s2, _ln2 in rr[i + 1:]:
                if (s2 - end) * bin_s > 3.0:
                    break
                if v2 in behaviours and v2 != v:
                    succ.setdefault(v, {}).setdefault(v2, 0)
                    succ[v][v2] += 1
                    break
    res["sequence"] = {a: {b: int(c) for b, c in d.items()} for a, d in succ.items()}
    return res


def score_match(audit: dict) -> float:
    """One number (lower = the unlabelled-video bouts look more like the labelled ones): mean |log ratio| of median
    bout length plus the flicker share. Usable to pick between candidate settings without extra labelling."""
    vals = []
    for b, r in audit["behaviours"].items():
        if r.get("labelled_median_s") and "predicted_median_s_unlabelled" in r:
            vals.append(abs(np.log(max(r["predicted_median_s_unlabelled"], 0.05) / r["labelled_median_s"]))
                        + r.get("flicker_frac_unlabelled", 0.0))
    return float(np.mean(vals)) if vals else float("nan")


def complete_video_report(X, y, w, sess_of_bin, complete_sessions, behaviours, min_conf=0.5, n_estimators=150):
    """TRUE coverage and leakage, over EVERY bin of the fully labelled videos (unmarked bins are real negatives).
    Leave-one-video-out among the videos that have labels: the scored video is never in training. This is the
    measurement sparse labels cannot give: how often a behaviour is called where it is NOT. Needs >= 1 complete video
    (>= 2 for the model to have other labelled videos to learn from). Returns {behaviour: {...}} or {} if impossible."""
    from guided import core as cg
    y = np.asarray(y, dtype=object)
    sess = np.asarray(sess_of_bin, dtype=object)
    out: dict = {}
    labelled_videos = sorted({s for s, v in zip(sess, y) if v not in ("", cg.NEITHER_MARK)})
    for s in complete_sessions:
        tr = (sess != s) & np.isin(sess, labelled_videos)
        if len({v for v in y[tr] if v in behaviours}) < 2:
            continue
        yy = np.where(tr, y, "")
        m = cg.train_guided(X, yy, np.where(tr, np.asarray(w, dtype=float), 0.0), n_estimators=n_estimators)
        rows = np.flatnonzero((sess == s) & (y != "") & (y != cg.NEITHER_MARK))     # only the fully labelled stretch
        if len(rows) == 0:
            continue
        P = m.predict_proba(X[rows])
        ok = np.isfinite(P).all(axis=1)
        cls = list(m.classes_)
        arg = np.array(cls, dtype=object)[np.nanargmax(np.where(ok[:, None], P, -1), axis=1)]
        conf = np.where(ok, P.max(axis=1), 0.0)
        pred = np.where((conf >= min_conf) & np.isin(arg, list(behaviours)), arg, "Other")
        truth = y[rows]
        for b in behaviours:
            d = out.setdefault(b, {"true_bins": 0, "called_bins": 0, "hit_bins": 0, "videos": []})
            tb, cb = truth == b, pred == b
            d["true_bins"] += int(tb.sum())
            d["called_bins"] += int(cb.sum())
            d["hit_bins"] += int((tb & cb).sum())
            d["videos"].append(s)
    for b, d in out.items():
        d["coverage_pct"] = round(100 * d["hit_bins"] / d["true_bins"], 1) if d["true_bins"] else None
        d["leakage_pct"] = (round(100 * (d["called_bins"] - d["hit_bins"]) / d["called_bins"], 1)
                            if d["called_bins"] else None)
    return out
