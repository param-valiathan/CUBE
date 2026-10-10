"""Guided labelling core (no Tk, no cube_core import at module level).

Implements GUIDED_LABELLING_CONTRACT.md section 2: label IO, time<->bin
alignment, the random-forest learner, query scoring/selection, cluster votes,
HMM smoothing and exports.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
BACKGROUND = "Background"
NEITHER = "Neither"
NEITHER_MARK = "__neither__"
_NON_BEHAVIOUR = ("", NEITHER_MARK, BACKGROUND)


# ----------------------------------------------------------------- label IO
@dataclass
class LabelSet:
    behaviours: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    path: str | None = None
    sha1: str = ""
    # videos the user declared FULLY labelled: every bin they did not mark is "none of my behaviours" (a real negative)
    complete_sessions: list = field(default_factory=list)
    # {session: [[start_s, end_s], ...]}: only THESE stretches of a video are fully labelled (e.g. its first minute)
    complete_spans: dict = field(default_factory=dict)


def _digest(behaviours, labels, complete=None, spans=None) -> str:
    payload = {"b": behaviours, "l": labels}
    if complete:                                   # only when used, so older digests / files are unchanged
        payload["c"] = sorted(complete)
    if spans:
        payload["cs"] = {k: [list(map(float, s)) for s in v] for k, v in sorted(spans.items())}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def load_labels(path) -> LabelSet:
    """Missing file -> empty LabelSet; corrupt / bad schema -> ValueError."""
    p = str(path)
    if not os.path.exists(p):
        return LabelSet([], [], p, _digest([], []))
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise ValueError(f"Cannot read labels file {p}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Labels file {p} is not a JSON object")
    ver = data.get("schema_version")
    if ver != SCHEMA_VERSION:
        raise ValueError(f"Unsupported labels schema_version {ver!r} "
                         f"(expected {SCHEMA_VERSION})")
    behaviours = data.get("behaviours", [])
    labels = data.get("labels", [])
    if not isinstance(behaviours, list) or not isinstance(labels, list):
        raise ValueError(f"Labels file {p}: 'behaviours'/'labels' must be lists")
    for i, lab in enumerate(labels):
        try:
            ok = (isinstance(lab, dict) and "session" in lab and "behaviour" in lab
                  and float(lab["start_s"]) < float(lab["end_s"]))
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"Labels file {p}: invalid label entry #{i}")
    complete = [str(s) for s in data.get("complete_sessions", []) if isinstance(s, str)]
    spans = {}
    for k, v in (data.get("complete_spans", {}) or {}).items():
        try:
            spans[str(k)] = [[float(a), float(b)] for a, b in v if float(b) > float(a)]
        except (TypeError, ValueError):
            continue
    return LabelSet(behaviours, labels, p, _digest(behaviours, labels, complete, spans), complete, spans)


def save_labels(ls: LabelSet, path) -> None:
    """Atomic write (.tmp + os.replace)."""
    p = str(path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION,
               "behaviours": ls.behaviours, "labels": ls.labels}
    if ls.complete_sessions:
        payload["complete_sessions"] = sorted(ls.complete_sessions)
    if ls.complete_spans:
        payload["complete_spans"] = {k: v for k, v in sorted(ls.complete_spans.items()) if v}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, p)
    ls.path = p
    ls.sha1 = _digest(ls.behaviours, ls.labels, ls.complete_sessions, ls.complete_spans)


# ------------------------------------------------------------ time -> bins
def _is_neither(lab: dict) -> bool:
    ans = lab.get("answer")
    return (str(lab.get("behaviour", "")).strip().lower() == NEITHER.lower()
            or (isinstance(ans, str) and ans.strip().lower() == "neither"))


def labels_to_bins(ls: LabelSet, session_bin_ranges: dict, fps,
                   n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, w). y is a str array ("" = unlabelled, "__neither__" for
    Neither answers); w is float (1.0, 0.5 on the two edge bins, 0 where
    unlabelled / neither). Later entries win on overlap. Unknown sessions are
    skipped; labels are clipped to the session's bin range."""
    y = np.full(n_bins, "", dtype=object)
    w = np.zeros(n_bins, dtype=float)
    for lab in ls.labels:
        sess = lab.get("session")
        rng = session_bin_ranges.get(sess)
        if rng is None:
            continue
        s0, s1 = int(rng[0]), int(rng[1])
        f = fps.get(sess) if isinstance(fps, dict) else fps
        f = float(f) if f else 0.0
        if f <= 0:
            continue
        win = max(1, int(round(f / 10.0)))
        t0, t1 = float(lab["start_s"]), float(lab["end_s"])
        if t1 <= t0:
            continue
        # 1e-6 guards float error: a time that IS a frame boundary (k*win/f) must not spill into the next bin
        f_first = int(math.floor(t0 * f + 1e-6))
        f_last = max(f_first, int(math.ceil(t1 * f - 1e-6)) - 1)   # end exclusive
        b0 = s0 + f_first // win
        b1 = s0 + f_last // win
        lo = max(b0, s0, 0)
        hi = min(b1, s1 - 1, n_bins - 1)
        if hi < lo:
            continue
        mark = NEITHER_MARK if _is_neither(lab) else str(lab["behaviour"])
        for b in range(lo, hi + 1):
            y[b] = mark
            w[b] = 0.0 if mark == NEITHER_MARK else 1.0
        if mark != NEITHER_MARK:
            if lo == b0:
                w[lo] = 0.5
            if hi == b1:
                w[hi] = 0.5
    # fully labelled video / stretch: everything NOT marked (incl. 'Neither' answers) is a real negative
    spans = {s: [[0.0, float("inf")]] for s in (getattr(ls, "complete_sessions", []) or [])}
    for s, v in (getattr(ls, "complete_spans", {}) or {}).items():
        spans.setdefault(s, []).extend(v)
    for sess, sp in spans.items():
        rng = session_bin_ranges.get(sess)
        if rng is None:
            continue
        s0, s1 = max(int(rng[0]), 0), min(int(rng[1]), n_bins)
        f = fps.get(sess) if isinstance(fps, dict) else fps
        f = float(f) if f else 0.0
        if f <= 0:
            continue
        win = max(1, int(round(f / 10.0)))
        for a, z in sp:
            lo = s0 + int(math.floor(float(a) * f + 1e-6)) // win
            hi = s1 - 1 if not np.isfinite(z) else min(s1 - 1, s0 + (int(math.ceil(float(z) * f - 1e-6)) - 1) // win)
            if hi < lo:
                continue
            seg = y[lo:hi + 1]
            free = np.array([(v == "" or v == NEITHER_MARK) for v in seg])
            seg[free] = BACKGROUND
            y[lo:hi + 1] = seg
            w[lo:hi + 1] = np.where(free, 1.0, w[lo:hi + 1])
    return y, w


# --------------------------------------------------------------- learner
class GuidedModel:
    def __init__(self, rf, classes):
        self._rf = rf
        self.classes_ = list(classes)

    def predict_proba(self, feats) -> np.ndarray:
        X = np.asarray(feats, dtype=np.float32)               # no float64 copy of the whole matrix
        good = np.flatnonzero(np.isfinite(X).all(axis=1))
        out = np.full((len(X), len(self.classes_)), np.nan)
        for a in range(0, len(good), 8192):                   # chunked: no full X[ok] copy either
            g = good[a:a + 8192]
            out[g] = self._rf.predict_proba(X[g])
        return out


def oversample_idx(y, cap: float = 4.0, seed: int = 0, skip=("Background",)) -> np.ndarray:
    """Row indices (into ``y``) with rare classes randomly duplicated towards the largest class.

    Each class smaller than the largest gets extra rows, drawn with replacement from its own rows, up to
    the size of the largest class but never more than ``(cap - 1)`` x its own count. The default
    (OVERSAMPLE_PARITY) is full parity; a finite cap such as 4.0 limits how far a tiny class is inflated.
    Originals are always kept. Evidence: EPM held-out macro-F1 +0.03..0.04 (80-92% of paired folds); OFT neutral
    and noisy on its own (shipped 4x cap missed the no-harm rule by 0.002 at 200 clips, parity lost a pool in
    1 of 8 splits) -- the default is the combination validated in wave 7, see build_context_features.
    Layered on class_weight="balanced_subsample"."""
    y = np.asarray(y, dtype=object)
    idx = [np.arange(len(y))]
    if len(y) == 0 or cap is None or cap <= 1.0:
        return idx[0]
    vals, cnts = np.unique(y.astype(str), return_counts=True)
    behav = np.array([str(v) not in skip for v in vals])
    # parity is towards the largest BEHAVIOUR; a (possibly huge) Background / "none of them" class is neither the
    # target size nor duplicated itself - with a fully labelled video it can be 50x bigger than a rare behaviour
    tgt = cnts[behav].max() if behav.any() else cnts.max()
    rng = np.random.default_rng(seed)
    ys = y.astype(str)
    for v, k in zip(vals, cnts):
        if str(v) in skip:
            continue
        if k < tgt:
            pool = np.flatnonzero(ys == v)
            idx.append(rng.choice(pool, int(min(tgt - k, (cap - 1.0) * k)), replace=True))
    return np.concatenate(idx)


OVERSAMPLE_PARITY = 1e9      # "no cap": every class is duplicated up to the largest class


def rf_n_jobs() -> int:
    """Forest parallelism: a quarter of the logical cores, at most 8. Measured: no speed gain above ~8 cores for
    these forest sizes, and this machine is documented as unstable under sustained all-core load (the project
    caps parallel dispatch at about half the cores)."""
    import os
    return max(1, min(8, (os.cpu_count() or 4) // 4))


def train_guided(feats, y, w=None, n_estimators=200, seed=0, oversample: bool = True,
                 oversample_cap: float = OVERSAMPLE_PARITY, class_weight="balanced_subsample") -> GuidedModel:
    """RandomForest(balanced_subsample) on labelled, finite rows; ``oversample=False`` = plain RF
    (kill switch / pre-wave-5 behaviour)."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(y, dtype=object)
    if len(y) != len(X):
        raise ValueError("feats and y length mismatch")
    keep = np.array([str(v) not in ("", NEITHER_MARK, "None", "nan") for v in y])
    if w is not None:
        w = np.asarray(w, dtype=float)
        keep &= w > 0
    rows = np.flatnonzero(keep)
    rows = rows[np.isfinite(X[rows]).all(axis=1)]            # finite check on the labelled rows only
    classes = sorted({str(v) for v in y[rows]})
    if len(classes) < 2:
        raise ValueError("need >=2 behaviours")
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                class_weight=class_weight,
                                random_state=seed, n_jobs=rf_n_jobs())
    Xk = X[rows]
    yk = np.array([str(v) for v in y[rows]])
    wk = None if w is None else w[rows]
    if oversample:
        ii = oversample_idx(yk, oversample_cap, seed)
        Xk, yk = Xk[ii], yk[ii]
        wk = None if wk is None else wk[ii]
    rf.fit(Xk, yk, sample_weight=wk)
    return GuidedModel(rf, rf.classes_)


# ---------------------------------------------------------------- scoring
def score_queries(model, feats, embedding, bout_ids=None, knn_k=15, proba=None) -> pd.DataFrame:
    """Per-bin (row index = bin) or, if bout_ids is given, per-bout scores.

    Per-bout mode: rows ordered by sorted unique non-negative bout id; columns
    gain bout_id, first_bin, n_bins. Negative ids are ignored. `embedding`
    must be (n_bins, d); per-bout embedding rows (centroids) are available via
    the returned attrs["bout_embedding"]."""
    if proba is None:
        proba = model.predict_proba(feats)                  # callers that already have it pass proba=
    classes = np.asarray(model.classes_, dtype=object)
    n = len(proba)
    valid = np.isfinite(proba).all(axis=1)
    top1_idx = np.full(n, -1)
    if valid.any():
        top1_idx[valid] = np.argmax(proba[valid], axis=1)

    # kNN-in-embedding disagreement of own top1 vs. neighbours' top1
    knn_dis = np.full(n, np.nan)
    if embedding is not None and valid.sum() > 2:
        from sklearn.neighbors import NearestNeighbors
        emb = np.asarray(embedding, dtype=float)
        vi = np.where(valid & np.isfinite(emb).all(axis=1))[0]
        if len(vi) > 2:
            k = min(knn_k, len(vi) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(emb[vi])
            _, nb = nn.kneighbors(emb[vi])
            nb_cls = top1_idx[vi][nb[:, 1:]]
            knn_dis[vi] = (nb_cls != top1_idx[vi][:, None]).mean(axis=1)

    def _frame(p, kd, extra=None):
        m = len(p)
        ok = np.isfinite(p).all(axis=1) if m else np.zeros(0, bool)
        maxprob = np.full(m, np.nan)
        margin = np.full(m, np.nan)
        t1 = np.full(m, "", dtype=object)
        t2 = np.full(m, "", dtype=object)
        if ok.any():
            srt = np.argsort(-p[ok], axis=1)
            ps = np.take_along_axis(p[ok], srt, axis=1)
            maxprob[ok] = ps[:, 0]
            margin[ok] = ps[:, 0] - (ps[:, 1] if ps.shape[1] > 1 else 0.0)
            t1[ok] = classes[srt[:, 0]]
            if srt.shape[1] > 1:
                t2[ok] = classes[srt[:, 1]]
        df = pd.DataFrame({"maxprob": maxprob, "margin": margin, "top1": t1,
                           "top2": t2, "knn_disagree": kd})
        df["uncertain"] = (df["maxprob"] < 0.5).fillna(False).astype(bool)
        if extra:
            for k_, v_ in extra.items():
                df[k_] = v_
        return df

    if bout_ids is None:
        return _frame(proba, knn_dis)

    ids = np.asarray(bout_ids)
    uniq = np.unique(ids[ids >= 0])
    P, KD, first, cnt, cent = [], [], [], [], []
    emb_arr = None if embedding is None else np.asarray(embedding, dtype=float)
    for b in uniq:
        idx = np.where(ids == b)[0]
        pv = proba[idx][valid[idx]]
        P.append(pv.mean(axis=0) if len(pv) else np.full(proba.shape[1], np.nan))
        kv = knn_dis[idx]
        KD.append(np.nanmean(kv) if np.isfinite(kv).any() else np.nan)
        first.append(int(idx[0]))
        cnt.append(len(idx))
        if emb_arr is not None:
            cent.append(np.nanmean(emb_arr[idx], axis=0))
    df = _frame(np.array(P).reshape(len(uniq), proba.shape[1]), np.array(KD, float),
                {"bout_id": uniq, "first_bin": first, "n_bins": cnt})
    if cent:
        df.attrs["bout_embedding"] = np.array(cent)
    return df


# --------------------------------------------------------------- selection
def select_queries(scores: pd.DataFrame, session_of_bin, labelled_sessions: set,
                   n: int, strategy: str = "ours",
                   quota_unlabelled_sessions: float = 0.5, embedding=None,
                   exclude=frozenset(), seed: int = 0, focus_p=None) -> list:
    """Return up to n row positions of `scores` (unlabelled-by-model-uncertainty
    candidates). `session_of_bin` and `embedding` must align with scores rows.
    asoid returns fewer than n if fewer rows have no class > 0.5."""
    rng = np.random.default_rng(seed)
    sess = np.asarray(session_of_bin, dtype=object)
    maxp = scores["maxprob"].to_numpy(float)
    valid = np.isfinite(maxp)
    if exclude:
        ex = np.zeros(len(scores), bool)
        ex[[i for i in exclude if 0 <= i < len(scores)]] = True
        valid &= ~ex
    if n <= 0 or not valid.any():
        return []

    if strategy == "random":
        pool = np.where(valid)[0]
        return [int(i) for i in rng.choice(pool, size=min(n, len(pool)), replace=False)]
    if strategy == "asoid":
        pool = np.where(valid & (maxp <= 0.5))[0]
        return [int(i) for i in rng.choice(pool, size=min(n, len(pool)), replace=False)] \
            if len(pool) else []
    if strategy != "ours":
        raise ValueError(f"unknown strategy {strategy!r}")

    margin = np.nan_to_num(scores["margin"].to_numpy(float), nan=1.0)
    kd = np.nan_to_num(scores["knn_disagree"].to_numpy(float), nan=0.0)
    util = 0.5 * (1.0 - np.nan_to_num(maxp, nan=1.0)) + 0.5 * (1.0 - margin) + 0.25 * kd
    if focus_p is not None and len(focus_p) == len(scores):
        # "ask me about X": the clips most informative about ONE behaviour are those where the model gives it a
        # middling probability (it could be X or not), provided there is some evidence of X at all
        fp = np.nan_to_num(np.asarray(focus_p, dtype=float), nan=0.0)
        f_util = np.where(fp >= 0.08, 1.0 - 2.0 * np.abs(fp - 0.5), 0.0) + 0.25 * kd
        if (f_util[valid] > 0.3).sum() >= max(3, n // 2):          # enough candidates; else fall back to the default
            util = f_util
            valid = valid & (f_util > 0.3)
    emb = None
    if embedding is not None:
        emb = np.asarray(embedding, dtype=float)
        if len(emb) != len(scores):
            emb = None

    chosen: list = []
    mind = np.full(len(scores), np.inf)

    def _pick(mask, k):
        cand = np.where(mask & valid)[0]
        cand = np.array([c for c in cand if c not in set(chosen)], dtype=int)
        if len(cand) == 0 or k <= 0:
            return
        pool = cand[np.argsort(-util[cand])[:max(10 * k, 50)]]
        for _ in range(min(k, len(pool))):
            if emb is not None and chosen:
                d = mind[pool]
                dn = d / d.max() if np.isfinite(d).all() and d.max() > 0 else np.zeros(len(pool))
                sc = util[pool] / max(util[pool].max(), 1e-9) + dn
            else:
                sc = util[pool]
            j = int(np.argmax(sc))
            p = int(pool[j])
            chosen.append(p)
            pool = np.delete(pool, j)
            if emb is not None:
                dist = np.linalg.norm(emb - emb[p], axis=1)
                mind[:] = np.minimum(mind, np.nan_to_num(dist, nan=np.inf))
            if len(pool) == 0:
                break

    unl = np.array([s not in labelled_sessions for s in sess], bool)
    if unl.any() and len(labelled_sessions) > 0:
        _pick(unl, int(round(quota_unlabelled_sessions * n)))
    _pick(np.ones(len(scores), bool), n - len(chosen))
    return chosen[:n]


# ------------------------------------------------- remove the animal / video identity
def centre_per_video(X, session_bin_ranges) -> np.ndarray:
    """Subtract each video's own per-feature median and divide by its inter-quartile range (computed from ALL of its
    bins, no labels needed). Labelled behaviours are often collected in different videos (different animals / sessions),
    so a classifier otherwise learns WHICH VIDEO a bin comes from; centring leaves only what differs inside a video.
    Real data (Xenium, leave-one-video-out): Licking coverage 23 -> 53 %, leakage 47 -> 30 %."""
    X = np.asarray(X, dtype=np.float32)
    out = np.array(X, dtype=np.float32, copy=True)
    for _k, (s0, s1) in clean_bin_ranges(session_bin_ranges).items():
        seg = out[s0:s1]
        if len(seg) < 3:
            continue
        with np.errstate(all="ignore"):
            med = np.nanmedian(seg, axis=0)
            iqr = np.nanpercentile(seg, 75, axis=0) - np.nanpercentile(seg, 25, axis=0)
        iqr = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.0)
        out[s0:s1] = (seg - np.where(np.isfinite(med), med, 0.0)) / iqr
    return out


def hybrid_features(X, session_bin_ranges) -> np.ndarray:
    """[absolute features | per-video-centred features]: absolute posture (Rearing) AND within-video deviation
    (Guarding / Licking). Best balance on the Xenium labels (Rearing 90/2.6, Licking 41/30, Guarding 72/65 cov/leak %)."""
    X = np.asarray(X, dtype=np.float32)
    return np.concatenate([X, centre_per_video(X, session_bin_ranges)], axis=1)


# ------------------------------------------------------------ cluster votes
def cluster_label_votes(cluster_labels, y, min_bins: int = 5) -> pd.DataFrame:
    cl = np.asarray(cluster_labels)
    yy = np.asarray(y, dtype=object)
    rows = []
    for c in np.unique(cl):
        if c < 0:
            continue
        ys = [str(v) for v in yy[cl == c] if str(v) not in _NON_BEHAVIOUR
              and v is not None]
        if len(ys) < min_bins:
            continue
        vc = pd.Series(ys).value_counts()
        purity = float(vc.iloc[0]) / len(ys)
        rows.append({"cluster": int(c), "n_labelled": len(ys),
                     "top_behaviour": str(vc.index[0]), "purity": purity,
                     "n_votes_by_behaviour": json.dumps(
                         {k: int(v) for k, v in vc.items()}),
                     "is_mixed": bool(purity < 0.8)})
    return pd.DataFrame(rows, columns=["cluster", "n_labelled", "top_behaviour",
                                       "purity", "n_votes_by_behaviour", "is_mixed"])


# ---------------------------------------------------------------- smoothing
def clean_bin_ranges(session_bin_ranges: dict) -> dict:
    """Normalise a real session_bin_ranges.json to {session: (start, end)}.

    Real files hold [start, end, video_path] values and a non-session
    ``_total_bins`` int entry; both are tolerated here."""
    out = {}
    for k, v in session_bin_ranges.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            out[k] = (int(v[0]), int(v[1]))
    return out


def smooth_guided_track(proba, session_bin_ranges, classes, n_iter=15) -> np.ndarray:
    """Soft-emission HMM smoothing per session; argmax class idx per bin
    (-1 for NaN rows / bins outside every session). Raises ImportError if
    hmmlearn is unavailable."""
    proba = np.asarray(proba, dtype=float)
    K = len(classes)
    out = np.full(len(proba), -1, dtype=int)
    ok = np.isfinite(proba).all(axis=1)
    out[ok] = np.argmax(proba[ok], axis=1)
    seqs, spans = [], []
    for _, (s0, s1) in clean_bin_ranges(session_bin_ranges).items():
        s0, s1 = int(s0), min(int(s1), len(proba))
        if s1 - s0 < 3 or K < 2:
            continue
        seg = proba[s0:s1].copy()
        seg[~np.isfinite(seg).all(axis=1)] = 1.0 / K
        seqs.append(seg)
        spans.append((s0, s1))
    if not seqs:
        return out
    from cube_core import train_hmm_soft, decode_hmm_soft   # lazy
    try:
        model = train_hmm_soft(seqs, K, n_iter=n_iter)
        for seg, (s0, s1) in zip(seqs, spans):
            st = decode_hmm_soft(model, seg)
            m = ok[s0:s1]
            out[s0:s1][m] = st[m]
    except ImportError:
        raise
    except Exception:
        pass   # keep the argmax fallback
    return out


# ------------------------------------------------------------------ exports
def export_mapping_tsv(votes: pd.DataFrame, path, min_purity=0.6, min_bins=5) -> None:
    rows = []
    for _, r in votes.iterrows():
        ok = (int(r["n_labelled"]) >= min_bins and float(r["purity"]) >= min_purity)
        rows.append({"bsoid_group_id": int(r["cluster"]),
                     "behaviour_group": str(r["top_behaviour"]) if ok else "",
                     "status": "assigned" if ok else "pending",
                     "example_count": 0, "gif_path": "", "mp4_paths": "",
                     "n_votes": int(r["n_labelled"]), "purity": float(r["purity"])})
    cols = ["bsoid_group_id", "behaviour_group", "status", "example_count",
            "gif_path", "mp4_paths", "n_votes", "purity"]
    p = str(path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = p + ".tmp"
    pd.DataFrame(rows, columns=cols).to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, p)


def export_guided_track_csv(track, classes, session_bin_ranges, path,
                            proba=None, fps=None) -> None:
    """Per-session rows. Optional extras (contract-compatible): `proba` fills
    maxprob (else NaN); `fps` (float or dict) gives time_s = bin*win100/fps
    (else nominal 0.1 s bins)."""
    track = np.asarray(track)
    rows = []
    for sess, (s0, s1) in clean_bin_ranges(session_bin_ranges).items():
        f = fps.get(sess) if isinstance(fps, dict) else fps
        win_s = (max(1, int(round(f / 10.0))) / float(f)) if f else 0.1
        for b in range(int(s0), min(int(s1), len(track))):
            k = int(track[b])
            rows.append({"session": sess, "bin": b - int(s0),
                         "time_s": (b - int(s0)) * win_s,
                         "behaviour": classes[k] if 0 <= k < len(classes) else "",
                         "maxprob": (float(np.nanmax(proba[b]))
                                     if proba is not None and np.isfinite(proba[b]).any()
                                     else float("nan"))})
    p = str(path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = p + ".tmp"
    pd.DataFrame(rows, columns=["session", "bin", "time_s", "behaviour",
                                "maxprob"]).to_csv(tmp, index=False)
    os.replace(tmp, p)



# ------------------------------------------------------- temporal context
def _session_fps(fps, sess, default=30.0) -> float:
    if isinstance(fps, dict):
        return float(fps.get(sess, default))
    return float(fps) if fps else default


def _context_blocks_pandas(src, ranges, fps, windows_s, min_valid_frac):
    """REFERENCE implementation (pandas rolling per session). Slower; kept as the fallback and as the
    ground truth the fast path is tested against. Returns [mean_w1, std_w1, mean_w2, std_w2, ..., delta]."""
    n, dd = src.shape
    blocks = [np.full((n, dd), np.nan, np.float32) for _ in range(2 * len(windows_s) + 1)]
    for sess, (s0, s1) in clean_bin_ranges(ranges).items():
        s0, s1 = max(0, s0), min(n, s1)
        if s1 - s0 < 2:
            continue
        f = _session_fps(fps, sess)
        bin_s = max(1, int(round(f / 10.0))) / f
        df = pd.DataFrame(src[s0:s1])
        bi = 0
        for ws in windows_s:
            w = max(1, int(round(ws / bin_s)))
            size = 2 * w + 1
            mp = max(1, int(math.ceil(min_valid_frac * size)))
            r = df.rolling(size, center=True, min_periods=mp)
            blocks[bi][s0:s1] = r.mean().to_numpy(np.float32)
            blocks[bi + 1][s0:s1] = r.std(ddof=0).to_numpy(np.float32)
            bi += 2
        w = max(1, int(round(max(windows_s) / bin_s)))
        mp = max(1, int(math.ceil(min_valid_frac * w)))
        prev = df.rolling(w, min_periods=mp).mean().shift(1)
        nxt = df[::-1].rolling(w, min_periods=mp).mean().shift(1)[::-1]
        blocks[-1][s0:s1] = (nxt - prev).to_numpy(np.float32)
    return blocks


def _rolling_cumsum(A, windows_s, bin_s, min_valid_frac, out_blocks, s0, s1):
    """Vectorised cumulative-sum rolling stats for ONE session; handles any NaN pattern."""
    m, d = A.shape
    V = np.isfinite(A)
    mu = np.nan_to_num(np.nanmean(A, axis=0)) if V.any() else np.zeros(d)
    Z = np.where(V, A - mu.astype(A.dtype), 0).astype(np.float64)
    zeros = np.zeros((1, d))
    S1 = np.vstack([zeros, np.cumsum(Z, axis=0)])
    S2 = np.vstack([zeros, np.cumsum(Z * Z, axis=0)])
    C = np.vstack([zeros, np.cumsum(V, axis=0, dtype=np.float64)])
    idx = np.arange(m)

    def window(lo, hi_excl):
        lo = np.clip(lo, 0, m)
        hi = np.maximum(np.clip(hi_excl, 0, m), lo)
        return C[hi] - C[lo], S1[hi] - S1[lo], lo, hi

    bi = 0
    for ws in windows_s:
        w = max(1, int(round(ws / bin_s)))
        mp = max(1, int(math.ceil(min_valid_frac * (2 * w + 1))))
        cnt, sm1, lo, hi = window(idx - w, idx + w + 1)
        s2 = S2[hi] - S2[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean0 = sm1 / cnt
            var = np.maximum(s2 / cnt - mean0 * mean0, 0.0)
        bad = cnt < mp
        mean = mean0 + mu
        mean[bad] = np.nan
        sd = np.sqrt(var)
        sd[bad] = np.nan
        out_blocks[bi][s0:s1] = mean
        out_blocks[bi + 1][s0:s1] = sd
        bi += 2
    w = max(1, int(round(max(windows_s) / bin_s)))
    mp = max(1, int(math.ceil(min_valid_frac * w)))
    cp, sm1p, _, _ = window(idx - w, idx)
    cn, sm1n, _, _ = window(idx + 1, idx + 1 + w)
    with np.errstate(invalid="ignore", divide="ignore"):
        prev = sm1p / cp
        nxt = sm1n / cn
    prev[cp < mp] = np.nan
    nxt[cn < mp] = np.nan
    out_blocks[-1][s0:s1] = nxt - prev            # the +mu cancels


def _rolling_uf(A, windows_s, bin_s, min_valid_frac, out_blocks, s0, s1):
    """Same statistics via scipy's C box filters (about 3x faster); valid when NaNs are WHOLE ROWS only (true for
    feats_all and PCA scores). Partially-NaN rows fall back to the generic cumsum path."""
    from scipy.ndimage import uniform_filter1d
    m, d = A.shape
    finite = np.isfinite(A)
    v = finite.all(axis=1)
    if not (v | ~finite.any(axis=1)).all():
        return _rolling_cumsum(A, windows_s, bin_s, min_valid_frac, out_blocks, s0, s1)
    mu = A[v].mean(axis=0, dtype=np.float64) if v.any() else np.zeros(d)
    Z = np.where(v[:, None], A - mu.astype(A.dtype), 0).astype(np.float32)
    vf = v.astype(np.float64)

    def box(arr, size):
        return uniform_filter1d(arr, size, axis=0, mode="constant", cval=0.0) * size

    bi = 0
    for ws in windows_s:
        w = max(1, int(round(ws / bin_s)))
        size, mp = 2 * w + 1, max(1, int(math.ceil(min_valid_frac * (2 * w + 1))))
        cnt = np.rint(box(vf, size))
        ok = cnt >= mp
        c = np.where(cnt > 0, cnt, 1.0).astype(np.float32)[:, None]
        mean = box(Z, size) / c
        var = np.maximum(box(Z * Z, size) / c - mean * mean, 0)
        mean = mean + mu.astype(np.float32)
        sd = np.sqrt(var)
        mean[~ok] = np.nan
        sd[~ok] = np.nan
        out_blocks[bi][s0:s1] = mean
        out_blocks[bi + 1][s0:s1] = sd
        bi += 2
    w = max(1, int(round(max(windows_s) / bin_s)))
    mp = max(1, int(math.ceil(min_valid_frac * w)))
    pz = np.zeros((w, d), np.float32)
    Zp = np.vstack([pz, Z, pz])
    vp = np.concatenate([np.zeros(w), vf, np.zeros(w)])
    Sw, Cw = box(Zp, w), np.rint(box(vp, w))

    def fwd(j):                                    # forward window [j, j+w-1] in session coordinates
        k = np.clip(j + w + w // 2, 0, len(Zp) - 1)
        return Sw[k], Cw[k]
    i = np.arange(m)
    sn, cn = fwd(i + 1)
    sp, cp = fwd(i - w)
    nxt = sn / np.where(cn > 0, cn, 1)[:, None]
    nxt[cn < mp] = np.nan
    prev = sp / np.where(cp > 0, cp, 1)[:, None]
    prev[cp < mp] = np.nan
    out_blocks[-1][s0:s1] = nxt - prev


def _context_blocks(src, ranges, fps, windows_s, min_valid_frac):
    """Per-session (never across sessions) rolling mean/std for each window plus one delta
    (mean of the next - previous max(windows_s)). NaN where a window has < min_valid_frac valid bins.
    Fast vectorised path; any unexpected failure falls back to the pandas reference."""
    try:
        n, dd = src.shape
        blocks = [np.full((n, dd), np.nan, np.float32) for _ in range(2 * len(windows_s) + 1)]
        for sess, (s0, s1) in clean_bin_ranges(ranges).items():
            s0, s1 = max(0, s0), min(n, s1)
            if s1 - s0 < 2:
                continue
            f = _session_fps(fps, sess)
            bin_s = max(1, int(round(f / 10.0))) / f
            _rolling_uf(src[s0:s1], windows_s, bin_s, min_valid_frac, blocks, s0, s1)
        return blocks
    except Exception:
        return _context_blocks_pandas(src, ranges, fps, windows_s, min_valid_frac)


def _neutral_fill(blocks, src, n_windows, valid_row):
    """Invalid-window rows get neutral values (mean = own value, std = 0, delta = 0) so no bin becomes NaN."""
    out = []
    for j, b in enumerate(blocks):
        kind = "mean" if (j < 2 * n_windows and j % 2 == 0) else ("delta" if j == 2 * n_windows else "std")
        neutral = src if kind == "mean" else 0.0
        out.append(np.where(np.isnan(b) & valid_row[:, None], neutral, b).astype(np.float32))
    return out


def _shared_pca(X, valid_row, k_max):
    """Exact top-k_max PCA scores of the valid rows from ONE d x d covariance eigendecomposition (NaN rows stay
    NaN). Components are nested, so every smaller k is a column slice of the same fit (the old code fitted two
    separate PCAs for the main and the extra-window blocks)."""
    Xv = np.nan_to_num(X[valid_row], nan=0.0).astype(np.float32, copy=False)
    Xc = Xv - Xv.mean(axis=0)
    cov = (Xc.T @ Xc).astype(np.float64) / max(1, len(Xc) - 1)
    _w, V = np.linalg.eigh(cov)
    V = V[:, ::-1][:, :k_max].astype(np.float32)
    Z = np.full((len(X), k_max), np.nan, np.float32)
    Z[valid_row] = Xc @ V
    return Z


def build_context_features(feats, session_bin_ranges, fps=30.0, *, enabled: bool = True,
                           windows_s=(0.5, 1.0), min_valid_frac: float = 0.5,
                           max_cols: int = 4000, extra_windows_s=(2.0, 4.0),
                           extra_pca_dim: int = 60) -> np.ndarray:
    """Base features + per-session temporal context (view-agnostic: 2D top/side and 3D).

    Per bin, within each session only (never across a session boundary): windowed mean and std over
    +-windows_s plus delta = mean(next max(windows_s)) - mean(prev ...). NaN-aware: a window needs
    >= ``min_valid_frac`` valid bins; invalid windows fall back to neutral values (mean = own value,
    std = 0, delta = 0) so no extra bins become NaN, and bins whose BASE row is NaN stay NaN.
    If base_cols * (1 + 2*len(windows_s) + 1) would exceed ``max_cols`` the context is built from a PCA
    of the base to keep memory bounded.

    ``extra_windows_s`` (default +-2 s and +-4 s, JAABA-style longer scales; () = off) adds the same blocks
    computed on a ``extra_pca_dim``-component PCA of the base. On its own this is weak (EPM +0.004..0.010, OFT
    -0.026 at 200 clips); it is a default only as part of the combination (longer windows + full-parity
    oversampling) that was the single arm passing the written "does not hurt" rule on BOTH top-view datasets
    (wave 7: OFT mean +0.006 over budgets, worst budget -0.007; EPM mean +0.034, worst +0.021).
    ``enabled=False`` returns ``feats`` unchanged (kill switch / blind behaviour)."""
    X = np.asarray(feats)
    if not enabled:
        return X
    X = X.astype(np.float32, copy=False)
    n, d = X.shape
    valid_row = ~np.isnan(X).all(axis=1)
    n_blocks = 1 + 2 * len(windows_s) + 1
    k_src = int(max(2, min(d, max_cols // n_blocks))) if (d * n_blocks > max_cols and valid_row.sum() > 2) else 0
    k_ext = int(max(2, min(d, extra_pca_dim))) if (extra_windows_s and valid_row.sum() > 2) else 0
    Zall = _shared_pca(X, valid_row, max(k_src, k_ext)) if max(k_src, k_ext) else None
    src = Zall[:, :k_src] if k_src else X
    out = [X] + _neutral_fill(_context_blocks(src, session_bin_ranges, fps, windows_s, min_valid_frac),
                              src, len(windows_s), valid_row)
    if k_ext:
        zs = Zall[:, :k_ext]
        out += _neutral_fill(_context_blocks(zs, session_bin_ranges, fps, tuple(extra_windows_s), min_valid_frac),
                             zs, len(extra_windows_s), valid_row)
    res = np.concatenate(out, axis=1)
    res[~valid_row, X.shape[1]:] = np.nan
    return res