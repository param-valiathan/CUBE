"""User-label constraints for the CUBE clustering engine (guided labelling, phase P4).

The engine (cube_core.BSoidEngine.run) builds ONE GuidedContext per run, only when
  * cfg["guided_constraints_enabled"] is True (kill switch), AND
  * cfg["guided_labels_path"] points at a readable labels file with >= 1 usable label.
Every hook in cube_core.py is `ctx = guided_active(); if ctx is not None: ...`, so with no labels attached the
hook bodies never run and outputs are identical to a blind run.

Constraints only ever make the engine MORE conservative: they can veto a prune / merge / split / reclaim that
the existing gates would have allowed, or add split candidates that the existing split gates must still
accept. They never force a merge, split or reassignment the existing gates reject.

Labels are per training row (same length / order as the cluster `labels` arrays the hooks see) and carry only
user behaviours: "Background" and "Neither" bins are None (excluded from purity logic). Behaviour names are
case/space-normalised. Half-weight edge bins (marks at an interval's edge) count 0.5 as evidence.

SAFETY: every query first checks that the `labels` array it is handed is EXACTLY the engine's training-row
array (same length). Hooks also fire on local split sub-fits and seed-sweep candidates whose arrays are shorter
or differently ordered; there the context has no opinion (never matches ids to unrelated rows).
Not carried into joblib worker processes (module global), so per-seed sweeps stay blind by design.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

_ACTIVE = None
BACKGROUND = "Background"
NEITHER_MARK = "__neither__"


def _norm(v):
    return str(v).strip().casefold()


class GuidedContext:
    def __init__(self, label_per_row, min_bins: int = 5, purity_gain: float = 0.05,
                 pure_thresh: float = 0.9, log=None, weights=None, min_frac: float = 0.005):
        rows = np.asarray(label_per_row, dtype=object)
        self._display: dict = {}                      # normalised name -> first original spelling (for reports)
        for v in rows:
            if v is not None:
                self._display.setdefault(_norm(v), str(v).strip())
        self.rows = np.array([None if v is None else _norm(v) for v in rows], dtype=object)
        self.lab_idx = np.flatnonzero(np.array([v is not None for v in self.rows]))
        self.lab_val = self.rows[self.lab_idx]
        w = np.ones(len(self.rows)) if weights is None else np.asarray(weights, dtype=float)
        self.lab_w = w[self.lab_idx] if len(w) == len(self.rows) else np.ones(len(self.lab_idx))
        self.names = sorted({str(v) for v in self.lab_val})
        code = {n: i for i, n in enumerate(self.names)}
        self.lab_code = np.array([code[str(v)] for v in self.lab_val], dtype=int)
        self.nb = len(self.names)
        self.min_bins = float(min_bins)
        self.min_frac = float(min_frac)
        self.purity_gain = float(purity_gain)
        self.pure_thresh = float(pure_thresh)
        self.log = log or (lambda m: None)
        self.actions: list = []          # audit trail -> guided_constraints_log.json
        self._seen: set = set()

    # ------------------------------------------------------------ helpers
    @property
    def n_labelled(self) -> int:
        return len(self.lab_idx)

    def behaviour_counts(self) -> dict:
        """{behaviour (original spelling): labelled training rows} for logs and the run summary."""
        c = Counter(self.lab_val)
        return {self._display.get(str(k), str(k)): int(v) for k, v in c.items()}

    def aligned(self, labels) -> bool:
        return len(labels) == len(self.rows)

    def note(self, kind: str, **kw):
        key = (kind, tuple(sorted((k, str(v)) for k, v in kw.items())))
        if key in self._seen:                         # the same refusal is re-evaluated on every pass: log once
            return
        self._seen.add(key)
        self.actions.append({"action": kind, **{k: (v.item() if hasattr(v, "item") else v) for k, v in kw.items()}})

    def _mat(self, labels):
        """(M, sizes): M[c, b] = weighted labelled bins of behaviour b in cluster c; sizes[c] = cluster size.
        Vectorised (one bincount each). None if `labels` is not the engine's training-row array."""
        labels = np.asarray(labels)
        if not self.aligned(labels) or self.nb == 0:
            return None
        ncl = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
        if ncl == 0:
            return None
        cl = labels[self.lab_idx].astype(int)
        ok = cl >= 0
        flat = cl[ok] * self.nb + self.lab_code[ok]
        M = np.bincount(flat, weights=self.lab_w[ok], minlength=ncl * self.nb).reshape(ncl, self.nb)
        sizes = np.bincount(labels[labels >= 0].astype(int), minlength=ncl)
        return M, sizes

    def _threshold(self, size):
        """Labelled bins of one behaviour needed before they count as evidence in a cluster of `size` bins:
        the absolute minimum AND a minimum coverage, so a handful of labels cannot vouch for a huge cluster."""
        return np.maximum(self.min_bins, np.ceil(self.min_frac * np.asarray(size, dtype=float)))

    def _evid(self, M, sizes):
        return M >= self._threshold(sizes)[:, None]

    def counts(self, labels) -> dict:
        """cluster id -> Counter(behaviour -> weighted labelled bins); noise excluded. {} if not aligned."""
        got = self._mat(labels)
        if got is None:
            return {}
        M, _s = got
        out = {}
        for c in np.flatnonzero(M.sum(axis=1) > 0):
            out[int(c)] = Counter({self.names[b]: float(M[c, b]) for b in np.flatnonzero(M[c] > 0)})
        return out

    # ----------------------------------------------------------- queries
    def protected_clusters(self, labels) -> set:
        """Clusters with real labelled evidence of a behaviour: they must survive pruning."""
        got = self._mat(labels)
        if got is None:
            return set()
        M, sizes = got
        return {int(c) for c in np.flatnonzero(self._evid(M, sizes).any(axis=1))}

    def dominant(self, labels) -> dict:
        """cluster id -> dominant EVIDENCED behaviour (clusters without evidence omitted)."""
        got = self._mat(labels)
        if got is None:
            return {}
        M, sizes = got
        ev = self._evid(M, sizes)
        out = {}
        for c in np.flatnonzero(ev.any(axis=1)):
            out[int(c)] = self.names[int(np.argmax(np.where(ev[c], M[c], -1)))]
        return out

    def merge_blocked(self, labels, a, b) -> bool:
        """Cannot-link: refuse a merge that would ADD a behavioural conflict. ``a`` / ``b`` are cluster ids or
        groups of ids (merges are transitive, so callers pass whole groups). Each side's evidenced behaviours
        are compared; the merge is blocked only when neither side already contains the other's behaviours
        (so merging a same-behaviour cluster into one that is already mixed is NOT blocked)."""
        got = self._mat(labels)
        if got is None:
            return False
        M, sizes = got
        ga = [int(x) for x in np.atleast_1d(a) if int(x) < len(sizes)]
        gb = [int(x) for x in np.atleast_1d(b) if int(x) < len(sizes)]
        if not ga or not gb:
            return False

        def evid(group):
            m = M[group].sum(axis=0)
            return set(np.flatnonzero(m >= self._threshold(sizes[group].sum())).tolist())
        ea, eb = evid(ga), evid(gb)
        union = ea | eb
        blocked = len(union) >= 2 and not (ea >= eb or eb >= ea)
        if blocked:
            self.note("merge_refused", a=ga, b=gb, behaviours=sorted(self.names[i] for i in union))
        return blocked

    def purity(self, cnt: Counter):
        n = sum(cnt.values())
        return (max(cnt.values()) / n) if n else None

    def mixed_clusters(self, labels, mixed_thresh: float = 0.8) -> set:
        """Clusters with enough labelled evidence of >= 2 behaviours (purity below mixed_thresh)."""
        got = self._mat(labels)
        if got is None:
            return set()
        M, sizes = got
        ev = self._evid(M, sizes)
        tot = M.sum(axis=1)
        out = set()
        for c in np.flatnonzero(ev.sum(axis=1) >= 2):
            if M[c].max() / max(tot[c], 1e-9) < mixed_thresh:
                out.add(int(c))
        return out

    def is_user_pure(self, labels, cid: int) -> bool:
        """True if this cluster's labels show ONE evidenced behaviour with purity >= pure_thresh."""
        got = self._mat(labels)
        if got is None or cid >= len(got[1]):
            return False
        M, sizes = got
        ev = self._evid(M, sizes)[cid]
        tot = M[cid].sum()
        return bool(ev.sum() == 1 and tot > 0 and M[cid].max() / tot >= self.pure_thresh)

    def reclaim_veto(self, labels, dest_cluster) -> np.ndarray:
        """Boolean mask over rows: True where moving that row to ``dest_cluster[row]`` must be vetoed.
        Only user-labelled rows are ever vetoed. A labelled row may not be moved into a cluster whose evidenced
        behaviour differs from its own label, nor pulled out of the cluster that matches its label."""
        labels = np.asarray(labels)
        veto = np.zeros(len(labels), bool)
        got = self._mat(labels)
        if got is None:
            return veto
        M, sizes = got
        ev = self._evid(M, sizes)
        dom = np.where(ev.any(axis=1), np.argmax(np.where(ev, M, -1), axis=1), -1)

        def dom_of(cl):
            cl = np.asarray(cl).astype(int)
            out = np.full(cl.shape, -1, dtype=int)
            ok = (cl >= 0) & (cl < len(dom))
            out[ok] = dom[cl[ok]]
            return out
        own = dom_of(labels[self.lab_idx])
        dst = dom_of(np.asarray(dest_cluster)[self.lab_idx])
        v = ((dst >= 0) & (dst != self.lab_code)) | ((own == self.lab_code) & (dst != self.lab_code))
        veto[self.lab_idx[v]] = True
        return veto

    def split_verdict(self, parent_rows, sub_labels) -> tuple:
        """Decide a candidate split of one cluster. ``parent_rows``: row indices (into the engine's training
        rows) of the parent cluster's members; ``sub_labels``: the proposed per-member sub-cluster labels (-1 =
        noise), same order. Returns (accept: bool, reason: str); (True, "no_evidence") = no opinion.
        Labelled bins a split sends to noise count AGAINST it (a split that dumps the minority behaviour
        into noise is not a purity gain)."""
        parent_rows = np.asarray(parent_rows)
        sub_labels = np.asarray(sub_labels)
        if parent_rows.size and parent_rows.max() >= len(self.rows):
            return True, "not_aligned"
        beh = self.rows[parent_rows]
        has = np.array([v is not None for v in beh])
        if has.sum() < self.min_bins:
            return True, "no_evidence"
        wmap = np.zeros(len(self.rows))
        wmap[self.lab_idx] = self.lab_w
        w = wmap[parent_rows]
        parent_cnt = Counter()
        for v, wi in zip(beh[has], w[has]):
            parent_cnt[str(v)] += wi
        thr = float(self._threshold(len(parent_rows)))
        ev = {b for b, n in parent_cnt.items() if n >= thr}
        total = sum(parent_cnt.values())
        par_pur = max(parent_cnt.values()) / total if total else 0.0
        if not ev:
            return True, "no_evidence"
        if len(ev) <= 1 and par_pur >= self.pure_thresh:
            return False, "pure_parent_untouched"             # user-pure clusters are never split
        # purity after: labelled weight kept in each child's own majority / ALL labelled weight (noise = loss)
        kept = 0.0
        for s in np.unique(sub_labels[sub_labels >= 0]):
            m = has & (sub_labels == s)
            if not m.any():
                continue
            c = Counter()
            for v, wi in zip(beh[m], w[m]):
                c[str(v)] += wi
            kept += max(c.values())
        after = kept / total if total else 0.0
        gain = after - par_pur
        if gain >= self.purity_gain:
            return True, f"purity_gain_{gain:.2f}"
        return False, f"no_purity_gain_{gain:.2f}"


# --------------------------------------------------------------- module state
def set_active(ctx) -> None:
    global _ACTIVE
    _ACTIVE = ctx


def clear() -> None:
    global _ACTIVE
    _ACTIVE = None


def guided_active():
    return _ACTIVE


# ------------------------------------------------------------------ builder
def load_global_rows(cfg: dict, session_ranges: dict, fps_map, n_bins: int, log=None):
    """User labels in GLOBAL pooled-bin space: (rows, LabelSet, weights) where rows is an object array of length
    n_bins holding a behaviour name or None (Background / Neither / unlabelled are None) and weights is the
    per-bin evidence weight (edge bins 0.5); or None for a blind run. Never raises: a missing / corrupt /
    inconsistent labels file logs a WARN and returns None."""
    log = log or (lambda m: None)
    try:
        if not bool(cfg.get("guided_constraints_enabled", True)):
            return None
        path = str(cfg.get("guided_labels_path", "") or "").strip()
        if not path:
            return None
        import os
        if not os.path.isfile(path):
            log(f"  [GUIDED] labels file not found ({path}) -- running blind.")
            return None
        import cube_guided as cg
        ls = cg.load_labels(path)
        if not ls.labels:
            log("  [GUIDED] labels file has no labels -- running blind.")
            return None
        known = set(session_ranges)
        unknown = sorted({l["session"] for l in ls.labels} - known)
        if unknown and len(unknown) == len({l["session"] for l in ls.labels}):
            log(f"  [GUIDED] WARN: none of the labelled sessions ({len(unknown)}) are in this run -- running blind.")
            return None
        if unknown:
            log(f"  [GUIDED] WARN: {len(unknown)} labelled session(s) are not in this run and are ignored: "
                f"{unknown[:3]}")
        y, w = cg.labels_to_bins(ls, session_ranges, fps_map, n_bins)
        rows = np.array([None if (v in ("", NEITHER_MARK, BACKGROUND) or v is None) else str(v)
                         for v in np.asarray(y, dtype=object)], dtype=object)
        if int(sum(v is not None for v in rows)) == 0:
            log("  [GUIDED] no labelled bin falls on a usable behaviour -- running blind.")
            return None
        return rows, ls, np.asarray(w, dtype=float)
    except Exception as e:                                       # noqa: BLE001
        log(f"  [GUIDED] WARN: could not use the labels file ({e}) -- running blind.")
        return None


def labelled_good_positions(rows_global, exclude_mask) -> np.ndarray:
    """Positions (into the GOOD, i.e. non-excluded, bins) of every user-labelled bin: the engine adds these to
    the training subsample so a labelled bin is never lost to train_frac subsampling."""
    lab = np.array([v is not None for v in rows_global], dtype=bool)
    good = ~np.asarray(exclude_mask, dtype=bool)
    return np.flatnonzero(lab[good])


def build_context(cfg: dict, session_ranges: dict, fps_map, n_bins: int, train_orig_bin_idx, n_samp: int,
                  log=None, preloaded=None):
    """Return a GuidedContext, or None (blind run). Never raises. ``preloaded`` = the (rows_global, LabelSet,
    weights) triple from load_global_rows(), reused so the engine reads the labels file only once per run."""
    log = log or (lambda m: None)
    try:
        got = preloaded if preloaded is not None else load_global_rows(cfg, session_ranges, fps_map, n_bins, log)
        if got is None:
            return None
        rows_global, ls, w_global = got
        if train_orig_bin_idx is not None:
            sel = np.asarray(train_orig_bin_idx)
            rows, w = rows_global[sel], w_global[sel]
        elif n_samp == n_bins:
            rows, w = rows_global, w_global
        else:
            log("  [GUIDED] WARN: cannot align labels to the training rows -- running blind.")
            return None
        if int(sum(v is not None for v in rows)) == 0:
            log("  [GUIDED] no labelled bin falls on a training row -- running blind.")
            return None
        ctx = GuidedContext(rows, min_bins=float(cfg.get("guided_min_labelled_bins", 5)),
                            purity_gain=float(cfg.get("guided_split_purity_gain", 0.05)),
                            pure_thresh=float(cfg.get("guided_pure_thresh", 0.9)),
                            min_frac=float(cfg.get("guided_min_labelled_frac", 0.005)),
                            weights=w, log=log)
        beh = ctx.behaviour_counts()
        log(f"  [GUIDED] labels: {len(ls.labels)} entries, {ctx.n_labelled} labelled training rows, "
            f"{len(beh)} behaviours {beh}; sha1 {ls.sha1[:10]}")
        return ctx
    except Exception as e:                                       # noqa: BLE001
        log(f"  [GUIDED] WARN: could not use the labels file ({e}) -- running blind.")
        return None
