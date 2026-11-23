"""Shared statistical policy for the CUBE Analyser (small-n friendly).

One module-level POLICY dict is read by every tab so the same options give the
same calls everywhere.  Pure functions only (no tkinter) so everything is unit
testable.  Defaults favour power at the low n typical of neuroscience while
keeping false discoveries controlled (BH-FDR is still applied; nothing here
removes multiple-testing correction).
"""
from __future__ import annotations

import warnings

import numpy as np

POLICY = {
    # Mark q < SUGGESTIVE_Q as "+" (suggestive) in addition to q < 0.05 (*, **, ***).
    "suggestive_tier": True,
    # Post-hoc pairs are tested/corrected only for clusters whose omnibus raw p < 0.05
    # (Fisher-protected), instead of one BH family over every pair of every cluster.
    "gate_posthoc": True,
    # Paired designs: paired t-test on log1p values when n is small (rank tests
    # cannot reach p < 0.05 at n <= 5 and are very conservative at n <= 9).
    "small_n_paired": True,
    # Repeated measures: use every animal with >= 2 levels (mixed model) instead of
    # only animals present at every level.
    "mixed_repeated": True,
    # One short note when n is too small for the chosen test to reach significance.
    "small_n_note": True,
    # Independent filtering: do not test features seen in fewer than
    # MIN_PRESENT_ANIMALS animals (they cannot be significant at small n and only
    # enlarge the BH family, costing power for every other feature).
    "prefilter_sparse": True,
}
MIN_PRESENT_ANIMALS = 3
SUGGESTIVE_Q = 0.10
SMALL_N_PAIRED_MAX = 9           # n <= this uses the paired t-test (when enabled)

OPTION_LABELS = {
    "suggestive_tier": "Show suggestive tier (+, q<0.10) and effect sizes",
    "gate_posthoc": "Post-hoc only where omnibus p<0.05 (more power)",
    "small_n_paired": "Paired t-test on log usage when n<10",
    "mixed_repeated": "Repeated measures: use all animals (mixed model)",
    "small_n_note": "Show small-n note",
    "prefilter_sparse": "Skip features seen in <3 animals (power)",
}

_listeners: list = []


def set_option(key: str, value: bool) -> None:
    POLICY[key] = bool(value)
    for fn in list(_listeners):
        try:
            fn()
        except Exception:
            pass


def on_change(fn) -> None:
    _listeners.append(fn)


def policy_dict() -> dict:
    d = dict(POLICY)
    d["suggestive_q"] = SUGGESTIVE_Q
    d["small_n_paired_max"] = SMALL_N_PAIRED_MAX
    return d


def write_policy(out_dir, extra: dict = None) -> bool:
    """Write stats_policy.json (the options in force) next to an export."""
    import json
    import pathlib
    try:
        d = policy_dict()
        d["min_present_animals"] = MIN_PRESENT_ANIMALS
        if extra:
            d.update(extra)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(out_dir) / "stats_policy.json").write_text(
            json.dumps(d, indent=2, default=str), encoding="utf-8")
        return True
    except Exception:
        return False


# ── significance marks / tiers ───────────────────────────────────────────────

def display_q() -> float:
    """Largest q that gets any mark."""
    return SUGGESTIVE_Q if POLICY["suggestive_tier"] else 0.05


def tier(q) -> str:
    try:
        q = float(q)
    except Exception:
        return "ns"
    if not np.isfinite(q):
        return "ns"
    if q < 0.05:
        return "confirmed"
    if POLICY["suggestive_tier"] and q < SUGGESTIVE_Q:
        return "suggestive"
    return "ns"


def mark(q, ns: str = "ns") -> str:
    """*** / ** / * for confirmed, + for suggestive, `ns` otherwise."""
    try:
        q = float(q)
    except Exception:
        return ns
    if not np.isfinite(q):
        return ns
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    if POLICY["suggestive_tier"] and q < SUGGESTIVE_Q:
        return "+"
    return ns


def legend_text(q_name: str = "q") -> str:
    s = f"* {q_name}<0.05  ** {q_name}<0.01  *** {q_name}<0.001"
    if POLICY["suggestive_tier"]:
        s += f"  + {q_name}<0.10 (suggestive)"
    return s


# ── paired tests ─────────────────────────────────────────────────────────────

def _transform(a, b):
    if np.all(a >= 0) and np.all(b >= 0):
        return np.log1p(a), np.log1p(b)
    return a, b


def paired_pvalue(va, vb, allow_small_n_t: bool = True):
    """(p, test_name) for two paired samples.  Wilcoxon signed-rank, or a paired
    t-test on log1p values when the policy asks for it and n is small."""
    from scipy import stats as st
    va = np.asarray(va, float)
    vb = np.asarray(vb, float)
    n = len(va)
    if n < 2 or np.allclose(va, vb):
        return 1.0, "none"
    if allow_small_n_t and POLICY["small_n_paired"] and n <= SMALL_N_PAIRED_MAX:
        a, b = _transform(va, vb)
        if not np.allclose(a, b):
            try:
                p = float(st.ttest_rel(a, b).pvalue)
                if np.isfinite(p):
                    return p, "paired_t_log1p"
            except Exception:
                pass
    try:
        return float(st.wilcoxon(va, vb).pvalue), "wilcoxon"
    except Exception:
        return 1.0, "wilcoxon"


def rm_omnibus_pvalue(aligned):
    """Omnibus test across >=3 matched levels: repeated-measures ANOVA on log1p
    when small and enabled, else Friedman.  Returns (stat, p, name)."""
    from scipy import stats as st
    X = np.column_stack([np.asarray(a, float) for a in aligned])
    n, k = X.shape
    if POLICY["small_n_paired"] and n <= SMALL_N_PAIRED_MAX and n >= 3 and k >= 3:
        Y = np.log1p(X) if np.all(X >= 0) else X
        gm = Y.mean()
        ss_lvl = n * ((Y.mean(0) - gm) ** 2).sum()
        ss_sub = k * ((Y.mean(1) - gm) ** 2).sum()
        ss_tot = ((Y - gm) ** 2).sum()
        ss_err = ss_tot - ss_lvl - ss_sub
        df1, df2 = k - 1, (k - 1) * (n - 1)
        if ss_err > 1e-12:
            F = (ss_lvl / df1) / (ss_err / df2)
            return float(F), float(st.f.sf(F, df1, df2)), "rm_anova_log1p"
    stat, p = st.friedmanchisquare(*[X[:, j] for j in range(k)])
    return float(stat), float(p), "friedman"


def mixed_level_pvalue(level_vals: dict):
    """LRT p for a level effect with a random intercept per animal, using every
    animal that has >=2 levels.  level_vals = {level: {subject_id: value}}.
    Returns (chi2, p, n_subjects) or None if it cannot be fit."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from scipy import stats as st
        rows = [(lvl, sid, float(v)) for lvl, d in level_vals.items() for sid, v in d.items()]
        df = pd.DataFrame(rows, columns=["lvl", "sid", "v"])
        cnt = df.groupby("sid")["lvl"].nunique()
        keep = cnt[cnt >= 2].index
        df = df[df["sid"].isin(keep)]
        if df["sid"].nunique() < 3 or df["lvl"].nunique() < 2:
            return None
        if (df["v"] >= 0).all():
            df["v"] = np.log1p(df["v"])
        if df["v"].std() == 0:
            return 0.0, 1.0, int(df["sid"].nunique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = smf.mixedlm("v ~ C(lvl)", df, groups=df["sid"]).fit(reml=False)
            m0 = smf.mixedlm("v ~ 1", df, groups=df["sid"]).fit(reml=False)
        chi2 = max(2 * (m1.llf - m0.llf), 0.0)
        dfree = df["lvl"].nunique() - 1
        p = float(st.chi2.sf(chi2, dfree))
        if not np.isfinite(p):
            return None
        return float(chi2), p, int(df["sid"].nunique())
    except Exception:
        return None


# ── effect sizes ─────────────────────────────────────────────────────────────

def hedges_g(x, y):
    """(g, lo, hi) independent groups; 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0, 0.0, 0.0
    g = (x.mean() - y.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return float(g), float(g - 1.96 * se), float(g + 1.96 * se)


def paired_dz(va, vb):
    """(dz, lo, hi) standardised paired difference; 95% CI."""
    d = np.asarray(va, float) - np.asarray(vb, float)
    n = len(d)
    if n < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan
    dz = d.mean() / d.std(ddof=1)
    se = np.sqrt(1.0 / n + dz ** 2 / (2 * n))
    return float(dz), float(dz - 1.96 * se), float(dz + 1.96 * se)


def one_sample_d_ci(vals, ref=0.0):
    v = np.asarray(vals, float)
    n = len(v)
    if n < 3 or v.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan
    d = (v.mean() - ref) / v.std(ddof=1)
    se = np.sqrt(1.0 / n + d ** 2 / (2 * n))
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se)


# ── small-n note ─────────────────────────────────────────────────────────────

def min_p_note(group_sizes: dict, paired: bool = False):
    """One short line when the rank test cannot reach p<0.05 at this n, else None."""
    if not POLICY["small_n_note"] or not group_sizes:
        return None
    ns = [n for n in group_sizes.values() if n]
    if not ns:
        return None
    nmin = min(ns)
    if paired and nmin <= 5:
        return (f"n={nmin} pairs: a Wilcoxon test cannot reach p<0.05 (min p={2 / 2 ** nmin:.3f}); "
                "the paired t-test on log usage is used" if POLICY["small_n_paired"]
                else f"n={nmin} pairs: a Wilcoxon test cannot reach p<0.05; enable the paired t-test option.")
    if not paired and len(ns) >= 2 and nmin <= 3:
        return f"n<=3 in a group: rank tests have very low power; read effect sizes and CIs."
    return None
