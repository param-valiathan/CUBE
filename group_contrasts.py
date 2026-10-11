"""Group-contrast statistics + themed figures for the Group Predictor tab.

Pure functions (no tkinter, no pyplot state).  Figure builders take a theme dict
``th`` with keys fig_bg / ax_bg / text / spine / muted (CUBE's ``T()``) so every
plot renders in both dark and light mode.  Every analysis returns plain
DataFrames so the caller can save each plot together with its CSV.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

PALETTE = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
           "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"]


# ── helpers ──────────────────────────────────────────────────────────────────

def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def proportions(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def clr(X) -> np.ndarray:
    """Centred log-ratio of row-normalised usage (zeros replaced by half the
    smallest non-zero proportion) - removes the sum-to-one constraint."""
    P = proportions(X)
    nz = P[P > 0]
    pseudo = 0.5 * nz.min() if nz.size else 1e-6
    P = np.where(P > 0, P, pseudo)
    P = P / P.sum(axis=1, keepdims=True)
    L = np.log(P)
    return L - L.mean(axis=1, keepdims=True)


def has_repeats(subjects) -> bool:
    named = [s for s in (subjects or []) if s]
    return len(named) != len(set(named))


def auc_boot_ci(y01, score, n_boot=1000, seed=0):
    """(AUC, lo, hi) with a stratified percentile bootstrap CI."""
    from sklearn.metrics import roc_auc_score
    y01 = np.asarray(y01).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y01)) < 2 or not np.isfinite(score).all():
        return np.nan, np.nan, np.nan
    auc = float(roc_auc_score(y01, score))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y01 == 1)[0], np.where(y01 == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y01[idx], score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, float(lo), float(hi)


def min_detectable_acc(perm_scores, alpha=0.05):
    """Balanced accuracy a model must exceed to reach p<alpha under its own null."""
    if perm_scores is None or len(perm_scores) == 0:
        return np.nan
    return float(np.quantile(np.asarray(perm_scores, dtype=float), 1 - alpha))


# ── omnibus tests ────────────────────────────────────────────────────────────

def _js(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def js_permutation_test(U, labels, n_perm=999, seed=0):
    """Mean pairwise Jensen-Shannon distance between group-mean usage
    distributions; labels permuted for the null.  Returns (stat, p, null)."""
    P = proportions(U)
    labels = np.asarray(labels, dtype=object)
    groups = sorted(set(labels))

    def stat(lab):
        means = [P[lab == g].mean(axis=0) for g in groups]
        return float(np.mean([_js(a, b) for a, b in itertools.combinations(means, 2)]))
    obs = stat(labels)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(labels)) for _ in range(n_perm)])
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1)), null


def permanova(Z, labels, n_perm=999, seed=0):
    """One-way PERMANOVA on Euclidean distances of Z (use CLR coordinates ->
    Aitchison).  Returns dict(F, R2, p, null)."""
    Z = np.asarray(Z, dtype=float)
    labels = np.asarray(labels, dtype=object)
    groups = sorted(set(labels))
    N, a = len(Z), len(groups)
    sst = float(((Z - Z.mean(0)) ** 2).sum())

    def pf(lab):
        ssw = sum(float(((Z[lab == g] - Z[lab == g].mean(0)) ** 2).sum()) for g in groups)
        ssa = sst - ssw
        return (ssa / (a - 1)) / (ssw / (N - a)) if ssw > 0 and N > a else np.nan, ssa
    F, ssa = pf(labels)
    rng = np.random.default_rng(seed)
    null = np.array([pf(rng.permutation(labels))[0] for _ in range(n_perm)])
    p = float((np.sum(null >= F) + 1) / (n_perm + 1)) if np.isfinite(F) else np.nan
    return {"F": float(F), "R2": float(ssa / sst) if sst > 0 else np.nan,
            "p": p, "null": null}


def pcoa(Z):
    """First two principal coordinates of Z + explained variance ratios."""
    Zc = np.asarray(Z, dtype=float)
    Zc = Zc - Zc.mean(0)
    U, S, _ = np.linalg.svd(Zc, full_matrices=False)
    var = S ** 2
    ev = var / var.sum() if var.sum() > 0 else var
    pcs = U[:, :2] * S[:2]
    if pcs.shape[1] < 2:
        pcs = np.hstack([pcs, np.zeros((len(pcs), 2 - pcs.shape[1]))])
    return pcs, ev[:2]


# ── per-feature contrasts ────────────────────────────────────────────────────

def hedges_g(x, y):
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


def _pvalue(x, y, sx, sy, repeated):
    """Mixed model (random intercept per animal) if repeated, else Mann-Whitney."""
    from scipy.stats import mannwhitneyu
    if repeated:
        try:
            import statsmodels.formula.api as smf
            df = pd.DataFrame({"v": np.r_[x, y], "grp": np.r_[np.ones(len(x)), np.zeros(len(y))],
                               "sid": list(sx) + list(sy)})
            if df["v"].std() > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.mixedlm("v ~ grp", df, groups=df["sid"]).fit(reml=True)
                p = float(fit.pvalues["grp"])
                if np.isfinite(p):
                    return p, "mixedlm"
        except Exception:
            pass
    if np.ptp(np.r_[x, y]) == 0:
        return 1.0, "mannwhitney"
    try:
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue), "mannwhitney"
    except Exception:
        return np.nan, "mannwhitney"


def feature_contrasts(U, feat_names, labels, contrasts, subjects=None) -> pd.DataFrame:
    """Effect size (Hedges g + CI), p (mixed model when animals repeat, else
    Mann-Whitney) and BH q per feature, per (A, B) contrast (A minus B)."""
    U = np.asarray(U, float)
    labels = np.asarray(labels, dtype=object)
    subj = np.asarray(subjects if subjects is not None else [""] * len(labels), dtype=object)
    rows = []
    for a, b in contrasts:
        ia, ib = labels == a, labels == b
        rep = has_repeats(list(subj[ia | ib]))
        block = []
        for j, fn in enumerate(feat_names):
            x, y = U[ia, j], U[ib, j]
            g, lo, hi = hedges_g(x, y)
            p, test = _pvalue(x, y, subj[ia], subj[ib], rep)
            block.append({"contrast": f"{a} vs {b}", "feature": fn, "hedges_g": g,
                          "g_lo": lo, "g_hi": hi, "mean_A": float(x.mean()),
                          "mean_B": float(y.mean()), "n_A": int(ia.sum()),
                          "n_B": int(ib.sum()), "test": test, "pval": p})
        q = bh([r["pval"] for r in block])
        for r, qv in zip(block, q):
            r["qval_bh"] = float(qv)
        rows += block
    return pd.DataFrame(rows)


# ── sequence-level features ──────────────────────────────────────────────────

SEQ_FEATURES = ["bouts_per_min", "mean_bout_s", "bout_cv", "usage_entropy_bits",
                "transition_entropy_bits", "dominant_cluster_share"]


def sequence_features(animals) -> pd.DataFrame:
    """Per-animal bout-structure descriptors from a['df'] (label, run_len)."""
    rows = []
    for a in animals:
        df, fps = a["df"], float(a["fps"])
        lab = df["label"].to_numpy()
        dur = df["run_len"].to_numpy(float) / fps
        tot = dur.sum()
        prop = pd.Series(dur).groupby(lab).sum().to_numpy() / tot if tot > 0 else np.array([1.0])
        h_use = float(-(prop[prop > 0] * np.log2(prop[prop > 0])).sum())
        trans = {}
        for s, d in zip(lab[:-1], lab[1:]):
            if s != d:
                trans.setdefault(s, {}).setdefault(d, 0)
                trans[s][d] += 1
        n_tr = sum(sum(v.values()) for v in trans.values())
        h_tr = 0.0
        for s, v in trans.items():
            cnt = np.array(list(v.values()), float)
            pr = cnt / cnt.sum()
            h_tr += (cnt.sum() / n_tr) * float(-(pr * np.log2(pr)).sum())
        rows.append({
            "animal": a["name"], "group": a.get("exp_group_orig", a["exp_group"]),
            "bouts_per_min": len(dur) / (tot / 60.0) if tot > 0 else np.nan,
            "mean_bout_s": float(dur.mean()) if len(dur) else np.nan,
            "bout_cv": float(dur.std() / dur.mean()) if len(dur) > 1 and dur.mean() > 0 else np.nan,
            "usage_entropy_bits": h_use,
            "transition_entropy_bits": h_tr if n_tr else np.nan,
            "dominant_cluster_share": float(prop.max()),
        })
    return pd.DataFrame(rows)


# ── extra classifiers (sklearn-compatible, picklable for loky) ───────────────

from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402


class BalancedShrinkageLDA(LinearDiscriminantAnalysis):
    """Shrinkage (Ledoit-Wolf) LDA with uniform class priors, so an imbalanced
    target-vs-rest split is not dominated by the larger class."""
    _no_sample_weight = True

    def __init__(self, shrinkage="auto"):
        super().__init__(solver="lsqr", shrinkage=shrinkage)

    def fit(self, X, y):
        k = len(np.unique(y))
        self.priors = np.full(k, 1.0 / k)
        return super().fit(X, y)


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """Partial-least-squares discriminant analysis (PLS regression on one-hot
    targets, argmax of the prior-corrected prediction).  coef_ is (n_classes,
    n_features) so the panel's importance code works unchanged."""
    _no_sample_weight = True

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        X = np.asarray(X, float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        Y = np.eye(k)[yi]
        self.prior_ = Y.mean(axis=0)
        p_full = X.shape[1]
        self.keep_ = np.nan_to_num(X.std(axis=0)) > 1e-10      # drop constant columns
        self.coef_ = np.zeros((k, p_full))
        self.pls_ = None
        if self.keep_.any() and k > 1:
            Xk = X[:, self.keep_]
            nc = int(max(1, min(self.n_components, np.linalg.matrix_rank(Xk), X.shape[0] - 1)))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.pls_ = PLSRegression(n_components=nc, scale=False).fit(Xk, Y)
                c = np.asarray(self.pls_.coef_)
                c = c if c.shape == (k, Xk.shape[1]) else c.T
                if np.isfinite(c).all():
                    self.coef_[:, self.keep_] = c
                else:
                    self.pls_ = None
            except Exception:
                self.pls_ = None
        return self

    def _scores(self, X):
        X = np.asarray(X, float)
        k = len(self.classes_)
        if self.pls_ is None:                       # degenerate fit: predict the prior
            return np.tile(self.prior_ - self.prior_ + 1.0 / k, (len(X), 1)) + 1e-9 * self.prior_
        s = np.asarray(self.pls_.predict(X[:, self.keep_])).reshape(len(X), -1)
        s = np.nan_to_num(s, nan=1.0 / k)
        return s - self.prior_ + 1.0 / k           # balance the classes

    def predict_proba(self, X):
        s = np.clip(self._scores(X), 1e-6, None)
        return s / s.sum(axis=1, keepdims=True)

    def decision_function(self, X):
        s = self._scores(X)
        return s[:, 1] - s[:, 0] if s.shape[1] == 2 else s

    def predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]


def pairwise_kappa(preds: dict) -> pd.DataFrame:
    """Cohen's kappa between each pair of classifiers' leave-one-out predictions."""
    from sklearn.metrics import cohen_kappa_score
    names = list(preds)
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            try:
                M.loc[a, b] = 1.0 if a == b else float(cohen_kappa_score(preds[a], preds[b]))
            except Exception:
                pass
    return M


# ── figures ──────────────────────────────────────────────────────────────────

def _fig(w, h, th):
    from matplotlib.figure import Figure
    return Figure(figsize=(w, h), dpi=100, facecolor=th["fig_bg"])


def _style(ax, th):
    ax.set_facecolor(th["ax_bg"])
    for s in ax.spines.values():
        s.set_color(th["spine"])
    ax.tick_params(colors=th["text"], labelsize=8)
    ax.xaxis.label.set_color(th["text"])
    ax.yaxis.label.set_color(th["text"])
    ax.title.set_color(th["text"])
    ax.grid(True, color=th["spine"], alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def group_colors(groups):
    return {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}


def plot_contrast_summary(df: pd.DataFrame, th: dict):
    """AUC (with bootstrap CI) per contrast, annotated with balanced accuracy and q."""
    n = len(df)
    fig = _fig(8.5, max(2.6, 0.55 * n + 1.6), th)
    ax = fig.add_subplot(111)
    _style(ax, th)
    y = np.arange(n)[::-1]
    ok = df["auc"].notna().to_numpy()
    ax.axvline(0.5, color=th["muted"], ls="--", lw=1)
    for yi, (_, r) in zip(y, df.iterrows()):
        if not np.isfinite(r["auc"]):
            continue
        sig = np.isfinite(r.get("qval_bh", np.nan)) and r["qval_bh"] < 0.05
        col = "#59A14F" if sig else "#4E79A7"
        ax.errorbar(r["auc"], yi, xerr=[[r["auc"] - r["auc_lo"]], [r["auc_hi"] - r["auc"]]],
                    fmt="o", color=col, ecolor=col, capsize=3, ms=7)
        ax.text(1.02, yi, f"bal.acc {r['bal_acc']:.2f}  p={r['pval']:.3g}  q={'-' if not np.isfinite(r.get('qval_bh', np.nan)) else format(r['qval_bh'], '.3g')}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=th["text"])
    ax.set_yticks(y)
    ax.set_yticklabels(df["contrast"], fontsize=8)
    ax.set_xlim(0.0, 1.06)
    ax.set_xlabel("ROC AUC (leave-one-animal-out, 95% bootstrap CI)  -  green: q<0.05")
    ax.set_title("Classifier contrasts", fontsize=10, loc="left", color=th["text"])
    fig.subplots_adjust(left=0.22, right=0.62, bottom=0.2, top=0.88)
    return fig


def plot_omnibus(js, perm, pc, ev, labels, th):
    """JS-divergence null, PERMANOVA null and PCoA of animals (Aitchison)."""
    labels = np.asarray(labels, dtype=object)
    cols = group_colors(sorted(set(labels)))
    fig = _fig(11, 3.6, th)
    for k, (title, obs, p, null) in enumerate([
            ("Usage JS distance", js[0], js[1], js[2]),
            ("PERMANOVA pseudo-F", perm["F"], perm["p"], perm["null"])]):
        ax = fig.add_subplot(1, 3, k + 1)
        _style(ax, th)
        ax.hist(null[np.isfinite(null)], bins=30, color="#8a8a8a", alpha=0.75)
        ax.axvline(obs, color="#E15759", lw=2)
        ax.set_title(f"{title}\nobs={obs:.3g}  p={p:.3g}", fontsize=9, loc="left", color=th["text"])
        ax.set_xlabel("permuted statistic")
    ax = fig.add_subplot(1, 3, 3)
    _style(ax, th)
    for g, c in cols.items():
        m = labels == g
        ax.scatter(pc[m, 0], pc[m, 1], s=34, color=c, label=g, edgecolor=th["fig_bg"], lw=0.6)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title(f"Aitchison PCA   R²={perm['R2']:.2f}", fontsize=9, loc="left", color=th["text"])
    leg = ax.legend(fontsize=7, facecolor=th["ax_bg"], edgecolor=th["spine"])
    for t in leg.get_texts():
        t.set_color(th["text"])
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.82, wspace=0.3)
    return fig


def plot_effects(eff: pd.DataFrame, th: dict, title: str, top=20):
    """Heatmap of Hedges g (rows=features, cols=contrasts); * q<0.05, ** q<0.01."""
    import matplotlib.cm as cm
    g = eff.pivot(index="feature", columns="contrast", values="hedges_g")
    q = eff.pivot(index="feature", columns="contrast", values="qval_bh").reindex_like(g)
    order = g.abs().max(axis=1).sort_values(ascending=False).index[:top]
    g, q = g.loc[order], q.loc[order]
    fig = _fig(max(6.5, 1.6 * g.shape[1] + 3.5), max(2.5, 0.28 * len(g) + 1.6), th)
    ax = fig.add_subplot(111)
    ax.set_facecolor(th["ax_bg"])
    lim = max(1.0, float(np.nanmax(np.abs(g.to_numpy()))))
    im = ax.imshow(g.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(g.shape[1]))
    ax.set_xticklabels(g.columns, rotation=20, ha="right", fontsize=8, color=th["text"])
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=8, color=th["text"])
    for s in ax.spines.values():
        s.set_color(th["spine"])
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v, qq = g.iat[i, j], q.iat[i, j]
            star = ("**" if qq < 0.01 else "*" if qq < 0.05
                    else "+" if (np.isfinite(qq) and qq < 0.10) else "")
            ax.text(j, i, f"{v:.1f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 * lim else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Hedges g (A - B)", color=th["text"], fontsize=8)
    cb.ax.tick_params(colors=th["text"], labelsize=7)
    ax.set_title(title, fontsize=9, loc="left", color=th["text"], wrap=True)
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
    return fig


def plot_forest(eff: pd.DataFrame, th: dict, title: str, top: int = 15):
    """Forest plot: effect size with 95% CI per feature, one panel per contrast.
    Needs columns feature, contrast, hedges_g, g_lo, g_hi, qval_bh.  Filled
    markers = q<0.05, ringed = suggestive (q<0.10), grey = not significant."""
    contrasts = list(dict.fromkeys(eff["contrast"]))
    n = len(contrasts)
    rows = min(top, int(eff.groupby("contrast").size().max()))
    fig = _fig(max(5.5, 4.2 * n), max(3.0, 0.32 * rows + 1.8), th)
    for k, c in enumerate(contrasts):
        d = eff[eff["contrast"] == c].copy()
        d = d.reindex(d["hedges_g"].abs().sort_values(ascending=False).index).head(top)
        d = d.iloc[::-1]
        ax = fig.add_subplot(1, n, k + 1)
        _style(ax, th)
        y = np.arange(len(d))
        ax.axvline(0, color=th["muted"], lw=1)
        for yi, (_, r) in zip(y, d.iterrows()):
            q = r.get("qval_bh", np.nan)
            sig = np.isfinite(q) and q < 0.05
            sug = np.isfinite(q) and 0.05 <= q < 0.10
            col = "#E15759" if sig else "#F28E2B" if sug else "#8a8a8a"
            ax.plot([r["g_lo"], r["g_hi"]], [yi, yi], color=col, lw=1.6)
            ax.plot(r["hedges_g"], yi, "o", color=(th["ax_bg"] if sug else col),
                    mec=col, ms=6, mew=1.6)
        ax.set_yticks(y)
        ax.set_yticklabels(d["feature"], fontsize=8)
        ax.set_xlabel("effect size (95% CI)")
        ax.set_title(c, fontsize=9, loc="left", color=th["text"])
    fig.suptitle(title, color=th["text"], fontsize=9, x=0.02, ha="left")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.86, wspace=0.45)
    return fig


def plot_agreement(summary: pd.DataFrame, kappa: pd.DataFrame, chance: float, th: dict):
    """Balanced accuracy per classifier (left) and pairwise prediction agreement (right)."""
    fig = _fig(10, 3.8, th)
    ax = fig.add_subplot(1, 2, 1)
    _style(ax, th)
    cols = [PALETTE[i % len(PALETTE)] for i in range(len(summary))]
    ax.bar(range(len(summary)), summary["bal_acc"], color=cols, alpha=0.9)
    for i, v in enumerate(summary["bal_acc"]):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=th["text"])
    ax.axhline(chance, color=th["muted"], ls="--", lw=1)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["classifier"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("leave-one-out balanced accuracy")
    ax.set_title("Same features, four classifiers", fontsize=10, loc="left", color=th["text"])
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor(th["ax_bg"])
    K = kappa.to_numpy(dtype=float)
    im = ax2.imshow(K, cmap="viridis", vmin=-0.2, vmax=1.0)
    n = len(kappa)
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(kappa.columns, rotation=30, ha="right", fontsize=7, color=th["text"])
    ax2.set_yticklabels(kappa.index, fontsize=7, color=th["text"])
    for i in range(n):
        for j in range(n):
            if np.isfinite(K[i, j]):
                ax2.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if K[i, j] < 0.6 else "black")
    for s in ax2.spines.values():
        s.set_color(th["spine"])
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cb.set_label("Cohen's kappa between predictions", color=th["text"], fontsize=8)
    cb.ax.tick_params(colors=th["text"], labelsize=7)
    ax2.set_title("Prediction agreement", fontsize=10, loc="left", color=th["text"])
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.27, top=0.88, wspace=0.55)
    return fig


def plot_sequence(seq: pd.DataFrame, th: dict):
    groups = sorted(seq["group"].unique())
    cols = group_colors(groups)
    fig = _fig(11, 5.2, th)
    rng = np.random.default_rng(0)
    for k, f in enumerate(SEQ_FEATURES):
        ax = fig.add_subplot(2, 3, k + 1)
        _style(ax, th)
        data = [seq.loc[seq["group"] == g, f].dropna().to_numpy() for g in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(cols[g])
            patch.set_alpha(0.35)
            patch.set_edgecolor(th["text"])
        for el in ("whiskers", "caps", "medians"):
            for ln in bp[el]:
                ln.set_color(th["text"])
        for i, (d, g) in enumerate(zip(data, groups), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=16, color=cols[g], zorder=3)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=7)
        ax.set_title(f.replace("_", " "), fontsize=9, loc="left", color=th["text"])
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.92, wspace=0.28, hspace=0.55)
    return fig

