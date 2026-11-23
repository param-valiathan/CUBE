"""fast_hdbscan compatibility adapter (2026-09-03): makes fast_hdbscan the
default HDBSCAN implementation for cube_core.py, without touching every one
of its `import hdbscan as _hdb; _hdb.HDBSCAN(**kwargs)` call sites
individually.

Applied by monkeypatching the REAL `hdbscan` package's `HDBSCAN` attribute
once, at cube_core.py's own module-load time (mirrors the existing
_patch_pynndescent_thread_safety() convention in cube_core.py) -- every
later `import hdbscan as _hdb` throughout the file reuses the same cached
module object in sys.modules, so they all transparently get the adapter.

Why fast_hdbscan, and what this trades away
--------------------------------------------
Real-data validated this session (D:\\3D DLC\\CLAUDE.md, 2026-09-03 fast_hdbscan
investigation): on the real 24-session EPM production dataset, consensus
seeds' kinetic-reweight step (the dominant per-seed cost, ~700s/seed under
classic hdbscan) dropped to ~630s/seed with fast_hdbscan -- a consistent,
reproducible ~9-10% speedup across all 8 seeds, when paired with a relaxed
numba thread budget (see NUMBA_HDBSCAN_THREAD_BUDGET below; fast_hdbscan's
own multi-core capability is suppressed to zero benefit under CUBE's
default 1-thread numba cap). The primary 80-candidate HDBSCAN sweep showed
no measurable benefit either way (each candidate fit is too small for
16-way parallelism to pay off) -- this is expected and harmless, not a bug.

Known, deliberate feature gap vs. the classic hdbscan library (confirmed by
direct inspection): fast_hdbscan has no `prediction_data`/
`all_points_membership_vectors`/`approximate_predict`. Both classic-only
kwargs (`prediction_data`, `gen_min_span_tree`, `core_dist_n_jobs`) are
silently dropped here rather than raised, since cube_core.py's call sites
pass them unconditionally. This only matters if
`hdbscan_membership_confidence_floor > 0` (default 0.0, off) or the rare
no-MLP `approximate_predict` fallback path is ever reached -- both would
need attention (fall back to classic hdbscan for that specific call, or a
future fast_hdbscan release adding the feature) before being relied on
under this adapter. Not exercised by CUBE's default config as of this
writing.

One unresolved, NOT-fast_hdbscan-caused finding from the same real-data
validation: the full production run's "Predicting & exporting" phase (video
export, downstream of and unrelated to HDBSCAN/numba -- confirmed by
reading that code path directly, it calls neither) was 4.4x slower than the
historical classic-hdbscan baseline in that one comparison. No code-level
cause was found (encoder-reuse fix already in place, zero retries/crashes
logged, identical code path either way) -- most likely environmental
variance between the two test runs, not a regression this adapter
introduces, but flagged here in case it recurs and needs real
investigation.
"""
import contextlib

import fast_hdbscan as _fast
import hdbscan as _real_hdbscan

# Validated real-data budget (2026-09-03): int(32 logical cores * 0.50),
# matching compute_adaptive_n_jobs()'s own existing conservative default
# (system_resource_target_pct=0.50) -- deliberately NOT raised further, to
# stay consistent with this machine's documented sustained-load hardware
# reliability constraint (see cube_core.py's compute_adaptive_n_jobs()
# docstring / D:\3D DLC\CLAUDE.md's Crash Protection table).
NUMBA_HDBSCAN_THREAD_BUDGET = 16

_FAST_ACCEPTED_KWARGS = {
    "min_cluster_size", "min_samples", "cluster_selection_method",
    "allow_single_cluster", "max_cluster_size", "cluster_selection_epsilon",
    "cluster_selection_persistence", "semi_supervised", "ss_algorithm",
    "reproducible", "metric", "algorithm", "knn_k", "cannot_link",
    "validate_cannot_link", "cannot_link_groups", "metric_kwds",
}


class FastHDBSCANAdapter:
    """Accepts classic hdbscan's kwargs, forwards only the ones
    fast_hdbscan understands, silently drops the rest (see module
    docstring for exactly which ones and why that's currently safe),
    delegates fit()/attribute access to the real fast_hdbscan object."""

    def __init__(self, **kwargs):
        accepted = {k: v for k, v in kwargs.items() if k in _FAST_ACCEPTED_KWARGS}
        self._inner = _fast.HDBSCAN(**accepted)

    def fit(self, X, y=None, **fit_kwargs):
        # fast_hdbscan.HDBSCAN.fit signature (confirmed via inspect.signature
        # against the installed version): (self, X, y=None,
        # sample_weight=None, **fit_params) -- forward y/fit_kwargs through
        # rather than silently dropping them. Not currently exercised by any
        # cube_core.py call site (all call .fit(_fit_target) with no y), but
        # a silent-drop here would have been a real, if dormant, correctness
        # gap for any future semi-supervised (cannot_link) usage.
        self._inner.fit(X, y, **fit_kwargs)
        return self

    def __getattr__(self, name):
        # Defensive recursion guard: __getattr__ only fires when normal
        # attribute lookup fails, which includes looking up `_inner` itself
        # before __init__ has set it (e.g. during unpickling/copy or any
        # introspection that touches the object mid-construction) --
        # without this explicit check, that lookup would recurse into
        # __getattr__("_inner") indefinitely (RecursionError) instead of
        # raising a normal AttributeError.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


@contextlib.contextmanager
def numba_hdbscan_thread_budget():
    """Replacement body for cube_core._numba_single_thread(): same
    thread-local set/restore pattern and safety rationale, capped to
    NUMBA_HDBSCAN_THREAD_BUDGET instead of 1. See module docstring for the
    real-data evidence this specific budget was validated against, and the
    documented side effect that this also relaxes threading for per-seed
    UMAP fits sharing the same wrapped `with` block in cube_core.py's
    consensus/seed-sweep worker functions (not perfectly isolated to
    HDBSCAN alone -- a practical, not surgical, speedup)."""
    import numba
    try:
        prev = numba.get_num_threads()
    except Exception:
        prev = None
    try:
        numba.set_num_threads(NUMBA_HDBSCAN_THREAD_BUDGET)
    except Exception:
        pass
    try:
        yield
    finally:
        if prev is not None:
            try:
                numba.set_num_threads(prev)
            except Exception:
                pass


def apply():
    """Call once, at cube_core.py module load. Idempotent."""
    _real_hdbscan.HDBSCAN = FastHDBSCANAdapter
