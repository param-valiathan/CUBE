# Clustering Improvement Options

This is a proposal/discussion document, not a changelog. **Nothing described here has been implemented.**
No pipeline files (`cube_core.py`, `cube.py`, `cube_analyser.py`, `cube_video_explorer.py`) have been
modified as part of writing this document. Each option below is written up with enough detail to be
implemented in a future session, after a decision on which to pursue.

Context: after this session's fixes (chronic-frequency auto-weighting, turned-away-frame exclusion from
bad-frame-fraction stats), a real-data test run produced 19 clusters at 43.6% noise — a solid improvement
over the pre-fix regression (61.9% noise), but still room to improve. Separately, the seed-stability sweep
(6 seeds) surfaced a real gap in the HDBSCAN `min_cluster_size` sweep/selection logic: 2 of 6 seeds
collapsed to 3 clusters because their sweep skipped over the preferred cluster-count range entirely.

---

## 1. Simple ways to improve clustering / reduce noise

Two options already available today, no code change needed at all:

| Try this | Current default | Why it might help |
|---|---|---|
| Raise `train_frac` | `0.3` (real run used 26% → 7291/28235 bins) | More UMAP/HDBSCAN training data gives a more representative density estimate. Try `0.4`–`0.5`. |
| Fix `umap_n_neighbors` higher | auto (17 on the real test run) | Larger neighborhoods preserve more global structure, typically yielding tighter, less fragmented clusters (at some cost to fine local detail). Try 25–40. |
| Nudge `umap_min_dist` up slightly | `0.1` | Spreads points more evenly, which can reduce tight spurious noise pockets. Existing code already warns `<0.05` risks degenerate DBCV — this is testing the other direction. Try `0.12`–`0.15`. Trade-off against cluster tightness, so treat as an A/B experiment rather than an assumed win. |

Two small, additive, **opt-in** cfg keys worth adding (both default to today's exact behavior — zero risk
to any existing run unless a user explicitly sets them):

1. **`hdbscan_cluster_selection_epsilon`** (new, default `0.0` = current behavior unchanged). HDBSCAN's
   standard mechanism for merging small/boundary clusters below a distance threshold into their nearest
   neighboring cluster — this is the most direct, well-established lever for reducing noise specifically.
   Currently `cluster_selection_epsilon` is never passed to `hdbscan.HDBSCAN(...)` in `run_hdbscan`
   (`cube_core.py:1474-1480` and `1492-1498`) at all, so this would be a new capability, not a behavior
   change for anyone who doesn't set it.
2. **`hdbscan_min_samples_override`** (new, default `0` = keep today's exact formula). `min_samples` is
   currently hardcoded as `max(5, mcs // 5)` (`cube_core.py:1477,1495`) with no way to override it. Lower
   `min_samples` is HDBSCAN's other standard noise-reduction knob (it directly relaxes how conservatively
   points get classified as outliers) — exposing an override lets a user trade some cluster purity for
   lower noise without touching the default for anyone else.

**Caveat, not a fix:** DBCV (`relative_validity_`) has been chronically non-finite/degenerate on the test
dataset across every run this session, forcing a silhouette-score fallback for cluster selection. The root
cause of the DBCV degeneracy itself is still unexplained — the existing `[DIAG]` near-duplicate-embedding-
point check has never fired despite the degeneracy persisting. Silhouette is known to structurally favor
fewer, well-separated clusters, yet noise remains meaningfully high anyway — some of the residual noise may
reflect genuinely graded/continuous behavioral transitions in this dataset rather than a fixable artifact of
clustering parameters. None of the options above should be expected to fully close that gap; it remains a
standing open investigation.

---

## 2. Better options for the sweep/selection logic

### Root cause (confirmed against real data)

`run_hdbscan`'s `min_cluster_size` sweep (`cube_core.py:1421-1424`, the `pcts` list) is spaced **linearly
in percentage-of-N units**. This does not guarantee dense sampling in the cluster-count range that
actually matters (`preferred_clusters_lo=12` to `preferred_clusters_hi=30` by default). Instrumented
against the real 3-folder test dataset (6-seed sweep):

| Seed | Cluster counts seen across the sweep | Landed in [12, 30]? |
|---|---|---|
| 42 | 2, 3, 4, 8, 19, 55, 67, 110 | 19 ✓ |
| 43 | 2, 3, 4, 7, 8, 15, 76, 117 | 15 ✓ |
| 44 | 2, 3, 4, 6, 7, 16, 17, 61, 80, 117 | 16, 17 ✓ |
| **45** | 2, 3, 6, **56**, 69, 114 | **none — jumps straight from 6 to 56** |
| **47** | 2, 3, 7, **58**, 75, 110 | **none — jumps straight from 7 to 58** |

For seeds 45/47, the boundary-fallback selector (`cube_core.py:1628-1638`, triggered when no candidate
lands in the preferred range) picked the closest-by-raw-cluster-count candidate — 3 clusters — rather than
anything reasonable, cratering that seed's result and dragging the sweep's mean pairwise ARI down.

### Proposed options (1 + 3 together is the recommended combination)

**1. (Recommended primary) Log-spaced sweep instead of linear-percentage spacing.**
Cluster count vs. `min_cluster_size` typically follows a power-law-ish curve, so log-spacing samples much
more densely at the small-`min_cluster_size`/high-cluster-count end — exactly where the target range
usually lives — while still covering the same overall dynamic range from `pct_lo` to `pct_hi`. Narrowly
scoped: only the `pcts` list-comprehension changes; every downstream candidate-scoring/selection code is
untouched. This is the lowest-risk of the three code-changing options since it only changes *which*
`min_cluster_size` values get tried, not how they get scored or picked.

**2. (Recommended complementary) Adaptive gap-fill when boundary-fallback triggers.**
When the "no candidate in `[pref_lo, pref_hi]`" branch fires (`cube_core.py:1628-1638`), bisect 2-4 extra
`min_cluster_size` values between the two candidates that bracket the gap, fit those, and only fall back to
the boundary pick if none of them land in range either. This directly targets the exact failure mode seen
on seeds 45/47 as a backstop, even in cases where log-spacing (option 1) alone doesn't fully close the gap.
Slightly more invasive than option 1 (adds a conditional extra-fit loop, meaning a small runtime cost only
on runs that actually hit this branch) but still fully isolated to the boundary-fallback code path.

**3. (Cheap, ship independently, zero behavior change) Log a warning whenever boundary-fallback triggers.**
Something like `[VALID-WARN] seed <n>: no min_cluster_size candidate fell inside the preferred cluster-count
range [12,30] — falling back to closest available (N clusters). Treat this seed's result with caution.`
Pure visibility, no algorithm change — makes this failure mode show up in normal run/seed-sweep logs
instead of silently picking a possibly-bad partition. Worth including regardless of whether options 1/2 are
implemented, and cheap enough to ship first/independently.

**4. (Considered, not recommended) Bias the boundary-fallback tie-break toward the *higher*-count
bracketing candidate** (over-clustered, which merge/refinement can safely consolidate) instead of pure
distance-to-boundary. Rejected as the primary fix: it's a band-aid on the selection heuristic rather than
addressing the sweep's actual resolution gap, and it changes existing tie-break behavior for *every* future
boundary-fallback case (larger blast radius than options 1-3, which only add sampling density/visibility).

### Suggested rollout if pursued

Ship option 3 (logging) first — it's a pure add, cannot regress anything, and immediately gives visibility
into how often this actually happens across real runs. Then implement option 1 (log-spaced sweep) and
re-run the same 6-seed real-data test to measure whether it alone closes the gap for seeds like 45/47
before deciding whether option 2's extra complexity is warranted.
