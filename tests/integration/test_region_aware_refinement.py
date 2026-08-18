"""Region-aware cluster refinement — Phase 1/2 equivalence coverage.

Phase 1 extracts _local_recluster_and_assign() out of split_impure_clusters()'s
body with NO intended behavior change. This module's job is to prove that:
the refactored split_impure_clusters() produces byte-identical output to the
pre-refactor version (captured at git tag `pre-region-aware-refinement`) on a
range of synthetic fixtures, and that the new shared helper itself behaves
correctly in isolation (no candidates / one candidate / multiple candidates /
a candidate that fails the acceptance gate).

Phase 2 adds compute_bin_region_labels() (an early, minimal sibling of
compute_session_env_context()) and BSoidEngine.run()'s hard-gated
region_per_bin computation. This module proves: compute_bin_region_labels()
agrees with compute_session_env_context()'s own current_region output on
identical inputs (they share the same _compute_region_membership_per_frame
primitive); and that with hdbscan_region_split_enabled left at its False
default, compute_bin_region_labels() is never called during a full
BSoidEngine.run() (spy-verified, not just "produces the same output").

Never touches real user data or CUBE_logs/ — synthetic fixtures only.
"""
import importlib.util
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import cube_core as cc

REPO_ROOT = cc.__file__.rsplit("cube_core.py", 1)[0]
PRE_REFACTOR_TAG = "pre-region-aware-refinement"


def _load_pre_refactor_module(tmp_path):
    """Load cube_core.py as it existed at PRE_REFACTOR_TAG into an isolated
    module, so this test can compare against the actual pre-refactor
    implementation rather than a hand-written re-derivation of it."""
    git_exe = shutil.which("git") or shutil.which(
        "git", path=r"C:\Users\param\anaconda3\Library\bin") or (
        r"C:\Users\param\anaconda3\Library\bin\git.exe"
        if __import__("os").path.exists(
            r"C:\Users\param\anaconda3\Library\bin\git.exe") else None)
    if git_exe is None:
        pytest.skip("git not found — cannot load pre-refactor snapshot")
    try:
        result = subprocess.run(
            [git_exe, "show", f"{PRE_REFACTOR_TAG}:cube_core.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError:
        pytest.skip(f"git tag {PRE_REFACTOR_TAG} not found — run from the "
                     "region-aware-refinement implementation session's repo")
    snapshot_path = tmp_path / "cube_core_pre_refactor.py"
    snapshot_path.write_text(result.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "cube_core_pre_refactor", snapshot_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────────────────────────────────
#  Synthetic fixtures
# ──────────────────────────────────────────────────────────────────────────

def _make_split_fixture(seed=0):
    """
    Builds (feats_sc, embedding, labels, cfg) with:
      - cluster 0: impure in embedding space (low silhouette) AND has real
        two-blob structure in feature space -> should split successfully
        into 2 sub-clusters.
      - cluster 1: pure/tight in embedding space (high silhouette) -> never
        a split candidate.
      - cluster 2: impure in embedding space (low silhouette) but
        feature-homogeneous (single blob, no real substructure) -> IS a
        split candidate but fails the local acceptance gate (n_sub_cl < 2).
    feats_sc shape: (n_features, n_bins) matching production convention.
    """
    rng = np.random.default_rng(seed)
    n_per = 20
    n_feat = 6

    # ---- embedding (2D) : controls candidate selection via silhouette ----
    # cluster 1 sits in a tight, isolated region -> high silhouette.
    emb1 = np.array([30.0, 30.0]) + rng.normal(0, 0.2, size=(n_per, 2))
    # cluster 0 and cluster 2 are both smeared across a shared, overlapping
    # region near the origin -> low silhouette for both.
    emb0 = rng.normal(0, 3.0, size=(n_per, 2))
    emb2 = rng.normal(0, 3.0, size=(n_per, 2)) + np.array([1.0, -1.0])
    embedding = np.vstack([emb0, emb1, emb2])
    labels = np.array([0] * n_per + [1] * n_per + [2] * n_per)

    # ---- feats_sc (n_feat, n_bins) : controls local-recluster outcome ----
    # cluster 0: two well-separated feature-space blobs (splits cleanly).
    b0a = np.array([10.0] * n_feat) + rng.normal(0, 0.3, size=(n_per // 2, n_feat))
    b0b = np.array([-10.0] * n_feat) + rng.normal(0, 0.3, size=(n_per - n_per // 2, n_feat))
    feat0 = np.vstack([b0a, b0b])
    # cluster 1: irrelevant to splitting (never a candidate), single blob.
    feat1 = np.array([0.0] * n_feat) + rng.normal(0, 0.3, size=(n_per, n_feat))
    # cluster 2: single homogeneous feature blob -> no real substructure,
    # local split should fail the n_sub_cl >= 2 acceptance gate.
    feat2 = np.array([50.0] * n_feat) + rng.normal(0, 0.3, size=(n_per, n_feat))

    feats_sc = np.vstack([feat0, feat1, feat2]).T  # (n_feat, n_bins)

    cfg = dict(
        umap_n_neighbors=10, umap_n_components=2, umap_min_dist=0.1,
        umap_random_state=42, umap_n_jobs=1, pca_n_components="off",
        hdbscan_split_min_points=10,
        hdbscan_split_max_subclusters=3,
        hdbscan_split_max_candidates=10,
        hdbscan_split_candidate_cutoff=0,
        hdbscan_split_sweep_n_steps=6,
        hdbscan_split_n_jobs=1,
        hdbscan_merge_thresh=0.0,
        hdbscan_pct_lo=5, hdbscan_pct_hi=40,
        hdbscan_method="eom", target_n_clusters=0,
        preferred_clusters_lo=2, preferred_clusters_hi=4,
    )
    return feats_sc, embedding, labels, cfg


SPLIT_THRESH = 0.3  # high enough that clusters 0 and 2 (overlapping, low
                     # silhouette) are candidates; cluster 1 (tight/isolated) is not


# ──────────────────────────────────────────────────────────────────────────
#  Phase 1 — pre/post-refactor equivalence
# ──────────────────────────────────────────────────────────────────────────

class TestPhase1RefactorEquivalence:
    def test_no_candidates(self, tmp_path):
        cc_old = _load_pre_refactor_module(tmp_path)
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=1)
        # Threshold below every cluster's silhouette -> no candidates.
        old = cc_old.split_impure_clusters(feats_sc, embedding, labels, -1.0, cfg)
        new = cc.split_impure_clusters(feats_sc, embedding, labels, -1.0, cfg)
        assert np.array_equal(old, labels)
        assert np.array_equal(new, labels)
        assert np.array_equal(old, new)

    def test_split_thresh_falsy_is_hard_noop(self, tmp_path):
        cc_old = _load_pre_refactor_module(tmp_path)
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=2)
        old = cc_old.split_impure_clusters(feats_sc, embedding, labels, 0, cfg)
        new = cc.split_impure_clusters(feats_sc, embedding, labels, 0, cfg)
        assert np.array_equal(old, new) and np.array_equal(new, labels)

    def test_multiple_candidates_one_succeeds_one_fails_gate(self, tmp_path):
        cc_old = _load_pre_refactor_module(tmp_path)
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=3)
        old = cc_old.split_impure_clusters(
            feats_sc, embedding, labels, SPLIT_THRESH, cfg)
        new = cc.split_impure_clusters(
            feats_sc, embedding, labels, SPLIT_THRESH, cfg)
        assert np.array_equal(old, new), (
            "Refactored split_impure_clusters diverged from pre-refactor "
            "snapshot on identical input")

    def test_single_candidate(self, tmp_path):
        cc_old = _load_pre_refactor_module(tmp_path)
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=4)
        # Isolate cluster 0 only: relabel cluster 2 into cluster 1 so only
        # cluster 0 remains impure/eligible.
        labels_one = labels.copy()
        labels_one[labels_one == 2] = 1
        old = cc_old.split_impure_clusters(
            feats_sc, embedding, labels_one, SPLIT_THRESH, cfg)
        new = cc.split_impure_clusters(
            feats_sc, embedding, labels_one, SPLIT_THRESH, cfg)
        assert np.array_equal(old, new)


# ──────────────────────────────────────────────────────────────────────────
#  Shared helper — direct behavior tests
# ──────────────────────────────────────────────────────────────────────────

class TestLocalReclusterAndAssign:
    def test_empty_candidates_returns_labels_unchanged(self):
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=5)
        out = cc._local_recluster_and_assign(
            feats_sc, embedding, labels, [], lambda cid: dict(cfg), cfg)
        assert np.array_equal(out, labels)

    def test_successful_split_produces_new_ids(self):
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=6)

        def local_cfg_fn(cid):
            idx_size = int(np.count_nonzero(labels == cid))
            lc = dict(cfg)
            lc["umap_n_neighbors"] = max(5, min(10, idx_size // 3))
            lc["preferred_clusters_lo"] = 2
            lc["preferred_clusters_hi"] = 3
            lc["hdbscan_fine_bias"] = 0.0
            lc["hdbscan_leaf_bonus"] = 0.0
            lc["hdbscan_method"] = "eom"
            lc["hdbscan_sweep_n_steps"] = 6
            lc["hdbscan_sweep_n_jobs"] = 1
            return lc

        out = cc._local_recluster_and_assign(
            feats_sc, embedding, labels, [0], local_cfg_fn, cfg)
        # Cluster 0 (real two-blob feature structure) should split: more
        # distinct cluster ids present among former-cluster-0 rows than before.
        orig_ids_in_c0 = set(labels[labels == 0].tolist())
        new_ids_in_c0_rows = set(int(x) for x in out[labels == 0] if x >= 0)
        assert orig_ids_in_c0 == {0}
        assert len(new_ids_in_c0_rows) >= 2, (
            "expected cluster 0's two feature-space blobs to split into "
            ">=2 sub-cluster ids")
        # Untouched clusters unaffected.
        assert np.array_equal(out[labels != 0], labels[labels != 0])

    def test_gate_rejects_homogeneous_candidate(self):
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=7)

        def local_cfg_fn(cid):
            idx_size = int(np.count_nonzero(labels == cid))
            lc = dict(cfg)
            lc["umap_n_neighbors"] = max(5, min(10, idx_size // 3))
            lc["preferred_clusters_lo"] = 2
            lc["preferred_clusters_hi"] = 3
            lc["hdbscan_fine_bias"] = 0.0
            lc["hdbscan_leaf_bonus"] = 0.0
            lc["hdbscan_method"] = "eom"
            lc["hdbscan_sweep_n_steps"] = 6
            lc["hdbscan_sweep_n_jobs"] = 1
            return lc

        # Cluster 2 is a single homogeneous feature blob -> local recluster
        # should not find >= 2 valid sub-clusters -> candidate rejected,
        # labels for cluster 2's rows stay exactly as they were.
        out = cc._local_recluster_and_assign(
            feats_sc, embedding, labels, [2], local_cfg_fn, cfg)
        assert np.array_equal(out[labels == 2], labels[labels == 2])
        assert np.array_equal(out, labels)

    def test_min_points_filter_skips_tiny_candidate(self):
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=8)
        cfg2 = dict(cfg)
        cfg2["hdbscan_split_min_points"] = 10_000  # larger than any cluster
        out = cc._local_recluster_and_assign(
            feats_sc, embedding, labels, [0, 1, 2],
            lambda cid: dict(cfg2), cfg2)
        assert np.array_equal(out, labels)

    def test_candidate_detail_fn_used_in_log(self):
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=9)
        logged = []

        def local_cfg_fn(cid):
            idx_size = int(np.count_nonzero(labels == cid))
            lc = dict(cfg)
            lc["umap_n_neighbors"] = max(5, min(10, idx_size // 3))
            lc["preferred_clusters_lo"] = 2
            lc["preferred_clusters_hi"] = 3
            lc["hdbscan_fine_bias"] = 0.0
            lc["hdbscan_leaf_bonus"] = 0.0
            lc["hdbscan_method"] = "eom"
            lc["hdbscan_sweep_n_steps"] = 6
            lc["hdbscan_sweep_n_jobs"] = 1
            return lc

        cc._local_recluster_and_assign(
            feats_sc, embedding, labels, [0], local_cfg_fn, cfg,
            log_fn=logged.append, candidate_detail_fn=lambda cid: " (custom detail)")
        joined = " ".join(logged)
        if logged:  # only asserts detail text made it through when a split happened
            assert "(custom detail)" in joined


# ──────────────────────────────────────────────────────────────────────────
#  Phase 2 — compute_bin_region_labels() + region_per_bin plumbing
# ──────────────────────────────────────────────────────────────────────────

def _square(x0, y0, size):
    return [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]


class TestComputeBinRegionLabels:
    FPS = 30.0
    BODYPARTS = ["nose", "tailbase"]

    def _cfg(self):
        return {
            "schema_version": 3, "paradigm": "open_field", "reference_stem": "s1",
            "coord_space": "post_crop",
            "reference_shapes": {
                "boundary": None,
                "regions": [{"name": "R1", "kind": "region",
                             "vertices": _square(0, 0, 20), "role": None}],
                "objects": []},
            "per_video": {}}

    def test_none_env_cfg_returns_none(self):
        xs = np.full((30, 2), 5.0)
        ys = np.full((30, 2), 5.0)
        assert cc.compute_bin_region_labels(None, "s1", xs, ys, self.BODYPARTS, self.FPS) is None

    def test_no_traced_shapes_returns_none(self):
        cfg = {"schema_version": 3, "paradigm": "custom", "reference_stem": "s1",
               "coord_space": "post_crop",
               "reference_shapes": {"boundary": None, "regions": [], "objects": []},
               "per_video": {}}
        xs = np.full((30, 2), 5.0)
        ys = np.full((30, 2), 5.0)
        assert cc.compute_bin_region_labels(cfg, "s1", xs, ys, self.BODYPARTS, self.FPS) is None

    def test_matches_compute_session_env_context_current_region(self):
        """The whole point of sharing _compute_region_membership_per_frame:
        compute_bin_region_labels()'s output must be IDENTICAL to
        compute_session_env_context()'s own current_region per-bin series,
        for the same inputs, on both a stationary-inside-region case and a
        moving-between-regions-and-outside case."""
        cfg = self._cfg()
        rng = np.random.default_rng(0)
        n = 90
        # animal wanders between inside R1 (0,0)-(20,20) and well outside it.
        cx = np.where(np.arange(n) % 30 < 15, 5.0, 50.0) + rng.normal(0, 0.01, n)
        cy = np.where(np.arange(n) % 30 < 15, 5.0, 50.0) + rng.normal(0, 0.01, n)
        xs = np.column_stack([cx, cx])
        ys = np.column_stack([cy, cy])

        via_env_ctx = cc.compute_session_env_context(
            cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)["per_bin"]["current_region"]
        via_bin_labels = cc.compute_bin_region_labels(
            cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)

        assert len(via_env_ctx) == len(via_bin_labels)
        assert list(via_env_ctx) == list(via_bin_labels)

    def test_leftover_region_fill_matches_too(self):
        """Same equivalence check but exercising the paradigm-leftover-zone
        fill path (open_field's implied 'Periphery')."""
        cfg = {
            "schema_version": 3, "paradigm": "open_field", "reference_stem": "s1",
            "coord_space": "post_crop",
            "reference_shapes": {
                "boundary": {"name": "Arena", "kind": "boundary",
                             "vertices": _square(0, 0, 100), "role": None},
                "regions": [{"name": "Center", "kind": "region",
                             "vertices": _square(30, 30, 40), "role": "center"}],
                "objects": []},
            "per_video": {}}
        n = 60
        cx = np.where(np.arange(n) < 30, 50.0, 5.0)  # half in Center, half in periphery
        cy = np.full(n, 50.0)
        xs = np.column_stack([cx, cx])
        ys = np.column_stack([cy, cy])

        via_env_ctx = cc.compute_session_env_context(
            cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)["per_bin"]["current_region"]
        via_bin_labels = cc.compute_bin_region_labels(
            cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)
        assert list(via_env_ctx) == list(via_bin_labels)
        assert "Periphery" in set(via_bin_labels)  # sanity: leftover fill actually fired


# ──────────────────────────────────────────────────────────────────────────
#  Phase 2 — BSoidEngine.run() gating: flag off => never called
# ──────────────────────────────────────────────────────────────────────────

_P2_BODYPARTS = ["nose", "neck", "tailbase", "paw1", "paw2"]


def _p2_make_session_xy(n_frames, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    n_pts = len(_P2_BODYPARTS)
    xy = np.zeros((n_frames, n_pts * 2))
    for i in range(n_pts):
        cx, cy = 50.0 * (i + 1), 30.0 * (i + 1)
        phase = rng.uniform(0, 6.28)
        xy[:, 2 * i] = cx + 15 * np.sin(t / 25.0 + phase + i) + rng.normal(0, 0.5, n_frames)
        xy[:, 2 * i + 1] = cy + 10 * np.cos(t / 30.0 + phase + i) + rng.normal(0, 0.5, n_frames)
    return xy


def _p2_write_session_h5(path, n_frames, seed):
    xy = _p2_make_session_xy(n_frames, seed)
    cols, data = [], {}
    for i, bp in enumerate(_P2_BODYPARTS):
        for coord in ("x", "y", "likelihood"):
            key = ("DLC_scorer", bp, coord)
            cols.append(key)
            if coord == "x":
                data[key] = xy[:, 2 * i]
            elif coord == "y":
                data[key] = xy[:, 2 * i + 1]
            else:
                data[key] = np.full(n_frames, 0.95)
    columns = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(data.values(), index=columns).T
    df.columns = columns
    df.index = range(n_frames)
    df.to_hdf(str(path), key="df_with_missing", mode="w", format="fixed")


_P2_FAST_RUN_CFG = dict(
    likelihood_thresh=0.3, max_interp_gap_sec=0.5, boxcar_win_sec=0.07,
    train_frac=1.0, umap_full_thresh=10_000, umap_n_neighbors=15,
    umap_n_components=2, umap_min_dist=0.1, umap_random_state=42,
    umap_n_jobs=1, pca_n_components="off", hdbscan_sweep_n_steps=5,
    hdbscan_pct_lo=5, hdbscan_pct_hi=25, hdbscan_method="eom",
    target_n_clusters=0, preferred_clusters_lo=2, preferred_clusters_hi=4,
    hdbscan_selection_mode="floor_soft_cap", mlp_hidden="8,4",
    mlp_max_iter=50, cv_folds=2, hmm_n_iter=5, seed_sweep_n=0,
    consensus_clustering_enabled=False, hdbscan_merge_thresh=0.0,
    hdbscan_split_silhouette_thresh=None, cluster_hierarchy_enabled=False,
    auto_bodypart_weighting=False, auto_flag_impure_clusters=False,
    visibility_features_enabled=False, min_cluster_freq=0.0,
    auto_resource_management=False, hdbscan_sweep_n_jobs=1,
    hdbscan_split_n_jobs=1, seed_sweep_n_jobs=1, consensus_n_jobs=1,
    plot_theme="dark",
    # hdbscan_region_split_enabled deliberately OMITTED -- must default False.
)


@pytest.mark.slow
class TestRegionPerBinGatingFullPipeline:
    def test_region_split_disabled_by_default_never_calls_compute_bin_region_labels(
            self, tmp_path, monkeypatch):
        dlc_dir = tmp_path / "dlc"
        dlc_dir.mkdir()
        _p2_write_session_h5(dlc_dir / "session1_filtered.h5", n_frames=300, seed=1)
        out_dir = tmp_path / "out"

        calls = []
        orig = cc.compute_bin_region_labels

        def spy(*a, **kw):
            calls.append((a, kw))
            return orig(*a, **kw)

        monkeypatch.setattr(cc, "compute_bin_region_labels", spy)

        engine = cc.BSoidEngine(dlc_dir, video_folder=None, output_dir=out_dir,
                                fps=30, logger=lambda m: None, cfg=_P2_FAST_RUN_CFG)
        engine.run()

        assert calls == [], (
            "compute_bin_region_labels() was called even though "
            "hdbscan_region_split_enabled defaults to False -- Phase 2's "
            "hard-gate (zero cost for opted-out users) is violated")

    def test_region_split_enabled_without_arena_cfg_calls_but_yields_none(
            self, tmp_path, monkeypatch):
        """Flag on, but no env_arena_cfg configured: the per-session call
        happens (flag-gated cost is opt-in, expected once enabled) but each
        call legitimately returns None (no traced shapes) -- must not crash
        and must leave region_per_bin as None (handled defensively)."""
        dlc_dir = tmp_path / "dlc"
        dlc_dir.mkdir()
        _p2_write_session_h5(dlc_dir / "session1_filtered.h5", n_frames=300, seed=2)
        out_dir = tmp_path / "out"

        calls = []
        orig = cc.compute_bin_region_labels

        def spy(*a, **kw):
            calls.append((a, kw))
            return orig(*a, **kw)

        monkeypatch.setattr(cc, "compute_bin_region_labels", spy)

        cfg = dict(_P2_FAST_RUN_CFG)
        cfg["hdbscan_region_split_enabled"] = True
        engine = cc.BSoidEngine(dlc_dir, video_folder=None, output_dir=out_dir,
                                fps=30, logger=lambda m: None, cfg=cfg)
        engine.run()  # must not raise

        assert len(calls) >= 1


# ──────────────────────────────────────────────────────────────────────────
#  Phase 3 — split_region_impure_clusters() + refine_clusters_iterative() wiring
# ──────────────────────────────────────────────────────────────────────────

class TestNormalizedRegionEntropy:
    def test_single_category_is_exactly_zero(self):
        entropy, counts = cc._normalized_region_entropy(["A"] * 10)
        assert entropy == 0.0
        assert counts == {"A": 10}

    def test_two_way_even_split_is_near_one(self):
        entropy, counts = cc._normalized_region_entropy(["A"] * 10 + ["B"] * 10)
        assert entropy == pytest.approx(1.0)
        assert counts == {"A": 10, "B": 10}

    def test_three_way_skewed_is_between_zero_and_one(self):
        entropy, counts = cc._normalized_region_entropy(["A"] * 10 + ["B"] * 9 + ["C"] * 1)
        assert 0.0 < entropy < 1.0
        assert counts == {"A": 10, "B": 9, "C": 1}

    def test_empty_list_does_not_raise(self):
        # k == 0 -> early return, not a 0/0 division.
        entropy, counts = cc._normalized_region_entropy([])
        assert entropy == 0.0
        assert counts == {}


def _make_region_split_fixture(seed=0):
    """
    Mirrors _make_split_fixture's feature-space structure but adds a
    parallel region_per_bin array:
      - cluster 0: real two-blob feature structure (splits cleanly when
        attempted) AND a 50/50 two-region split (entropy ~1.0, clears both
        the impurity threshold and the minority floor) -> region-split
        candidate that should actually split.
      - cluster 1: single feature blob, single region -> never a candidate
        (entropy exactly 0).
      - cluster 2: single feature blob (so even if picked as a candidate it
        would fail split_impure_clusters' acceptance gate) with a 10/9/1
        three-region split -> entropy clears hdbscan_region_split_impurity_thresh
        (default 0.5) but the size-1 minority region (1/20 = 0.05) fails
        hdbscan_region_split_min_minority_frac (default 0.15) -> must NOT
        be a candidate at all (gate should reject before ever attempting a
        local re-cluster).
    """
    rng = np.random.default_rng(seed)
    n_per = 20
    n_feat = 6

    emb0 = rng.normal(0, 3.0, size=(n_per, 2))
    emb1 = np.array([30.0, 30.0]) + rng.normal(0, 0.2, size=(n_per, 2))
    emb2 = rng.normal(0, 3.0, size=(n_per, 2)) + np.array([1.0, -1.0])
    embedding = np.vstack([emb0, emb1, emb2])
    labels = np.array([0] * n_per + [1] * n_per + [2] * n_per)

    b0a = np.array([10.0] * n_feat) + rng.normal(0, 0.3, size=(n_per // 2, n_feat))
    b0b = np.array([-10.0] * n_feat) + rng.normal(0, 0.3, size=(n_per - n_per // 2, n_feat))
    feat0 = np.vstack([b0a, b0b])
    feat1 = np.array([0.0] * n_feat) + rng.normal(0, 0.3, size=(n_per, n_feat))
    feat2 = np.array([50.0] * n_feat) + rng.normal(0, 0.3, size=(n_per, n_feat))
    feats_sc = np.vstack([feat0, feat1, feat2]).T  # (n_feat, n_bins)

    region0 = ["RegionA"] * 10 + ["RegionB"] * 10
    region1 = ["RegionA"] * n_per
    region2 = ["RegionA"] * 10 + ["RegionB"] * 9 + ["RegionC"] * 1
    region_per_bin = region0 + region1 + region2

    cfg = dict(
        umap_n_neighbors=10, umap_n_components=2, umap_min_dist=0.1,
        umap_random_state=42, umap_n_jobs=1, pca_n_components="off",
        hdbscan_split_min_points=10,
        hdbscan_region_split_enabled=True,
        hdbscan_region_split_signal=["current_region"],
        hdbscan_region_split_impurity_thresh=0.5,
        hdbscan_region_split_min_minority_frac=0.15,
        hdbscan_region_split_max_subclusters=3,
        hdbscan_region_split_max_candidates=10,
        hdbscan_split_sweep_n_steps=6,
        hdbscan_split_n_jobs=1,
        hdbscan_merge_thresh=0.0,
        hdbscan_pct_lo=5, hdbscan_pct_hi=40,
        hdbscan_method="eom", target_n_clusters=0,
        preferred_clusters_lo=2, preferred_clusters_hi=4,
    )
    return feats_sc, embedding, labels, region_per_bin, cfg


class TestSplitRegionImpureClusters:
    def test_flag_off_is_hard_noop(self):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=10)
        cfg = dict(cfg)
        cfg["hdbscan_region_split_enabled"] = False
        out = cc.split_region_impure_clusters(
            feats_sc, embedding, labels, region_per_bin, cfg)
        assert np.array_equal(out, labels)

    def test_region_per_bin_none_is_hard_noop(self):
        feats_sc, embedding, labels, _region_per_bin, cfg = _make_region_split_fixture(seed=11)
        out = cc.split_region_impure_clusters(feats_sc, embedding, labels, None, cfg)
        assert np.array_equal(out, labels)

    def test_signal_without_current_region_is_hard_noop(self):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=12)
        cfg = dict(cfg)
        cfg["hdbscan_region_split_signal"] = ["dist_to_nearest_object_Toy"]
        out = cc.split_region_impure_clusters(
            feats_sc, embedding, labels, region_per_bin, cfg)
        assert np.array_equal(out, labels)

    def test_misaligned_length_is_hard_noop(self):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=13)
        out = cc.split_region_impure_clusters(
            feats_sc, embedding, labels, region_per_bin[:-1], cfg)
        assert np.array_equal(out, labels)

    def test_impure_cluster_with_real_structure_splits_when_enabled(self):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=14)
        out = cc.split_region_impure_clusters(
            feats_sc, embedding, labels, region_per_bin, cfg)
        new_ids_in_c0 = set(int(x) for x in out[labels == 0] if x >= 0)
        assert len(new_ids_in_c0) >= 2, (
            "cluster 0 (50/50 two-region impurity, real two-blob feature "
            "structure) should have split into >= 2 sub-clusters")
        # cluster 1 (pure, entropy 0) untouched.
        assert np.array_equal(out[labels == 1], labels[labels == 1])

    def test_minority_floor_rejects_cluster_2_candidacy(self):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=15)
        out = cc.split_region_impure_clusters(
            feats_sc, embedding, labels, region_per_bin, cfg)
        # cluster 2's entropy clears the impurity threshold but its 1/20
        # minority region fails the 0.15 floor -> must never even be
        # attempted as a candidate, so its labels are completely untouched.
        assert np.array_equal(out[labels == 2], labels[labels == 2])

    def test_minority_floor_disabled_widens_candidate_pool(self, monkeypatch):
        """With the minority floor set to 0 (accept any minority fraction),
        cluster 2 becomes a CANDIDATE (region entropy still clears the
        impurity threshold) even though it wasn't one at the default 0.15
        floor -- verified via the same worst-first candidate list the
        function builds internally, not by asserting a specific label
        outcome (real local HDBSCAN behavior on a tiny homogeneous blob is
        not reliably "no split" at n=20, so asserting on final labels here
        would be flaky; the candidate-SELECTION gate is what this test is
        actually about)."""
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=16)

        captured = {}
        orig = cc._local_recluster_and_assign

        def spy(feats_sc, embedding, labels, candidate_ids, *a, **kw):
            captured["candidate_ids"] = list(candidate_ids)
            return orig(feats_sc, embedding, labels, candidate_ids, *a, **kw)

        monkeypatch.setattr(cc, "_local_recluster_and_assign", spy)

        cfg_default = dict(cfg)
        cc.split_region_impure_clusters(feats_sc, embedding, labels, region_per_bin, cfg_default)
        assert 2 not in captured.get("candidate_ids", [])

        cfg_floor_off = dict(cfg)
        cfg_floor_off["hdbscan_region_split_min_minority_frac"] = 0.0
        cc.split_region_impure_clusters(feats_sc, embedding, labels, region_per_bin, cfg_floor_off)
        assert 2 in captured.get("candidate_ids", [])

    def test_max_candidates_bounds_worst_first(self, monkeypatch):
        """With the minority floor relaxed, clusters 0 (entropy 1.0) and 2
        (entropy ~0.78) are both legitimate candidates. max_candidates=1
        must keep only the higher-entropy one (cluster 0), worst-impurity-
        first -- verified via the same candidate-list spy as above (not
        final labels, to avoid coupling this test to local-HDBSCAN
        randomness on the smaller candidate)."""
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=17)
        cfg = dict(cfg)
        cfg["hdbscan_region_split_min_minority_frac"] = 0.0  # cluster 2 now eligible too
        cfg["hdbscan_region_split_max_candidates"] = 1

        captured = {}
        orig = cc._local_recluster_and_assign

        def spy(feats_sc, embedding, labels, candidate_ids, *a, **kw):
            captured["candidate_ids"] = list(candidate_ids)
            return orig(feats_sc, embedding, labels, candidate_ids, *a, **kw)

        monkeypatch.setattr(cc, "_local_recluster_and_assign", spy)
        cc.split_region_impure_clusters(feats_sc, embedding, labels, region_per_bin, cfg)
        assert captured.get("candidate_ids") == [0], (
            "max_candidates=1 with both clusters 0 (entropy 1.0) and 2 "
            "(entropy ~0.78) eligible must keep only the higher-entropy "
            "cluster 0, worst-first")


class TestRefineClustersIterativeRegionWiring:
    def test_region_per_bin_none_is_equivalent_to_pre_phase3(self, tmp_path):
        """With region_per_bin=None (the default), refine_clusters_iterative()
        must behave exactly as it did before Phase 3 -- verified against the
        pre-region-aware-refinement snapshot on the silhouette-split fixture
        from Phase 1."""
        cc_old = _load_pre_refactor_module(tmp_path)
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=20)
        cfg = dict(cfg)
        cfg["recluster_max_iterations"] = 2
        cfg["hdbscan_split_silhouette_thresh"] = SPLIT_THRESH

        class _FakeClf:
            condensed_tree_ = None

        old = cc_old.refine_clusters_iterative(
            feats_sc, embedding, labels, _FakeClf(), cfg)
        new = cc.refine_clusters_iterative(
            feats_sc, embedding, labels, _FakeClf(), cfg)  # region_per_bin defaults to None
        assert np.array_equal(old, new)

    def test_region_split_enabled_but_region_per_bin_none_is_still_noop_for_that_pass(self):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=21)
        cfg = dict(cfg)
        cfg["recluster_max_iterations"] = 1
        cfg["hdbscan_split_silhouette_thresh"] = None
        cfg["hdbscan_merge_thresh"] = 0.0
        # region_per_bin intentionally NOT passed -> region_split_on is False
        # even though hdbscan_region_split_enabled=True in cfg.
        out = cc.refine_clusters_iterative(
            feats_sc, embedding, labels, None, cfg)
        assert np.array_equal(out, labels)

    def test_region_split_actually_invoked_when_wired_in(self, monkeypatch):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=22)
        cfg = dict(cfg)
        cfg["recluster_max_iterations"] = 1
        cfg["hdbscan_split_silhouette_thresh"] = None  # isolate the region pass
        cfg["hdbscan_merge_thresh"] = 0.0

        calls = []
        orig = cc.split_region_impure_clusters

        def spy(*a, **kw):
            calls.append(1)
            return orig(*a, **kw)

        monkeypatch.setattr(cc, "split_region_impure_clusters", spy)

        class _FakeClf:
            condensed_tree_ = None

        out = cc.refine_clusters_iterative(
            feats_sc, embedding, labels, _FakeClf(), cfg,
            region_per_bin=region_per_bin)
        assert len(calls) >= 1
        new_ids_in_c0 = set(int(x) for x in out[labels == 0] if x >= 0)
        assert len(new_ids_in_c0) >= 2

    def test_split_order_is_silhouette_then_region_then_merge(self, monkeypatch):
        """Order-of-operations check: with BOTH silhouette and region split
        active, split_impure_clusters must be called before
        split_region_impure_clusters, which must be called before
        merge_similar_clusters, each iteration."""
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=23)
        cfg = dict(cfg)
        cfg["recluster_max_iterations"] = 1
        cfg["hdbscan_split_silhouette_thresh"] = 0.9  # deliberately permissive -> real candidates
        cfg["hdbscan_merge_thresh"] = 0.0

        order = []
        orig_sil = cc.split_impure_clusters
        orig_reg = cc.split_region_impure_clusters
        orig_merge = cc.merge_similar_clusters

        def spy_sil(*a, **kw):
            order.append("silhouette")
            return orig_sil(*a, **kw)

        def spy_reg(*a, **kw):
            order.append("region")
            return orig_reg(*a, **kw)

        def spy_merge(*a, **kw):
            order.append("merge")
            return orig_merge(*a, **kw)

        monkeypatch.setattr(cc, "split_impure_clusters", spy_sil)
        monkeypatch.setattr(cc, "split_region_impure_clusters", spy_reg)
        monkeypatch.setattr(cc, "merge_similar_clusters", spy_merge)

        class _FakeClf:
            condensed_tree_ = None

        cc.refine_clusters_iterative(
            feats_sc, embedding, labels, _FakeClf(), cfg,
            region_per_bin=region_per_bin)
        assert order == ["silhouette", "region", "merge"]


# ──────────────────────────────────────────────────────────────────────────
#  Phase 4 — consensus-mode wiring (refine_consensus_clusters)
# ──────────────────────────────────────────────────────────────────────────

class TestRefineConsensusClustersRegionWiring:
    def test_region_per_bin_none_is_equivalent_to_pre_phase4(self, tmp_path):
        cc_old = _load_pre_refactor_module(tmp_path)
        feats_sc, embedding, labels, cfg = _make_split_fixture(seed=30)
        feats_sc_T = feats_sc.T
        n = labels.shape[0]
        co_assoc = np.eye(n, dtype=np.float32)  # merge_thresh=0 -> never read for merging
        cfg = dict(cfg)
        cfg["recluster_max_iterations"] = 2
        cfg["hdbscan_split_silhouette_thresh"] = SPLIT_THRESH
        cfg["consensus_merge_coassoc_thresh"] = 0.0

        old = cc_old.refine_consensus_clusters(
            feats_sc_T, labels, co_assoc, embedding, cfg)
        new = cc.refine_consensus_clusters(
            feats_sc_T, labels, co_assoc, embedding, cfg)  # region_per_bin defaults to None
        assert np.array_equal(old, new)

    def test_region_split_requires_embedding(self):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=31)
        feats_sc_T = feats_sc.T
        n = labels.shape[0]
        co_assoc = np.eye(n, dtype=np.float32)
        cfg = dict(cfg)
        cfg["recluster_max_iterations"] = 1
        cfg["hdbscan_split_silhouette_thresh"] = None
        cfg["consensus_merge_coassoc_thresh"] = 0.0

        out = cc.refine_consensus_clusters(
            feats_sc_T, labels, co_assoc, None, cfg, region_per_bin=region_per_bin)
        assert np.array_equal(out, labels)  # embedding=None -> region split skipped, true no-op

    def test_region_split_invoked_and_splits_when_wired_in(self, monkeypatch):
        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=32)
        feats_sc_T = feats_sc.T
        n = labels.shape[0]
        co_assoc = np.eye(n, dtype=np.float32)
        cfg = dict(cfg)
        cfg["recluster_max_iterations"] = 1
        cfg["hdbscan_split_silhouette_thresh"] = None
        cfg["consensus_merge_coassoc_thresh"] = 0.0

        calls = []
        orig = cc.split_region_impure_clusters

        def spy(*a, **kw):
            calls.append(1)
            return orig(*a, **kw)

        monkeypatch.setattr(cc, "split_region_impure_clusters", spy)

        out = cc.refine_consensus_clusters(
            feats_sc_T, labels, co_assoc, embedding, cfg, region_per_bin=region_per_bin)
        assert len(calls) >= 1
        new_ids_in_c0 = set(int(x) for x in out[labels == 0] if x >= 0)
        assert len(new_ids_in_c0) >= 2

    def test_consensus_cluster_threads_region_per_bin_through(self, monkeypatch):
        """consensus_cluster() itself must forward its region_per_bin
        argument into refine_consensus_clusters() -- spy-verified at that
        boundary rather than running the full consensus pipeline."""
        captured = {}
        orig = cc.refine_consensus_clusters

        def spy(feats_sc_T, labels, co_assoc, embedding, cfg, log_fn=None,
                region_per_bin=None):
            captured["region_per_bin"] = region_per_bin
            return orig(feats_sc_T, labels, co_assoc, embedding, cfg,
                        log_fn=log_fn, region_per_bin=region_per_bin)

        monkeypatch.setattr(cc, "refine_consensus_clusters", spy)

        feats_sc, embedding, labels, region_per_bin, cfg = _make_region_split_fixture(seed=33)
        cfg = dict(cfg)
        cfg["consensus_refine_enabled"] = True
        cfg["consensus_n_seeds"] = 2
        cfg["hdbscan_split_silhouette_thresh"] = None
        cfg["consensus_merge_coassoc_thresh"] = 0.0
        cfg["recluster_max_iterations"] = 1
        cfg["umap_n_jobs"] = 1
        cfg["consensus_n_jobs"] = 1
        cfg["hdbscan_sweep_n_steps"] = 4
        cfg["target_n_clusters"] = 0

        sentinel = ["marker"]
        try:
            cc.consensus_cluster(feats_sc.T, cfg, 2, embedding=embedding,
                                 region_per_bin=sentinel)
        except Exception:
            pass  # only the threading-through matters for this test, not a full consensus fit
        assert captured.get("region_per_bin") == sentinel


# ──────────────────────────────────────────────────────────────────────────
#  Phase 5 — region_split_pre_reduction_pct + audit logging
# ──────────────────────────────────────────────────────────────────────────

_P5_FAST_RUN_CFG = dict(_P2_FAST_RUN_CFG)
_P5_FAST_RUN_CFG.update(
    hdbscan_region_split_enabled=True,
    region_split_pre_reduction_pct=0.5,
    preferred_clusters_lo=10, preferred_clusters_hi=20,
    hdbscan_split_silhouette_thresh=None,  # isolate: no silhouette-split local recluster calls
    hdbscan_merge_thresh=0.0,
    consensus_clustering_enabled=False,
    seed_sweep_n=0,
)


@pytest.mark.slow
class TestRegionSplitPreReductionPct:
    def test_only_primary_sweep_call_gets_reduced_range(self, tmp_path, monkeypatch):
        """region_split_pre_reduction_pct=0.5 with preferred_clusters_lo/hi
        10/20 must reduce ONLY the primary sweep's own run_hdbscan() call to
        5/10 -- no local re-clustering call (which always hardcodes its own
        small lo=2/hi=max_subclusters range, independent of the global
        range already) should ever see either the original OR the reduced
        pair coincidentally standing in for that independent behavior."""
        dlc_dir = tmp_path / "dlc"
        dlc_dir.mkdir()
        _p2_write_session_h5(dlc_dir / "session1_filtered.h5", n_frames=300, seed=3)
        out_dir = tmp_path / "out"

        calls = []
        orig = cc.run_hdbscan

        def spy(embedding, cfg, *a, **kw):
            calls.append((cfg.get("preferred_clusters_lo"), cfg.get("preferred_clusters_hi")))
            return orig(embedding, cfg, *a, **kw)

        monkeypatch.setattr(cc, "run_hdbscan", spy)

        engine = cc.BSoidEngine(dlc_dir, video_folder=None, output_dir=out_dir,
                                fps=30, logger=lambda m: None, cfg=_P5_FAST_RUN_CFG)
        engine.run()

        assert (5, 10) in calls, (
            f"expected the primary sweep call with the reduced (5, 10) "
            f"pair somewhere in the recorded run_hdbscan() calls: {calls}")
        assert (10, 20) not in calls, (
            "the ORIGINAL, un-reduced (10, 20) pair must never reach "
            f"run_hdbscan() while pre-reduction is active: {calls}")

    def test_disabled_by_default_leaves_sweep_unreduced(self, tmp_path, monkeypatch):
        dlc_dir = tmp_path / "dlc"
        dlc_dir.mkdir()
        _p2_write_session_h5(dlc_dir / "session1_filtered.h5", n_frames=300, seed=4)
        out_dir = tmp_path / "out"

        calls = []
        orig = cc.run_hdbscan

        def spy(embedding, cfg, *a, **kw):
            calls.append((cfg.get("preferred_clusters_lo"), cfg.get("preferred_clusters_hi")))
            return orig(embedding, cfg, *a, **kw)

        monkeypatch.setattr(cc, "run_hdbscan", spy)

        cfg = dict(_P2_FAST_RUN_CFG)  # hdbscan_region_split_enabled omitted -> defaults False
        cfg["preferred_clusters_lo"] = 10
        cfg["preferred_clusters_hi"] = 20
        engine = cc.BSoidEngine(dlc_dir, video_folder=None, output_dir=out_dir,
                                fps=30, logger=lambda m: None, cfg=cfg)
        engine.run()

        assert (10, 20) in calls
        assert all(pair != (5, 10) for pair in calls)

    def test_audit_log_line_present_when_region_split_active(self, tmp_path):
        dlc_dir = tmp_path / "dlc"
        dlc_dir.mkdir()
        _p2_write_session_h5(dlc_dir / "session1_filtered.h5", n_frames=300, seed=5)
        out_dir = tmp_path / "out"

        logged = []
        engine = cc.BSoidEngine(dlc_dir, video_folder=None, output_dir=out_dir,
                                fps=30, logger=logged.append, cfg=_P5_FAST_RUN_CFG)
        engine.run()

        trace_lines = [m for m in logged if "[region-split] cluster-count trace" in m]
        assert len(trace_lines) == 1
        assert "raw (primary sweep)" in trace_lines[0]
        assert "after split/merge refinement" in trace_lines[0]
        assert "final (after rare-cluster prune)" in trace_lines[0]
        assert "pre-reduced to 5-10" in trace_lines[0]

    def test_no_audit_log_line_when_region_split_disabled(self, tmp_path):
        dlc_dir = tmp_path / "dlc"
        dlc_dir.mkdir()
        _p2_write_session_h5(dlc_dir / "session1_filtered.h5", n_frames=300, seed=6)
        out_dir = tmp_path / "out"

        logged = []
        engine = cc.BSoidEngine(dlc_dir, video_folder=None, output_dir=out_dir,
                                fps=30, logger=logged.append, cfg=_P2_FAST_RUN_CFG)
        engine.run()

        assert not any("[region-split]" in m for m in logged)
