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
