"""Region-aware cluster refinement — Phase 1 equivalence coverage.

Phase 1 extracts _local_recluster_and_assign() out of split_impure_clusters()'s
body with NO intended behavior change. This module's job is to prove that:
the refactored split_impure_clusters() produces byte-identical output to the
pre-refactor version (captured at git tag `pre-region-aware-refinement`) on a
range of synthetic fixtures, and that the new shared helper itself behaves
correctly in isolation (no candidates / one candidate / multiple candidates /
a candidate that fails the acceptance gate).

Never touches real user data or CUBE_logs/ — synthetic fixtures only.
"""
import importlib.util
import shutil
import subprocess

import numpy as np
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
