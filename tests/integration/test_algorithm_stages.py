"""Phase T6: light integration/smoke tests for published-algorithm stages
(Tier 2 -- do NOT re-validate UMAP/HDBSCAN/MLP/HMM's own correctness).

Covers run_umap, run_hdbscan (reduced sweep), train_mlp (tiny synthetic
data, short max_iter), train_hmm/decode_hmm (seeded, low iterations).
Assertions are limited to: no crash/hang, correct output types/shapes,
determinism under a fixed seed. Never asserts a "correct" clustering or
classification result.
"""
import numpy as np

import cube_core as cc


def make_blob_embedding(n_per_blob=50, n_blobs=3, seed=0, dim=2, spread=0.3):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_blobs, dim))
    parts = [centers[i] + rng.normal(0, spread, size=(n_per_blob, dim))
             for i in range(n_blobs)]
    return np.vstack(parts).astype(float)


# ──────────────────────────────────────────────────────────────────────────
#  run_umap
# ──────────────────────────────────────────────────────────────────────────

class TestRunUmap:
    def test_output_shape_and_type(self):
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(200, 40))  # (n_samples, n_features)
        cfg = dict(umap_n_neighbors=15, umap_n_components=2,
                  umap_min_dist=0.1, umap_random_state=42, umap_n_jobs=1,
                  pca_n_components="off")
        reducer, embedding = cc.run_umap(feats, cfg)
        assert embedding.shape == (200, 2)
        assert np.isfinite(embedding).all()

    def test_determinism_same_seed_identical_output(self):
        rng = np.random.default_rng(1)
        feats = rng.normal(size=(150, 30))
        cfg = dict(umap_n_neighbors=15, umap_n_components=2,
                  umap_min_dist=0.1, umap_random_state=7, umap_n_jobs=1,
                  pca_n_components="off")
        _, emb1 = cc.run_umap(feats, cfg)
        _, emb2 = cc.run_umap(feats, cfg)
        assert np.allclose(emb1, emb2)

    def test_pca_pre_reduction_triggers_when_auto_and_high_dim(self):
        rng = np.random.default_rng(2)
        # n_samples/n_features < 5 and n_features > 50 -> auto PCA triggers
        feats = rng.normal(size=(30, 100))
        cfg = dict(umap_n_neighbors=5, umap_n_components=2,
                  umap_min_dist=0.1, umap_random_state=42, umap_n_jobs=1,
                  pca_n_components="auto")
        _, embedding = cc.run_umap(feats, cfg)
        assert embedding.shape == (30, 2)


# ──────────────────────────────────────────────────────────────────────────
#  run_hdbscan  (reduced sweep for speed)
# ──────────────────────────────────────────────────────────────────────────

FAST_HDBSCAN_CFG = dict(
    hdbscan_sweep_n_steps=5,
    hdbscan_pct_lo=5,
    hdbscan_pct_hi=20,
    hdbscan_method="eom",
    target_n_clusters=0,
    preferred_clusters_lo=2,
    preferred_clusters_hi=4,
)


class TestRunHdbscan:
    def test_no_crash_correct_output_types(self):
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        clf, labels, score, score_label = cc.run_hdbscan(embedding, FAST_HDBSCAN_CFG)
        assert labels.shape[0] == embedding.shape[0]
        assert labels.dtype.kind in ("i", "u")
        assert isinstance(score, float) or score is None or np.isnan(score) \
            or isinstance(score, (int, np.floating))

    def test_determinism_same_input_same_cfg(self):
        embedding = make_blob_embedding(n_per_blob=40, n_blobs=3, seed=4)
        _, labels1, score1, _ = cc.run_hdbscan(embedding, FAST_HDBSCAN_CFG)
        _, labels2, score2, _ = cc.run_hdbscan(embedding, FAST_HDBSCAN_CFG)
        assert np.array_equal(labels1, labels2)

    def test_tolerates_tiny_degenerate_input(self):
        # Small/degenerate embedding -- DBCV may be non-finite; must not crash.
        embedding = make_blob_embedding(n_per_blob=10, n_blobs=2, seed=5, spread=0.01)
        clf, labels, score, score_label = cc.run_hdbscan(embedding, FAST_HDBSCAN_CFG)
        assert labels.shape[0] == embedding.shape[0]


# ──────────────────────────────────────────────────────────────────────────
#  run_hdbscan -- Option 3 (HDBSCAN sweep tree-reuse perf) equivalence suite
#
#  hdbscan_tree_reuse_enabled=True (_fit_one_new, default) must produce
#  BIT-FOR-BIT identical labels/scores to hdbscan_tree_reuse_enabled=False
#  (_fit_one_legacy, verbatim pre-change path) for the same embedding/cfg.
#  This is the mandatory gate for Option 3 -- see
#  HDBSCAN_Sweep_Performance_Implementation_Plan.md's "Verification" section.
# ──────────────────────────────────────────────────────────────────────────

import pytest


def _wide_bucket_cfg(selection_mode="legacy"):
    # hdbscan_pct_lo/hi spread wide enough that mcs // 5 crosses several
    # integer-division boundaries -- exercises real multi-bucket reuse.
    return dict(
        hdbscan_sweep_n_steps=20,
        hdbscan_pct_lo=2,
        hdbscan_pct_hi=40,
        hdbscan_method="both",
        target_n_clusters=0,
        preferred_clusters_lo=2,
        preferred_clusters_hi=6,
        hdbscan_selection_mode=selection_mode,
    )


def _one_bucket_cfg(selection_mode="legacy"):
    # Narrow pct range -- every mcs collapses to one min_samples bucket.
    return dict(
        hdbscan_sweep_n_steps=6,
        hdbscan_pct_lo=10,
        hdbscan_pct_hi=12,
        hdbscan_method="both",
        target_n_clusters=0,
        preferred_clusters_lo=2,
        preferred_clusters_hi=6,
        hdbscan_selection_mode=selection_mode,
    )


def _unique_bucket_cfg(selection_mode="legacy"):
    # A grid where one step's min_samples value is unique (isolated bucket
    # among otherwise-clustered mcs steps).
    return dict(
        hdbscan_sweep_n_steps=8,
        hdbscan_pct_lo=5,
        hdbscan_pct_hi=45,
        hdbscan_method="eom",
        target_n_clusters=0,
        preferred_clusters_lo=2,
        preferred_clusters_hi=6,
        hdbscan_selection_mode=selection_mode,
    )


def _degenerate_embedding(seed=5):
    # Reuses the existing tiny/near-duplicate blob pattern from
    # test_tolerates_tiny_degenerate_input.
    return make_blob_embedding(n_per_blob=10, n_blobs=2, seed=seed, spread=0.01)


def _assert_equivalent(embedding, cfg):
    cfg_new = dict(cfg, hdbscan_tree_reuse_enabled=True)
    cfg_old = dict(cfg, hdbscan_tree_reuse_enabled=False)
    clf_new, labels_new, score_new, label_kind_new = cc.run_hdbscan(
        embedding.copy(), cfg_new)
    clf_old, labels_old, score_old, label_kind_old = cc.run_hdbscan(
        embedding.copy(), cfg_old)
    assert np.array_equal(labels_new, labels_old)
    assert label_kind_new == label_kind_old
    if np.isnan(score_old):
        assert np.isnan(score_new)
    else:
        assert score_new == pytest.approx(score_old, abs=1e-9)
    return clf_new, labels_new, score_new


class TestRunHdbscanTreeReuseEquivalence:
    """Option 3 mandatory equivalence gate: hdbscan_tree_reuse_enabled=True
    vs False must select the same winner and produce identical output."""

    # -- Case 1: standard, wide bucket spread, both eom/leaf --------------
    def test_case1_standard_wide_bucket_spread(self):
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        _assert_equivalent(embedding, _wide_bucket_cfg())

    # -- Case 2: all candidates collapse to one min_samples bucket --------
    def test_case2_all_candidates_one_bucket(self):
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        _assert_equivalent(embedding, _one_bucket_cfg())

    # -- Case 3: a bucket of size 1 (unique min_samples step) --------------
    def test_case3_unique_size_one_bucket(self):
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        _assert_equivalent(embedding, _unique_bucket_cfg())

    # -- Case 4: degenerate embedding -- both paths hit the same fallback --
    def test_case4_degenerate_embedding(self):
        embedding = _degenerate_embedding()
        _assert_equivalent(embedding, _wide_bucket_cfg())

    # -- Case 5: best_clf functional check (real refit, not just a tuple) --
    def test_case5_best_clf_is_fully_functional(self):
        import hdbscan as _hdb
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        clf, labels, score, _ = cc.run_hdbscan(
            embedding.copy(), dict(_wide_bucket_cfg(), hdbscan_tree_reuse_enabled=True))
        assert clf.condensed_tree_ is not None
        pred, strengths = _hdb.approximate_predict(clf, embedding[:5])
        assert pred.shape[0] == 5

    # -- Case 6: non-mutation of the cached tree/MST across repeated reuse -
    def test_case6_cached_tree_not_mutated_across_reuse(self):
        import hdbscan as _hdb
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        sl_tree, mst = cc._build_tree(
            min_samples=5, embedding=embedding, metric="euclidean",
            core_dist_n_jobs=None, cache={})
        sl_tree_before = sl_tree.copy()
        mst_before = mst.copy()
        _hdb.hdbscan_._tree_to_labels(
            None, sl_tree, min_cluster_size=5, cluster_selection_method="eom")
        _hdb.hdbscan_._tree_to_labels(
            None, sl_tree, min_cluster_size=15, cluster_selection_method="leaf")
        assert np.array_equal(sl_tree, sl_tree_before)
        assert np.array_equal(mst, mst_before)

    # -- Case 7: determinism under tree_reuse=True -------------------------
    def test_case7_determinism_under_tree_reuse(self):
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        cfg = dict(_wide_bucket_cfg(), hdbscan_tree_reuse_enabled=True)
        _, labels1, score1, _ = cc.run_hdbscan(embedding.copy(), cfg)
        _, labels2, score2, _ = cc.run_hdbscan(embedding.copy(), cfg)
        assert np.array_equal(labels1, labels2)
        if np.isnan(score1):
            assert np.isnan(score2)
        else:
            assert score2 == pytest.approx(score1, abs=1e-9)

    # -- Case 8: _dbcv_from_mst isolated unit test vs real relative_validity_
    def test_case8_dbcv_from_mst_matches_relative_validity(self):
        import hdbscan as _hdb
        embedding = make_blob_embedding(n_per_blob=40, n_blobs=3, seed=9)
        clf = _hdb.HDBSCAN(min_cluster_size=10, min_samples=5,
                            gen_min_span_tree=True).fit(embedding)
        mst_raw = clf.minimum_spanning_tree_.to_numpy()
        ported_score = cc._dbcv_from_mst(clf.labels_, mst_raw)
        real_score = clf.relative_validity_
        if np.isnan(real_score):
            assert np.isnan(ported_score)
        else:
            assert ported_score == pytest.approx(real_score, abs=1e-9)

    # -- Case 9: cases 1-4 under BOTH hdbscan_selection_mode values --------
    @pytest.mark.parametrize("selection_mode", ["legacy", "floor_soft_cap"])
    @pytest.mark.parametrize("cfg_factory,embedding_factory", [
        (_wide_bucket_cfg, lambda: make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)),
        (_one_bucket_cfg, lambda: make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)),
        (_unique_bucket_cfg, lambda: make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)),
        (_wide_bucket_cfg, _degenerate_embedding),
    ])
    def test_case9_both_selection_modes(self, selection_mode, cfg_factory, embedding_factory):
        embedding = embedding_factory()
        cfg = cfg_factory(selection_mode=selection_mode)
        _assert_equivalent(embedding, cfg)

    # -- Case 10 (regression, found on real data): the DBCV canary must not
    # false-fire under the DEFAULT hdbscan_merge_thresh=0.08 config when the
    # winning candidate is a "leaf" method -- the pre-existing (pre-Option-3)
    # eom/leaf tie-breaking nudge adds +hdbscan_leaf_bonus to leaf candidates'
    # ranking score, which the canary must subtract back out before comparing
    # against the refit's real (bonus-free) relative_validity_. Confirmed on
    # a real 29k-bin dataset: swept score - refit score == hdbscan_leaf_bonus
    # (0.03) exactly, whenever the winner was a leaf candidate -- not a real
    # DBCV divergence.
    def test_case10_leaf_bonus_does_not_trigger_canary_false_positive(self):
        embedding = make_blob_embedding(n_per_blob=200, n_blobs=5, seed=11)
        logs = []
        cfg = dict(
            hdbscan_sweep_n_steps=20, hdbscan_pct_lo=2, hdbscan_pct_hi=40,
            hdbscan_method="leaf",   # force the winner to be a leaf candidate
            target_n_clusters=0, preferred_clusters_lo=2, preferred_clusters_hi=8,
            hdbscan_merge_thresh=0.08, hdbscan_leaf_bonus=0.03,   # real defaults
            hdbscan_tree_reuse_enabled=True,
        )
        clf, labels, score, _ = cc.run_hdbscan(embedding.copy(), cfg, log_fn=logs.append)
        canary_lines = [l for l in logs if "tree-reuse DBCV canary" in l]
        assert canary_lines == [], (
            f"canary false-fired despite the leaf-bonus offset being a known, "
            f"non-buggy ranking-only nudge: {canary_lines}")
        # The RETURNED score matches legacy's own (bonus-inclusive) convention
        # -- it is NOT expected to equal the refit clf's raw relative_validity_
        # when the leaf bonus was applied (see case 12: that's the whole point
        # -- the returned value must match _tree_reuse=False, not "the most
        # numerically pure" value). Sanity-check the bonus relationship directly.
        assert score - clf.relative_validity_ == pytest.approx(0.03, abs=1e-9)

    # -- Case 12 (regression, found on real data): the RETURNED score, not
    # just the labels, must be equivalent between hdbscan_tree_reuse_enabled
    # =True and =False when the winner is a leaf candidate under the default
    # hdbscan_merge_thresh=0.08. An earlier version of the winner-refit block
    # reassigned best_score to the refit's own (bonus-free) relative_validity_
    # -- labels stayed identical, but the returned SCORE silently diverged
    # from legacy by exactly hdbscan_leaf_bonus (confirmed on a real 29k-bin
    # dataset: True returned 0.151, False returned 0.181, same winning
    # candidate/labels). Case 10 alone couldn't catch this -- it only checks
    # tree_reuse=True's internal self-consistency, never compares against
    # tree_reuse=False directly under leaf-bonus conditions.
    def test_case12_returned_score_equivalent_under_leaf_bonus(self):
        embedding = make_blob_embedding(n_per_blob=200, n_blobs=5, seed=11)
        cfg = dict(
            hdbscan_sweep_n_steps=20, hdbscan_pct_lo=2, hdbscan_pct_hi=40,
            hdbscan_method="leaf",   # force the winner to be a leaf candidate
            target_n_clusters=0, preferred_clusters_lo=2, preferred_clusters_hi=8,
            hdbscan_merge_thresh=0.08, hdbscan_leaf_bonus=0.03,   # real defaults
        )
        _assert_equivalent(embedding, cfg)

    # -- Case 11 (regression, Aug 2026): the sequential sweep branch must
    # log progress too, not just the parallel branch. hdbscan_sweep_n_jobs
    # defaults to 1 (sequential), so this is the DEFAULT path on every real
    # run -- before this fix it produced zero log output for its entire
    # duration (confirmed live on a real 20-session dataset: 37+ minutes of
    # total silence in the equivalent consensus/seed-sweep call sites while
    # the process was still alive and actively computing).
    def test_case11_sequential_sweep_logs_progress(self):
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        logs = []
        cfg = dict(_wide_bucket_cfg(), hdbscan_sweep_n_jobs=1)
        cc.run_hdbscan(embedding.copy(), cfg, log_fn=logs.append)
        assert any("[hdbscan-sweep] sweeping" in l for l in logs), logs
        assert any("[hdbscan-sweep]" in l and "candidates fit" in l for l in logs), logs


# ──────────────────────────────────────────────────────────────────────────
#  seed_sweep_stability -- zero prior coverage (per the perf plan's own
#  research findings); this is the foundational smoke-test baseline that
#  Option 4's tests diff against, added before layering Option 4 on top.
# ──────────────────────────────────────────────────────────────────────────

def make_blob_features(n_per_blob=50, n_blobs=3, n_features=8, seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(n_blobs, n_features))
    parts = [centers[i] + rng.normal(0, spread, size=(n_per_blob, n_features))
             for i in range(n_blobs)]
    return np.vstack(parts).astype(float)


SEED_SWEEP_CFG = dict(
    umap_n_neighbors=10, umap_n_components=2, umap_min_dist=0.1,
    umap_random_state=42, umap_n_jobs=1, pca_n_components="off",
    hdbscan_sweep_n_steps=6, hdbscan_pct_lo=5, hdbscan_pct_hi=20,
    hdbscan_method="eom", target_n_clusters=0,
    preferred_clusters_lo=2, preferred_clusters_hi=4,
    seed_sweep_n_jobs=1,
)


class TestSeedSweepStabilityFoundational:
    def test_n_seeds_below_2_returns_empty_dict(self):
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        assert cc.seed_sweep_stability(feats, SEED_SWEEP_CFG, n_seeds=1) == {}
        assert cc.seed_sweep_stability(feats, SEED_SWEEP_CFG, n_seeds=0) == {}

    def test_return_shape_and_keys(self):
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        result = cc.seed_sweep_stability(feats, SEED_SWEEP_CFG, n_seeds=3)
        assert result != {}
        for key in ("seeds", "counts", "ari", "labels", "mean_ari",
                    "stable_counts", "dbcv"):
            assert key in result
        assert len(result["seeds"]) == len(result["counts"]) == len(result["labels"])
        assert result["ari"].shape == (len(result["seeds"]), len(result["seeds"]))

    def test_mean_ari_in_valid_range(self):
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        result = cc.seed_sweep_stability(feats, SEED_SWEEP_CFG, n_seeds=3)
        mean_ari = result["mean_ari"]
        if not np.isnan(mean_ari):
            assert -1.0 <= mean_ari <= 1.0

    # Regression (Aug 2026): the sequential dispatch branch (seed_sweep_n_jobs
    # =1, forced here but also the real fallback under RAM pressure) must log
    # progress per seed, not just the parallel branch -- before this fix it
    # was silent for the whole call's duration.
    def test_sequential_dispatch_logs_progress(self):
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        logs = []
        cfg = dict(SEED_SWEEP_CFG, seed_sweep_n_jobs=1)
        cc.seed_sweep_stability(feats, cfg, n_seeds=3, log_fn=logs.append)
        assert any("[seed-sweep] running" in l for l in logs), logs
        assert any("[seed-sweep]" in l and "done — seed" in l for l in logs), logs


# ──────────────────────────────────────────────────────────────────────────
#  consensus_cluster -- sequential-dispatch progress-logging regression.
# ──────────────────────────────────────────────────────────────────────────

class TestConsensusClusterLogging:
    # Regression (Aug 2026, found live on a real 20-session dataset: 37+
    # minutes of total silence with the process still alive and actively
    # computing). consensus_n_jobs=1 (sequential -- the real fallback under
    # RAM pressure, or whenever pinned explicitly) must log progress per
    # seed, not just the parallel branch.
    def test_sequential_dispatch_logs_progress(self):
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        logs = []
        cfg = dict(SEED_SWEEP_CFG, consensus_n_jobs=1)
        result = cc.consensus_cluster(feats, cfg, n_seeds=3, log_fn=logs.append)
        assert result is not None
        assert any("[consensus] running" in l for l in logs), logs
        assert any("[consensus]" in l and "done — seed" in l for l in logs), logs


# ──────────────────────────────────────────────────────────────────────────
#  Option 4 -- coarser grid for diagnostic (seed/consensus) sweeps only.
#  Both new *_sweep_n_steps keys default to 0 = inherit hdbscan_sweep_n_steps
#  (true no-op); this suite exercises the nonzero-override plumbing and
#  confirms the primary run_hdbscan() call path is unaffected either way.
# ──────────────────────────────────────────────────────────────────────────

class TestOption4SeedConsensusSweepNSteps:
    def test_seed_sweep_n_steps_override_does_not_crash_shape_unaffected(self):
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        cfg = dict(SEED_SWEEP_CFG, hdbscan_seed_sweep_n_steps=3)
        result = cc.seed_sweep_stability(feats, cfg, n_seeds=3)
        assert result != {}
        for key in ("seeds", "counts", "ari", "labels", "mean_ari",
                    "stable_counts", "dbcv"):
            assert key in result

    def test_consensus_one_seed_reads_override_key(self):
        # _consensus_one_seed is the Option 4 call site for consensus_cluster;
        # exercise it directly (consensus_cluster itself needs no cfg changes
        # per the plan -- only its per-seed helper reads the new key).
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=2)
        cfg = dict(SEED_SWEEP_CFG, hdbscan_consensus_sweep_n_steps=3)
        result = cc._consensus_one_seed(7, feats, cfg, n_samp=feats.shape[0])
        assert result is not None
        assert result["labels"].shape[0] == feats.shape[0]

    def test_primary_run_hdbscan_unaffected_by_new_keys_when_unset(self):
        # run_hdbscan() itself never reads hdbscan_seed_sweep_n_steps /
        # hdbscan_consensus_sweep_n_steps -- only _seed_sweep_one_seed /
        # _consensus_one_seed do, building their own cfg overrides. Presence
        # of the (default, 0) keys in cfg must not change run_hdbscan()'s
        # own output at all.
        embedding = make_blob_embedding(n_per_blob=60, n_blobs=3, seed=3)
        cfg_plain = dict(FAST_HDBSCAN_CFG)
        cfg_with_keys = dict(FAST_HDBSCAN_CFG,
                              hdbscan_seed_sweep_n_steps=0,
                              hdbscan_consensus_sweep_n_steps=0)
        _, labels_plain, score_plain, _ = cc.run_hdbscan(embedding.copy(), cfg_plain)
        _, labels_keys, score_keys, _ = cc.run_hdbscan(embedding.copy(), cfg_with_keys)
        assert np.array_equal(labels_plain, labels_keys)
        if np.isnan(score_plain):
            assert np.isnan(score_keys)
        else:
            assert score_keys == pytest.approx(score_plain, abs=1e-9)

    def test_seed_sweep_override_zero_is_true_noop_vs_unset(self):
        # hdbscan_seed_sweep_n_steps=0 (explicit) must behave identically to
        # the key being absent entirely -- both mean "inherit
        # hdbscan_sweep_n_steps". Compare _seed_sweep_one_seed's own result
        # directly (single seed, deterministic).
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        cfg_unset = dict(SEED_SWEEP_CFG)
        cfg_zero = dict(SEED_SWEEP_CFG, hdbscan_seed_sweep_n_steps=0)
        r_unset = cc._seed_sweep_one_seed(42, feats, cfg_unset, feats.shape[0])
        r_zero = cc._seed_sweep_one_seed(42, feats, cfg_zero, feats.shape[0])
        assert r_unset is not None and r_zero is not None
        assert np.array_equal(r_unset["labels"], r_zero["labels"])


# ──────────────────────────────────────────────────────────────────────────
#  Option 5 -- shared-subsample opt-in for seed_sweep_stability() only.
#  seed_sweep_train_frac=1.0 (default) must be a true no-op: the default-
#  is-a-true-no-op guarantee is the one thing that must be airtight here,
#  per the plan's Verification section.
# ──────────────────────────────────────────────────────────────────────────

class TestOption5SeedSweepTrainFrac:
    def test_default_train_frac_is_true_noop_full_array_used(self, monkeypatch):
        # No seed_sweep_train_frac key at all -- every seed must still see
        # the FULL, unsubsampled feats_sc_T array (identical to pre-Option-5
        # behaviour, which never subsampled at all).
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        cfg = dict(SEED_SWEEP_CFG)  # seed_sweep_train_frac absent
        captured = []
        orig = cc._seed_sweep_one_seed

        def spy(s, feats_sc_T, cfg_, n_total, progress_cb=None):
            captured.append(feats_sc_T.shape[0])
            return orig(s, feats_sc_T, cfg_, n_total, progress_cb=progress_cb)

        monkeypatch.setattr(cc, "_seed_sweep_one_seed", spy)
        cc.seed_sweep_stability(feats, cfg, n_seeds=3)
        assert len(captured) == 3
        assert all(n == feats.shape[0] for n in captured)

    def test_explicit_train_frac_one_matches_unset(self, monkeypatch):
        feats = make_blob_features(n_per_blob=40, n_blobs=3, seed=1)
        cfg = dict(SEED_SWEEP_CFG, seed_sweep_train_frac=1.0)
        captured = []
        orig = cc._seed_sweep_one_seed

        def spy(s, feats_sc_T, cfg_, n_total, progress_cb=None):
            captured.append(feats_sc_T.shape[0])
            return orig(s, feats_sc_T, cfg_, n_total, progress_cb=progress_cb)

        monkeypatch.setattr(cc, "_seed_sweep_one_seed", spy)
        cc.seed_sweep_stability(feats, cfg, n_seeds=3)
        assert all(n == feats.shape[0] for n in captured)

    def test_subsample_size_matches_train_frac(self):
        feats = make_blob_features(n_per_blob=100, n_blobs=3, seed=1)
        cfg = dict(SEED_SWEEP_CFG, seed_sweep_train_frac=0.5, seed_sweep_n_jobs=1)
        result = cc.seed_sweep_stability(feats, cfg, n_seeds=3)
        assert result != {}
        expected_n = int(round(feats.shape[0] * 0.5))
        for lbls in result["labels"]:
            assert lbls.shape[0] == expected_n

    def test_same_subsample_used_across_all_seeds_within_one_call(self, monkeypatch):
        feats = make_blob_features(n_per_blob=100, n_blobs=3, seed=1)
        cfg = dict(SEED_SWEEP_CFG, seed_sweep_train_frac=0.5, seed_sweep_n_jobs=1)
        captured = []
        orig = cc._seed_sweep_one_seed

        def spy(s, feats_sc_T, cfg_, n_total, progress_cb=None):
            captured.append(feats_sc_T.copy())
            return orig(s, feats_sc_T, cfg_, n_total, progress_cb=progress_cb)

        monkeypatch.setattr(cc, "_seed_sweep_one_seed", spy)
        cc.seed_sweep_stability(feats, cfg, n_seeds=3)
        assert len(captured) == 3
        for arr in captured[1:]:
            assert np.array_equal(arr, captured[0])

    def test_subsample_deterministic_across_separate_calls(self, monkeypatch):
        feats = make_blob_features(n_per_blob=100, n_blobs=3, seed=1)
        cfg = dict(SEED_SWEEP_CFG, seed_sweep_train_frac=0.5, seed_sweep_n_jobs=1)
        orig = cc._seed_sweep_one_seed

        def make_spy(bucket):
            def spy(s, feats_sc_T, cfg_, n_total, progress_cb=None):
                bucket.append(feats_sc_T.copy())
                return orig(s, feats_sc_T, cfg_, n_total, progress_cb=progress_cb)
            return spy

        bucket1 = []
        monkeypatch.setattr(cc, "_seed_sweep_one_seed", make_spy(bucket1))
        cc.seed_sweep_stability(feats, cfg, n_seeds=2)

        bucket2 = []
        monkeypatch.setattr(cc, "_seed_sweep_one_seed", make_spy(bucket2))
        cc.seed_sweep_stability(feats, cfg, n_seeds=2)

        assert np.array_equal(bucket1[0], bucket2[0])

    def test_train_frac_with_mcs_anchor_full_behaves_sanely(self):
        # Interaction case flagged by the plan's Risk Assessment: mcs
        # anchored to the ORIGINAL n_total (not the subsample) combined with
        # a subsample must not crash or produce a nonsensical (empty/wrong-
        # shape) result.
        feats = make_blob_features(n_per_blob=100, n_blobs=3, seed=1)
        cfg = dict(SEED_SWEEP_CFG, seed_sweep_train_frac=0.5,
                    hdbscan_mcs_anchor="full", seed_sweep_n_jobs=1)
        result = cc.seed_sweep_stability(feats, cfg, n_seeds=3)
        assert result != {}
        expected_n = int(round(feats.shape[0] * 0.5))
        for lbls in result["labels"]:
            assert lbls.shape[0] == expected_n

    def test_seed_sweep_stability_bootstrap_untouched(self):
        # Regression guard: seed_sweep_stability_bootstrap() must remain
        # standalone -- zero cfg keys introduced by this plan, and this
        # option's key must never be referenced inside it (conflating the
        # two would silently change what seed_sweep_stability_bootstrap
        # measures).
        import inspect
        src = inspect.getsource(cc.seed_sweep_stability_bootstrap)
        assert "seed_sweep_train_frac" not in src


# ──────────────────────────────────────────────────────────────────────────
#  train_mlp
# ──────────────────────────────────────────────────────────────────────────

class TestTrainMlp:
    def _make_labeled_data(self, seed=0, n_per_class=40, n_features=10, n_classes=3):
        rng = np.random.default_rng(seed)
        feats = []
        labels = []
        for c in range(n_classes):
            center = rng.normal(0, 3, size=n_features)
            feats.append(center + rng.normal(0, 0.5, size=(n_per_class, n_features)))
            labels.append(np.full(n_per_class, c))
        X = np.vstack(feats).T  # (n_features, n_samples) per train_mlp's expected feats_sc
        y = np.concatenate(labels)
        return X, y

    def test_returns_fitted_classifier_and_cv_scores(self):
        feats_sc, labels = self._make_labeled_data(seed=10)
        cfg = dict(mlp_hidden="8,4", mlp_max_iter=50, umap_random_state=42, cv_folds=3)
        clf, scores = cc.train_mlp(feats_sc, labels, cfg)
        assert clf is not None
        assert scores.size > 0
        assert np.all((scores >= 0) & (scores <= 1))

    def test_fewer_than_2_classes_returns_none(self):
        feats_sc = np.random.default_rng(11).normal(size=(10, 20))
        labels = np.zeros(20, dtype=int)
        cfg = dict(mlp_hidden="8,4", mlp_max_iter=50, umap_random_state=42)
        clf, scores = cc.train_mlp(feats_sc, labels, cfg)
        assert clf is None
        assert np.allclose(scores, 0.0)

    def test_noise_label_minus_one_excluded(self):
        feats_sc, labels = self._make_labeled_data(seed=12)
        # add some noise-labeled bins
        noise_feats = np.random.default_rng(13).normal(size=(feats_sc.shape[0], 5))
        feats_sc_ext = np.hstack([feats_sc, noise_feats])
        labels_ext = np.concatenate([labels, np.full(5, -1)])
        cfg = dict(mlp_hidden="8,4", mlp_max_iter=50, umap_random_state=42, cv_folds=3)
        clf, scores = cc.train_mlp(feats_sc_ext, labels_ext, cfg)
        assert clf is not None
        assert clf.classes_.min() >= 0  # noise (-1) excluded from training classes

    def test_determinism_same_seed(self):
        feats_sc, labels = self._make_labeled_data(seed=14)
        cfg = dict(mlp_hidden="8,4", mlp_max_iter=50, umap_random_state=42, cv_folds=3)
        clf1, _ = cc.train_mlp(feats_sc, labels, cfg)
        clf2, _ = cc.train_mlp(feats_sc, labels, cfg)
        assert np.allclose(clf1.predict_proba(feats_sc.T),
                          clf2.predict_proba(feats_sc.T))


# ──────────────────────────────────────────────────────────────────────────
#  train_hmm / decode_hmm
# ──────────────────────────────────────────────────────────────────────────

def make_label_sequences(n_clusters=3, seq_len=200, n_seqs=2, seed=0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_seqs):
        # simple persistent-state random walk (favors self-transitions)
        seq = [rng.integers(0, n_clusters)]
        for _ in range(seq_len - 1):
            if rng.random() < 0.85:
                seq.append(seq[-1])
            else:
                seq.append(rng.integers(0, n_clusters))
        seqs.append(np.array(seq))
    return seqs


class TestTrainDecodeHmm:
    def test_train_hmm_smoothing_mode_shapes(self):
        n_clusters = 3
        seqs = make_label_sequences(n_clusters=n_clusters, seed=20)
        model = cc.train_hmm(seqs, n_clusters, n_iter=10)
        assert model.emissionprob_.shape == (n_clusters, n_clusters)
        assert model.transmat_.shape == (n_clusters, n_clusters)
        assert model.cube_smoothing_mode is True

    def test_decode_hmm_output_shape(self):
        n_clusters = 3
        seqs = make_label_sequences(n_clusters=n_clusters, seed=21)
        model = cc.train_hmm(seqs, n_clusters, n_iter=10)
        frame_labels = seqs[0]
        decoded = cc.decode_hmm(model, frame_labels)
        assert decoded.shape == frame_labels.shape
        assert decoded.dtype.kind in ("i", "u")
        assert set(np.unique(decoded)).issubset(set(range(n_clusters)))

    def test_train_hmm_determinism_same_input(self):
        # Regression test for a fixed bug: train_hmm() used to construct
        # hmmlearn.CategoricalHMM without passing random_state, so its 's'
        # (startprob) init drew from the global numpy legacy RandomState each
        # call, making two back-to-back calls with identical inputs diverge
        # by ~1e-5 in transmat_/emissionprob_ -- unlike run_umap/train_mlp,
        # which both already accepted and used a seed. Fixed by adding a
        # random_state parameter (default 42, config key hmm_random_state)
        # threaded into the CategoricalHMM constructor.
        n_clusters = 3
        seqs = make_label_sequences(n_clusters=n_clusters, seed=22)
        model1 = cc.train_hmm(seqs, n_clusters, n_iter=10)
        model2 = cc.train_hmm(seqs, n_clusters, n_iter=10)
        assert np.allclose(model1.transmat_, model2.transmat_)
        assert np.allclose(model1.emissionprob_, model2.emissionprob_)

    def test_train_hmm_determinism_with_manually_seeded_global_rng(self):
        # Workaround for the above bug: seeding numpy's GLOBAL legacy RNG
        # immediately before each call achieves determinism today, since
        # hmmlearn falls back to the global RandomState when none is passed
        # explicitly. This is a workaround, not a fix -- documents the
        # current escape hatch for anyone needing reproducible HMM training.
        n_clusters = 3
        seqs = make_label_sequences(n_clusters=n_clusters, seed=25)
        np.random.seed(123)
        model1 = cc.train_hmm(seqs, n_clusters, n_iter=10)
        np.random.seed(123)
        model2 = cc.train_hmm(seqs, n_clusters, n_iter=10)
        assert np.allclose(model1.transmat_, model2.transmat_)
        assert np.allclose(model1.emissionprob_, model2.emissionprob_)

    def test_decode_hmm_handles_out_of_range_labels(self):
        # frame_labels containing -1 (unclassified) or an out-of-range id
        # must be sanitised, not crash (see _sanitize_labels_for_hmm).
        n_clusters = 3
        seqs = make_label_sequences(n_clusters=n_clusters, seed=23)
        model = cc.train_hmm(seqs, n_clusters, n_iter=10)
        dirty = seqs[0].copy()
        dirty[0:5] = -1
        decoded = cc.decode_hmm(model, dirty)
        assert decoded.shape == dirty.shape

    def test_macro_state_mode_fewer_states_than_clusters(self):
        n_clusters = 4
        seqs = make_label_sequences(n_clusters=n_clusters, seed=24)
        model = cc.train_hmm(seqs, n_clusters, n_states=2, n_iter=10)
        assert model.emissionprob_.shape == (2, n_clusters)
        assert model.cube_smoothing_mode is False
