"""Phase T6: light integration/smoke tests for published-algorithm stages
(Tier 2 -- do NOT re-validate UMAP/HDBSCAN/MLP/HMM's own correctness).

Covers run_umap, run_hdbscan (reduced sweep), train_mlp (tiny synthetic
data, short max_iter), train_hmm/decode_hmm (seeded, low iterations).
Assertions are limited to: no crash/hang, correct output types/shapes,
determinism under a fixed seed. Never asserts a "correct" clustering or
classification result.
"""
import numpy as np
import pytest

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

    @pytest.mark.xfail(
        reason="BUG: train_hmm() constructs hmmlearn.CategoricalHMM without "
               "passing random_state, so its 's' (startprob) init draws from "
               "the global numpy legacy RandomState each call. Two back-to-"
               "back calls with IDENTICAL inputs/cfg usually produce "
               "measurably different transmat_/emissionprob_ (differences on "
               "the order of 1e-5, i.e. real EM-trajectory divergence, not "
               "float noise) -- unlike run_umap/train_mlp, which both accept "
               "and use a seed. This breaks the 'seeded, low iteration count' "
               "determinism the test plan expects at this stage, and quietly "
               "undermines any reproducibility claim for a saved bsoid_model.pkl "
               "trained via train_hmm. strict=False: whether this coincidentally "
               "passes depends on the ambient global numpy RNG state left by "
               "whichever tests ran earlier in the same process, so it is not "
               "reliably reproducible as a hard failure across every run order "
               "-- the underlying bug (no explicit seed control) is real and "
               "confirmed either way; see test_train_hmm_determinism_with_"
               "manually_seeded_global_rng below for the deterministic repro.",
        strict=False)
    def test_train_hmm_determinism_same_input(self):
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
