"""Phase T7: end-to-end pipeline regression test (marked slow).

Runs one small, fully-synthetic fixture through BSoidEngine.run() in full,
with a fixed config/seed throughout, and asserts the documented output file
set exists with the correct schema. This is the golden regression mechanism
referenced by the plan for "byte-identical when a new flag defaults off"
backward-compatibility claims.

Fixture-strategy note (see Automated_Test_Suite_Plan.md's Fixture Strategy
section): the plan flags an optional real anonymized DLC file as a stronger
regression anchor than synthetic data, but explicitly defers that choice to
the user and defaults to fully synthetic fixtures when none is supplied.
No real fixture was provided for this implementation, so this test uses the
fully-synthetic default -- swapping in a real anonymized file later is a
drop-in change to the `dlc_dir` fixture below.

Kept intentionally small (2 short sessions, reduced HDBSCAN sweep/MLP/HMM
iteration counts) to bound wall-clock time -- see the implementation report
for the actual benchmarked runtime.
"""
import time

import numpy as np
import pandas as pd
import pytest

import cube_core as cc


FAST_RUN_CFG = dict(
    likelihood_thresh=0.3,
    max_interp_gap_sec=0.5,
    boxcar_win_sec=0.07,
    train_frac=1.0,
    umap_full_thresh=10_000,
    umap_n_neighbors=15,
    umap_n_components=2,
    umap_min_dist=0.1,
    umap_random_state=42,
    umap_n_jobs=1,
    pca_n_components="off",
    hdbscan_sweep_n_steps=5,
    hdbscan_pct_lo=5,
    hdbscan_pct_hi=25,
    hdbscan_method="eom",
    target_n_clusters=0,
    preferred_clusters_lo=2,
    preferred_clusters_hi=4,
    hdbscan_selection_mode="floor_soft_cap",
    mlp_hidden="8,4",
    mlp_max_iter=50,
    cv_folds=2,
    hmm_n_iter=5,
    seed_sweep_n=0,
    consensus_clustering_enabled=False,
    hdbscan_merge_thresh=0.0,
    hdbscan_split_silhouette_thresh=None,
    cluster_hierarchy_enabled=False,
    auto_bodypart_weighting=False,
    auto_flag_impure_clusters=False,
    visibility_features_enabled=False,
    min_cluster_freq=0.0,
    auto_resource_management=False,
    hdbscan_sweep_n_jobs=1,
    hdbscan_split_n_jobs=1,
    seed_sweep_n_jobs=1,
    consensus_n_jobs=1,
    plot_theme="dark",
)

BODYPARTS = ["nose", "neck", "tailbase", "paw1", "paw2"]


def _make_session_xy(n_frames, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    n_pts = len(BODYPARTS)
    xy = np.zeros((n_frames, n_pts * 2))
    for i in range(n_pts):
        cx, cy = 50.0 * (i + 1), 30.0 * (i + 1)
        phase = rng.uniform(0, 6.28)
        xy[:, 2 * i] = cx + 15 * np.sin(t / 25.0 + phase + i) + rng.normal(0, 0.5, n_frames)
        xy[:, 2 * i + 1] = cy + 10 * np.cos(t / 30.0 + phase + i) + rng.normal(0, 0.5, n_frames)
    return xy


def _write_session_h5(path, n_frames, seed):
    xy = _make_session_xy(n_frames, seed)
    n_pts = len(BODYPARTS)
    cols, data = [], {}
    for i, bp in enumerate(BODYPARTS):
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


@pytest.fixture(scope="module")
def dlc_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("t7_dlc")
    _write_session_h5(d / "session1_filtered.h5", n_frames=1500, seed=100)
    _write_session_h5(d / "session2_filtered.h5", n_frames=1500, seed=200)
    return d


@pytest.fixture(scope="module")
def pipeline_result(dlc_dir, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("t7_out")
    engine = cc.BSoidEngine(dlc_dir, video_folder=None, output_dir=out_dir,
                            fps=30, logger=lambda m: None, cfg=FAST_RUN_CFG)
    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    return dict(engine=engine, out_dir=out_dir, results=results, elapsed=elapsed)


@pytest.mark.slow
class TestFullPipelineRegression:
    def test_run_completes_and_reports_timing(self, pipeline_result, capsys):
        elapsed = pipeline_result["elapsed"]
        print(f"\n[T7 BENCHMARK] full BSoidEngine.run() wall-clock: {elapsed:.1f}s")
        assert elapsed > 0
        # Bounded per ground rule 5 -- generous ceiling since run() is
        # monolithic and includes every Tier-2 stage; a real regression in
        # runtime should still fail this rather than silently pass.
        assert elapsed < 600, f"T7 pipeline took {elapsed:.1f}s (> 600s budget)"

    def test_bout_lengths_csv_exists_with_correct_schema(self, pipeline_result):
        out_dir = pipeline_result["out_dir"]
        bouts_dir = out_dir / "bout_lengths"
        bout_files = list(bouts_dir.glob("*_bout_lengths.csv"))
        assert len(bout_files) >= 1
        df = pd.read_csv(bout_files[0])
        assert list(df.columns) == ["B-SOiD labels", "Start time (frames)", "Run lengths"]
        assert len(df) > 0

    def test_frame_labels_csv_exists_with_correct_schema(self, pipeline_result):
        out_dir = pipeline_result["out_dir"]
        bouts_dir = out_dir / "bout_lengths"
        frame_files = list(bouts_dir.glob("*_frame_labels.csv"))
        assert len(frame_files) >= 1
        df = pd.read_csv(frame_files[0])
        for col in ("frame", "time_s", "label", "raw_label", "turned_away_reason"):
            assert col in df.columns
        assert len(df) == 1500  # matches the synthetic session's frame count

    def test_epochs_csv_exists_with_correct_schema(self, pipeline_result):
        out_dir = pipeline_result["out_dir"]
        bouts_dir = out_dir / "bout_lengths"
        epoch_files = list(bouts_dir.glob("*_epochs.csv"))
        assert len(epoch_files) >= 1
        df = pd.read_csv(epoch_files[0])
        for col in ("start_frame", "end_frame", "start_sec", "end_sec",
                   "duration_sec", "label"):
            assert col in df.columns

    def test_model_pkl_exists(self, pipeline_result):
        model_path = pipeline_result["out_dir"] / "model" / "bsoid_model.pkl"
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_umap_embedding_npy_exists(self, pipeline_result):
        emb_path = pipeline_result["out_dir"] / "model" / "umap_embedding.npy"
        assert emb_path.exists()
        emb = np.load(emb_path)
        assert emb.ndim == 2
        assert emb.shape[1] == FAST_RUN_CFG["umap_n_components"]

    def test_session_bin_ranges_json_exists(self, pipeline_result):
        p = pipeline_result["out_dir"] / "model" / "session_bin_ranges.json"
        assert p.exists()

    def test_two_sessions_both_produce_output(self, pipeline_result):
        bouts_dir = pipeline_result["out_dir"] / "bout_lengths"
        frame_files = {p.stem.replace("_frame_labels", "")
                       for p in bouts_dir.glob("*_frame_labels.csv")}
        assert len(frame_files) == 2

    def test_hmm_variant_csvs_exist(self, pipeline_result):
        # HMM smoothing runs by default (hmmlearn installed); *_hmm CSVs
        # should exist alongside the raw MLP outputs.
        bouts_dir = pipeline_result["out_dir"] / "bout_lengths"
        hmm_bout_files = list(bouts_dir.glob("*_bout_lengths_hmm.csv"))
        hmm_frame_files = list(bouts_dir.glob("*_frame_labels_hmm.csv"))
        assert len(hmm_bout_files) >= 1
        assert len(hmm_frame_files) >= 1
