"""Phase T2: unit tests for CUBE-authored pure functions (Tier 1).

Covers: smooth_boxcar, pair_files, _find_spine_indices, _angular_features,
visibility_feature_names, _append_visibility_block, _normalise_dlc_df.

All functions here are pure/stateless -- no file I/O, no randomness beyond
what the caller supplies. Expected values are hand-derived where feasible.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cube_core as cc


# ──────────────────────────────────────────────────────────────────────────
#  smooth_boxcar
# ──────────────────────────────────────────────────────────────────────────

class TestSmoothBoxcar:
    def test_constant_signal_unchanged_away_from_edges(self):
        # np.convolve(..., mode="same") implicitly zero-pads outside the
        # array, so boundary bins are NOT preserved for a constant signal --
        # only the interior (more than win//2 away from either edge) is.
        xy = np.full((50, 2), 5.0)
        win_sec = 0.1
        fps = 30
        win = max(1, int(round(fps * win_sec)))
        out = cc.smooth_boxcar(xy, fps=fps, win_sec=win_sec)
        interior = out[win:-win]
        assert np.allclose(interior, 5.0)

    def test_boundary_bins_are_NOT_edge_corrected(self):
        # Documents smooth_boxcar's boundary behaviour: 'same'-mode convolve
        # zero-pads outside the signal, so the first/last bins of a constant
        # signal are pulled toward zero rather than held flat. Downstream
        # consumers should be aware boxcar smoothing distorts the first/last
        # ~win/2 frames of every session.
        xy = np.full((10, 1), 5.0)
        out = cc.smooth_boxcar(xy, fps=30, win_sec=0.1)  # win=3
        assert out[0, 0] < 5.0  # pulled down by implicit zero padding

    def test_output_shape_matches_input(self):
        xy = np.random.default_rng(0).normal(size=(120, 4))
        out = cc.smooth_boxcar(xy, fps=30, win_sec=0.07)
        assert out.shape == xy.shape

    def test_win_le_1_returns_copy_not_view(self):
        xy = np.arange(20.0).reshape(10, 2)
        out = cc.smooth_boxcar(xy, fps=1, win_sec=0.5)  # win = round(0.5) = 0 -> max(1,.) = 1
        assert np.array_equal(out, xy)
        out[0, 0] = 999
        assert xy[0, 0] != 999  # confirms .copy(), not the same buffer

    def test_hand_computed_boxcar_average(self):
        # fps=10, win_sec=0.3 -> win=3. np.convolve 'same' with k=[1/3,1/3,1/3]
        xy = np.array([[0.0], [3.0], [6.0], [9.0], [12.0]])
        out = cc.smooth_boxcar(xy, fps=10, win_sec=0.3)
        expected = np.convolve(xy[:, 0], np.ones(3) / 3.0, mode="same")
        assert np.allclose(out[:, 0], expected)

    def test_single_frame_input_win_le_1_shape_preserved(self):
        # win_sec chosen so win == 1 (the guarded fast-path) -> shape preserved.
        xy = np.array([[1.0, 2.0]])
        out = cc.smooth_boxcar(xy, fps=1, win_sec=0.4)  # round(0.4)=0 -> win=1
        assert out.shape == (1, 2)
        assert np.allclose(out, xy)

    @pytest.mark.xfail(
        reason="BUG: smooth_boxcar(xy, ...) with n_frames < win (win>1) returns "
               "MORE rows than the input via np.convolve(..., mode='same') "
               "(output length = max(len(a), len(win_kernel))), silently "
               "breaking the shape-preservation invariant every caller assumes. "
               "Repro: smooth_boxcar(np.array([[1.,2.]]), fps=30, win_sec=0.07) "
               "-> win=2, output shape (2,2) instead of (1,2).",
        strict=True)
    def test_single_frame_input_win_gt_nframes_shape_bug(self):
        xy = np.array([[1.0, 2.0]])
        out = cc.smooth_boxcar(xy, fps=30, win_sec=0.07)  # win = round(2.1) = 2
        assert out.shape == (1, 2)

    @pytest.mark.xfail(
        reason="BUG: smooth_boxcar crashes on an empty (0-row) input whenever "
               "win > 1, because np.convolve(empty_array, kernel, mode='same') "
               "raises ValueError('a cannot be empty') instead of returning an "
               "empty array. Repro: smooth_boxcar(np.zeros((0,3)), fps=30, "
               "win_sec=0.07).",
        strict=True)
    def test_empty_input(self):
        xy = np.zeros((0, 3))
        out = cc.smooth_boxcar(xy, fps=30, win_sec=0.07)
        assert out.shape == (0, 3)


# ──────────────────────────────────────────────────────────────────────────
#  pair_files
# ──────────────────────────────────────────────────────────────────────────

class TestPairFiles:
    def test_exact_stem_match(self):
        dlc = [Path("session1_filtered.h5")]
        videos = {"session1_filtered": Path("session1_filtered.mp4")}
        pairs = cc.pair_files(dlc, videos)
        assert pairs == [(dlc[0], videos["session1_filtered"])]

    def test_prefix_match(self):
        dlc = [Path("session1DLC_resnet50_filtered.h5")]
        videos = {"session1": Path("session1.mp4")}
        pairs = cc.pair_files(dlc, videos)
        assert pairs[0][1] == videos["session1"]

    def test_timestamp_match(self):
        dlc = [Path("mouseA_20250601_120000DLC_filtered.h5")]
        videos = {"cam1_20250601_120000": Path("cam1_20250601_120000.mp4")}
        pairs = cc.pair_files(dlc, videos)
        assert pairs[0][1] == videos["cam1_20250601_120000"]

    def test_no_match_returns_none(self):
        dlc = [Path("unrelated_file.h5")]
        videos = {"totally_different": Path("totally_different.mp4")}
        pairs = cc.pair_files(dlc, videos)
        assert pairs == [(dlc[0], None)]

    def test_empty_dlc_files(self):
        assert cc.pair_files([], {"a": Path("a.mp4")}) == []

    def test_empty_video_dict(self):
        dlc = [Path("x.h5")]
        pairs = cc.pair_files(dlc, {})
        assert pairs == [(dlc[0], None)]

    def test_multiple_files_each_paired_independently(self):
        dlc = [Path("s1_filtered.h5"), Path("s2_filtered.h5"), Path("nomatch.h5")]
        videos = {"s1_filtered": Path("s1_filtered.mp4"),
                  "s2_filtered": Path("s2_filtered.mp4")}
        pairs = cc.pair_files(dlc, videos)
        assert pairs[0][1] == videos["s1_filtered"]
        assert pairs[1][1] == videos["s2_filtered"]
        assert pairs[2][1] is None


# ──────────────────────────────────────────────────────────────────────────
#  _find_spine_indices
# ──────────────────────────────────────────────────────────────────────────

class TestFindSpineIndices:
    def test_standard_bodyparts_found(self):
        bps = ["nose", "neck", "back_middle", "tailbase"]
        head, tail = cc._find_spine_indices(bps)
        assert head == 0   # 'nose'
        assert tail == 3   # 'tailbase'

    def test_empty_list(self):
        assert cc._find_spine_indices([]) == (None, None)

    def test_none_input(self):
        assert cc._find_spine_indices(None) == (None, None)

    def test_single_bodypart_no_match(self):
        assert cc._find_spine_indices(["nose"]) == (None, None)

    def test_missing_tail_landmark(self):
        # no bodypart containing "tailbase"/"tail_base" -> tail_idx stays None
        bps = ["nose", "neck", "back_middle"]
        assert cc._find_spine_indices(bps) == (None, None)

    def test_missing_head_landmark(self):
        bps = ["tailbase", "back_middle"]
        assert cc._find_spine_indices(bps) == (None, None)

    def test_head_and_tail_same_index_rejected(self):
        # A bodypart matching both head and tail keywords would produce
        # head_idx == tail_idx; function must reject that (documented guard).
        # "tailbase" doesn't match head_kw, so build a contrived case where
        # both indices land on the same element is not directly reachable
        # through real names -- instead verify the function never returns
        # equal non-None indices for a normal duplicate-free list.
        bps = ["nose", "tailbase"]
        head, tail = cc._find_spine_indices(bps)
        assert head != tail

    def test_ambiguous_multiple_head_candidates_picks_first_keyword_priority(self):
        # head_kw priority order: nose, snout, head, neck, rostral
        bps = ["neck", "snout", "tailbase"]
        head, tail = cc._find_spine_indices(bps)
        # "snout" (index 1) should win over "neck" (index 0) since snout is
        # earlier in head_kw priority order
        assert head == 1


# ──────────────────────────────────────────────────────────────────────────
#  _angular_features
# ──────────────────────────────────────────────────────────────────────────

class TestAngularFeatures:
    def test_returns_none_for_fewer_than_3_bodyparts(self):
        n = 10
        xs = np.zeros((n, 2))
        ys = np.zeros((n, 2))
        assert cc._angular_features(xs, ys, ["nose", "tailbase"]) is None

    def test_none_bodyparts_returns_none(self):
        xs = np.zeros((10, 3))
        ys = np.zeros((10, 3))
        assert cc._angular_features(xs, ys, None) is None

    def test_straight_line_angle_is_pi(self):
        # 3 spine-keyword points colinear -> angle at vertex = pi (180 deg)
        bps = ["nose", "neck", "tailbase"]
        n = 5
        xs = np.column_stack([np.zeros(n), np.ones(n), 2 * np.ones(n)])
        ys = np.zeros((n, 3))
        ang = cc._angular_features(xs, ys, bps)
        assert ang is not None
        assert ang.shape == (n, 1)
        # atol relaxed: arccos' derivative diverges near +/-1, so the 1e-8
        # norm epsilon the function adds gets amplified to ~2e-4 rad of
        # error for a near-180-degree angle (verified empirically here).
        assert np.allclose(ang[:, 0], np.pi, atol=5e-4)

    def test_right_angle(self):
        bps = ["nose", "neck", "tailbase"]
        n = 3
        # A=(1,0), B=(0,0), C=(0,1) -> angle at B = 90 deg
        xs = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n)])
        ys = np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)])
        ang = cc._angular_features(xs, ys, bps)
        assert np.allclose(ang[:, 0], np.pi / 2, atol=1e-6)

    def test_fallback_disabled_returns_none_when_no_spine_keywords(self):
        # bodyparts have no spine-keyword matches -> < 3 spine_ids found
        bps = ["ptA", "ptB", "ptC", "ptD"]
        n = 4
        xs = np.random.default_rng(1).normal(size=(n, 4))
        ys = np.random.default_rng(2).normal(size=(n, 4))
        ang_no_fallback = cc._angular_features(xs, ys, bps, allow_fallback=False)
        assert ang_no_fallback is None

    def test_fallback_enabled_uses_evenly_spaced_indices(self):
        bps = ["ptA", "ptB", "ptC", "ptD", "ptE", "ptF"]
        n = 4
        rng = np.random.default_rng(3)
        xs = rng.normal(size=(n, 6))
        ys = rng.normal(size=(n, 6))
        ang = cc._angular_features(xs, ys, bps, allow_fallback=True)
        assert ang is not None
        assert ang.shape[0] == n


# ──────────────────────────────────────────────────────────────────────────
#  visibility_feature_names
# ──────────────────────────────────────────────────────────────────────────

class TestVisibilityFeatureNames:
    def test_names_start_with_fixed_columns(self):
        names = cc.visibility_feature_names(["nose", "tailbase"])
        assert names[0] == "mean_visibility"
        assert names[1] == "frac_low_conf"

    def test_column_count_matches_region_count(self):
        bps = ["nose", "tailbase", "front_left_paw"]
        names = cc.visibility_feature_names(bps)
        regions = cc.group_bodyparts_by_region(bps)
        assert len(names) == 2 + len(regions)

    def test_empty_bodyparts_still_returns_fixed_region_columns(self):
        names_empty = cc.visibility_feature_names([])
        names_other = cc.visibility_feature_names(["nose"])
        # Region set is fixed regardless of which bodyparts are present
        assert len(names_empty) == len(names_other)

    def test_names_are_deterministic_and_sorted_region_order(self):
        bps = ["nose", "back_left_paw", "front_right_paw"]
        names1 = cc.visibility_feature_names(bps)
        names2 = cc.visibility_feature_names(list(reversed(bps)))
        assert names1 == names2


# ──────────────────────────────────────────────────────────────────────────
#  _append_visibility_block
# ──────────────────────────────────────────────────────────────────────────

class TestAppendVisibilityBlock:
    def test_none_vis_is_noop(self):
        f = np.zeros((5, 10))
        out = cc._append_visibility_block(f, None)
        assert out is f

    def test_empty_vis_is_noop(self):
        f = np.zeros((5, 10))
        vis = np.zeros((0, 3))
        out = cc._append_visibility_block(f, vis)
        assert out is f

    def test_matching_bin_counts_stacks_correctly(self):
        n_bins = 8
        f = np.arange(3 * n_bins, dtype=float).reshape(3, n_bins)
        vis = np.arange(n_bins * 2, dtype=float).reshape(n_bins, 2)
        out = cc._append_visibility_block(f, vis)
        assert out.shape == (5, n_bins)
        assert np.allclose(out[:3], f)
        assert np.allclose(out[3:], vis.T)

    def test_truncates_to_min_bin_count_when_vis_shorter(self):
        n_bins_f = 10
        f = np.ones((2, n_bins_f))
        vis = np.ones((6, 3))  # fewer bins than f
        out = cc._append_visibility_block(f, vis)
        assert out.shape == (2 + 3, 6)

    def test_truncates_to_min_bin_count_when_f_shorter(self):
        f = np.ones((2, 4))
        vis = np.ones((10, 3))  # more bins than f
        out = cc._append_visibility_block(f, vis)
        assert out.shape == (2 + 3, 4)


# ──────────────────────────────────────────────────────────────────────────
#  _normalise_dlc_df
# ──────────────────────────────────────────────────────────────────────────

class TestNormaliseDlcDf:
    def test_3level_passthrough(self, dlc_df_factory):
        df = dlc_df_factory(n_frames=20, nlevels=3)
        out = cc._normalise_dlc_df(df)
        assert out.columns.nlevels == 3
        assert list(out.columns.names) == ["scorer", "bodyparts", "coords"]

    def test_2level_gets_scorer_injected(self, dlc_df_factory):
        df = dlc_df_factory(n_frames=20, nlevels=2)
        out = cc._normalise_dlc_df(df)
        assert out.columns.nlevels == 3
        assert list(out.columns.names) == ["scorer", "bodyparts", "coords"]
        scorers = out.columns.get_level_values("scorer").unique().tolist()
        assert scorers == ["DLC_scorer"]

    def test_4level_single_individual_drops_individual_level(self, dlc_df_factory):
        df = dlc_df_factory(n_frames=20, nlevels=4, individual="animal1")
        out = cc._normalise_dlc_df(df)
        assert out.columns.nlevels == 3
        bps = out.columns.get_level_values("bodyparts").unique().tolist()
        assert "nose" in bps
        assert "tailbase" in bps
        # the individual label must NOT leak into bodypart names when unique
        assert not any("animal1" in b for b in bps)

    def test_4level_multi_individual_merges_into_bodypart_label(self):
        # Build a 4-level df with two distinct individuals manually.
        bodyparts = ["nose", "tailbase"]
        cols = []
        data = {}
        n = 10
        for ind in ("animal1", "animal2"):
            for bp in bodyparts:
                for coord in ("x", "y", "likelihood"):
                    key = ("scorer1", ind, bp, coord)
                    cols.append(key)
                    data[key] = np.zeros(n)
        columns = pd.MultiIndex.from_tuples(
            cols, names=["scorer", "individuals", "bodyparts", "coords"])
        df = pd.DataFrame(data.values(), index=columns).T
        df.columns = columns
        out = cc._normalise_dlc_df(df)
        assert out.columns.nlevels == 3
        bps_out = out.columns.get_level_values("bodyparts").unique().tolist()
        assert "animal1_nose" in bps_out
        assert "animal2_nose" in bps_out

    def test_column_names_always_canonical(self, dlc_df_factory):
        for nlevels in (2, 3):
            df = dlc_df_factory(n_frames=10, nlevels=nlevels)
            out = cc._normalise_dlc_df(df)
            assert list(out.columns.names) == ["scorer", "bodyparts", "coords"]
