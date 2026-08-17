"""Phase T4: feature extraction tests (Tier 1, highest-value area).

Covers extract_features_v2, extract_features_3d, compute_session_visibility_
block, compute_visibility_features. This is CUBE's most novel and highest
silent-corruption-risk logic, so expected values are hand-derived from the
actual binning/feature-block formulas in cube_core.py rather than just
smoke-checked.
"""
from itertools import combinations
from math import comb

import numpy as np
import pytest

import cube_core as cc


# Unambiguous 5-bodypart set: exactly 3 spine keyword matches (nose, neck,
# tailbase) -> 1 angular feature column, no "back"-substring ambiguity from
# paw names (see _angular_features' keyword-priority algorithm).
BP5 = ["nose", "neck", "tailbase", "paw1", "paw2"]


def make_xy(n_frames, n_pts, seed=0, moving=True):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    xy = np.zeros((n_frames, n_pts * 2))
    for i in range(n_pts):
        cx, cy = 50.0 * (i + 1), 30.0 * (i + 1)
        if moving:
            xy[:, 2 * i] = cx + 10 * np.sin(t / 20.0 + i) + rng.normal(0, 0.1, n_frames)
            xy[:, 2 * i + 1] = cy + 8 * np.cos(t / 25.0 + i) + rng.normal(0, 0.1, n_frames)
        else:
            xy[:, 2 * i] = cx
            xy[:, 2 * i + 1] = cy
    return xy


def expected_n_features_v2(n_pts, fps, n_angles, long_lag_drift=False):
    n_pair = comb(n_pts, 2)
    win100 = max(1, int(round(fps / 10)))
    use_fine = fps >= 60
    f100 = n_pair + n_pts + n_pts          # dist + vel + accel
    f_coarse = n_pair + n_pts              # dist + vel, no accel
    f_fine = n_pair + n_pts if use_fine else 0
    f_withinbin = 2 * n_pts if win100 > 1 else 0
    f_persist = 4 if long_lag_drift else 2
    total = f100 + f_withinbin + f_persist + f_coarse + f_fine + n_angles
    return total


class TestExtractFeaturesV2Shape:
    @pytest.mark.parametrize("fps", [30, 60, 120])
    def test_output_shape_matches_bin_count_formula(self, fps):
        n_frames = 1200
        xy = make_xy(n_frames, len(BP5), seed=1)
        feats = cc.extract_features_v2(xy, fps, bodyparts=BP5)
        win100 = max(1, int(round(fps / 10)))
        expected_n_bins = n_frames // win100
        expected_n_feat = expected_n_features_v2(len(BP5), fps, n_angles=1)
        assert feats.shape == (expected_n_feat, expected_n_bins)

    def test_fine_scale_block_absent_below_60fps(self):
        xy = make_xy(900, len(BP5), seed=2)
        feats_30 = cc.extract_features_v2(xy, fps=30, bodyparts=BP5)
        feats_59 = cc.extract_features_v2(xy, fps=59, bodyparts=BP5)
        assert feats_30.shape[0] == expected_n_features_v2(len(BP5), 30, 1)
        assert feats_59.shape[0] == expected_n_features_v2(len(BP5), 59, 1)
        assert feats_30.shape[0] == feats_59.shape[0]  # both below the 60fps gate

    def test_fine_scale_block_present_at_60fps(self):
        xy = make_xy(1200, len(BP5), seed=3)
        feats_59 = cc.extract_features_v2(xy, fps=59, bodyparts=BP5)
        feats_60 = cc.extract_features_v2(xy, fps=60, bodyparts=BP5)
        assert feats_60.shape[0] > feats_59.shape[0]

    def test_too_short_recording_raises(self):
        xy = make_xy(5, len(BP5), seed=4)  # far fewer frames than 1 bin @ 30fps
        with pytest.raises(ValueError):
            cc.extract_features_v2(xy, fps=30, bodyparts=BP5)

    def test_no_bodyparts_no_angular_block(self):
        xy = make_xy(600, len(BP5), seed=5)
        feats = cc.extract_features_v2(xy, fps=30, bodyparts=None)
        expected = expected_n_features_v2(len(BP5), 30, n_angles=0)
        assert feats.shape[0] == expected

    def test_long_lag_drift_adds_two_persist_columns(self):
        xy = make_xy(1200, len(BP5), seed=6)
        feats_default = cc.extract_features_v2(xy, fps=30, bodyparts=BP5,
                                               long_lag_drift=False)
        feats_long = cc.extract_features_v2(xy, fps=30, bodyparts=BP5,
                                            long_lag_drift=True)
        assert feats_long.shape[0] == feats_default.shape[0] + 2
        assert feats_long.shape[1] == feats_default.shape[1]


class TestExtractFeaturesV2Scaling:
    def test_body_normalise_scales_pairwise_distances_by_inverse_spine_length(self):
        # Static (non-moving) points -> spine length is constant across bins,
        # so body_normalise should divide every distance feature by that
        # exact constant.
        n_frames = 900
        xy = make_xy(n_frames, len(BP5), seed=7, moving=False)
        feats_raw = cc.extract_features_v2(xy, fps=30, bodyparts=BP5,
                                           body_normalise=False)
        feats_norm = cc.extract_features_v2(xy, fps=30, bodyparts=BP5,
                                            body_normalise=True)
        head_idx, tail_idx = cc._find_spine_indices(BP5)
        assert head_idx is not None and tail_idx is not None
        nose_x, nose_y = xy[0, 2 * head_idx], xy[0, 2 * head_idx + 1]
        tail_x, tail_y = xy[0, 2 * tail_idx], xy[0, 2 * tail_idx + 1]
        spine_len = max(np.hypot(nose_x - tail_x, nose_y - tail_y), 10.0)

        # First pairwise-distance column (bodypart 0 vs 1) in the f100 block
        # is row 0 of the transposed feature matrix.
        raw_col0 = feats_raw[0, :]
        norm_col0 = feats_norm[0, :]
        # Distances are constant (static points) so this ratio should be
        # uniform across all bins.
        ratio = raw_col0 / norm_col0
        assert np.allclose(ratio, spine_len, rtol=1e-6)

    def test_bodypart_weights_scale_velocity_by_wi(self):
        # Give bodypart 0 a 2x weight; with all points static except bodypart
        # 0 moving, only bodypart 0's velocity feature should double.
        n_frames = 300
        n_pts = len(BP5)
        xy = make_xy(n_frames, n_pts, seed=8, moving=False).copy()
        # Inject a small constant per-frame displacement onto bodypart 0 only.
        t = np.arange(n_frames)
        xy[:, 0] += 0.5 * t  # bodypart 0's x drifts linearly -> nonzero velocity

        feats_unweighted = cc.extract_features_v2(
            xy, fps=30, bodyparts=BP5, body_normalise=False)
        feats_weighted = cc.extract_features_v2(
            xy, fps=30, bodyparts=BP5, body_normalise=False,
            bodypart_weights={"nose": 2.0})

        n_pair = comb(n_pts, 2)
        # f100 block column order: [pairwise dists (n_pair)] + [velocity (n_pts)] + [accel (n_pts)]
        vel_row_bp0 = n_pair + 0   # bodypart 0's velocity row index within f100
        raw_vel = feats_unweighted[vel_row_bp0, :]
        weighted_vel = feats_weighted[vel_row_bp0, :]
        nonzero = np.abs(raw_vel) > 1e-9
        assert nonzero.any()
        assert np.allclose(weighted_vel[nonzero] / raw_vel[nonzero], 2.0, rtol=1e-6)

    def test_default_weights_are_bit_identical_to_unweighted(self):
        xy = make_xy(600, len(BP5), seed=9)
        feats_a = cc.extract_features_v2(xy, fps=30, bodyparts=BP5,
                                         bodypart_weights=None)
        feats_b = cc.extract_features_v2(xy, fps=30, bodyparts=BP5,
                                         bodypart_weights={})
        assert np.array_equal(feats_a, feats_b)


# ──────────────────────────────────────────────────────────────────────────
#  extract_features_3d
# ──────────────────────────────────────────────────────────────────────────

def make_xyz(n_frames, n_pts, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    xyz = np.zeros((n_frames, n_pts * 3))
    for i in range(n_pts):
        cx, cy, cz = 50.0 * (i + 1), 30.0 * (i + 1), 10.0 * (i + 1)
        xyz[:, 3 * i] = cx + 10 * np.sin(t / 20.0 + i) + rng.normal(0, 0.1, n_frames)
        xyz[:, 3 * i + 1] = cy + 8 * np.cos(t / 25.0 + i) + rng.normal(0, 0.1, n_frames)
        xyz[:, 3 * i + 2] = cz + 5 * np.sin(t / 30.0 + i) + rng.normal(0, 0.1, n_frames)
    return xyz


def expected_n_features_3d(n_pts, fps, long_lag_drift=False, long_scale_bins=False):
    n_pair = comb(n_pts, 2)
    use_fine = fps >= 60
    f100 = n_pair + n_pts + n_pts
    f_persist = 4 if long_lag_drift else 2
    f_fine = n_pair + n_pts if use_fine else 0
    f_coarse = n_pair + n_pts
    f_500 = n_pair + n_pts if long_scale_bins else 0
    f_1000 = n_pair + n_pts if long_scale_bins else 0
    return f100 + f_persist + f_fine + f_coarse + f_500 + f_1000


class TestExtractFeatures3D:
    @pytest.mark.parametrize("fps", [30, 60, 120])
    def test_output_shape(self, fps):
        n_frames = 1200
        n_pts = 4
        xyz = make_xyz(n_frames, n_pts, seed=10)
        feats = cc.extract_features_3d(xyz, fps, bodyparts=None)
        win100 = max(1, int(round(fps / 10)))
        expected_bins = n_frames // win100
        expected_feat = expected_n_features_3d(n_pts, fps)
        assert feats.shape == (expected_feat, expected_bins)

    def test_too_short_raises(self):
        xyz = make_xyz(5, 4, seed=11)
        with pytest.raises(ValueError):
            cc.extract_features_3d(xyz, fps=30)

    def test_long_scale_bins_adds_two_blocks(self):
        n_frames = 3000  # long enough for 500ms/1000ms windows to be valid
        n_pts = 4
        xyz = make_xyz(n_frames, n_pts, seed=12)
        feats_default = cc.extract_features_3d(xyz, fps=30, long_scale_bins=False)
        feats_long = cc.extract_features_3d(xyz, fps=30, long_scale_bins=True)
        n_pair = comb(n_pts, 2)
        added = 2 * (n_pair + n_pts)
        assert feats_long.shape[0] == feats_default.shape[0] + added
        assert feats_long.shape[1] == feats_default.shape[1]


# ──────────────────────────────────────────────────────────────────────────
#  compute_visibility_features / compute_session_visibility_block
# ──────────────────────────────────────────────────────────────────────────

class TestVisibilityFeatures:
    def test_shape_matches_bin_formula(self):
        n_frames = 300
        n_pts = 4
        bodyparts = ["nose", "neck", "tailbase", "front_left_paw"]
        ll = np.full((n_frames, n_pts), 0.9)
        win = 10
        thresh = cc.compute_adaptive_visibility_threshold(ll, 0.3, 10)
        out = cc.compute_visibility_features(ll, bodyparts, win, thresh)
        n_regions = len(cc.group_bodyparts_by_region(bodyparts))
        assert out.shape == (n_frames // win, 2 + n_regions)

    def test_empty_ll_returns_zero_rows(self):
        out = cc.compute_visibility_features(
            np.zeros((0, 0)), [], 10, np.array([]))
        assert out.shape[0] == 0

    def test_low_confidence_bins_flagged(self):
        n_frames = 100
        n_pts = 2
        bodyparts = ["nose", "tailbase"]
        ll = np.full((n_frames, n_pts), 0.9)
        ll[50:, :] = 0.05   # second half fully occluded
        win = 10
        thresh = cc.compute_adaptive_visibility_threshold(ll, 0.3, 10)
        out = cc.compute_visibility_features(ll, bodyparts, win, thresh)
        # first half bins: frac_low_conf ~ 0; second half: frac_low_conf ~ 1
        first_half = out[:5, 1]
        second_half = out[5:, 1]
        assert np.allclose(first_half, 0.0, atol=1e-6)
        assert np.allclose(second_half, 1.0, atol=1e-6)

    def test_session_visibility_block_none_when_ll_missing(self):
        assert cc.compute_session_visibility_block(None, ["nose"], fps=30) is None
        assert cc.compute_session_visibility_block(
            np.zeros((0, 1)), ["nose"], fps=30) is None
        assert cc.compute_session_visibility_block(
            np.ones((10, 1)), [], fps=30) is None

    def test_session_visibility_block_shape(self):
        n_frames = 300
        bodyparts = ["nose", "tailbase"]
        ll = np.full((n_frames, len(bodyparts)), 0.9)
        out = cc.compute_session_visibility_block(ll, bodyparts, fps=30)
        assert out is not None
        win = max(1, int(round(30 / 10)))
        n_regions = len(cc.group_bodyparts_by_region(bodyparts))
        assert out.shape == (n_frames // win, 2 + n_regions)


class TestAppendVisibilityBlockIntegration:
    def test_visibility_block_appends_onto_real_v2_features(self):
        n_frames = 900
        bodyparts = BP5
        xy = make_xy(n_frames, len(bodyparts), seed=13)
        feats = cc.extract_features_v2(xy, fps=30, bodyparts=bodyparts)
        ll = np.full((n_frames, len(bodyparts)), 0.9)
        vis = cc.compute_session_visibility_block(ll, bodyparts, fps=30)
        combined = cc._append_visibility_block(feats, vis)
        n_regions = len(cc.group_bodyparts_by_region(bodyparts))
        assert combined.shape[0] == feats.shape[0] + 2 + n_regions
        # bin count should be min of the two (should already match exactly)
        assert combined.shape[1] == min(feats.shape[1], vis.shape[0])
