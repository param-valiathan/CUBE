"""v6 Part 1 (Kinematic Directedness): unit tests for the new pure
CUBE-authored functions in cube_core.py -- enrich_bouts_from_bin_source
(Step 2, the shared per-bin-to-bout join utility) and
compute_bout_directedness (Step 3, per-bout straightness/speed/heading
metrics). Both are pure/stateless; expected values are hand-derived.
"""
import numpy as np
import pandas as pd
import pytest

import cube_core as cc


def bout_df(rows):
    """rows: list of (label, start_frame, run_len)."""
    return pd.DataFrame(rows, columns=["B-SOiD labels", "Start time (frames)", "Run lengths"])


# ──────────────────────────────────────────────────────────────────────────
#  enrich_bouts_from_bin_source (Step 2)
# ──────────────────────────────────────────────────────────────────────────

class TestEnrichBoutsFromBinSource:
    # win=10 frames/bin; session occupies global bins [5..14] (bin_offset=5).
    WIN = 10
    BIN_OFFSET = 5
    ARR = np.arange(20, dtype=float)  # global per-bin array, values 0..19

    def test_single_array_mean_aggregation_matches_hand_computed(self):
        df = bout_df([
            (0, 0, 10),   # frames 0-9   -> local bin 0   -> global bin 5
            (1, 10, 20),  # frames 10-29 -> local bins 1-2 -> global bins 6-7
            (2, 90, 10),  # frames 90-99 -> local bin 9   -> global bin 14
        ])
        out = cc.enrich_bouts_from_bin_source(
            df, self.ARR, bin_offset=self.BIN_OFFSET, win=self.WIN,
            agg_fns=lambda seg: seg.mean(), out_col_names="mean_val")
        assert out.loc[0, "mean_val"] == pytest.approx(5.0)
        assert out.loc[1, "mean_val"] == pytest.approx((6 + 7) / 2)
        assert out.loc[2, "mean_val"] == pytest.approx(14.0)

    def test_bout_spanning_exactly_one_bin(self):
        df = bout_df([(0, 20, 5)])  # frames 20-24, entirely within local bin 2 -> global bin 7
        out = cc.enrich_bouts_from_bin_source(
            df, self.ARR, bin_offset=self.BIN_OFFSET, win=self.WIN,
            agg_fns=lambda seg: seg.mean())
        assert out.loc[0, "value"] == pytest.approx(7.0)

    def test_bout_at_start_of_session_range_clamped_not_negative(self):
        # A start_frame that would map below bin_offset (shouldn't normally
        # happen, but the clamp mirrors attach_centroid_distance exactly).
        df = bout_df([(0, 0, 1)])
        out = cc.enrich_bouts_from_bin_source(
            df, self.ARR, bin_offset=0, win=self.WIN,
            agg_fns=lambda seg: seg.mean())
        assert out.loc[0, "value"] == pytest.approx(0.0)

    def test_bout_beyond_array_end_clamps_to_last_valid_bin(self):
        df = bout_df([(0, 1000, 10)])  # far beyond the 20-bin array
        out = cc.enrich_bouts_from_bin_source(
            df, self.ARR, bin_offset=self.BIN_OFFSET, win=self.WIN,
            agg_fns=lambda seg: seg.mean())
        assert out.loc[0, "value"] == pytest.approx(19.0)

    def test_multi_column_dict_aggregator_case(self):
        df = bout_df([(0, 0, 10)])
        arr_b = self.ARR * 2
        out = cc.enrich_bouts_from_bin_source(
            df, {"a": self.ARR, "b": arr_b},
            bin_offset=self.BIN_OFFSET, win=self.WIN,
            agg_fns={"a": lambda s: s.mean(), "b": lambda s: s.max()},
            out_col_names={"a": "col_a", "b": "col_b"})
        assert out.loc[0, "col_a"] == pytest.approx(5.0)
        assert out.loc[0, "col_b"] == pytest.approx(10.0)

    def test_empty_bout_df_returns_empty_with_columns_present(self):
        out = cc.enrich_bouts_from_bin_source(
            bout_df([]), self.ARR, bin_offset=self.BIN_OFFSET, win=self.WIN,
            agg_fns=lambda s: s.mean(), out_col_names="mean_val")
        assert out.empty
        assert "mean_val" in out.columns

    def test_aggregator_exception_yields_nan_not_crash(self):
        df = bout_df([(0, 0, 10)])
        def _boom(seg):
            raise ValueError("synthetic aggregator failure")
        out = cc.enrich_bouts_from_bin_source(
            df, self.ARR, bin_offset=self.BIN_OFFSET, win=self.WIN,
            agg_fns=_boom, out_col_names="v")
        assert np.isnan(out.loc[0, "v"])

    def test_does_not_mutate_input_dataframe(self):
        df = bout_df([(0, 0, 10)])
        cols_before = list(df.columns)
        cc.enrich_bouts_from_bin_source(
            df, self.ARR, bin_offset=self.BIN_OFFSET, win=self.WIN,
            agg_fns=lambda s: s.mean(), out_col_names="mean_val")
        assert list(df.columns) == cols_before


# ──────────────────────────────────────────────────────────────────────────
#  compute_bout_directedness (Step 3)
# ──────────────────────────────────────────────────────────────────────────

class TestComputeBoutDirectedness:
    FPS = 30

    def test_straight_line_path_has_straightness_near_one(self):
        n = 100
        xy = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
        df = bout_df([(0, 0, 100)])
        out = cc.compute_bout_directedness(df, xy, self.FPS)
        assert out.loc[0, "straightness_ratio"] == pytest.approx(1.0)
        assert out.loc[0, "heading_consistency"] > 0.99
        assert out.loc[0, "net_displacement_px"] == pytest.approx(99.0)
        assert out.loc[0, "path_length_px"] == pytest.approx(99.0)
        assert out.loc[0, "mean_speed_px_s"] == pytest.approx(99.0 / (100 / self.FPS))

    def test_random_walk_path_has_low_straightness(self):
        n = 100
        rng = np.random.default_rng(42)
        steps = rng.normal(0, 1, size=(n - 1, 2))
        xy = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
        df = bout_df([(0, 0, 100)])
        out = cc.compute_bout_directedness(df, xy, self.FPS)
        # A random walk's net displacement grows like sqrt(n) while path
        # length grows like n, so straightness/heading consistency should be
        # well below the straight-line path's ~1.0.
        assert out.loc[0, "straightness_ratio"] < 0.5
        assert out.loc[0, "heading_consistency"] < 0.5

    def test_min_bout_length_guard_produces_nan_on_short_bout(self):
        # default min_bout_frames=5; this bout has only 3 frames.
        xy = np.column_stack([np.arange(3, dtype=float), np.zeros(3)])
        df = bout_df([(0, 0, 3)])
        out = cc.compute_bout_directedness(df, xy, self.FPS)
        assert np.isnan(out.loc[0, "straightness_ratio"])
        assert np.isnan(out.loc[0, "heading_consistency"])
        # net displacement/path length/mean speed remain well-defined and computed
        assert out.loc[0, "net_displacement_px"] == pytest.approx(2.0)
        assert out.loc[0, "path_length_px"] == pytest.approx(2.0)

    def test_single_frame_bout_no_crash(self):
        xy = np.array([[5.0, 5.0]])
        df = bout_df([(0, 0, 1)])
        out = cc.compute_bout_directedness(df, xy, self.FPS)
        assert out.loc[0, "net_displacement_px"] == 0.0
        assert out.loc[0, "path_length_px"] == 0.0
        assert np.isnan(out.loc[0, "straightness_ratio"])

    def test_empty_bout_df_returns_empty_with_columns_present(self):
        xy = np.zeros((10, 2))
        out = cc.compute_bout_directedness(bout_df([]), xy, self.FPS)
        assert out.empty
        for col in ("net_displacement_px", "path_length_px", "straightness_ratio",
                    "mean_speed_px_s", "heading_consistency"):
            assert col in out.columns

    def test_zero_path_length_yields_nan_straightness_not_divzero(self):
        # bout stays perfectly still -- path_length is 0, so straightness
        # (net/path) must be NaN rather than a ZeroDivisionError/inf.
        xy = np.full((10, 2), 3.0)
        df = bout_df([(0, 0, 10)])
        out = cc.compute_bout_directedness(df, xy, self.FPS)
        assert out.loc[0, "path_length_px"] == 0.0
        assert np.isnan(out.loc[0, "straightness_ratio"])
        assert np.isnan(out.loc[0, "heading_consistency"])
