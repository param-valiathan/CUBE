"""CUBE_Analyser_Paradigm_Reporting_Plan.md (v6 part 3): unit coverage for
cube_analyser.py's pure, fixture-able logic -- the new one-sample-vs-
reference statistical primitive and the arena/region cross-tab's
aggregation logic (ground rule 13's required test-suite split: GUI panel
construction itself is covered by the manual/headless-Tk QA described in
the implementation report, not here). matplotlib is not exercised.
"""
import numpy as np
import pandas as pd
import pytest

import cube_analyser as ca


class TestOneSampleTest:
    def test_mean_far_from_ref_is_significant(self):
        vals = [0.30, 0.32, 0.28, 0.31, 0.29, 0.33]
        stat, p = ca.one_sample_test(vals, ref=0.0)
        assert p < 0.001
        assert stat > 0

    def test_mean_at_ref_is_not_significant(self):
        vals = [0.01, -0.02, 0.015, -0.01, 0.005, -0.005]
        stat, p = ca.one_sample_test(vals, ref=0.0)
        assert p > 0.3

    def test_fewer_than_two_values_returns_nan_pval_one(self):
        stat, p = ca.one_sample_test([0.5], ref=0.0)
        assert np.isnan(stat)
        assert p == 1.0
        stat, p = ca.one_sample_test([], ref=0.0)
        assert np.isnan(stat)
        assert p == 1.0

    def test_wilcoxon_mode_runs_and_flags_significant_shift(self):
        vals = [0.4, 0.45, 0.42, 0.38, 0.5, 0.41]
        stat, p = ca.one_sample_test(vals, ref=0.0, use_wilcoxon=True)
        assert p < 0.05

    def test_wilcoxon_all_equal_to_ref_returns_zero_stat_p_one(self):
        stat, p = ca.one_sample_test([0.0, 0.0, 0.0, 0.0], ref=0.0, use_wilcoxon=True)
        assert stat == 0.0
        assert p == 1.0

    def test_nan_values_are_dropped_before_testing(self):
        vals = [0.3, np.nan, 0.32, 0.29, np.nan, 0.31]
        stat, p = ca.one_sample_test(vals, ref=0.0)
        stat2, p2 = ca.one_sample_test([0.3, 0.32, 0.29, 0.31], ref=0.0)
        assert stat == pytest.approx(stat2)
        assert p == pytest.approx(p2)


class TestRunOneSampleStatistics:
    def test_two_groups_one_significant_one_not_bh_corrected(self):
        group_values = {
            "Control": [0.30, 0.32, 0.28, 0.31, 0.29, 0.33],
            "Treated": [0.01, -0.02, 0.015, -0.01, 0.005, -0.005],
        }
        df = ca.run_one_sample_statistics(group_values, ref=0.0)
        assert set(df["exp_group"]) == {"Control", "Treated"}
        assert "qval" in df.columns
        ctrl = df[df["exp_group"] == "Control"].iloc[0]
        treat = df[df["exp_group"] == "Treated"].iloc[0]
        assert ctrl["qval"] < 0.05
        assert treat["qval"] > 0.05
        assert ctrl["n"] == 6
        assert ctrl["test_type"] == "ttest_1samp"

    def test_empty_group_values_returns_empty_df_with_columns(self):
        df = ca.run_one_sample_statistics({}, ref=0.0)
        assert df.empty
        for col in ("exp_group", "n", "mean", "ref", "stat", "pval",
                    "effect_size_cohens_d", "test_type", "qval"):
            assert col in df.columns

    def test_group_with_single_value_gets_n1_pval1_no_crash(self):
        df = ca.run_one_sample_statistics({"Solo": [0.5]}, ref=0.0)
        row = df[df["exp_group"] == "Solo"].iloc[0]
        assert row["n"] == 1
        assert row["pval"] == 1.0
        assert np.isnan(row["effect_size_cohens_d"])

    def test_effect_size_is_nan_for_zero_variance_group(self):
        df = ca.run_one_sample_statistics({"Flat": [0.5, 0.5, 0.5, 0.5]}, ref=0.0)
        row = df[df["exp_group"] == "Flat"].iloc[0]
        assert np.isnan(row["effect_size_cohens_d"])

    def test_use_wilcoxon_flag_propagates_to_test_type(self):
        df = ca.run_one_sample_statistics(
            {"G": [0.1, 0.2, 0.15, 0.12, -0.05, 0.18]}, ref=0.0, use_wilcoxon=True)
        assert df.iloc[0]["test_type"] == "wilcoxon_1samp"


class TestAggregateRegionTimeByLabel:
    def test_hand_computed_weighted_mean(self):
        rows = [
            {"label": 0, "run_length": 100, "pct_time_in_region": {"arm_A": 1.0, "arm_B": 0.0}},
            {"label": 0, "run_length": 300, "pct_time_in_region": {"arm_A": 0.0, "arm_B": 1.0}},
        ]
        agg = ca.aggregate_region_time_by_label(rows, ["arm_A", "arm_B"])
        # weighted mean: arm_A = (100*1.0 + 300*0.0)/400 = 0.25; arm_B = 0.75
        assert agg[0]["arm_A"] == pytest.approx(0.25)
        assert agg[0]["arm_B"] == pytest.approx(0.75)

    def test_region_absent_from_a_row_contributes_zero_not_skipped(self):
        rows = [
            {"label": "Control", "run_length": 100, "pct_time_in_region": {"arm_A": 1.0}},
        ]
        agg = ca.aggregate_region_time_by_label(rows, ["arm_A", "arm_B"])
        assert agg["Control"]["arm_A"] == pytest.approx(1.0)
        assert agg["Control"]["arm_B"] == pytest.approx(0.0)

    def test_zero_or_negative_weight_rows_excluded(self):
        rows = [
            {"label": 0, "run_length": 0, "pct_time_in_region": {"arm_A": 1.0}},
            {"label": 0, "run_length": -5, "pct_time_in_region": {"arm_A": 1.0}},
        ]
        agg = ca.aggregate_region_time_by_label(rows, ["arm_A"])
        assert agg == {}

    def test_multiple_labels_kept_independent(self):
        rows = [
            {"label": "Control", "run_length": 100, "pct_time_in_region": {"arm_A": 0.8}},
            {"label": "Treated", "run_length": 100, "pct_time_in_region": {"arm_A": 0.2}},
        ]
        agg = ca.aggregate_region_time_by_label(rows, ["arm_A"])
        assert agg["Control"]["arm_A"] == pytest.approx(0.8)
        assert agg["Treated"]["arm_A"] == pytest.approx(0.2)

    def test_empty_rows_returns_empty_dict(self):
        assert ca.aggregate_region_time_by_label([], ["arm_A"]) == {}

    def test_empty_region_names_yields_empty_per_label_dict(self):
        rows = [{"label": 0, "run_length": 100, "pct_time_in_region": {"arm_A": 1.0}}]
        agg = ca.aggregate_region_time_by_label(rows, [])
        assert agg == {0: {}}


class TestAggregateRegionTimeByLabelAndAnimal:
    def test_two_animals_same_label_kept_independent(self):
        rows = [
            {"label": "Control", "animal": "M1", "run_length": 100,
             "pct_time_in_region": {"arm_A": 1.0, "arm_B": 0.0}},
            {"label": "Control", "animal": "M2", "run_length": 100,
             "pct_time_in_region": {"arm_A": 0.0, "arm_B": 1.0}},
        ]
        out = ca.aggregate_region_time_by_label_and_animal(rows, ["arm_A", "arm_B"])
        assert out["Control"]["M1"]["arm_A"] == pytest.approx(1.0)
        assert out["Control"]["M2"]["arm_A"] == pytest.approx(0.0)
        # per-animal values must NOT be averaged into a single pooled entry --
        # that's what aggregate_region_time_by_label already does.
        assert set(out["Control"].keys()) == {"M1", "M2"}

    def test_one_animal_multiple_rows_weighted_mean_matches_pooled(self):
        rows = [
            {"label": 0, "animal": "M1", "run_length": 100,
             "pct_time_in_region": {"arm_A": 1.0}},
            {"label": 0, "animal": "M1", "run_length": 300,
             "pct_time_in_region": {"arm_A": 0.0}},
        ]
        out = ca.aggregate_region_time_by_label_and_animal(rows, ["arm_A"])
        assert out[0]["M1"]["arm_A"] == pytest.approx(0.25)

    def test_empty_rows_returns_empty_dict(self):
        assert ca.aggregate_region_time_by_label_and_animal([], ["arm_A"]) == {}


class TestApproximateRegionOutline:
    def test_fewer_than_three_points_returns_empty(self):
        cx = np.array([1.0, 2.0])
        cy = np.array([1.0, 2.0])
        mask = np.array([True, True])
        assert ca.approximate_region_outline(cx, cy, mask) == []

    def test_convex_hull_of_a_square_returns_four_vertices(self):
        cx = np.array([0.0, 0.0, 10.0, 10.0, 5.0])
        cy = np.array([0.0, 10.0, 0.0, 10.0, 5.0])
        mask = np.array([True, True, True, True, True])
        verts = ca.approximate_region_outline(cx, cy, mask)
        assert len(verts) == 4  # the interior point (5,5) is not a hull vertex

    def test_nan_points_excluded_not_crashing(self):
        cx = np.array([0.0, 0.0, 10.0, 10.0, np.nan])
        cy = np.array([0.0, 10.0, 0.0, 10.0, np.nan])
        mask = np.array([True, True, True, True, True])
        verts = ca.approximate_region_outline(cx, cy, mask)
        assert len(verts) == 4


class TestSessionStemFromBoutPath:
    def test_strips_hmm_suffix(self):
        assert ca._session_stem_from_bout_path(
            "mouse1_bout_lengths_hmm.csv") == "mouse1"

    def test_strips_raw_suffix(self):
        assert ca._session_stem_from_bout_path(
            "mouse1_bout_lengths.csv") == "mouse1"

    def test_leaves_unrelated_stem_untouched(self):
        assert ca._session_stem_from_bout_path("mouse1.csv") == "mouse1"


class TestScalarMetricComputeFn:
    def test_wraps_scalar_into_cluster_indexed_dataframe(self):
        fn = ca._scalar_metric_compute_fn("alt_pct")
        df = fn({"alt_pct": 42.5})
        assert list(df.index) == [0]
        assert df.index.name == "cluster_id"
        assert df.loc[0, "alt_pct"] == 42.5

    def test_missing_key_yields_nan_not_keyerror(self):
        fn = ca._scalar_metric_compute_fn("missing_key")
        df = fn({})
        assert np.isnan(df.loc[0, "missing_key"])
