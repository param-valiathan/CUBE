"""Phase T9 (optional, lower priority): headless logic tests for
cube_analyser.py's pure stats functions.

Given cube_analyser.py's unusually strong logic/UI separation, this covers
a handful of its module-level statistics functions with small synthetic
per-animal bout DataFrames: merge_bouts, compute_metrics, session_duration,
compute_combined, compute_transition_matrix. Figure-builder functions
(build_volcano_figure, etc.) are NOT covered here -- per the plan, T9 is
explicitly lower priority than T2-T8, and those functions require a full
theme/stats-dataframe contract that would need substantially more setup for
comparatively low additional risk coverage (this module already has strong
logic/UI separation; a plotting function's bugs tend to be visually loud,
consistent with the plan's own "test where bugs are silent" philosophy).
matplotlib is not exercised in this file, so no Agg backend setup is needed.
"""
import numpy as np
import pandas as pd
import pytest

import cube_analyser as ca


def make_bout_df(rows):
    """rows: list of (label, start_frame, run_len)."""
    return pd.DataFrame(rows, columns=["label", "start_frame", "run_len"])


class TestMergeBouts:
    def test_adjacent_bouts_of_same_label_merge(self):
        # merge_bouts filters to only the given label(s) BEFORE merging, so
        # a same-label bout is only "adjacent" if its start is within 1
        # frame of the previous (same-label) bout's end -- an intervening
        # different-label bout doesn't count as a bridge unless the frame
        # gap between the two SAME-label bouts is itself <= 1.
        df = make_bout_df([(0, 0, 10), (1, 10, 5), (0, 11, 10)])
        evs = ca.merge_bouts(df, {0})
        assert len(evs) == 1
        assert evs[0] == {"start": 0, "end": 21}

    def test_non_adjacent_bouts_stay_separate(self):
        df = make_bout_df([(0, 0, 10), (1, 10, 20), (0, 40, 10)])
        evs = ca.merge_bouts(df, {0})
        assert len(evs) == 2
        assert evs[0] == {"start": 0, "end": 10}
        assert evs[1] == {"start": 40, "end": 50}

    def test_empty_when_label_absent(self):
        df = make_bout_df([(1, 0, 10)])
        assert ca.merge_bouts(df, {0}) == []

    def test_multiple_labels_merged_together(self):
        # merge_bouts merges any bout whose label is IN the given set,
        # regardless of which specific label within the set it is.
        df = make_bout_df([(0, 0, 10), (2, 10, 10), (1, 20, 10)])
        evs = ca.merge_bouts(df, {0, 2})
        assert len(evs) == 1
        assert evs[0] == {"start": 0, "end": 20}


class TestComputeMetrics:
    def test_basic_metrics_hand_computed(self):
        fps = 10
        df = make_bout_df([(0, 0, 10), (1, 10, 20), (0, 40, 10)])
        groups = {"behaviorA": {"labels": [0]}}
        out = ca.compute_metrics(df, groups, fps)
        m = out["behaviorA"]
        assert m["frequency"] == 2
        # durations: (10-0)/10=1.0s, (50-40)/10=1.0s
        assert m["total_duration"] == pytest.approx(2.0)
        assert m["mean_bout"] == pytest.approx(1.0)
        assert m["latency"] == pytest.approx(0.0)

    def test_no_matching_events_returns_zeroed_dict(self):
        fps = 10
        df = make_bout_df([(1, 0, 10)])
        groups = {"behaviorA": {"labels": [0]}}
        out = ca.compute_metrics(df, groups, fps)
        m = out["behaviorA"]
        assert m["total_duration"] == 0.0
        assert m["frequency"] == 0
        assert m["latency"] is None
        assert m["events"] == []

    def test_multiple_groups_computed_independently(self):
        fps = 10
        df = make_bout_df([(0, 0, 10), (1, 10, 10)])
        groups = {"A": {"labels": [0]}, "B": {"labels": [1]}}
        out = ca.compute_metrics(df, groups, fps)
        assert out["A"]["frequency"] == 1
        assert out["B"]["frequency"] == 1


class TestSessionDuration:
    def test_session_duration_matches_last_bout_end(self):
        fps = 10
        df = make_bout_df([(0, 0, 10), (1, 10, 30)])
        assert ca.session_duration(df, fps) == pytest.approx(40 / 10)


class TestComputeCombined:
    def _make_animal(self, uid, name, exp_group, fps=10):
        df = make_bout_df([(0, 0, 10), (1, 10, 10)])
        return dict(uid=uid, name=name, df=df, fps=fps, exp_group=exp_group)

    def test_empty_animal_data_raises(self):
        with pytest.raises(ValueError):
            ca.compute_combined([], groups={"A": {"labels": [0]}})

    def test_missing_required_keys_raises(self):
        with pytest.raises(ValueError):
            ca.compute_combined([{"uid": 1}], groups={"A": {"labels": [0]}})

    def test_duplicate_uid_raises(self):
        a1 = self._make_animal(1, "a1", "ctrl")
        a2 = self._make_animal(1, "a2", "ctrl")
        with pytest.raises(ValueError):
            ca.compute_combined([a1, a2], groups={"A": {"labels": [0]}})

    def test_basic_grand_aggregate_shape(self):
        a1 = self._make_animal(1, "a1", "ctrl")
        a2 = self._make_animal(2, "a2", "treated")
        result = ca.compute_combined([a1, a2], groups={"A": {"labels": [0]}})
        assert "records" in result
        assert "grand" in result
        assert set(result["grand"].keys()) == {"ctrl", "treated"}
        assert result["grand"]["ctrl"]["A"]["frequency"]["n"] == 1


class TestComputeTransitionMatrix:
    def test_row_stochastic_normalisation(self):
        # label sequence via start_frame ordering: 0 -> 1 -> 0 -> 2
        df = make_bout_df([(0, 0, 5), (1, 5, 5), (0, 10, 5), (2, 15, 5)])
        mat, cids = ca.compute_transition_matrix(df)
        assert cids == [0, 1, 2]
        # each row should sum to 1 (or 0 if the cluster never transitions out)
        row_sums = mat.sum(axis=1)
        for i, rs in enumerate(row_sums):
            if mat[i].any():
                assert rs == pytest.approx(1.0)

    def test_self_transitions_ignored(self):
        # consecutive bouts with the SAME label should not count as a transition
        df = make_bout_df([(0, 0, 5), (0, 5, 5), (1, 10, 5)])
        mat, cids = ca.compute_transition_matrix(df)
        i0 = cids.index(0)
        i1 = cids.index(1)
        assert mat[i0, i0] == 0.0
        assert mat[i0, i1] == pytest.approx(1.0)

    def test_single_cluster_no_transitions(self):
        df = make_bout_df([(0, 0, 5), (0, 5, 5)])
        mat, cids = ca.compute_transition_matrix(df)
        assert cids == [0]
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0


class TestClusterKinematicsJoin:
    """v6 K1: compute_cluster_kinematics.csv wiring into cube_analyser.py's
    per-cluster metrics table (find_cluster_kinematics/load_cluster_kinematics/
    compute_per_cluster_metrics' kinematics_df param). Covers the plan's own
    verification bullets: file found + joined correctly, and graceful
    degradation (no crash, NaN not KeyError) when the file is absent."""

    def _make_run(self, tmp_path, write_kinematics=True):
        out_dir = tmp_path / "run1"
        bout_dir = out_dir / "bout_lengths"
        bout_dir.mkdir(parents=True)
        bout_path = bout_dir / "sess1_bout_lengths_hmm.csv"
        make_bout_df([(0, 0, 10), (1, 10, 10), (2, 20, 10)]).rename(
            columns={"label": "B-SOiD labels", "start_frame": "Start time (frames)",
                     "run_len": "Run lengths"}
        ).to_csv(bout_path, index=False)
        if write_kinematics:
            pd.DataFrame({
                "cluster_id": [0, 1, 2],
                "n_frames": [100, 50, 50],
                "mean_speed_px_s": [1.5, 2.5, 3.5],
                "mean_body_elongation_px": [10.0, 11.0, 12.0],
                "mean_angular_velocity_rad_s": [0.1, 0.2, 0.3],
            }).to_csv(out_dir / "cluster_kinematics.csv", index=False)
        return bout_path

    def test_find_and_load_kinematics_when_present(self, tmp_path):
        bout_path = self._make_run(tmp_path, write_kinematics=True)
        found = ca.find_cluster_kinematics(bout_path)
        assert found is not None and found.name == "cluster_kinematics.csv"
        kin = ca.load_cluster_kinematics(bout_path)
        assert kin is not None
        assert list(kin.index) == [0, 1, 2]
        assert kin.loc[1, "mean_speed_px_s"] == pytest.approx(2.5)

    def test_load_kinematics_returns_none_when_absent(self, tmp_path):
        bout_path = self._make_run(tmp_path, write_kinematics=False)
        assert ca.find_cluster_kinematics(bout_path) is None
        assert ca.load_cluster_kinematics(bout_path) is None

    def test_compute_per_cluster_metrics_joins_kinematics_columns(self, tmp_path):
        bout_path = self._make_run(tmp_path, write_kinematics=True)
        df = ca.load_csv(bout_path)
        kin = ca.load_cluster_kinematics(bout_path)
        pcm = ca.compute_per_cluster_metrics(df, fps=30, kinematics_df=kin)
        assert pcm.loc[0, "mean_speed_px_s"] == pytest.approx(1.5)
        assert pcm.loc[1, "mean_body_elongation_px"] == pytest.approx(11.0)
        assert pcm.loc[2, "mean_angular_velocity_rad_s"] == pytest.approx(0.3)
        # pre-existing columns untouched
        assert pcm.loc[0, "frequency"] == 1

    def test_compute_per_cluster_metrics_degrades_gracefully_without_kinematics(self, tmp_path):
        bout_path = self._make_run(tmp_path, write_kinematics=False)
        df = ca.load_csv(bout_path)
        # No kinematics_df at all (legacy call signature) -- columns are
        # present (so metric-selection UI never KeyErrors) but NaN.
        pcm = ca.compute_per_cluster_metrics(df, fps=30)
        assert "mean_speed_px_s" in pcm.columns
        assert np.isnan(pcm.loc[0, "mean_speed_px_s"])
        # Core columns unaffected by the missing kinematics file.
        assert pcm.loc[0, "frequency"] == 1
        assert pcm.loc[0, "total_duration"] == pytest.approx(10 / 30)
