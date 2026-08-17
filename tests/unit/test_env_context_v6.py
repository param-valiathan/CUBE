"""v6 Part 2 (Environmental Context): unit tests for the new pure
CUBE-authored functions in cube_core.py -- resolve_env_shapes (Step 1/4
shape resolution: reference + per_video transform/override + role_overrides),
the point-in-polygon / nearest-edge-distance geometry primitives,
compute_session_env_context (Step 4: per-bin time series, session summaries,
paradigm-gated derived metrics), and detect_approach_avoid_events (Step 7).
All pure/stateless; expected values are hand-derived or hand-verified.
"""
import numpy as np
import pandas as pd
import pytest

import cube_core as cc


def _square(x0, y0, size):
    return [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]


# ──────────────────────────────────────────────────────────────────────────
#  resolve_env_shapes (Step 1 data model + Step 4 resolution)
# ──────────────────────────────────────────────────────────────────────────

class TestResolveEnvShapes:
    def _cfg(self, per_video=None):
        return {
            "schema_version": 3, "paradigm": "custom",
            "reference_stem": "ref", "coord_space": "post_crop",
            "reference_shapes": {
                "boundary": {"name": "Arena", "kind": "boundary",
                             "vertices": _square(0, 0, 100), "role": None},
                "regions": [{"name": "R1", "kind": "region",
                             "vertices": _square(10, 10, 20), "role": "arm"}],
                "objects": [{"name": "O1", "kind": "object",
                             "vertices": _square(50, 50, 10), "role": "novel"}],
            },
            "per_video": per_video or {},
        }

    def test_none_or_empty_cfg_returns_empty_list(self):
        assert cc.resolve_env_shapes(None, "any") == []
        assert cc.resolve_env_shapes({}, "any") == []

    def test_identity_default_for_unlisted_video(self):
        shapes = cc.resolve_env_shapes(self._cfg(), "unrelated_stem")
        names = {s["name"] for s in shapes}
        assert names == {"Arena", "R1", "O1"}
        r1 = next(s for s in shapes if s["name"] == "R1")
        assert r1["vertices"] == _square(10, 10, 20)
        assert r1["role"] == "arm"

    def test_transform_mode_translates_every_shape(self):
        cfg = self._cfg(per_video={"vid2": {"mode": "transform", "translate": [5, -3]}})
        shapes = cc.resolve_env_shapes(cfg, "vid2")
        r1 = next(s for s in shapes if s["name"] == "R1")
        assert r1["vertices"] == [(15, 7), (35, 7), (35, 27), (15, 27)]
        # reference untouched
        assert cfg["reference_shapes"]["regions"][0]["vertices"] == _square(10, 10, 20)

    def test_override_mode_substitutes_full_shape_set(self):
        override_shapes = {"boundary": None,
                            "regions": [{"name": "R_override", "kind": "region",
                                         "vertices": _square(0, 0, 5), "role": None}],
                            "objects": []}
        cfg = self._cfg(per_video={"vid3": {"mode": "override", "shapes": override_shapes}})
        shapes = cc.resolve_env_shapes(cfg, "vid3")
        assert [s["name"] for s in shapes] == ["R_override"]

    def test_role_overrides_independent_of_position_mode(self):
        cfg = self._cfg(per_video={"vid4": {"mode": "transform", "translate": [0, 0],
                                             "role_overrides": {"O1": "familiar"}}})
        shapes = cc.resolve_env_shapes(cfg, "vid4")
        o1 = next(s for s in shapes if s["name"] == "O1")
        assert o1["role"] == "familiar"
        r1 = next(s for s in shapes if s["name"] == "R1")
        assert r1["role"] == "arm"  # untouched


# ──────────────────────────────────────────────────────────────────────────
#  Point-in-polygon / nearest-edge-distance geometry primitives
# ──────────────────────────────────────────────────────────────────────────

class TestGeometryPrimitives:
    SQUARE = _square(0, 0, 10)  # (0,0)-(10,0)-(10,10)-(0,10)

    def test_points_in_polygon_inside_and_outside(self):
        xs = np.array([5.0, 15.0, -1.0])
        ys = np.array([5.0, 5.0, 5.0])
        inside = cc._points_in_polygon(xs, ys, self.SQUARE)
        assert list(inside) == [True, False, False]

    def test_points_in_polygon_degenerate_shape_returns_all_false(self):
        xs = np.array([1.0])
        ys = np.array([1.0])
        assert list(cc._points_in_polygon(xs, ys, [(0, 0), (1, 1)])) == [False]

    def test_nearest_edge_distance_center_and_outside(self):
        xs = np.array([5.0, 20.0])
        ys = np.array([5.0, 5.0])
        d = cc._nearest_edge_distances(xs, ys, self.SQUARE)
        assert d[0] == pytest.approx(5.0)   # center of a 10x10 square -> 5 to nearest edge
        assert d[1] == pytest.approx(10.0)  # 20,5 -> 10 px from the x=10 edge


# ──────────────────────────────────────────────────────────────────────────
#  compute_session_env_context (Step 4)
# ──────────────────────────────────────────────────────────────────────────

class TestComputeSessionEnvContext:
    FPS = 30.0
    BODYPARTS = ["nose", "tailbase"]

    def _stationary_xy(self, cx, cy, n_frames):
        xs = np.full((n_frames, 2), cx)
        ys = np.full((n_frames, 2), cy)
        return xs, ys

    def test_none_env_cfg_is_noop(self):
        xs, ys = self._stationary_xy(5, 5, 30)
        assert cc.compute_session_env_context(None, "s1", xs, ys, self.BODYPARTS, self.FPS) == {}

    def test_no_traced_shapes_is_noop(self):
        cfg = {"schema_version": 3, "paradigm": "custom", "reference_stem": "s1",
               "coord_space": "post_crop",
               "reference_shapes": {"boundary": None, "regions": [], "objects": []},
               "per_video": {}}
        xs, ys = self._stationary_xy(5, 5, 30)
        assert cc.compute_session_env_context(cfg, "s1", xs, ys, self.BODYPARTS, self.FPS) == {}

    def test_time_in_region_matches_hand_computed(self):
        # animal sits inside "R1" the whole 1-second (30-frame) session.
        cfg = {"schema_version": 3, "paradigm": "open_field", "reference_stem": "s1",
               "coord_space": "post_crop",
               "reference_shapes": {
                   "boundary": None,
                   "regions": [{"name": "R1", "kind": "region",
                                "vertices": _square(0, 0, 20), "role": None}],
                   "objects": []},
               "per_video": {}}
        xs, ys = self._stationary_xy(5, 5, 30)
        out = cc.compute_session_env_context(cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)
        assert out["summary"]["time_in_region_sec"]["R1"] == pytest.approx(1.0)
        assert out["summary"]["region_entries_count"]["R1"] == 1

    def test_y_maze_alternation_with_hub_excluded(self):
        arm_a, arm_b, arm_c = _square(0, 0, 10), _square(100, 0, 10), _square(50, 100, 10)
        hub = _square(40, 40, 20)
        cfg = {"schema_version": 3, "paradigm": "y_maze", "reference_stem": "s1",
               "coord_space": "post_crop",
               "reference_shapes": {
                   "boundary": None,
                   "regions": [
                       {"name": "Arm A", "kind": "region", "vertices": arm_a, "role": "arm"},
                       {"name": "Arm B", "kind": "region", "vertices": arm_b, "role": "arm"},
                       {"name": "Arm C", "kind": "region", "vertices": arm_c, "role": "arm"},
                       {"name": "Hub",   "kind": "region", "vertices": hub,   "role": "hub"},
                   ],
                   "objects": []},
               "per_video": {}}
        centers = {"Arm A": (5, 5), "Arm B": (105, 5), "Arm C": (55, 105), "Hub": (50, 50)}
        seq = ["Arm A", "Hub", "Arm B", "Hub", "Arm C", "Hub", "Arm A"]
        per_seg = 9  # 3 bins/segment @ 100ms bins, 30fps -> 9 frames
        xs_l, ys_l = [], []
        for s in seq:
            cx, cy = centers[s]
            xs_l += [cx] * per_seg
            ys_l += [cy] * per_seg
        n = len(xs_l)
        xs = np.column_stack([xs_l, xs_l])
        ys = np.column_stack([ys_l, ys_l])
        out = cc.compute_session_env_context(cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)
        # arm sequence (hub excluded): A, B, C, A -> triplets (A,B,C)=alt, (B,C,A)=alt -> 100%
        assert out["derived"]["spontaneous_alternation_pct"] == pytest.approx(100.0)
        assert out["derived"]["n_arm_entries"] == 4

    def test_y_maze_alternation_skipped_below_min_arms(self):
        arm_a = _square(0, 0, 10)
        cfg = {"schema_version": 3, "paradigm": "y_maze", "reference_stem": "s1",
               "coord_space": "post_crop",
               "reference_shapes": {
                   "boundary": None,
                   "regions": [{"name": "Arm A", "kind": "region", "vertices": arm_a, "role": "arm"}],
                   "objects": []},
               "per_video": {}}
        xs, ys = self._stationary_xy(5, 5, 30)
        out = cc.compute_session_env_context(cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)
        assert "spontaneous_alternation_pct" not in out["derived"]

    def test_novel_object_discrimination_index_uses_role_not_name(self):
        # object literally named "Banana" tagged role=novel must still work.
        obj_novel = {"name": "Banana", "kind": "object", "vertices": _square(0, 0, 6), "role": "novel"}
        obj_fam   = {"name": "Rock",   "kind": "object", "vertices": _square(100, 100, 6), "role": "familiar"}
        cfg = {"schema_version": 3, "paradigm": "novel_object", "reference_stem": "s1",
               "coord_space": "post_crop",
               "reference_shapes": {"boundary": None, "regions": [],
                                     "objects": [obj_novel, obj_fam]},
               "per_video": {}}
        # animal spends the whole session investigating "Banana" (nose inside it)
        xs, ys = self._stationary_xy(3, 3, 60)
        out = cc.compute_session_env_context(cfg, "s1", xs, ys, self.BODYPARTS, self.FPS)
        assert out["derived"]["discrimination_index"] == pytest.approx(1.0)

    def test_per_video_role_override_flips_place_preference_index(self):
        chamber_a = {"name": "Chamber A", "kind": "region", "vertices": _square(0, 0, 10), "role": "paired"}
        chamber_b = {"name": "Chamber B", "kind": "region", "vertices": _square(100, 0, 10), "role": "unpaired"}
        cfg = {"schema_version": 3, "paradigm": "place_preference", "reference_stem": "ref",
               "coord_space": "post_crop",
               "reference_shapes": {"boundary": None, "regions": [chamber_a, chamber_b], "objects": []},
               "per_video": {"vid_flip": {"mode": "transform", "translate": [0, 0],
                                          "role_overrides": {"Chamber A": "unpaired",
                                                              "Chamber B": "paired"}}}}
        xs, ys = self._stationary_xy(5, 5, 30)  # sits in Chamber A the whole time
        out_ref  = cc.compute_session_env_context(cfg, "ref", xs, ys, self.BODYPARTS, self.FPS)
        out_flip = cc.compute_session_env_context(cfg, "vid_flip", xs, ys, self.BODYPARTS, self.FPS)
        assert out_ref["derived"]["preference_index"] == pytest.approx(1.0)
        assert out_flip["derived"]["preference_index"] == pytest.approx(-1.0)


# ──────────────────────────────────────────────────────────────────────────
#  detect_approach_avoid_events (Step 7)
# ──────────────────────────────────────────────────────────────────────────

class TestDetectApproachAvoidEvents:
    FPS = 30.0
    WIN = 3

    def _bout_df(self):
        return pd.DataFrame({
            "B-SOiD labels": [1, 2, 3],
            "Start time (frames)": [0, 30, 60],
            "Run lengths": [30, 30, 20],
            "straightness_ratio": [0.95, 0.1, 0.05],
            "mean_speed_px_s": [200.0, 3.0, 4.0],
        })

    def _env_pb(self, n_bins=37):
        dist = [50.0] * n_bins
        for b in range(8, 12):
            if b < n_bins:
                dist[b] = 5.0
        return {"current_region": [None] * n_bins,
                "dist_to_nearest_object_nose": {"Obj A": dist}}

    def test_detects_one_event_with_correct_target(self):
        out = cc.detect_approach_avoid_events(self._bout_df(), self._env_pb(), self.FPS, self.WIN)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["approach_target_object"] == "Obj A"
        assert row["start_frame"] == 0 and row["end_frame"] == 29
        assert row["next_bout_label"] == 2

    def test_missing_kinematics_columns_is_noop(self):
        df = self._bout_df().drop(columns=["straightness_ratio"])
        out = cc.detect_approach_avoid_events(df, self._env_pb(), self.FPS, self.WIN)
        assert out.empty
        assert list(out.columns) == [
            "start_frame", "end_frame", "duration_s", "straightness_ratio",
            "mean_speed_px_s", "next_bout_label", "next_bout_mean_speed_px_s",
            "approach_target_region", "approach_target_object"]

    def test_empty_env_pb_is_noop(self):
        out = cc.detect_approach_avoid_events(self._bout_df(), {}, self.FPS, self.WIN)
        assert out.empty

    def test_no_directed_bout_followed_by_slow_bout_yields_no_events(self):
        df = self._bout_df().copy()
        df["straightness_ratio"] = [0.1, 0.1, 0.1]  # nothing qualifies as directed
        out = cc.detect_approach_avoid_events(df, self._env_pb(), self.FPS, self.WIN)
        assert out.empty
