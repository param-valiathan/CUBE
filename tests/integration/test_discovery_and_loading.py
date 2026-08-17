"""Phase T3: file discovery, pairing, and loading tests (Tier 1).

Covers: find_dlc_files, find_videos (via tmp_path fake file trees),
pair_files matching logic (see also test_core_pure.py for pure-logic
cases), and load_dlc_file against synthetic .h5/.csv fixtures --
specifically the interpolation-vs-flat-hold behavior across a short gap
(interpolate) vs a long gap (flat-hold per max_interp_gap_frames).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cube_core as cc


# ──────────────────────────────────────────────────────────────────────────
#  find_dlc_files
# ──────────────────────────────────────────────────────────────────────────

class TestFindDlcFiles:
    def test_prefers_filtered_h5_over_csv_and_others(self, tmp_path):
        (tmp_path / "sessionA_filtered.h5").touch()
        (tmp_path / "sessionB_filtered.csv").touch()
        (tmp_path / "random.h5").touch()
        found = cc.find_dlc_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "sessionA_filtered.h5"

    def test_falls_back_to_filtered_csv_when_no_filtered_h5(self, tmp_path):
        (tmp_path / "sessionB_filtered.csv").touch()
        (tmp_path / "unrelated.h5").touch()
        found = cc.find_dlc_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "sessionB_filtered.csv"

    def test_falls_back_to_any_dlc_file_when_none_filtered(self, tmp_path):
        (tmp_path / "raw_session.h5").touch()
        (tmp_path / "raw_session2.csv").touch()
        found = cc.find_dlc_files(tmp_path)
        names = sorted(p.name for p in found)
        assert names == ["raw_session.h5", "raw_session2.csv"]

    def test_excludes_bsoid_prefixed_and_un_filtered_and_bout_lengths(self, tmp_path):
        (tmp_path / "BSOID_output.h5").touch()
        (tmp_path / "sessionUN_filtered.h5").touch()
        (tmp_path / "sessionA_bout_lengths.csv").touch()
        found = cc.find_dlc_files(tmp_path)
        assert found == []

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert cc.find_dlc_files(tmp_path) == []

    def test_recursive_search_into_subdirectories(self, tmp_path):
        sub = tmp_path / "nested" / "deeper"
        sub.mkdir(parents=True)
        (sub / "session_filtered.h5").touch()
        found = cc.find_dlc_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "session_filtered.h5"


# ──────────────────────────────────────────────────────────────────────────
#  find_videos
# ──────────────────────────────────────────────────────────────────────────

class TestFindVideos:
    def test_finds_supported_video_extensions(self, tmp_path):
        for name in ("a.mp4", "b.avi", "c.mov", "d.mkv", "e.wmv", "f.txt"):
            (tmp_path / name).touch()
        videos = cc.find_videos(tmp_path)
        assert set(videos.keys()) == {"a", "b", "c", "d", "e"}

    def test_empty_folder(self, tmp_path):
        assert cc.find_videos(tmp_path) == {}

    def test_recursive(self, tmp_path):
        sub = tmp_path / "vids"
        sub.mkdir()
        (sub / "clip1.mp4").touch()
        videos = cc.find_videos(tmp_path)
        assert "clip1" in videos
        assert videos["clip1"] == sub / "clip1.mp4"


# ──────────────────────────────────────────────────────────────────────────
#  find_dlc_files + find_videos + pair_files end-to-end on a fake tree
# ──────────────────────────────────────────────────────────────────────────

class TestDiscoveryPairingIntegration:
    def test_full_discovery_and_pairing_flow(self, tmp_path):
        (tmp_path / "mouse1_20250601_120000_filtered.h5").touch()
        (tmp_path / "mouse1_20250601_120000.mp4").touch()
        (tmp_path / "mouse2_nomatch_filtered.h5").touch()

        dlc = cc.find_dlc_files(tmp_path)
        videos = cc.find_videos(tmp_path)
        pairs = cc.pair_files(dlc, videos)

        by_stem = {p.stem: v for p, v in pairs}
        assert by_stem["mouse1_20250601_120000_filtered"] is not None
        assert by_stem["mouse2_nomatch_filtered"] is None


# ──────────────────────────────────────────────────────────────────────────
#  load_dlc_file — real file I/O, interpolation / flat-hold behavior
# ──────────────────────────────────────────────────────────────────────────

class TestLoadDlcFile:
    def test_load_h5_basic_shape(self, tmp_path, dlc_df_factory, write_h5):
        df = dlc_df_factory(n_frames=100, nlevels=3)
        path = write_h5(tmp_path / "session_filtered.h5", df)
        xy, bodyparts, fps_hint = cc.load_dlc_file(path)
        assert xy.shape == (100, len(bodyparts) * 2)
        assert len(bodyparts) == 6

    def test_load_csv_basic_shape(self, tmp_path, dlc_df_factory, write_csv):
        df = dlc_df_factory(n_frames=80, nlevels=3)
        path = write_csv(tmp_path / "session_filtered.csv", df)
        xy, bodyparts, fps_hint = cc.load_dlc_file(path)
        assert xy.shape[0] == 80
        assert xy.shape[1] == len(bodyparts) * 2

    def test_fps_hint_extracted_from_filename(self, tmp_path, dlc_df_factory, write_h5):
        df = dlc_df_factory(n_frames=50, nlevels=3)
        path = write_h5(tmp_path / "session_60fps_filtered.h5", df)
        _, _, fps_hint = cc.load_dlc_file(path)
        assert fps_hint == 60.0

    def test_no_fps_hint_when_absent(self, tmp_path, dlc_df_factory, write_h5):
        df = dlc_df_factory(n_frames=50, nlevels=3)
        path = write_h5(tmp_path / "plain_session_filtered.h5", df)
        _, _, fps_hint = cc.load_dlc_file(path)
        assert fps_hint is None

    def test_short_gap_is_linearly_interpolated(self, tmp_path, dlc_df_factory, write_h5):
        # A short (5-frame) low-likelihood gap with max_interp_gap_frames=10
        # should be linearly interpolated (a ramp), not flat-held.
        n = 100
        gap = slice(40, 45)
        df = dlc_df_factory(n_frames=n, nlevels=3, low_ll_slice=gap, seed=1)
        path = write_h5(tmp_path / "gap_short_filtered.h5", df)
        xy, bodyparts, _, ll_fracs, flat_held, ll = cc.load_dlc_file(
            path, likelihood_thresh=0.3, max_interp_gap_frames=10,
            return_quality=True)
        # The gap should NOT be recorded as flat-held for any bodypart
        assert not any(mask[gap].any() for mask in flat_held)
        # x column should vary smoothly (linear ramp) across the gap, i.e.
        # it should lie strictly between the values flanking the gap for the
        # first bodypart's x column (col 0).
        left_val = xy[gap.start - 1, 0]
        right_val = xy[gap.stop, 0]
        mid_val = xy[(gap.start + gap.stop) // 2, 0]
        lo, hi = sorted([left_val, right_val])
        assert lo - 1e-6 <= mid_val <= hi + 1e-6

    def test_long_gap_is_flat_held_not_ramped(self, tmp_path, dlc_df_factory, write_h5):
        # A long (30-frame) low-likelihood gap with max_interp_gap_frames=10
        # must be flat-held (split at the midpoint), not linearly ramped.
        n = 150
        gap = slice(50, 80)  # 30 frames > cap of 10
        df = dlc_df_factory(n_frames=n, nlevels=3, low_ll_slice=gap, seed=2)
        path = write_h5(tmp_path / "gap_long_filtered.h5", df)
        xy, bodyparts, _, ll_fracs, flat_held, ll = cc.load_dlc_file(
            path, likelihood_thresh=0.3, max_interp_gap_frames=10,
            return_quality=True)
        # bodypart 0's mask should mark the long gap as flat-held
        assert flat_held[0][gap].all()
        # Flat-hold means the value is constant across each half of the
        # split gap (not a smooth ramp) -- verify no in-between "ramp" values
        # by checking most of the first half of the gap equals the left
        # boundary value exactly.
        left_val = xy[gap.start - 1, 0]
        first_half = xy[gap.start:(gap.start + gap.stop) // 2, 0]
        assert np.allclose(first_half, left_val)

    def test_max_interp_gap_frames_none_means_legacy_interpolate_all(
            self, tmp_path, dlc_df_factory, write_h5):
        n = 150
        gap = slice(50, 80)
        df = dlc_df_factory(n_frames=n, nlevels=3, low_ll_slice=gap, seed=3)
        path = write_h5(tmp_path / "gap_legacy_filtered.h5", df)
        xy, bodyparts, _, ll_fracs, flat_held, ll = cc.load_dlc_file(
            path, likelihood_thresh=0.3, max_interp_gap_frames=None,
            return_quality=True)
        assert not any(mask.any() for mask in flat_held)

    def test_return_quality_false_by_default(self, tmp_path, dlc_df_factory, write_h5):
        df = dlc_df_factory(n_frames=30, nlevels=3)
        path = write_h5(tmp_path / "s_filtered.h5", df)
        result = cc.load_dlc_file(path)
        assert len(result) == 3   # xy, bodyparts, fps_hint only

    def test_ll_fracs_reflects_injected_low_likelihood_gap(
            self, tmp_path, dlc_df_factory, write_h5):
        n = 100
        gap = slice(0, 20)  # 20/100 = 0.2 frac below threshold
        df = dlc_df_factory(n_frames=n, nlevels=3, low_ll_slice=gap, seed=4)
        path = write_h5(tmp_path / "frac_filtered.h5", df)
        _, bodyparts, _, ll_fracs, _, _ = cc.load_dlc_file(
            path, likelihood_thresh=0.3, return_quality=True)
        for bp in bodyparts:
            assert ll_fracs[bp] == pytest.approx(0.2, abs=1e-6)

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "session.txt"
        bad.write_text("not a dlc file")
        with pytest.raises(ValueError):
            cc.load_dlc_file(bad)
