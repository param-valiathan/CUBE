"""Phase T5: config, DEFAULTS, and compat_mode logic tests (Tier 1).

Covers BSoidEngine.__init__'s config-merging behavior:
    self._cfg = {**DEFAULTS, **(cfg or {})}
    self._explicit_cfg_keys tracking
    compat_mode="legacy_v2" revert-only-unset-keys logic

This mechanism underpins every "byte-identical when a new flag defaults
off" backward-compatibility guarantee in the pipeline, so it is tested
directly against BSoidEngine's real __init__ rather than reimplemented.
Only __init__ is exercised here -- run() is NOT called (see Phase T7 for
that).
"""
import pytest

import cube_core as cc


def make_engine(tmp_path, cfg=None):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir(exist_ok=True)
    out_dir = tmp_path / "out"
    return cc.BSoidEngine(csv_dir, video_folder=None, output_dir=out_dir,
                          fps=30, cfg=cfg)


class TestConfigMerging:
    def test_no_cfg_uses_pure_defaults(self, tmp_path):
        eng = make_engine(tmp_path, cfg=None)
        assert eng._cfg == cc.BSoidEngine.DEFAULTS
        assert eng._explicit_cfg_keys == set()

    def test_empty_cfg_dict_uses_pure_defaults(self, tmp_path):
        eng = make_engine(tmp_path, cfg={})
        assert eng._cfg == cc.BSoidEngine.DEFAULTS
        assert eng._explicit_cfg_keys == set()

    def test_partial_override_merges_with_defaults(self, tmp_path):
        eng = make_engine(tmp_path, cfg={"likelihood_thresh": 0.5})
        assert eng._cfg["likelihood_thresh"] == 0.5
        # every other key should still be the documented default
        for k, v in cc.BSoidEngine.DEFAULTS.items():
            if k != "likelihood_thresh":
                assert eng._cfg[k] == v
        assert eng._explicit_cfg_keys == {"likelihood_thresh"}

    def test_unknown_cfg_key_is_kept_verbatim(self, tmp_path):
        # cfg dicts may carry keys DEFAULTS doesn't define (e.g. future/
        # experimental flags) -- __init__ must not silently drop them.
        eng = make_engine(tmp_path, cfg={"some_future_flag": 123})
        assert eng._cfg["some_future_flag"] == 123
        assert "some_future_flag" in eng._explicit_cfg_keys

    def test_multiple_overrides_all_tracked_as_explicit(self, tmp_path):
        cfg = {"likelihood_thresh": 0.5, "umap_min_dist": 0.2,
              "mlp_max_iter": 10}
        eng = make_engine(tmp_path, cfg=cfg)
        assert eng._explicit_cfg_keys == set(cfg.keys())
        for k, v in cfg.items():
            assert eng._cfg[k] == v


class TestCompatModeLegacyV2:
    def test_legacy_v2_reverts_all_unset_legacy_keys(self, tmp_path):
        eng = make_engine(tmp_path, cfg={"compat_mode": "legacy_v2"})
        for k, v in cc.BSoidEngine._LEGACY_V2_DEFAULTS.items():
            assert eng._cfg[k] == v, f"{k} was not reverted to legacy default"

    def test_legacy_v2_explicit_override_wins_over_legacy_default(self, tmp_path):
        # umap_min_dist's legacy default is 0.0; explicitly requesting 0.2
        # under legacy_v2 must still win (explicit overrides always win).
        eng = make_engine(tmp_path, cfg={
            "compat_mode": "legacy_v2",
            "umap_min_dist": 0.2,
        })
        assert eng._cfg["umap_min_dist"] == 0.2
        # untouched legacy keys still revert
        assert eng._cfg["hdbscan_mcs_anchor"] == "full"
        assert eng._cfg["angular_fallback"] is True

    def test_legacy_v2_explicit_override_of_boolean_legacy_key(self, tmp_path):
        eng = make_engine(tmp_path, cfg={
            "compat_mode": "legacy_v2",
            "auto_bodypart_weighting": True,   # legacy default is False
        })
        assert eng._cfg["auto_bodypart_weighting"] is True

    def test_current_compat_mode_keeps_v2_1_defaults(self, tmp_path):
        eng = make_engine(tmp_path, cfg={"compat_mode": "current"})
        assert eng._cfg["umap_min_dist"] == cc.BSoidEngine.DEFAULTS["umap_min_dist"]
        assert eng._cfg["umap_min_dist"] != cc.BSoidEngine._LEGACY_V2_DEFAULTS["umap_min_dist"]

    def test_default_compat_mode_is_current_behavior(self, tmp_path):
        # No compat_mode key at all -> DEFAULTS["compat_mode"] == "current",
        # so legacy reversion must NOT fire.
        eng = make_engine(tmp_path, cfg=None)
        assert eng._cfg.get("compat_mode") == "current"
        assert eng._cfg["umap_min_dist"] == 0.1
        # hdbscan_mcs_anchor's DEFAULTS value ("embedding") differs from its
        # _LEGACY_V2_DEFAULTS value ("full") -- under "current" mode (no
        # legacy reversion) it must stay at the DEFAULTS value.
        assert eng._cfg["hdbscan_mcs_anchor"] == cc.BSoidEngine.DEFAULTS["hdbscan_mcs_anchor"]
        assert eng._cfg["hdbscan_mcs_anchor"] != cc.BSoidEngine._LEGACY_V2_DEFAULTS["hdbscan_mcs_anchor"]

    def test_legacy_v2_all_five_documented_keys_present_in_legacy_defaults(self):
        # Pin the documented key set so a future edit to _LEGACY_V2_DEFAULTS
        # is caught by this test rather than silently changing scope.
        expected_keys = {
            "umap_min_dist", "hdbscan_mcs_anchor", "angular_fallback",
            "hdbscan_merge_thresh", "hdbscan_split_silhouette_thresh",
            "auto_bodypart_weighting", "turned_away_exclude_from_bad_frac",
        }
        assert expected_keys == set(cc.BSoidEngine._LEGACY_V2_DEFAULTS.keys())


class TestHdbscanSweepPerfCfgKeys:
    """HDBSCAN_Sweep_Performance_Implementation_Plan.md's new cfg keys
    must default to true no-ops and must NOT collide with
    _LEGACY_V2_DEFAULTS -- Option 3 changes *how* output is computed, not
    numeric defaults. (Options 4/5 keys are added in follow-up phases and
    get their own tests in this class then.)
    """

    def test_hdbscan_tree_reuse_enabled_default_true_not_in_legacy_defaults(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_tree_reuse_enabled"] is True
        assert "hdbscan_tree_reuse_enabled" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_hdbscan_seed_sweep_n_steps_default_zero_not_in_legacy_defaults(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_seed_sweep_n_steps"] == 0
        assert "hdbscan_seed_sweep_n_steps" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_hdbscan_consensus_sweep_n_steps_default_zero_not_in_legacy_defaults(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_consensus_sweep_n_steps"] == 0
        assert "hdbscan_consensus_sweep_n_steps" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_seed_sweep_train_frac_default_one_not_in_legacy_defaults(self):
        assert cc.BSoidEngine.DEFAULTS["seed_sweep_train_frac"] == 1.0
        assert "seed_sweep_train_frac" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_pinned_five_key_legacy_defaults_set_undisturbed(self):
        # Re-assert the exact same pinned set as
        # test_legacy_v2_all_five_documented_keys_present_in_legacy_defaults
        # (this plan's own keys must not have been added to it).
        expected_keys = {
            "umap_min_dist", "hdbscan_mcs_anchor", "angular_fallback",
            "hdbscan_merge_thresh", "hdbscan_split_silhouette_thresh",
            "auto_bodypart_weighting", "turned_away_exclude_from_bad_frac",
        }
        assert expected_keys == set(cc.BSoidEngine._LEGACY_V2_DEFAULTS.keys())


class TestRegionAwareRefinementCfgKeys:
    """Region_Aware_Refinement_Implementation_Plan.md's new cfg keys must
    default to true no-ops and must NOT collide with _LEGACY_V2_DEFAULTS --
    this feature changes *how* refinement runs when explicitly opted into,
    not a pre-2.1 numeric default compat_mode="legacy_v2" needs to revert to.
    """

    def test_hdbscan_region_split_enabled_default_false_not_in_legacy_defaults(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_region_split_enabled"] is False
        assert "hdbscan_region_split_enabled" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_hdbscan_region_split_signal_default(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_region_split_signal"] == ["current_region"]
        assert "hdbscan_region_split_signal" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_hdbscan_region_split_impurity_thresh_default(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_region_split_impurity_thresh"] == 0.5
        assert "hdbscan_region_split_impurity_thresh" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_hdbscan_region_split_min_minority_frac_default(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_region_split_min_minority_frac"] == 0.15
        assert "hdbscan_region_split_min_minority_frac" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_hdbscan_region_split_max_subclusters_default(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_region_split_max_subclusters"] == 3
        assert "hdbscan_region_split_max_subclusters" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_hdbscan_region_split_max_candidates_default(self):
        assert cc.BSoidEngine.DEFAULTS["hdbscan_region_split_max_candidates"] == 10
        assert "hdbscan_region_split_max_candidates" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_region_split_pre_reduction_pct_default_off(self):
        assert cc.BSoidEngine.DEFAULTS["region_split_pre_reduction_pct"] == 0.0
        assert "region_split_pre_reduction_pct" not in cc.BSoidEngine._LEGACY_V2_DEFAULTS

    def test_pinned_legacy_defaults_set_undisturbed(self):
        # Re-assert the exact same pinned set as the other two "...five..."-
        # named tests above (misleadingly named -- 7 keys, not 5 -- but this
        # plan's own new keys must not have been added to it either way).
        expected_keys = {
            "umap_min_dist", "hdbscan_mcs_anchor", "angular_fallback",
            "hdbscan_merge_thresh", "hdbscan_split_silhouette_thresh",
            "auto_bodypart_weighting", "turned_away_exclude_from_bad_frac",
        }
        assert expected_keys == set(cc.BSoidEngine._LEGACY_V2_DEFAULTS.keys())


class TestEngineConstructionSideEffects:
    def test_output_subdirs_created(self, tmp_path):
        eng = make_engine(tmp_path)
        assert eng._out_bouts.is_dir()
        assert eng._out_model.is_dir()
        assert eng._out_plots.is_dir()
        assert eng._out_videos.is_dir()

    def test_csv_folder_accepts_single_path_or_list(self, tmp_path):
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        eng1 = cc.BSoidEngine(csv_dir, output_dir=tmp_path / "out1")
        eng2 = cc.BSoidEngine([csv_dir], output_dir=tmp_path / "out2")
        assert eng1._csv_folders == [csv_dir]
        assert eng2._csv_folders == [csv_dir]

    def test_video_folder_none_by_default(self, tmp_path):
        eng = make_engine(tmp_path)
        assert eng.video_folder is None
        assert eng._vid_folders == []
