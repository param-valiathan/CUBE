"""Shared pytest fixtures for the CUBE test suite.

Fixture generators are added incrementally as later phases need them
(synthetic DLC-format DataFrames, tmp_path file trees, small .h5/.csv
files). Nothing here touches real user data or CUBE_logs/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────────
#  Synthetic bodypart / trajectory generators
# ──────────────────────────────────────────────────────────────────────────

STANDARD_BODYPARTS = ["nose", "neck", "back_middle", "tailbase",
                      "front_left_paw", "back_right_paw"]


def make_synthetic_xy(n_frames=600, bodyparts=None, seed=0, noise=1.0):
    """(n_frames, n_pts*2) smooth-ish synthetic trajectory, deterministic."""
    bodyparts = bodyparts if bodyparts is not None else STANDARD_BODYPARTS
    n_pts = len(bodyparts)
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    xy = np.zeros((n_frames, n_pts * 2), dtype=float)
    for i in range(n_pts):
        cx, cy = 100.0 + 30 * i, 200.0 + 20 * i
        xy[:, 2 * i]     = cx + 15 * np.sin(t / 37.0 + i) + rng.normal(0, noise, n_frames)
        xy[:, 2 * i + 1] = cy + 10 * np.cos(t / 53.0 + i) + rng.normal(0, noise, n_frames)
    return xy


@pytest.fixture
def synthetic_xy():
    return make_synthetic_xy()


@pytest.fixture
def standard_bodyparts():
    return list(STANDARD_BODYPARTS)


def make_dlc_multiindex_df(n_frames=300, bodyparts=None, scorer="DLC_scorer",
                            seed=0, nlevels=3, individual="animal1",
                            low_ll_slice=None, low_ll_value=0.1):
    """Build a synthetic DLC-format pd.DataFrame with a 2/3/4-level MultiIndex.

    low_ll_slice: optional slice(start, stop) of frames whose likelihood
    column is forced to `low_ll_value` (for interpolation-gap tests).
    """
    bodyparts = bodyparts if bodyparts is not None else STANDARD_BODYPARTS
    rng = np.random.default_rng(seed)
    n_pts = len(bodyparts)
    xy = make_synthetic_xy(n_frames, bodyparts, seed=seed)
    ll = np.full((n_frames, n_pts), 0.95, dtype=float)
    if low_ll_slice is not None:
        ll[low_ll_slice, :] = low_ll_value

    cols = []
    data = {}
    for i, bp in enumerate(bodyparts):
        for coord in ("x", "y", "likelihood"):
            if nlevels == 4:
                key = (scorer, individual, bp, coord)
            elif nlevels == 3:
                key = (scorer, bp, coord)
            elif nlevels == 2:
                key = (bp, coord)
            else:
                raise ValueError(nlevels)
            cols.append(key)
            if coord == "x":
                data[key] = xy[:, 2 * i]
            elif coord == "y":
                data[key] = xy[:, 2 * i + 1]
            else:
                data[key] = ll[:, i]

    if nlevels == 4:
        names = ["scorer", "individuals", "bodyparts", "coords"]
    elif nlevels == 3:
        names = ["scorer", "bodyparts", "coords"]
    else:
        names = ["bodyparts", "coords"]

    columns = pd.MultiIndex.from_tuples(cols, names=names)
    df = pd.DataFrame(data.values(), index=columns).T
    df.columns = columns
    df.index = range(n_frames)
    return df


@pytest.fixture
def dlc_df_factory():
    return make_dlc_multiindex_df


def write_dlc_h5(path: Path, df: pd.DataFrame):
    df.to_hdf(str(path), key="df_with_missing", mode="w", format="fixed")
    return path


def write_dlc_csv(path: Path, df: pd.DataFrame):
    df.to_csv(str(path))
    return path


@pytest.fixture
def write_h5():
    return write_dlc_h5


@pytest.fixture
def write_csv():
    return write_dlc_csv
