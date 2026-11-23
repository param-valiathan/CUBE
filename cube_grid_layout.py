"""
Camera-grid layout for the composite (quad / reprojection) QC videos.

Historically the composites supported 2, 3 or 4 streams (hstack / 3-pane xstack / 2x2 xstack) and silently fell into the
3-input branch for any other count, so a 5- or 6-camera rig failed with an ffmpeg error and lost its example video.  This
module keeps the 2/3/4-camera filters BYTE-IDENTICAL and adds a general grid for 5+ cameras (3 columns, as many rows as
needed, panes shrunk so the composite stays a sensible size).  No dependencies; shared by cube_3d_dlc.py and
reproject_3d_to_2d_overlay.py.
"""
from __future__ import annotations

MANY_CAMERAS_PANE_MAX_W = 960      # pane width cap when n >= 5 (6 x 1280 px would be 3840 px wide)
GRID_COLS_MANY = 3


def pane_size(target_w: int, target_h: int, n: int) -> tuple[int, int]:
    """Size every pane is scaled to.  Unchanged for n <= 4; for n >= 5 the panes are shrunk (even dimensions)."""
    if n <= 4 or target_w <= MANY_CAMERAS_PANE_MAX_W:
        return target_w, target_h
    w = MANY_CAMERAS_PANE_MAX_W
    h = int(round(target_h * w / target_w / 2.0)) * 2
    return w, h


def grid_shape(n: int) -> tuple[int, int]:
    """(columns, rows) of the layout used for n streams."""
    if n == 2:
        return 2, 1
    if n == 3:
        return 2, 2          # existing 3-pane layout: two on top, one below-left
    if n == 4:
        return 2, 2
    cols = GRID_COLS_MANY
    return cols, -(-n // cols)


def stack_filter(ins: str, n: int, out_label: str) -> str:
    """The ffmpeg filter fragment that tiles the n labelled inputs `ins` ("[s0][s1]...") into one stream `[out_label]`.
    All panes must already have the same size.  n = 2, 3, 4 reproduce the historical filters exactly."""
    if n == 4:
        return f"{ins}xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[{out_label}]"
    if n == 2:
        return f"{ins}hstack=inputs=2[{out_label}]"
    if n == 3:
        return f"{ins}xstack=inputs=3:layout=0_0|w0_0|0_h0[{out_label}]"
    cols, _rows = grid_shape(n)
    cells = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = "0" if c == 0 else "+".join(f"w{j}" for j in range(c))
        y = "0" if r == 0 else "+".join(f"h{cols * j}" for j in range(r))
        cells.append(f"{x}_{y}")
    return f"{ins}xstack=inputs={n}:layout={'|'.join(cells)}:fill=black[{out_label}]"
