# Plan: Side-by-Side UMAP Evolution Video Export

## Context

The user wants a publication-quality video showing the original behavioural recording on the left and a 3D UMAP that gradually builds up (B-SOiD style) on the right, time-aligned frame-by-frame. The key addition over the B-SOiD paper is that **cumulative transition arrows** between cluster centroids grow thicker each time a transition is observed, so the viewer sees both the accumulating point cloud AND the emergent transition structure simultaneously.

Only one representative video needs to be processed this way — it is an export operation, not part of the main pipeline.

User choices confirmed:
- **UMAP background**: Session-only buildup (only this session's points accumulate from scratch)
- **Transitions**: Cumulative arrows (centroid-to-centroid arrows thicken with each observed transition)

---

## Files to Modify

| File | Change |
|------|--------|
| `cube_core.py` | Add `create_umap_evolution_video()` + save session bin ranges during pipeline |
| `cube.py` | Add "Export UMAP Evolution Video" button and handler |

---

## Step 1 — Track Per-Session Bin Ranges in the Pipeline

**Where:** Inside `BSoidEngine` (cube_core.py), at the point where per-session feature arrays are stacked into the global feature matrix before UMAP.

**What to save:** `model/session_bin_ranges.json`
```json
{
  "session_name_1": [0, 843],
  "session_name_2": [843, 1701],
  ...
}
```

This lets the export function slice `umap_embedding[start:end]` to get exactly this session's 3D coordinates, without re-running UMAP.

The mapping is trivial to compute at stack time:
```python
offset = 0
for name, feat in session_feature_dict.items():
    n_bins = feat.shape[0]
    ranges[name] = [offset, offset + n_bins]
    offset += n_bins
```

Save alongside `umap_embedding.npy`. No change to existing downstream code needed.

---

## Step 2 — New Function `create_umap_evolution_video()` in `cube_core.py`

### Signature
```python
def create_umap_evolution_video(
    video_path: Path,
    embedding: np.ndarray,        # (n_session_bins, 3) — pre-sliced to this session
    umap_labels: np.ndarray,      # (n_session_bins,)
    frame_labels: np.ndarray,     # (n_video_frames,) per-frame cluster IDs
    source_fps: float,
    out_path: Path,
    output_fps: float = 15.0,
    umap_panel_width: int = 640,
    elev: float = 20.0,
    azim: float = -60.0,
    palette: list = None,
) -> Path:
```

### Algorithm

**Phase 0 — Setup**
- Compute `bin_stride = max(1, round(source_fps / 10.0))` (100 ms bins = source_fps/10 frames)
- Compute cluster centroids: mean of `embedding[umap_labels == c]` for each unique cluster `c`
- Compute fixed axis limits from the full session embedding (add 5% padding)
- Build transition event list: scan `frame_labels` for consecutive changes → `[(frame_idx, from_c, to_c), ...]`
- Build cumulative transition count array `T_cum[from_c, to_c]` incremented at each transition event

**Phase 1 — Pre-render UMAP panel frames (the bottleneck)**

For each bin index `b` in `0 ... n_session_bins`:
1. Determine which transitions have occurred by bin `b` (all transitions where `frame_idx // bin_stride <= b`)
2. Render a matplotlib Axes3D figure (Agg backend):
   - Scatter all bins `0..b`, coloured by cluster using existing `PALETTE`
   - Highlight current bin's point with a larger white-outlined marker
   - Draw cumulative arrows between centroids: for each `(from_c, to_c)` where count > 0, draw a quiver arrow using the same shaft+quiver pattern from `plot_umap_3d_transitions()` (cube_core.py ~line 1918), width ∝ `count / max_count`
3. Rasterise via `canvas.draw(); buf = fig.canvas.tostring_rgb()` → reshape to `(H, W, 3)` numpy array
4. Cache in dict `umap_frames[b]`

Reuse a single `Figure` / `FigureCanvasAgg` object across iterations (call `ax.cla()` between frames) to avoid matplotlib object creation overhead.

Estimated time: ~0.15 s/render × 1800 bins (10 min session) ≈ 4–5 minutes pre-render.

**Phase 2 — Assemble output video**

```
output_width = video_panel_width + umap_panel_width
output_height = video_panel_height  # matched to source video height (capped at 720px)
```

Use existing `_open_writer()` (cube_core.py line 2522) for codec fallback.

Frame loop (mirrors `create_labeled_video()` pattern, cube_core.py line 2714):
```
src_fi = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if src_fi % step == 0:
        bin_idx = min(src_fi // bin_stride, n_session_bins - 1)
        umap_img = umap_frames[bin_idx]        # pre-rendered numpy array
        vid_panel = resize_frame(frame, target_h)
        umap_panel = resize_to_panel(umap_img, umap_panel_width, target_h)
        combined = np.concatenate([vid_panel, umap_panel], axis=1)
        # Overlay: cluster label + timestamp (bottom strip, same style as create_labeled_video)
        writer.write(combined)
    src_fi += 1
```

### Layout
```
┌─────────────────────┬──────────────────┐
│                     │                  │
│   Original video    │   3D UMAP        │
│   (left, ~60%)      │   buildup        │
│                     │   (right, ~40%)  │
├─────────────────────┴──────────────────┤
│  Cluster: 3   t = 00:42   ████░░░░░░   │  ← bottom strip
└─────────────────────────────────────────┘
```

Output: `videos/umap_evolution/<video_stem>_umap_evolution.mp4`

---

## Step 3 — GUI Button in `cube.py`

- Add an **"Export UMAP Evolution Video"** button in the video/export area of the GUI (Step 4 or Step 5 panel, wherever labeled video export lives)
- On click:
  1. File dialog: user picks the representative video from the project
  2. Load `model/umap_embedding.npy`, `model/umap_labels.npy`, `model/session_bin_ranges.json`
  3. Match selected video stem → slice indices from `session_bin_ranges.json`
  4. Load matching `bout_lengths/<stem>_frame_labels_hmm.csv` (fall back to non-HMM)
  5. Call `create_umap_evolution_video()` in a background thread (same pattern as existing video exports)
  6. Log progress to `LogPanel` with phase updates ("Pre-rendering UMAP frames… 42%", "Assembling video…")

---

## Reuse Points

| Existing code | Reused for |
|---------------|------------|
| `_open_writer()` (line 2522) | Video codec/fallback handling |
| `create_labeled_video()` frame loop (line 2714) | Sequential frame read + step pattern |
| `plot_umap_3d_transitions()` arrow drawing logic (line 1918–1951) | Shaft + quiver style for cumulative arrows |
| `PALETTE` (module-level) | Cluster colours consistent with rest of project |
| `labels_to_bouts()` (line 1655) | Derive transition events from frame_labels |

---

## Verification

1. Run the full CUBE pipeline on a test dataset to confirm `session_bin_ranges.json` is created correctly and indices match the embedding shape.
2. Trigger "Export UMAP Evolution Video" from the GUI on a short (~2 min) test session.
3. Open the output video and verify:
   - Left panel shows original video playing at correct speed
   - Right panel 3D cloud grows from 0 points to full session cloud as video progresses
   - Cumulative arrows between cluster centroids appear and thicken at the correct moments
   - Timestamps in the overlay match the actual video time
4. Spot-check by pausing at a known transition frame and confirming the arrow count matches the number of transitions observed up to that point.
