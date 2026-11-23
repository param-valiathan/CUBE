"""
cube_umap_video.py -- fast GPU version of CUBE's UMAP-evolution video (original recording | 3-D UMAP build-up).

Why the old renderer was slow: for EVERY 100 ms bin it cleared a matplotlib mplot3d figure and re-scattered the whole cloud
so far (up to ~8,000 points per cluster) -> work grows with the SQUARE of the session length, plus per-frame 3-D axes
layout, a CPU decode, OpenCV text and a software mp4v encode.  mplot3d is also the engine behind CUBE's earlier native crashes.

Tricks used here (same picture, a fraction of the work):
  * the camera never moves and the cloud only grows -> every point is drawn ONCE into a persistent GPU layer
    (cost grows linearly, not quadratically); a frame only adds the newest bins, the current-bin marker and <= 10 arrows;
  * the 3-D box / grid / axis labels are drawn once;
  * the recording is decoded, resized and frame-skipped inside ffmpeg (only the frames that are shown cross the pipe);
  * panels are composited and captioned on the GPU (glyph atlas), then encoded with NVENC (H.264) instead of mp4v.

Public entry: render(...) -- same arguments as cube_core.create_umap_evolution_video plus theme colours; raises on any
failure so the caller can fall back to the matplotlib renderer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _rgb(c):
    from matplotlib.colors import to_rgb
    return tuple(float(v) for v in to_rgb(c))


def _mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


def render(video_path, embedding, umap_labels, frame_labels, source_fps, out_path, output_fps=15.0,
           umap_panel_width=640, elev=20.0, azim=-60.0, palette=None, progress_cb=None, draw_cluster_overlay=True,
           bg="#0d0d1a", text_col="white", tick_col="#aaaacc", ffmpeg_exe=None, logger=None):
    import torch
    import cube_gpu_video as gv

    dev = gv.device()
    ff = ffmpeg_exe or gv.get_ffmpeg()
    pal = [_rgb(c) for c in (palette or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])]
    col = lambda c: pal[int(c) % len(pal)]
    BG, TXT, TICK = _rgb(bg), _rgb(text_col), _rgb(tick_col)

    emb = np.asarray(embedding, float)
    ulab = np.asarray(umap_labels, int)
    flab = np.asarray(frame_labels, int)
    n_bins = emb.shape[0]
    if n_bins == 0 or emb.shape[1] < 3:
        raise ValueError("empty / non-3D embedding")
    bin_stride = max(1, int(round(source_fps / 10.0)))
    valid = ulab >= 0
    uniq = sorted(set(ulab[valid]))
    if not uniq:
        raise ValueError("no clustered bins")
    centroids = {u: emb[valid & (ulab == u)].mean(0) for u in uniq}
    chance = 1.0 / max(1, len(uniq) - 1)
    events = [((fi // bin_stride), int(flab[fi - 1]), int(flab[fi])) for fi in range(1, len(flab))
              if flab[fi] != flab[fi - 1] and flab[fi] >= 0 and flab[fi - 1] >= 0]

    # ---- geometry -------------------------------------------------------------------------------------------
    sw, sh, _sfps, total = gv.video_info(video_path)
    target_h = min(sh, 720) - min(sh, 720) % 2
    vid_w = int(sw * target_h / sh); vid_w -= vid_w % 2
    pw = int(umap_panel_width) - int(umap_panel_width) % 2
    out_w, out_h = vid_w + pw, target_h
    step = max(1, int(round(source_fps / output_fps)))
    u = target_h / 720.0                                    # px scale for markers / text

    lo, hi = emb.min(0), emb.max(0)
    pad = (hi - lo) * 0.05 + 1e-6
    lo, hi = lo - pad, hi + pad
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    norm = lambda p: np.atleast_2d((np.asarray(p, float) - mid) / half) * np.array([1.0, 1.0, 0.75])

    a, e = math.radians(azim), math.radians(elev)
    Rv = np.array([-math.sin(a), math.cos(a), 0.0])
    Uv = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    Dv = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-0.75, 0.75)], float)
    cq = np.stack([corners @ Rv, corners @ Uv], 1)
    clo, chi = cq.min(0), cq.max(0)
    margin_top = 26 * u
    sc = 0.86 * min(pw / (chi[0] - clo[0]), (target_h - margin_top - 24 * u) / (chi[1] - clo[1]))
    cx, cy = pw / 2, margin_top + (target_h - margin_top) / 2 - 6 * u
    ccen = (clo + chi) / 2

    def px(P):
        P = np.atleast_2d(P)
        q = np.stack([P @ Rv, P @ Uv], 1)
        return np.stack([cx + (q[:, 0] - ccen[0]) * sc, cy - (q[:, 1] - ccen[1]) * sc], 1)

    # ---- static layer: box, back-face grid, axis labels ---------------------------------------------------------
    canvas = gv.FrameCanvas(pw, target_h, dev)
    R = canvas.R
    canvas.fill(BG)
    edge = _mix(BG, TICK, 0.28)
    grid = _mix(BG, TICK, 0.14)
    atlas_s = gv.TextAtlas(max(8, 13 * u), dev)
    atlas_t = gv.TextAtlas(max(9, 14 * u), dev)

    def faces():
        # the three faces farthest from the viewer
        out = []
        for ax_i in range(3):
            sgn = -1.0 if Dv[ax_i] > 0 else 1.0               # face on the far side along this axis
            val = sgn * (0.75 if ax_i == 2 else 1.0)
            idx = [i for i in range(3) if i != ax_i]
            q = []
            for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3); p[ax_i] = val
                p[idx[0]] = s1 * (0.75 if idx[0] == 2 else 1.0)
                p[idx[1]] = s2 * (0.75 if idx[1] == 2 else 1.0)
                q.append(p)
            out.append(np.array(q))
        return out

    for q in faces():
        qp = px(q)
        rows = [[qp[i][0], qp[i][1], qp[(i + 1) % 4][0], qp[(i + 1) % 4][1], 0.7 * u, *edge, 1.0, 1e9, 0.0] for i in range(4)]
        for t in (0.25, 0.5, 0.75):
            for (i0, i1, j0, j1) in ((0, 1, 3, 2), (1, 2, 0, 3)):
                p0, p1 = qp[i0] * (1 - t) + qp[j0] * t, qp[i1] * (1 - t) + qp[j1] * t
                rows.append([p0[0], p0[1], p1[0], p1[1], 0.5 * u, *grid, 1.0, 1e9, 0.0])
        R.segs_multi(rows)
    # axis titles + numeric ticks on the three edges nearest the viewer / left side (drawn once)
    front = np.array([1.0 if Dv[0] > 0 else -1.0, 1.0 if Dv[1] > 0 else -1.0, -0.75])
    tick_lbl = []

    def tick_txt(v):
        return ("%.0f" % v) if abs(v) >= 10 or abs(v - round(v)) < 1e-6 else ("%.1f" % v)

    for t in np.linspace(-1, 1, 5):
        tick_lbl.append((tick_txt(mid[0] + t * half[0]), px(np.array([t, front[1], -0.75]))[0] + np.array([0, 11 * u])))
        tick_lbl.append((tick_txt(mid[1] + t * half[1]), px(np.array([front[0], t, -0.75]))[0] + np.array([12 * u, 4 * u])))
    verts = [(x, y) for x in (-1.0, 1.0) for y in (-1.0, 1.0)]
    lx, ly = min(verts, key=lambda xy: px(np.array([xy[0], xy[1], 0.0]))[0][0])
    for t in np.linspace(-1, 1, 5):
        tick_lbl.append((tick_txt(mid[2] + t / 0.75 * half[2] * 0.75 if False else mid[2] + t * half[2]),
                         px(np.array([lx, ly, t * 0.75]))[0] + np.array([-14 * u, 0])))
    titles = [("UMAP 1", px(np.array([0.0, front[1], -0.75]))[0] + np.array([0, 28 * u])),
              ("UMAP 2", px(np.array([front[0], 0.0, -0.75]))[0] + np.array([40 * u, 14 * u])),
              ("UMAP 3", px(np.array([lx, ly, 0.0]))[0] + np.array([-46 * u, -6 * u]))]
    atlas_k = gv.TextAtlas(max(7, 10 * u), dev)
    for txt, (x, y) in tick_lbl:
        atlas_k.blit(R.canvas, txt, x - atlas_k.width(txt) / 2, y - atlas_k.line_h / 2, TICK, outline_rgb=BG)
    for txt, (x, y) in titles:
        x = min(max(x, atlas_s.width(txt) / 2 + 6), pw - atlas_s.width(txt) / 2 - 6)      # keep inside the panel
        atlas_s.blit(R.canvas, txt, x - atlas_s.width(txt) / 2, y - atlas_s.line_h / 2, TICK, outline_rgb=BG)
    static = R.canvas.clone()

    # ---- the cloud layer (persistent) ---------------------------------------------------------------------------
    R.canvas.copy_(static)
    r_pt = max(2.2, 3.4 * u)
    r_cur = max(4.0, 6.0 * u)
    drawn = -1
    cursor = 0
    T_cum: dict = {}

    def add_points(upto):
        nonlocal drawn
        if upto <= drawn:
            return
        lo_i = drawn + 1
        idx = np.arange(lo_i, upto + 1)
        idx = idx[ulab[idx] >= 0]
        if len(idx):
            q = px(norm(emb[idx]))
            # draw far points first so nearer points sit on top (cheap depth sort for this batch)
            depth = norm(emb[idx]) @ Dv
            order = np.argsort(depth)
            rows = [[q[j][0], q[j][1], r_pt, 0.0, *col(ulab[idx[j]]), 0.82] for j in order]
            R.circ_multi(rows)
        drawn = upto

    def arrows():
        src_tot = {}
        for (s, _t), v in T_cum.items():
            src_tot[s] = src_tot.get(s, 0) + v
        ab = {}
        for (s, t), cnt in T_cum.items():
            if s in centroids and t in centroids:
                p = cnt / max(1, src_tot.get(s, 1))
                if p > chance:
                    ab[(s, t)] = p
        if not ab:
            return []
        rng = max(max(ab.values()) - chance, 1e-9)
        rows = []
        for (s, t), p in sorted(ab.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            rel = (p - chance) / rng
            lw, al = 0.6 + rel * 1.4, 0.18 + rel * 0.22
            sc_, tc_ = norm(centroids[s])[0], norm(centroids[t])[0]
            d = tc_ - sc_
            m = sc_ + 0.8 * d
            a0, a1, a2 = px(sc_)[0], px(m)[0], px(tc_)[0]
            c = col(s)
            rows.append([a0[0], a0[1], a1[0], a1[1], lw * u * 0.5, *c, al * 1.15, 1e9, 0.0])
            v = a2 - a1
            n = np.linalg.norm(v) + 1e-9
            v, perp = v / n, np.array([-v[1], v[0]]) / n
            for sgn in (-1, 1):
                hd = a2 - v * 8 * u + perp * sgn * 4 * u
                rows.append([a2[0], a2[1], hd[0], hd[1], lw * u * 0.4, *c, al * 1.3, 1e9, 0.0])
        return rows

    # ---- compose + encode ---------------------------------------------------------------------------------------
    big = gv.TextAtlas(max(16, int(max(1.2, target_h / 500.0) * 30)), dev, bold=True, outline=3)
    ts_atlas = gv.TextAtlas(max(12, int(22 * u)), dev, bold=True)
    outp = Path(out_path).with_suffix(".mp4")
    n_out = (total + step - 1) // step

    def run(prefer_gpu):
        nonlocal drawn, cursor
        drawn, cursor = -1, 0
        T_cum.clear()
        R.canvas.copy_(static)
        reader = gv.GpuVideoReader(video_path, ff, size=(vid_w, target_h), every=step, dev=dev)
        writer = gv.GpuVideoWriter(outp, output_fps, out_w, out_h, ff, prefer_gpu=prefer_gpu)
        try:
            for i, fr in enumerate(reader):
                src_fi = i * step
                b = min(src_fi // bin_stride, n_bins - 1)
                while cursor < len(events) and events[cursor][0] <= b:
                    _, fc, tc = events[cursor]
                    T_cum[(fc, tc)] = T_cum.get((fc, tc), 0) + 1
                    cursor += 1
                add_points(b)
                snap = R.canvas.clone()
                if ulab[b] >= 0:
                    qc = px(norm(emb[b]))[0]
                    R.circ_multi([[qc[0], qc[1], r_cur + 1.6 * u, 0.0, *TXT, 1.0],
                                  [qc[0], qc[1], r_cur, 0.0, *col(ulab[b]), 1.0]])
                R.segs_multi(arrows())
                atlas_t.blit(R.canvas, f"UMAP  bin {b + 1}/{n_bins}", 10 * u, 6 * u, TICK, outline_rgb=BG)
                panel = R.canvas
                left = fr.float().div_(255.0)
                frame = torch.cat([left, panel], 1)
                if draw_cluster_overlay:
                    lbl = int(flab[min(src_fi, len(flab) - 1)])
                    text = f"Cluster {lbl}"
                    strip_h = int(big.line_h + 14 * u)
                    roi = frame[0:strip_h, 0:vid_w]
                    roi.mul_(0.45)                                            # dark 55 % backing strip
                    big.blit(frame, text, (vid_w - big.width(text)) / 2, 6 * u, col(lbl), outline_rgb=(0.12, 0.12, 0.12))
                    frame[target_h - int(40 * u):target_h, 0:int(260 * u)] = 0.0
                    t = src_fi / source_fps
                    ts_atlas.blit(frame, f"t={int(t // 60):02d}:{t % 60:05.2f}", 8 * u, target_h - int(34 * u), (0.78, 0.78, 0.78),
                                  outline_rgb=(0, 0, 0))
                writer.write((frame.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8))
                R.canvas.copy_(snap)
                if progress_cb and i % max(1, n_out // 50) == 0:
                    progress_cb("Assembling video", min(1.0, i / max(1, n_out)))
            reader.close()
            writer.close()
        except Exception:
            writer.abort()
            reader.close()
            raise
        return writer.encoder

    enc = gv.run_with_encoder_fallback(run, logger)
    if progress_cb:
        progress_cb("Assembling video", 1.0)
    return outp
