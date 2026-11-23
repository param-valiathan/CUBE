"""
cube_gpu_video.py -- shared GPU video toolkit for CUBE's exported videos.

* GpuVideoReader : ffmpeg decode (optional NVDEC) -> raw frames -> torch uint8 tensors on the GPU, with the
                   resize / fps change done inside ffmpeg so far fewer pixels cross the pipe.
* FrameCanvas    : float RGB canvas on the GPU with the analytic anti-aliased drawing of cube_skeleton_render.Raster
                   (ordered batches of segments / discs / rings).  Nothing is drawn by OpenCV on the CPU.
* TextAtlas      : glyph tiles rendered once (PIL), blitted on the GPU (optional outline) -- captions, banners, clocks.
* GpuVideoWriter : raw frames -> ffmpeg h264_nvenc (libx264 fallback), atomic .tmp rename, bounded queue.
* run_with_encoder_fallback : try NVENC, repeat once with libx264 if the GPU encoder refuses (session limit).

Safe by construction: no matplotlib / mplot3d in any frame loop, bounded queues, one ffmpeg per stream, every function
raises on failure so the caller can fall back to its previous CPU implementation.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np

_NVENC_CACHE: dict = {}


def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def nvenc_ok(ffmpeg_exe: str) -> bool:
    """One cached probe for a working h264_nvenc encoder (256x256: NVENC rejects tiny frames)."""
    if ffmpeg_exe not in _NVENC_CACHE:
        try:
            r = subprocess.run([ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                               capture_output=True, timeout=30)
            _NVENC_CACHE[ffmpeg_exe] = r.returncode == 0
        except Exception:
            _NVENC_CACHE[ffmpeg_exe] = False
    return _NVENC_CACHE[ffmpeg_exe]


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def video_info(path):
    """(width, height, fps, n_frames) via OpenCV (header only)."""
    import cv2
    c = cv2.VideoCapture(str(path))
    try:
        if not c.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        return (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(c.get(cv2.CAP_PROP_FPS)) or 30.0, int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        c.release()


# --------------------------------------------------------------------------- reader
class GpuVideoReader:
    """Iterate frames as uint8 GPU tensors (H, W, 3) RGB, or (H, W, 1) when gray=True.

    size=(w, h) and fps are applied inside ffmpeg.  hwaccel: True / False / "auto" (NVDEC for >= 2 Mpx sources:
    measured slower than software decode on 720p clips)."""

    def __init__(self, path, ffmpeg_exe=None, size=None, fps=None, gray=False, hwaccel="auto", dev=None,
                 max_frames=None, queue_size=8, every=1):
        self.path = str(path)
        self.ff = ffmpeg_exe or get_ffmpeg()
        self.src_w, self.src_h, self.src_fps, self.src_n = video_info(path)
        self.w, self.h = (int(size[0]), int(size[1])) if size else (self.src_w, self.src_h)
        self.gray, self.dev, self.max_frames = gray, dev or device(), max_frames
        hw = (self.src_w * self.src_h >= 2_000_000) if hwaccel == "auto" else bool(hwaccel)
        vf = []
        if every and int(every) > 1:                    # keep every Nth frame inside ffmpeg (exact, no dup / drop)
            vf.append(f"select=not(mod(n\\,{int(every)}))")
        if fps:
            vf.append(f"fps={fps:g}")
        if (self.w, self.h) != (self.src_w, self.src_h):
            vf.append(f"scale={self.w}:{self.h}:flags=bilinear")
        cmd = [self.ff, "-hide_banner", "-loglevel", "error"]
        if hw:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-i", self.path, "-an", "-sn"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if every and int(every) > 1:
            cmd += ["-fps_mode", "passthrough"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "pipe:1"]
        self.nbytes = self.w * self.h * (1 if gray else 3)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.nbytes * 2)
        self._q = queue.Queue(maxsize=queue_size)
        self._err = []
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self):
        n = 0
        try:
            while self.max_frames is None or n < self.max_frames:
                buf = self._proc.stdout.read(self.nbytes)
                if len(buf) < self.nbytes:
                    break
                self._q.put(np.frombuffer(buf, np.uint8))
                n += 1
        except Exception as e:                          # noqa: BLE001
            self._err.append(e)
        finally:
            self._q.put(None)

    def __iter__(self):
        import torch
        c = 1 if self.gray else 3
        while True:
            a = self._q.get()
            if a is None:
                break
            yield torch.from_numpy(a.reshape(self.h, self.w, c).copy()).to(self.dev)
        self.close()

    def close(self):
        try:
            if self._proc.poll() is None:
                self._proc.kill()
            err = self._proc.stderr.read().decode(errors="replace")[-300:] if self._proc.stderr else ""
            self._proc.wait(timeout=10)
        except Exception:
            err = ""
        if self._err:
            raise RuntimeError(f"video read failed: {self._err[0]}")
        return err


# --------------------------------------------------------------------------- writer
class GpuVideoWriter:
    """Pipe raw RGB frames (torch uint8 (H,W,3) or numpy) to ffmpeg; NVENC with libx264 fallback; atomic rename."""

    def __init__(self, path, fps, w, h, ffmpeg_exe=None, prefer_gpu=True, bitrate="8M", crf=20, encoder=None):
        """encoder: an ALREADY-RESOLVED "h264_nvenc" / "libx264" (skips the NVENC probe -- required inside isolated child
        processes, where a fresh probe would itself open a transient NVENC session)."""
        self.path = Path(path)
        self.tmp = self.path.with_name(f"{self.path.stem}.tmp{self.path.suffix}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ff = ffmpeg_exe or get_ffmpeg()
        self.w, self.h = int(w), int(h)
        self.gpu = (encoder == "h264_nvenc") if encoder else bool(prefer_gpu and nvenc_ok(self.ff))
        self.encoder = "h264_nvenc" if self.gpu else "libx264"
        enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0"] if self.gpu
               else ["-c:v", "libx264", "-threads", "2", "-crf", str(crf), "-preset", "veryfast"])
        cmd = [self.ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.w}x{self.h}", "-r", f"{float(fps):g}", "-i", "pipe:0", "-an"] + enc + \
              ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.tmp)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._q = queue.Queue(maxsize=6)
        self._err = []
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self):
        try:
            while True:
                b = self._q.get()
                if b is None:
                    return
                self._proc.stdin.write(b)
        except Exception as e:                          # noqa: BLE001  (encoder died)
            self._err.append(e)
            while self._q.get() is not None:
                pass

    def write(self, frame):
        if self._err:
            raise RuntimeError(f"{self.encoder} stopped: {self._err[0]}")
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        self._q.put(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self):
        self._q.put(None)
        self._t.join(timeout=120)
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        rc = self._proc.wait(timeout=180)
        err = self._proc.stderr.read().decode(errors="replace")[-400:]
        if rc != 0 or self._err:
            self.tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{self.encoder} failed (rc={rc}): {err}")
        self.tmp.replace(self.path)
        return self.path

    def abort(self):
        try:
            self._q.put(None)
            self._proc.kill()
        except Exception:
            pass
        self.tmp.unlink(missing_ok=True)


def run_with_encoder_fallback(fn, logger=None):
    """fn(prefer_gpu: bool) -> result.  NVENC first; if anything in the GPU-encoded run fails, once more with libx264."""
    try:
        return fn(True)
    except Exception as e:                              # noqa: BLE001
        if logger is not None:
            try:
                logger.info(f"  GPU-encoded run failed ({str(e)[:160]}); retrying with the CPU encoder")
            except Exception:
                pass
        return fn(False)


# --------------------------------------------------------------------------- canvas + text
class FrameCanvas:
    """Float RGB canvas (H, W, 3, 0..1) on the GPU with Raster's anti-aliased drawing, at the video's own resolution."""

    def __init__(self, w, h, dev=None):
        from cube_skeleton_render import Raster
        self.dev = dev or device()
        self.R = Raster(int(w), int(h), self.dev, ss=1)
        self.w, self.h = int(w), int(h)

    @property
    def img(self):
        return self.R.canvas

    def load_rgb(self, u8):                             # (H, W, 3) uint8
        self.R.canvas.copy_(u8.float().div_(255.0))

    def load_gray(self, u8):                            # (H, W, 1) uint8 -> neutral RGB
        self.R.canvas.copy_(u8.float().div_(255.0).expand(-1, -1, 3))

    def fill(self, rgb):
        self.R.canvas[:] = self.R.col(rgb)

    def rect(self, x0, y0, x1, y1, rgb, alpha=1.0):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.w, int(x1)), min(self.h, int(y1))
        if x1 > x0 and y1 > y0:
            roi = self.R.canvas[y0:y1, x0:x1]
            roi.lerp_(self.R.col(rgb), alpha)

    def to_u8(self):
        return (self.R.canvas.clamp(0, 1) * 255.0 + 0.5).to(self.R.t.uint8)


class TextAtlas:
    """Glyphs rendered once with PIL and blitted on the GPU.  outline>0 adds a dark stroke (legible on any footage)."""

    CHARS = [chr(c) for c in range(32, 127)] + ["\u00b7", "\u2264", "\u00b0"]

    def __init__(self, size_px, dev=None, bold=False, outline=0):
        import torch
        from PIL import Image, ImageDraw, ImageFont
        self.t, self.dev = torch, dev or device()
        fp = None
        for p in ((r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf") if bold else ()) + \
                (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
            if os.path.exists(p):
                fp = p
                break
        if fp is None:
            try:
                import matplotlib
                fp = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
            except Exception:
                fp = None
        font = ImageFont.truetype(fp, max(6, int(size_px))) if fp else ImageFont.load_default()
        asc, desc = font.getmetrics()
        self.pad = int(outline) + 1
        self.line_h = asc + desc + 2 * self.pad
        self.tiles, self.adv = {}, {}
        for ch in self.CHARS:
            adv = font.getlength(ch)
            w = int(np.ceil(adv)) + 2 * self.pad
            fill = Image.new("L", (w, self.line_h), 0)
            ImageDraw.Draw(fill).text((self.pad, self.pad), ch, font=font, fill=255)
            stroke = Image.new("L", (w, self.line_h), 0)
            if outline:
                ImageDraw.Draw(stroke).text((self.pad, self.pad), ch, font=font, fill=255, stroke_width=int(outline),
                                            stroke_fill=255)
            self.tiles[ch] = (torch.from_numpy(np.asarray(fill, np.float32) / 255.0).to(self.dev),
                              torch.from_numpy(np.asarray(stroke, np.float32) / 255.0).to(self.dev))
            self.adv[ch] = adv
        self.outline = outline

    def width(self, text):
        return sum(self.adv.get(c, 0) for c in text)

    def blit(self, img, text, x, y, rgb=(1, 1, 1), outline_rgb=(0, 0, 0)):
        """img: float (H, W, 3) tensor; (x, y) = top-left of the text line."""
        t = self.t
        col = t.tensor(rgb, device=self.dev, dtype=t.float32).view(1, 1, 3)
        ocol = t.tensor(outline_rgb, device=self.dev, dtype=t.float32).view(1, 1, 3)
        H, W = img.shape[:2]
        cx = float(x)
        for ch in text:
            tile = self.tiles.get(ch)
            if tile is None:
                continue
            fill, stroke = tile
            h, w = fill.shape
            x0, y0 = int(round(cx)) - self.pad, int(y) - self.pad
            xs, ys = max(0, x0), max(0, y0)
            xe, ye = min(W, x0 + w), min(H, y0 + h)
            if xe > xs and ye > ys:
                sl = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
                roi = img[ys:ye, xs:xe]
                if self.outline:
                    roi.lerp_(ocol, stroke[sl].unsqueeze(-1))
                roi.lerp_(col, fill[sl].unsqueeze(-1))
            cx += self.adv[ch]
        return cx
