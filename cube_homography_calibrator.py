# -*- coding: utf-8 -*-
"""
cube_homography_calibrator.py — Two/three-camera floor-plane homography
calibrator (TwoCam_Homography_BoneConstraint_Fusion_Plan.md §4, Phase 2).

Standalone window (like cube_video_explorer.py): pick a frame from each of
two camera views of the same Doric rig, click 4+ shared physical reference
points (arena floor corners plus a wall/height point) in both views, then
compute and cache the floor-plane homography relating them. Not yet
auto-triggered from Step 2 -- that wiring is a later phase of the plan.

Run standalone:
    python   (with the CUBE environment active) cube_homography_calibrator.py
    (the path is saved in cube_env_path.txt by install_cube.bat)

Requirements: pillow and OpenCV (already in the CUBE env).
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cube_core


def _raise_window(win):
    """Bring a newly created window to the front on Windows (see the
    matching helper in cube.py / cube_video_explorer.py for why)."""
    try:
        win.lift()
        win.focus_force()
        win.attributes("-topmost", True)
        win.after(150, lambda: win.attributes("-topmost", False))
    except Exception:
        pass


class _CameraPane(tk.Frame):
    """One camera view: video picker, frame slider, and a click-capturing
    canvas. Clicked points are stored in *self.points* in ORIGINAL video
    pixel coordinates (not canvas/display coordinates), so they're directly
    usable by cube_core.fit_floor_homography() regardless of how the frame
    was scaled to fit the canvas.
    """

    CANVAS_W, CANVAS_H = 480, 360

    def __init__(self, parent, label: str, on_point_added):
        super().__init__(parent, bd=1, relief="groove")
        self._on_point_added = on_point_added
        self.points: list[tuple[float, float]] = []
        self._cap = None
        self._frame_bgr = None
        self._photo = None
        self._scale = 1.0
        self.video_path: "str | None" = None

        tk.Label(self, text=label, font=("Segoe UI", 11, "bold")).pack(pady=(6, 2))

        btn_row = tk.Frame(self)
        btn_row.pack(fill="x", padx=6)
        tk.Button(btn_row, text="Load Video…", command=self._pick_video).pack(side="left")
        self._path_lbl = tk.Label(btn_row, text="(no video loaded)", fg="#666")
        self._path_lbl.pack(side="left", padx=8)

        self._frame_var = tk.IntVar(value=0)
        self._frame_scale = tk.Scale(self, from_=0, to=0, orient="horizontal",
                                      variable=self._frame_var, showvalue=True,
                                      command=lambda _v: self._show_frame(int(self._frame_var.get())))
        self._frame_scale.pack(fill="x", padx=6)

        self.canvas = tk.Canvas(self, width=self.CANVAS_W, height=self.CANVAS_H, bg="#222")
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind("<Button-1>", self._on_click)

        tk.Button(self, text="Undo last point", command=self._undo_point).pack(pady=(0, 6))

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Select camera video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str):
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Homography Calibrator", f"Could not open video:\n{path}")
            return
        self._cap = cap
        self.video_path = path
        n_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        self._frame_scale.configure(to=n_frames)
        self._path_lbl.configure(text=Path(path).name)
        self._show_frame(0)

    def _show_frame(self, frame_idx: int):
        import cv2
        from PIL import Image, ImageTk
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok:
            return
        self._frame_bgr = frame
        h, w = frame.shape[:2]
        self._scale = min(self.CANVAS_W / w, self.CANVAS_H / h)
        disp_w, disp_h = int(w * self._scale), int(h * self._scale)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((disp_w, disp_h))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._redraw_points()

    def _redraw_points(self):
        self.canvas.delete("pt")
        for i, (ox, oy) in enumerate(self.points, 1):
            cx, cy = ox * self._scale, oy * self._scale
            r = 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     outline="#00ff88", width=2, tags="pt")
            self.canvas.create_text(cx + 8, cy - 8, text=str(i), fill="#00ff88",
                                     font=("Segoe UI", 9, "bold"), tags="pt")

    def _on_click(self, event):
        if self._frame_bgr is None or self._scale == 0:
            return
        orig_x, orig_y = event.x / self._scale, event.y / self._scale
        self.points.append((orig_x, orig_y))
        self._redraw_points()
        self._on_point_added()

    def _undo_point(self):
        if self.points:
            self.points.pop()
            self._redraw_points()
            self._on_point_added()


def _build_ui(root, cam_a_label=None, cam_b_label=None, on_saved=None):
    root.title("CUBE — Two-Camera Floor-Plane Homography Calibrator")
    root.geometry("1040x760")

    tk.Label(root, text=("Click the same real-world point in both camera views — start with the "
                          "4 floor corners, then one point partway up a wall.\n"
                          "Use 3–5 frames from different parts of the video so the points aren't "
                          "all bunched together."),
             justify="left", wraplength=1000, font=("Segoe UI", 10)).pack(pady=(8, 4), padx=10, anchor="w")

    panes_row = tk.Frame(root)
    panes_row.pack(padx=10, pady=6)

    status = {"pair_pending": False}

    def _on_point_added():
        na, nb = len(pane_a.points), len(pane_b.points)
        pair_count_var.set(f"Paired points: {min(na, nb)}   (view A: {na}, view B: {nb})")
        compute_btn.configure(state=("normal" if min(na, nb) >= 4 else "disabled"))

    pane_a = _CameraPane(panes_row, "Camera A", _on_point_added)
    pane_a.pack(side="left", padx=6)
    pane_b = _CameraPane(panes_row, "Camera B", _on_point_added)
    pane_b.pack(side="left", padx=6)

    cam_label_row = tk.Frame(root)
    cam_label_row.pack(pady=(0, 4))
    tk.Label(cam_label_row, text="Camera A label (e.g. cam0):").pack(side="left")
    cam_a_label_var = tk.StringVar(value=cam_a_label or "")
    tk.Entry(cam_label_row, textvariable=cam_a_label_var, width=10).pack(side="left", padx=(2, 16))
    tk.Label(cam_label_row, text="Camera B label (e.g. cam1):").pack(side="left")
    cam_b_label_var = tk.StringVar(value=cam_b_label or "")
    tk.Entry(cam_label_row, textvariable=cam_b_label_var, width=10).pack(side="left", padx=(2, 0))

    pair_count_var = tk.StringVar(value="Paired points: 0   (view A: 0, view B: 0)")
    tk.Label(root, textvariable=pair_count_var, font=("Segoe UI", 10, "italic")).pack()

    result_holder = {"result": None}
    result_var = tk.StringVar(value="")
    result_lbl = tk.Label(root, textvariable=result_var, font=("Segoe UI", 10), fg="#0a6", justify="left")
    result_lbl.pack(pady=(4, 4))

    def _compute():
        na, nb = len(pane_a.points), len(pane_b.points)
        n = min(na, nb)
        if n < 4:
            messagebox.showwarning("Homography Calibrator",
                                    "Need at least 4 paired points in both views.")
            return
        try:
            result = cube_core.fit_floor_homography(pane_a.points[:n], pane_b.points[:n])
        except ValueError as e:
            messagebox.showerror("Homography Calibrator", f"Could not fit homography:\n{e}")
            return
        result_holder["result"] = result
        err = result["reproj_error_px"]
        if err < 3.0:
            quality = f"Floor-plane estimate looks accurate to within ~{err:.1f} px for paws/tail."
            result_lbl.configure(fg="#0a6")
        elif err < 10.0:
            quality = (f"Floor-plane estimate is usable but noisier than ideal "
                       f"(~{err:.1f} px). Consider re-clicking a few points.")
            result_lbl.configure(fg="#b8860b")
        else:
            quality = (f"WARNING: reprojection error is large (~{err:.1f} px, "
                       f"max {result['max_error_px']:.1f} px) — the clicked points may not "
                       f"actually correspond, or lie too close to a line. Re-click and try again.")
            result_lbl.configure(fg="#c00")
        result_var.set(f"{quality}\n({n} point pairs, RMS {err:.2f} px, max {result['max_error_px']:.2f} px)")
        save_btn.configure(state="normal")

    def _cam_labels():
        cam_a = cam_a_label_var.get().strip()
        cam_b = cam_b_label_var.get().strip()
        if not cam_a or not cam_b:
            messagebox.showwarning("Homography Calibrator",
                                    "Enter both camera labels (e.g. cam0 / cam1) first.")
            return None
        return cam_a, cam_b

    def _save():
        result = result_holder.get("result")
        if result is None:
            return
        labels = _cam_labels()
        if labels is None:
            return
        cam_a, cam_b = labels
        result = dict(result)
        result["cam_src"], result["cam_dst"] = cam_a, cam_b
        rig_id = cube_core.camera_pair_rig_id(cam_a, cam_b)
        path = cube_core.save_cached_homography(rig_id, result)
        messagebox.showinfo("Homography Calibrator",
                             f"Saved homography for {cam_a}<->{cam_b} to:\n{path}")
        if on_saved is not None:
            on_saved(rig_id, result)

    def _load():
        labels = _cam_labels()
        if labels is None:
            return
        cam_a, cam_b = labels
        rig_id = cube_core.camera_pair_rig_id(cam_a, cam_b)
        cached = cube_core.load_cached_homography(rig_id)
        if cached is None:
            messagebox.showinfo("Homography Calibrator",
                                 f"No cached homography found for {cam_a}<->{cam_b}.")
            return
        result_holder["result"] = cached
        err = cached.get("reproj_error_px", float("nan"))
        source = ("auto-calibrated from tracked poses" if cached.get("method") == "pose_auto"
                   else "manually clicked")
        result_var.set(f"Loaded cached homography for {cam_a}<->{cam_b} ({source}, "
                        f"{cached.get('n_points', '?')} points, RMS {err:.2f} px at capture time).")
        save_btn.configure(state="normal")

    btn_row = tk.Frame(root)
    btn_row.pack(pady=(2, 6))
    compute_btn = tk.Button(btn_row, text="Compute Homography", state="disabled", command=_compute)
    compute_btn.pack(side="left", padx=4)

    save_btn = tk.Button(btn_row, text="Save to Cache", state="disabled", command=_save)
    save_btn.pack(side="left", padx=4)
    tk.Button(btn_row, text="Load Cached", command=_load).pack(side="left", padx=4)

    return {"pane_a": pane_a, "pane_b": pane_b, "result_holder": result_holder}


def launch_homography_calibrator(parent=None, cam_a_label=None, cam_b_label=None, on_saved=None):
    """
    Launch the calibrator window. If *parent* is given (an existing Tk
    widget from cube.py), opens as a Toplevel; otherwise runs as its own
    Tk root (standalone launch).

    cam_a_label, cam_b_label : pre-fill the two camera-label fields (e.g.
        when cube.py already knows which two cam_labels a detected group
        needs calibrated) -- the user can still edit them.
    on_saved : optional callable(rig_id, result) invoked right after a
        successful "Save to Cache" -- lets a blocking caller (cube.py's
        Step 2 auto-trigger) detect that calibration actually completed.
    """
    if parent is not None:
        win = tk.Toplevel(parent)
        win.transient(parent)
    else:
        win = tk.Tk()
    _build_ui(win, cam_a_label=cam_a_label, cam_b_label=cam_b_label, on_saved=on_saved)
    win.after(10, lambda: _raise_window(win))
    return win


if __name__ == "__main__":
    app = launch_homography_calibrator()
    app.mainloop()
