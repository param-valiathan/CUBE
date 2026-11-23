"""Guided labelling tab (Tk view over cube_guided_session.GuidedController).

One screen, one obvious next action: 1 Pick video -> 2 Mark behaviours -> 3 Answer questions -> 4 Save.
All heavy work (loading a run, retraining, scoring, exporting) runs in a worker thread; Tk is only
touched from the UI thread (workers hand results back through a queue polled with after()).
"""
from __future__ import annotations

import colorsys
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np

import cube_guided_session as gs
from cube_busy_overlay import BusyOverlay
from cube_label_player import LabelPlayer

PREFS = Path.home() / ".cube_guided_prefs.json"
STEPS = [
    ("1  Pick video", "Pick a video on the left (double-click it, or select it and press Enter)."),
    ("2  Mark behaviours", "Scrub, press Start [S] where a behaviour begins and End [E] where it ends, then click its "
                           "coloured behaviour button (or press its number). To change one later, click the segment on the "
                           "bottom row, then another button. Mark at least 2 behaviours."),
    ("3  Answer questions", "Pick the behaviour that matches the clip (CUBE's guess is starred), or Neither / not sure."),
    ("4  Save", "Your labels save automatically. Export the guided track and cluster mapping when done."),
]
HELP = ("Mouse: click the timeline to jump | mouse wheel = 1 frame (Shift = 1 s, Ctrl = zoom) | drag on the "
        "bottom row = new segment | drag a segment edge = resize | right-click a segment = relabel/delete\n"
        "Keys: Left/Right = 1 frame | Shift+Left/Right = 1 s | Space = play/pause | S / E = start / end | "
        "1-9 = pick behaviour | Delete = remove segment | Ctrl+Z / Ctrl+Y = undo / redo | Ctrl+S = save")


class _Tip:
    def __init__(self, widget, text):
        self.w, self.text, self.tw = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _e=None):
        if self.tw or not self.text:
            return
        self.tw = tk.Toplevel(self.w)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{self.w.winfo_rootx() + 12}+{self.w.winfo_rooty() + self.w.winfo_height() + 4}")
        tk.Label(self.tw, text=self.text, bg="#fffde7", fg="#222", relief="solid", bd=1, justify="left",
                 wraplength=320, font=("Segoe UI", 9), padx=6, pady=3).pack()

    def _hide(self, _e=None):
        if self.tw is not None:
            try:
                self.tw.destroy()
            except Exception:
                pass
            self.tw = None


def _cluster_palette(n=64):
    out = {}
    for i in range(-1, n):
        if i < 0:
            out[i] = "#9e9e9e"
        else:
            r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.36, 0.93)
            out[i] = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    return out


class GuidedTab(tk.Frame):
    def __init__(self, master, theme: dict, on_use_labels_changed=None, **kw):
        super().__init__(master, bg=theme["bg"], **kw)
        self.T = theme
        self.on_use_labels_changed = on_use_labels_changed
        self.data = None
        self.ctrl = None
        self.cur_key = None
        self.run_dir = None
        self._jobs: queue.Queue = queue.Queue()
        self._poll_id = None
        self._step = 1
        self._questions: list = []
        self._qi = 0
        self._answered_in_round = 0
        self._cand_key = None
        self._ex_key = None
        self._busy = False
        self._view = "msg"             # which stacked view is on top: msg / label / q
        self._selected_beh = None
        self._saved_sha = None         # label-set digest at open / last save (unsaved-changes prompt)
        self._hist: list = []          # question-card undo stack: ('answer'|'skip', Query)
        self._qgen = 0                 # bumped whenever the user leaves question mode / opens another video
        self._welcome_pending = False
        self._shutdown = False
        self.use_labels_var = tk.BooleanVar(master=self, value=True)
        self.show_clusters_var = tk.BooleanVar(master=self, value=True)
        self._palette = _cluster_palette()
        self._build()
        self._poll_id = self.after(100, self._poll_jobs)

    # ------------------------------------------------------------ building
    def _btn(self, parent, text, cmd, bg=None, fg=None, **kw):
        T = self.T
        return tk.Button(parent, text=text, command=cmd, takefocus=0, bg=bg or T["btn"], fg=fg or T["text"],
                         activebackground=T["btn_hover"], relief="flat", padx=10, pady=5, cursor="hand2",
                         font=("Segoe UI", 9, "bold" if bg else "normal"), **kw)

    def _build(self):
        T = self.T
        strip = tk.Frame(self, bg=T["panel"], highlightbackground=T["border"], highlightthickness=1)
        strip.pack(fill="x", padx=6, pady=(6, 3))
        row = tk.Frame(strip, bg=T["panel"])
        row.pack(fill="x", padx=8, pady=(6, 0))
        self.step_lbls = []
        for i, (name, _txt) in enumerate(STEPS):
            lb = tk.Label(row, text=name, font=("Segoe UI", 10, "bold"), padx=10, pady=3, bg=T["panel"],
                          fg=T["text_dim"])
            lb.pack(side="left")
            self.step_lbls.append(lb)
            if i < len(STEPS) - 1:
                tk.Label(row, text="->", bg=T["panel"], fg=T["text_dim"]).pack(side="left")
        help_btn = self._btn(row, "?  Help", self._show_welcome)
        help_btn.pack(side="right")
        _Tip(help_btn, "Show the quick-start guide again.")
        self.step_text = tk.Label(strip, text="", anchor="w", bg=T["panel"], fg=T["text"], font=("Segoe UI", 10))
        self.step_text.pack(fill="x", padx=14, pady=(0, 6))
        strip.bind("<Configure>", lambda e: self.step_text.config(wraplength=max(300, e.width - 40)))

        body = tk.Frame(self, bg=T["bg"])
        body.pack(fill="both", expand=True, padx=6, pady=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # left: videos
        left = tk.Frame(body, bg=T["panel"], width=290, highlightbackground=T["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 4))
        left.pack_propagate(False)
        tk.Label(left, text="Videos", font=("Segoe UI", 11, "bold"), bg=T["panel"], fg=T["text"]
                 ).pack(anchor="w", padx=8, pady=(8, 2))
        self.search_var = tk.StringVar(master=self)
        self._ph_active = False
        ent = tk.Entry(left, textvariable=self.search_var, relief="flat", bd=0, highlightthickness=1,
                       highlightbackground=T["border"], highlightcolor=T["accent"], font=("Segoe UI", 9))
        ent.pack(fill="x", padx=8, ipady=3)
        self._ph_text = "Search videos ..."

        def _ph_on(_e=None):
            if not self.search_var.get():
                self._ph_active = True
                ent.config(fg=T["text_dim"])
                self.search_var.set(self._ph_text)

        def _ph_off(_e=None):
            if self._ph_active:
                self._ph_active = False
                self.search_var.set("")
                ent.config(fg=T["text"])
        ent.bind("<FocusIn>", _ph_off)
        ent.bind("<FocusOut>", _ph_on)
        self.search_var.trace_add("write", lambda *_: self._populate_sessions())
        _ph_on()
        self.tree = ttk.Treeview(left, columns=("st",), show="tree headings", selectmode="browse", height=18)
        self.tree.heading("#0", text="Video")
        self.tree.heading("st", text="State")
        self.tree.column("#0", width=190)
        self.tree.column("st", width=70, anchor="center")
        self.tree.tag_configure("missing", foreground="#9e9e9e")
        self.tree.pack(fill="both", expand=True, padx=8, pady=6)
        self.tree.bind("<Double-1>", lambda e: self._open_selected())
        self.tree.bind("<Return>", lambda e: self._open_selected())
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.complete_var = tk.BooleanVar(master=self, value=False)
        self.complete_cb = tk.Checkbutton(
            left, text="This video is fully labelled", variable=self.complete_var, command=self._complete_toggled,
            bg=T["panel"], fg=T["text"], anchor="w", selectcolor=T["panel"], wraplength=260, justify="left",
            state="disabled")
        self.complete_cb.pack(fill="x", padx=8, pady=(6, 0))
        _Tip(self.complete_cb, "Tick ONLY when you have marked EVERY instance of all your behaviours in this video. "
                               "Everything not marked then counts as 'none of my behaviours' - the best way to teach CUBE "
                               "what is NOT your behaviour, and the only way to measure false positives.")
        self.tree_note = tk.Label(left, text="", wraplength=270, justify="left", anchor="nw", bg=T["panel"],
                                  fg=T["text_dim"], font=("Segoe UI", 8))
        self.tree_note.pack(fill="x", padx=8, pady=(0, 8))

        # centre: stacked views
        centre = tk.Frame(body, bg=T["bg"])
        centre.grid(row=0, column=1, sticky="nsew")
        centre.rowconfigure(0, weight=1)
        centre.columnconfigure(0, weight=1)
        self.view_label = tk.Frame(centre, bg=T["bg"])
        self.view_q = tk.Frame(centre, bg=T["panel"], highlightbackground=T["border"], highlightthickness=1)
        self.view_msg = tk.Frame(centre, bg=T["panel"])
        self.view_busy = BusyOverlay(centre, T)            # "CUBE is working": logo + progress bar + status
        for v in (self.view_label, self.view_q, self.view_msg, self.view_busy):
            v.grid(row=0, column=0, sticky="nsew")
        self.player = LabelPlayer(self.view_label, T, tile_size=(760, 420))
        self.player.pack(fill="both", expand=True)
        self.player.on_segment_changed = self._on_segments_changed
        _box = tk.Frame(self.view_msg, bg=T["panel"])
        _box.place(relx=0.5, rely=0.42, anchor="center")
        self.msg_lbl = tk.Label(_box, text="Open a CUBE run to start.", bg=T["panel"], fg=T["text"],
                                font=("Segoe UI", 12), wraplength=620, justify="center")
        self.msg_lbl.pack()
        self.btn_open = self._btn(_box, "Open a CUBE run ...", self._open_run_dialog, T["accent"], "#ffffff")
        self.btn_open.pack(pady=(14, 0))
        _Tip(self.btn_open, "Pick the run's 'CUBE results ...' folder, or its 'CUBE 3 - Clusters' folder.")
        self._build_question_card()
        self._raise('msg')

        # right: behaviours
        right = tk.Frame(body, bg=T["panel"], width=300, highlightbackground=T["border"], highlightthickness=1)
        right.grid(row=0, column=2, sticky="ns", padx=(4, 0))
        right.pack_propagate(False)
        tk.Label(right, text="Behaviours", font=("Segoe UI", 11, "bold"), bg=T["panel"], fg=T["text"]
                 ).pack(anchor="w", padx=8, pady=(8, 2))
        self.beh_frame = tk.Frame(right, bg=T["panel"])
        self.beh_frame.pack(fill="x", padx=8)
        r1 = tk.Frame(right, bg=T["panel"])
        r1.pack(fill="x", padx=8, pady=4)
        b = self._btn(r1, "+ New behaviour", self._new_behaviour, T["accent"], "#ffffff")
        b.pack(fill="x")
        _Tip(b, "Name a behaviour you want CUBE to learn, e.g. Rearing.")
        b2 = self._btn(r1, "+ Background / other", self._add_background)
        b2.pack(fill="x", pady=(3, 0))
        _Tip(b2, "Background / other: things that are none of your behaviours. Mark a few stretches of it "
                 "so CUBE learns what 'nothing of interest' looks like.")
        r2 = tk.Frame(right, bg=T["panel"])
        r2.pack(fill="x", padx=8)
        self._btn(r2, "Rename", self._rename_behaviour).pack(side="left")
        self._btn(r2, "Delete", self._delete_behaviour, fg=T["accent4"]).pack(side="left", padx=4)
        self.card = tk.Label(right, text="", justify="left", anchor="w", wraplength=270, bg=T["panel2"],
                             fg=T["text"], font=("Segoe UI", 9, "bold"), padx=8, pady=6)
        self.card.pack(fill="x", padx=8, pady=8)
        cb = tk.Checkbutton(right, text="Use my labels in the next CUBE run", variable=self.use_labels_var,
                            command=self._use_labels_toggled, bg=T["panel"], fg=T["text"], anchor="w",
                            selectcolor=T["panel"], wraplength=230, justify="left")
        cb.pack(fill="x", padx=8)
        _Tip(cb, "When on, the next clustering run uses your labels to protect and refine clusters. "
                 "Turn off to run exactly as before.")
        cb2 = tk.Checkbutton(right, text="Show CUBE clusters on the timeline", variable=self.show_clusters_var,
                             command=self._clusters_toggled, bg=T["panel"], fg=T["text"], anchor="w",
                             selectcolor=T["panel"])
        cb2.pack(fill="x", padx=8)
        self.btn_q = self._btn(right, "Start questions", self._start_questions, T["accent2"], "#ffffff")
        self.btn_q.pack(fill="x", padx=8, pady=(10, 2))
        _Tip(self.btn_q, "CUBE shows short clips it is unsure of and asks which of your behaviours each one shows.")
        _fr = tk.Frame(right, bg=T["panel"])
        _fr.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(_fr, text="Ask about:", bg=T["panel"], fg=T["text_dim"], font=("Segoe UI", 9)).pack(side="left")
        self.focus_var = tk.StringVar(master=self, value=self._FOCUS_ANY)
        self.focus_cb = ttk.Combobox(_fr, textvariable=self.focus_var, state="readonly", width=24,
                                     values=[self._FOCUS_ANY])
        self.focus_cb.pack(side="left", padx=(6, 0))
        _Tip(self.focus_cb, "Choose a behaviour that CUBE gets wrong or that you have few labels for (e.g. a rare "
                            "one): the questions will show clips that could be that behaviour.")
        self.q_reason = tk.Label(right, text="", wraplength=270, justify="left", anchor="w", bg=T["panel"],
                                 fg=T["text_dim"], font=("Segoe UI", 8))
        self.q_reason.pack(fill="x", padx=8, pady=(0, 6))
        self.btn_x = self._btn(right, "Save results", self._export)
        self.btn_x.pack(fill="x", padx=8)
        self.btn_p = self._btn(right, "Produce plots ...", self._produce_plots)
        self.btn_p.pack(fill="x", padx=8, pady=(6, 0))
        _Tip(self.btn_p, "After you re-run CUBE with your labels: exports before/after UMAPs, Coverage / Leakage / "
                         "Ratio before vs after, a hierarchy with your behaviour names, and the behaviour groups "
                         "for the Behaviour Analyser, into the run's plots/guided folder.")
        self.help_toggle = self._btn(right, "Controls  (show/hide)", self._toggle_help)
        self.help_toggle.pack(fill="x", padx=8, pady=(10, 2))
        self.help_lbl = tk.Label(right, text=HELP, wraplength=270, justify="left", anchor="nw", bg=T["panel2"],
                                 fg=T["text_dim"], font=("Segoe UI", 8), padx=6, pady=4)
        self._help_shown = False

        self.status = tk.Label(self, text="", anchor="w", bg=T["panel"], fg=T["text_dim"], font=("Segoe UI", 9),
                               padx=10)
        self.status.pack(fill="x", padx=6, pady=(3, 6))
        self._set_step(1)
        self._refresh_behaviours()
        self.bind("<Map>", self._on_map)
        self.bind("<Unmap>", self._on_unmap)

    def _on_map(self, e=None):
        if e is not None and e.widget is not self:
            return
        if self._welcome_pending and self.ctrl is not None:
            self._welcome_pending = False
            self.after(200, self._maybe_welcome)       # only when the user actually opens this tab

    def _on_unmap(self, e=None):
        if e is not None and e.widget is not self:
            return
        for p in (self.player, self.p_cand, self.p_ex):  # nothing keeps decoding behind another tab
            try:
                p.pause()
            except Exception:
                pass

    def _build_question_card(self):
        T, q = self.T, self.view_q
        self.q_title = tk.Label(q, text="", font=("Segoe UI", 16, "bold"), bg=T["panel"], fg=T["text"])
        self.q_title.pack(pady=(10, 0))
        self.q_sub = tk.Label(q, text="", font=("Segoe UI", 10), bg=T["panel"], fg=T["text_dim"])
        self.q_sub.pack()
        tiles = tk.Frame(q, bg=T["panel"])
        tiles.pack(fill="both", expand=True, padx=8, pady=6)
        tiles.columnconfigure((0, 1), weight=1)
        tiles.rowconfigure(1, weight=1)
        tk.Label(tiles, text="This clip", bg=T["panel"], fg=T["text_dim"]).grid(row=0, column=0)
        self.q_ex_title = tk.Label(tiles, text="For comparison", bg=T["panel"], fg=T["text_dim"])
        self.q_ex_title.grid(row=0, column=1)
        self.p_cand = LabelPlayer(tiles, T, tile_size=(420, 260), compact=True)
        self.p_ex = LabelPlayer(tiles, T, tile_size=(420, 260), compact=True)
        self.p_cand.grid(row=1, column=0, sticky="nsew", padx=3)
        self.p_ex.grid(row=1, column=1, sticky="nsew", padx=3)
        # one answer button per behaviour the user defined (rebuilt for every question), then the
        # always-available Neither / Skip / Stop row
        self.q_beh_frame = tk.Frame(q, bg=T["panel"])
        self.q_beh_frame.pack(pady=(6, 2))
        btns = tk.Frame(q, bg=T["panel"])
        btns.pack(pady=(2, 6))
        self.qb_back = self._btn(btns, "< Back  [B]", self._back)
        self.qb_nei = self._btn(btns, "Neither / not sure  [X]", lambda: self._answer("neither"))
        self.qb_skip = self._btn(btns, "Skip  [S]", self._skip)
        self.qb_stop = self._btn(btns, "Stop for now", self._stop_questions)
        for w in (self.qb_back, self.qb_nei, self.qb_skip, self.qb_stop):
            w.pack(side="left", padx=5)
        self._q_buttons: list = []
        self.q_prog = tk.Label(q, text="", bg=T["panel"], fg=T["text_dim"])
        self.q_prog.pack(pady=(0, 8))
        q.bind("<Key>", self._question_key)
        for w in (self.p_cand, self.p_ex):
            w.bind("<Key>", self._question_key, add="+")

    # ------------------------------------------------------------- helpers
    def _raise(self, which):
        # stacked views stay 'mapped' in Tk even when covered, so remember which one is on top
        self._view = which
        {"msg": self.view_msg, "label": self.view_label, "q": self.view_q, "busy": self.view_busy}[which].tkraise()

    # ------------------------------------------------- "CUBE is working" overlay
    _FOCUS_ANY = "Any behaviour (CUBE chooses)"

    def _focus_value(self):
        v = self.focus_var.get() if hasattr(self, "focus_var") else ""
        return None if v in ("", self._FOCUS_ANY) else v

    def _busy_begin(self, title, text=""):
        if self._view != "busy":
            self._view_before = self._view
        self.view_busy.begin(title, text)
        self._raise("busy")

    def _busy_end(self, restore=False):
        """Stop the timer. ``restore=True`` also puts the previous view back (use when nothing else is raised)."""
        self.view_busy.end()
        if restore and self._view == "busy":
            self._raise(getattr(self, "_view_before", "msg"))

    def _progress(self, fraction, text=None):
        """Thread-safe: worker threads call this; the UI thread applies it (same queue as every other job)."""
        self._jobs.put((lambda a: self.view_busy.update_progress(*a), (fraction, text)))

    def _sync_progress(self, fraction, text=None):
        """For work that runs on the UI thread itself (e.g. saving while the window closes)."""
        self.view_busy.update_progress(fraction, text)
        self.update_idletasks()
        self.update()

    def _status(self, msg, error=False):
        self.status.config(text=msg, fg=self.T["accent4"] if error else self.T["text_dim"])

    def _saved_status(self):
        if self.ctrl is not None and self.ctrl.last_error:
            self._status(f"NOT SAVED: {self.ctrl.last_error}  (your latest change is only in memory; it will be "
                         f"retried on the next change)", error=True)
        else:
            self._status(f"Saved {time.strftime('%H:%M:%S')}")

    @staticmethod
    def _pl(n, word):
        return f"{n} {word}" + ("" if n == 1 else "s")

    @staticmethod
    def _display(key, width=34):
        """Video label for the list: no '_prep_' prefix, middle-ellipsis when long (full name in the note)."""
        k = key
        while k.startswith("_prep_"):
            k = k[len("_prep_"):]
        return k if len(k) <= width else k[: width // 2 - 1] + "..." + k[-(width // 2 - 2):]

    def _set_step(self, n):
        self._step = n
        for i, lb in enumerate(self.step_lbls, 1):
            on = i == n
            lb.config(bg=self.T["accent"] if on else self.T["panel"],
                      fg="#ffffff" if on else (self.T["text"] if i < n else self.T["text_dim"]))
        self.step_text.config(text=STEPS[n - 1][1])

    def _show_msg(self, text):
        self.msg_lbl.config(text=text)
        self._raise('msg')

    def _run_bg(self, fn, done, err=None):
        """Run fn() in a worker thread; done(result) / err(exc) run on the UI thread."""
        def work():
            try:
                res = fn()
                self._jobs.put((done, res))
            except Exception as e:                                 # noqa: BLE001
                self._jobs.put(((err or self._default_err), e))
                self._log_error(traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _poll_jobs(self):
        self._poll_id = None
        try:
            while True:
                cb, arg = self._jobs.get_nowait()
                try:
                    cb(arg)
                except Exception:                                   # noqa: BLE001
                    self._log_error(traceback.format_exc())
        except queue.Empty:
            pass
        if not self._shutdown:
            self._poll_id = self.after(100, self._poll_jobs)

    def _default_err(self, e):
        self._busy = False
        self._status(f"Something went wrong: {e}. Details are in the guided folder's error log.")

    def _log_error(self, text):
        try:
            d = (self.run_dir or Path.home()) / "guided"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "guided_error.log", "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
        except Exception:
            pass

    # --------------------------------------------------------------- run IO
    def set_run(self, model_dir, bout_dir=None, video_dirs=(), run_dir=None, use_labels=True):
        self.run_dir = Path(run_dir) if run_dir else Path(model_dir).parent
        self.use_labels_var.set(bool(use_labels))
        self._busy_begin("Loading your CUBE run", "Opening the run ...")
        self._status("Loading ...")

        def load():
            d = gs.GuidedData(model_dir, bout_dir, video_dirs, progress=self._progress)
            self._progress(0.97, "Setting up your labels ...")
            return d, gs.GuidedController(d, self.run_dir, log=self._log_error)

        def done(res):
            self._busy_end()
            self.data, self.ctrl = res
            self._saved_sha = self.ctrl.ls.sha1
            self._populate_sessions()
            self._refresh_behaviours()
            self._refresh_card()
            self.player.set_behaviours(self.ctrl.ls.behaviours)
            run_name = Path(self.run_dir).parent.name if self.run_dir else ""
            if self.ctrl.load_warning:
                self._status(self.ctrl.load_warning, error=True)
            elif self.data.problems:
                self._status("Loaded with notes: " + "; ".join(self.data.problems[:2]))
            else:
                self._status(f"Loaded {len(self.data.sessions)} videos, {self.data.n_bins} analysed time bins"
                             + (f"  -  run: {run_name}" if run_name else "") + ".")
            self._show_msg("Pick a video on the left to start labelling.")
            self._use_labels_toggled()
            self._welcome_pending = True
            if self.winfo_ismapped():
                self._on_map()

        def err(e):
            self._busy_end()
            msg = str(e) if isinstance(e, (FileNotFoundError, ValueError)) else \
                "This run's data could not be read. See guided_error.log in the run's guided folder."
            self._show_msg(msg)
            self._status("Guided labelling is unavailable for this run.")

        self._run_bg(load, done, err)

    def open_layout(self, lay):
        self.set_run(lay.model_dir, lay.bout_dir, [lay.video_root], lay.run_dir, True)

    def _open_run_dialog(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(parent=self, title="Select a finished CUBE run (CUBE results ... or "
                                                          "CUBE 3 - Clusters)")
        if not folder:
            return
        lay = gs.find_run_layout(folder)
        if lay is None:
            self._show_msg("That folder does not contain a finished CUBE clustering run.\n\nLooked for "
                           "model\\session_bin_ranges.json in it and around it. Pick the run's 'CUBE results ...' "
                           "folder or its 'CUBE 3 - Clusters' folder (Step 3 must have finished).")
            return
        self.open_layout(lay)

    def _populate_sessions(self):
        if self.data is None:
            return
        flt = "" if self._ph_active else self.search_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        for s in self.data.sessions:
            if flt and flt not in s.key.lower():
                continue
            st = self.ctrl.session_status(s.key) if self.ctrl else "untouched"
            tags = () if s.video else ("missing",)
            self.tree.insert("", "end", iid=s.key, text=self._display(s.key),
                             values=(st if s.video else "no video",), tags=tags)

    def _on_tree_select(self, _e=None):
        sel = self.tree.selection()
        if sel and self.data:
            s = self.data.session(sel[0])
            self.tree_note.config(text=s.video_reason if not s.video else s.key)

    def _open_selected(self):
        sel = self.tree.selection()
        if sel:
            self._open_session(sel[0])

    def _open_session(self, key):
        if self.data is None or self.ctrl is None:
            return
        s = self.data.session(key)
        if not s.video:
            self._show_msg(s.video_reason)
            self._status("This video can't be opened - see the note under the list.")
            return
        self._stop_questions(silent=True)
        self._raise('label')
        dropped = self.player.unassigned_count() if self.player._loaded else 0
        ok = self.player.load(s.video, s.fps, s.frame_clusters if self.show_clusters_var.get() else None,
                              self._palette)
        if not ok:
            self._status("Could not open this video file.")
            return
        self.cur_key = key
        self.complete_cb.config(state="normal")
        self.complete_var.set(self.ctrl.is_complete(key))
        self.player.set_behaviours(self.ctrl.ls.behaviours)
        self.player.set_segments(self.ctrl.segments_for(key))
        self.player.set_readonly_segments(self.ctrl.query_segments_for(key))
        self.player.focus_set()
        self._set_step(2 if self._step < 3 else self._step)
        self._status(f"Opened {key}." + (f"  {dropped} segment(s) in the previous video had no behaviour "
                                          "and were not saved." if dropped else "")
                     + (f"  {self.player.fps_note}" if self.player.fps_note else ""))

    # ------------------------------------------------------------ behaviours
    def _refresh_behaviours(self):
        for w in self.beh_frame.winfo_children():
            w.destroy()
        T = self.T
        bl = self.ctrl.ls.behaviours if self.ctrl else []
        try:                                         # keep the "Ask about" choices in step with the behaviours
            names = [b["name"] for b in bl if b["name"] != "Background"]
            self.focus_cb.configure(values=[self._FOCUS_ANY] + names)
            if self.focus_var.get() not in [self._FOCUS_ANY] + names:
                self.focus_var.set(self._FOCUS_ANY)
        except Exception:                            # noqa: BLE001
            pass
        if not bl:
            tk.Label(self.beh_frame, text="No behaviours yet - click + New behaviour.", bg=T["panel"],
                     fg=T["text_dim"], wraplength=230, justify="left").pack(anchor="w")
            return
        secs = self.ctrl.summary()["seconds"]
        for i, b in enumerate(bl, 1):
            sel = b["name"] == self._selected_beh
            rbg = T["sel_row"] if sel else T["panel"]
            row = tk.Frame(self.beh_frame, bg=rbg, cursor="hand2")
            row.pack(fill="x", pady=1)
            tk.Label(row, width=2, bg=b["color"]).pack(side="left", padx=(0, 6))
            tk.Label(row, text=f"{i}  {b['name']}", bg=rbg, fg=T["text"], anchor="w",
                     font=("Segoe UI", 9, "bold" if sel else "normal")).pack(side="left")
            tk.Label(row, text=f"{secs.get(b['name'], 0.0):.0f} s", bg=rbg, fg=T["text_dim"]
                     ).pack(side="right")
            for w in [row] + list(row.winfo_children()):
                w.bind("<Button-1>", lambda e, n=b["name"]: self._select_behaviour(n))

    def _select_behaviour(self, name):
        """Select a behaviour in the list (for Rename / Delete). It does NOT relabel the selected segment:
        that is done by the coloured buttons under the video or the number keys."""
        self._selected_beh = name
        self.after_idle(self._refresh_behaviours)

    def _selected(self):
        return self._selected_beh

    def _new_behaviour(self):
        if self.ctrl is None:
            return
        name = simpledialog.askstring("New behaviour", "Name of the behaviour (e.g. Rearing):", parent=self)
        if not name:
            return
        if not self.ctrl.add_behaviour(name):
            messagebox.showinfo("Name not available", "That name is empty or already used.", parent=self)
            return
        self._after_behaviours_changed()

    def _add_background(self):
        if self.ctrl and self.ctrl.add_behaviour(gs.BACKGROUND):
            self._after_behaviours_changed()

    def _rename_behaviour(self):
        n = self._selected()
        if not (self.ctrl and n):
            return
        new = simpledialog.askstring("Rename behaviour", f"New name for '{n}':", parent=self)
        if new and self.ctrl.rename_behaviour(n, new.strip()):
            self._after_behaviours_changed()
            if self.cur_key:
                self.player.set_segments(self.ctrl.segments_for(self.cur_key))

    def _delete_behaviour(self):
        n = self._selected()
        if not (self.ctrl and n):
            return
        if messagebox.askyesno("Delete behaviour",
                               f"Delete '{n}' and ALL its labelled segments?\nThis cannot be undone.", parent=self):
            self.ctrl.delete_behaviour(n)
            self._selected_beh = None
            self._after_behaviours_changed()
            if self.cur_key:
                self.player.set_segments(self.ctrl.segments_for(self.cur_key))

    def _after_behaviours_changed(self):
        self.player.set_behaviours(self.ctrl.ls.behaviours)
        self._refresh_behaviours()
        self._refresh_card()
        self._saved_status()

    def _advice_note(self) -> str:
        try:
            stop, msg = self.ctrl.stop_advice()
        except Exception:                                          # noqa: BLE001
            return ""
        return ("  " + ("STOP? " if stop else "") + msg) if msg else ""

    def _complete_toggled(self):
        if not (self.ctrl and self.cur_key):
            return
        self.player.pause()
        self.ctrl.set_complete(self.cur_key, bool(self.complete_var.get()))
        self._refresh_card()
        self._saved_status()
        if self.complete_var.get():
            self._status("Marked fully labelled: everything you did not mark in this video now counts as 'none of my "
                         "behaviours'. Untick if you have not marked every instance.")

    def _tab_key(self, e):
        if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Text)):
            return None
        if e.char and e.char.isdigit() and e.char != "0" and self.ctrl:
            i = int(e.char) - 1
            if i < len(self.ctrl.ls.behaviours):
                self._select_behaviour(self.ctrl.ls.behaviours[i]["name"])
        return None

    # ----------------------------------------------------------- label events
    def _on_segments_changed(self, segs):
        if not (self.ctrl and self.cur_key):
            return
        self.ctrl.set_manual_segments(self.cur_key, segs)
        self._refresh_behaviours()
        self._refresh_card()
        self._populate_sessions_keep_selection()
        self._saved_status()
        if self._step == 2 and self.ctrl.can_train()[0]:
            self.step_text.config(text="Nice - you can keep marking, or press Start questions on the right.")

    def _populate_sessions_keep_selection(self):
        """Update the state column in place (no rebuild, so the list never jumps while labelling)."""
        if not (self.ctrl and self.data):
            return
        for k in self.tree.get_children():
            if self.data.session(k).video:
                self.tree.set(k, "st", self.ctrl.session_status(k))

    def _refresh_card(self):
        if not self.ctrl:
            self.card.config(text="")
            return
        s = self.ctrl.summary()
        line = (f"{self._pl(s['n_behaviours'], 'behaviour')}  |  {self._pl(s['n_bouts'], 'bout')}  |  "
                f"{self._pl(s['n_sessions'], 'video')} labelled")
        cov = self.ctrl.coverage_line()
        warn = self.ctrl.label_design_warning()
        self.card.config(text=line + (("\n" + cov) if cov else "") + (("\n! " + warn) if warn else ""),
                         fg=self.T["accent3"] if (cov and self.ctrl.coverage()["unusable"]) else self.T["text"])
        self._update_start_button()

    def _update_start_button(self):
        if not self.ctrl:
            return
        ok, why = self.ctrl.can_train()
        pending = self.player.unassigned_count() if self.player._loaded else 0
        if pending:
            ok, why = False, "Pick a behaviour for the highlighted segment first."
        T = self.T
        self.btn_q.config(state="normal" if ok else "disabled", bg=T["accent2"] if ok else T["btn"],
                          fg="#ffffff" if ok else T["text_dim"], disabledforeground=T["text_dim"])
        self.q_reason.config(text="" if ok else why)

    def _use_labels_toggled(self):
        if self.on_use_labels_changed and self.ctrl:
            self.on_use_labels_changed(str(self.ctrl.labels_path) if self.use_labels_var.get() else "")

    def _clusters_toggled(self):
        if self.cur_key and self.data:
            s = self.data.session(self.cur_key)
            self.player.cluster_labels = s.frame_clusters if self.show_clusters_var.get() else None
            self.player._redraw()

    def _toggle_help(self):
        if self._help_shown:
            self.help_lbl.pack_forget()
        else:
            self.help_lbl.pack(fill="x", padx=8)
        self._help_shown = not self._help_shown

    # -------------------------------------------------------------- questions
    def _start_questions(self):
        if self.ctrl is None:
            return
        if self._busy:
            self._status("Still working on the previous step - please wait a moment.")
            return
        ok, why = self.ctrl.can_train()
        if not ok:
            self._status(why)
            messagebox.showinfo("Not enough labels yet", why, parent=self)
            return
        self._busy = True
        self._qgen += 1
        gen = self._qgen
        self.player.pause()
        self._status("Training on your labels ...")
        self._busy_begin("Training on your labels", "The first round prepares features for the whole run "
                                                    "(10-20 s on a large run).")

        focus = self._focus_value()

        def work():
            self.ctrl.focus = focus
            self.ctrl.progress = lambda f, t: self._progress(0.7 * f, t)
            info = self.ctrl.retrain()
            self.ctrl.progress = lambda f, t: self._progress(0.7 + 0.3 * f, t)
            qs_ = self.ctrl.next_queries(10)
            try:                                  # stop rule: honest leakage on the fully labelled video(s)
                self._progress(0.95, "Measuring false positives on your fully labelled video ...")
                self.ctrl.honest_check()
            except Exception:                     # noqa: BLE001 - never block the questions
                pass
            return info, qs_

        def done(res):
            self._busy = False
            self._busy_end()
            self.ctrl.progress = None
            if gen != self._qgen:        # user left question mode / opened a video meanwhile
                return
            info, qs = res
            f1 = info.get("cv_f1")
            note = (f" (held-out check F1 about {f1:.2f})" if f1 is not None else "") + self._advice_note()
            if not qs:
                self._show_msg("No more useful questions - you can stop.\n" + note)
                self._set_step(4)
                return
            self._questions, self._qi, self._answered_in_round = qs, 0, 0
            self._hist = []
            self._set_step(3)
            self._status("Model trained" + note)
            self._raise('q')
            self._show_question()

        self._run_bg(work, done, lambda e: (setattr(self, "_busy", False), self._busy_end(), self._show_msg(
            "Training did not work: " + str(e) if isinstance(e, ValueError) else
            "Training failed. Details are in the guided folder's error log."), self._status("Training failed.")))

    def _show_question(self):
        if self._qi >= len(self._questions):
            return self._round_done()
        q = self._questions[self._qi]
        self.q_title.config(text=f"Question {self._qi + 1} of {len(self._questions)}:  which behaviour is this clip?")
        self.q_sub.config(text=f"CUBE's guess: {q.top1} ({q.maxprob:.0%} sure)"
                               + (f", then {q.top2}" if q.top2 else "")
                               + f"  |  {q.session}  {q.start_s:.1f}-{q.end_s:.1f} s")
        self._build_answer_buttons(q)
        self.q_prog.config(text="")
        s = self.data.session(q.session)
        if not s.video:
            self._status(f"Skipped a question: video for {q.session} not found.")
            self._qi += 1
            return self._show_question()
        if self._cand_key != s.video:
            self._cand_key = s.video if self.p_cand.load(s.video, s.fps) else None
        self.p_cand.play_range(int(round(q.start_s * s.fps)), int(round(q.end_s * s.fps)))
        shown = False
        if q.exemplar:
            es = self.data.session(q.exemplar[0])
            if es.video:
                if self._ex_key != es.video:
                    self._ex_key = es.video if self.p_ex.load(es.video, es.fps) else None
                if self._ex_key:
                    self.p_ex.play_range(int(round(q.exemplar[1] * es.fps)), int(round(q.exemplar[2] * es.fps)))
                    shown = True
        if shown:
            self.q_ex_title.config(text=f"CUBE's guess, for comparison: {q.top1}")
        else:                                        # never leave the previous question's exemplar looping
            self.p_ex.pause()
            self.p_ex.clear_range()
            self.p_ex.video.config(image="", text=f"No example of {q.top1} to show yet")
            self.q_ex_title.config(text="No comparison clip")
        self.view_q.focus_set()

    def _build_answer_buttons(self, q):
        """One button per behaviour the user defined (keys 1-9); CUBE's guess is marked with a star."""
        for w in self.q_beh_frame.winfo_children():
            w.destroy()
        T = self.T
        names = self.ctrl.behaviour_names() if self.ctrl else []
        colour = {b["name"]: b.get("color", "#757575") for b in self.ctrl.ls.behaviours} if self.ctrl else {}
        per_row = 4
        for i, name in enumerate(names):
            guess = name == q.top1
            label = f"{'* ' if guess else ''}{name}" + (f"  [{i + 1}]" if i < 9 else "")
            b = self._btn(self.q_beh_frame, label, lambda n=name: self._answer(n), colour.get(name), "#ffffff")
            if guess:
                b.config(relief="solid", bd=3)                 # CUBE's guess: starred and outlined
            b.grid(row=i // per_row, column=i % per_row, padx=4, pady=3, sticky="ew")
            self.q_beh_frame.columnconfigure(i % per_row, weight=1, uniform="ans")
        self.q_beh_frame.update_idletasks()

    def _question_key(self, e):
        if self._view == 'q':
            k = (e.keysym or "").lower()
            if k == "x":
                self._answer("neither")
            elif k == "s":
                self._skip()
            elif k == "b":
                self._back()
            elif len(k) == 1 and k in "123456789" and self.ctrl:
                names = self.ctrl.behaviour_names()
                if int(k) <= len(names):
                    self._answer(names[int(k) - 1])

    def _answer(self, a):
        """a = a behaviour name the user defined, or 'neither'."""
        if not self._questions or self._qi >= len(self._questions):
            return
        q = self._questions[self._qi]
        self.ctrl.answer(q, a)
        self._hist.append(("answer", q))
        self._answered_in_round += 1
        self._qi += 1
        self._refresh_card()
        self._saved_status()
        self._show_question()

    def _skip(self):
        if self._questions and self._qi < len(self._questions):
            self.ctrl.skip(self._questions[self._qi])          # never asked again
            self._hist.append(("skip", self._questions[self._qi]))
        self._qi += 1
        self._show_question()

    def _back(self):
        """Undo the previous answer (or skip) and show that question again."""
        if not self._hist or self._qi <= 0:
            self._status("Nothing to go back to in this round.")
            return
        kind, _q = self._hist.pop()
        if kind == "answer":
            self.ctrl.undo_last_answer()
            self._answered_in_round = max(0, self._answered_in_round - 1)
        elif self.ctrl.skipped:
            self.ctrl.skipped.pop()
        self._qi -= 1
        self._refresh_card()
        self._show_question()

    def _round_done(self):
        self.p_cand.pause()
        self.p_ex.pause()
        self.ctrl.end_round()
        if self._answered_in_round == 0:
            self._stop_questions()
            return
        self._busy = True
        gen = self._qgen
        self._status("Updating the model with your answers ...")
        self._busy_begin("Learning from your answers", "Retraining the model ...")

        focus = self._focus_value()

        def work():
            self.ctrl.focus = focus
            self.ctrl.progress = lambda f, t: self._progress(0.7 * f, t)
            info = self.ctrl.retrain()
            self.ctrl.progress = lambda f, t: self._progress(0.7 + 0.3 * f, t)
            qs_ = self.ctrl.next_queries(10)
            try:                                  # stop rule: honest leakage on the fully labelled video(s)
                self._progress(0.95, "Measuring false positives on your fully labelled video ...")
                self.ctrl.honest_check()
            except Exception:                     # noqa: BLE001 - never block the questions
                pass
            return info, qs_

        def done(res):
            self._busy = False
            self._busy_end()
            self.ctrl.progress = None
            if gen != self._qgen:        # user pressed Stop / opened a video while the model retrained
                return
            info, qs = res
            f1 = info.get("cv_f1")
            note = (f"Held-out check F1 about {f1:.2f}. " if f1 is not None else "") + self._advice_note().strip() + " "
            if not qs:
                self._questions = []
                self._show_msg(f"{note}No more useful questions - you can stop.")
                self._set_step(4)
                return
            self._questions, self._qi, self._answered_in_round = qs, 0, 0
            self._hist = []
            self._status(f"{note}{len(qs)} more questions ready.")
            self._raise('q')
            self._show_question()

        self._run_bg(work, done, lambda e: (setattr(self, "_busy", False), self._busy_end(), self._stop_questions(),
                                            self._status("Could not update the model. Your answers are saved.")))

    def _stop_questions(self, silent=False):
        self._qgen += 1                 # invalidate any in-flight retrain result
        self.p_cand.pause()
        self.p_ex.pause()
        self._questions = []
        if self._view == 'q':
            self._raise('label')
        if not silent:
            self._set_step(4)
            self._status("Stopped. Your answers are saved.")

    # ---------------------------------------------------------------- export
    def _export(self):
        if self.ctrl is None:
            return
        if self._busy:
            self._status("Still working on the previous step - please wait a moment.")
            return
        ok, why = self.ctrl.can_train()
        if not ok:
            self._status(why)
            return
        self._busy = True
        self._status("Saving results ...")
        self._busy_begin("Saving your results", "Exporting the behaviour track and the cluster mapping ...")

        def done(res):
            self._busy = False
            self._busy_end(restore=True)
            self.ctrl.progress = None
            self._saved_sha = self.ctrl.ls.sha1
            self._set_step(4)
            mixed = res.get("mixed_clusters", [])
            gdir = res.get("guided_behaviours_dir")
            self._status("Saved: " + Path(res["track_csv"]).name + " and " + Path(res["mapping_tsv"]).name +
                         (f". Guided behaviours for the Behaviour Analyser: {gdir}" if gdir else "") +
                         (f". Mixed clusters to review: {mixed}" if mixed else ""))

        def export_work():
            self.ctrl.progress = self._progress
            return self.ctrl.export_all()

        self._run_bg(export_work, done, lambda e: (setattr(self, "_busy", False), self._busy_end(restore=True),
                                                  self._status(f"Could not save results: {e}", error=True)))

    # ------------------------------------------------------------ produce plots
    def _before_run_dir(self, ask=True):
        """The blind run to compare against: the snapshot a labelled run keeps, else ask for a folder."""
        snap = Path(self.run_dir) / "guided" / "before_run"
        if (snap / "model" / "session_bin_ranges.json").is_file():
            return snap
        if not ask:
            return None
        from tkinter import filedialog
        if not messagebox.askyesno(
                "Before / after", "To show what your labels changed, I need the EARLIER run (before you used your "
                "labels).\n\nSelect its 'CUBE 3 - Clusters' folder?\n\n(No = produce the after-only plots.)",
                parent=self):
            return None
        f = filedialog.askdirectory(parent=self, title="Select the earlier run's 'CUBE 3 - Clusters' folder")
        if not f:
            return None
        p = Path(f)
        for cand in (p, p / "CUBE 3 - Clusters"):
            if (cand / "model" / "session_bin_ranges.json").is_file():
                return cand
        messagebox.showwarning("Not a CUBE run", "That folder has no model/session_bin_ranges.json.", parent=self)
        return None

    def _produce_plots(self):
        if self.ctrl is None:
            return
        if self._busy:
            self._status("Still working on the previous step - please wait a moment.")
            return
        if self.ctrl.summary()["n_bouts"] == 0:
            self._status("Label some behaviours first - the plots compare the clusters with your labels.")
            return
        before = self._before_run_dir(ask=True)
        self.ctrl.save()
        self._busy = True
        self._status("Producing plots ... this takes a minute.")
        self._busy_begin("Producing before / after plots", "Starting ...")
        import cube_guided_report as rep
        run_dir, labels_path = Path(self.run_dir), str(self.ctrl.labels_path)

        def work():
            return rep.produce_plots(run_dir, labels_path, before_dir=before, log=self._log_error,
                                     progress=self._progress)

        def done(res):
            self._busy = False
            self._busy_end(restore=True)
            self._saved_sha = self.ctrl.ls.sha1
            out = res.get("out_dir") or str(run_dir / "plots" / "guided")
            extra = f" {res['before']}" if "before" in res and isinstance(res["before"], str) else ""
            multi = [b for b, c in res.get("behaviour_groups", {}).items() if len(c) > 1]
            self._status(f"Plots written to {out}.  Behaviour groups saved for the analyser"
                         + (f" ({', '.join(multi)} made of several clusters)." if multi else ".") + extra)
            try:
                import os
                os.startfile(out)
            except Exception:
                pass

        self._run_bg(work, done, lambda e: (setattr(self, "_busy", False), self._busy_end(restore=True),
                                            self._status(f"Could not produce plots: {e}", error=True)))

    def unsaved_results(self) -> bool:
        return bool(self.ctrl is not None and self._saved_sha is not None
                    and self.ctrl.ls.sha1 != self._saved_sha and self.ctrl.summary()["n_bouts"] > 0)

    def confirm_close(self) -> bool:
        """False = the user chose to stay. Asks to save when labels changed since the last save / plots."""
        self.flush()
        if not self.unsaved_results():
            return True
        a = messagebox.askyesnocancel(
            "Save your results?",
            "You changed your labels since the results were last saved.\n\nSave now? This exports your behaviour "
            "groups for the Behaviour Analyser and, if a labelled CUBE run exists, the before/after plots.\n\n"
            "Yes = save and close     No = close without saving     Cancel = stay", parent=self)
        if a is None:
            return False
        if a:
            try:
                self._busy_begin("Saving your results before closing", "Exporting ...")
                self.update()
                self.ctrl.progress = lambda f, t: self._sync_progress(0.5 * f, t)
                self.ctrl.export_all()
                self.ctrl.progress = None
                import cube_guided_report as rep
                rep.produce_plots(Path(self.run_dir), str(self.ctrl.labels_path),
                                  before_dir=self._before_run_dir(ask=False), log=self._log_error,
                                  progress=lambda f, t=None: self._sync_progress(0.5 + 0.5 * f, t))
                self._saved_sha = self.ctrl.ls.sha1
                self._busy_end(restore=True)
            except Exception as e:                              # noqa: BLE001
                self.ctrl.progress = None
                self._busy_end(restore=True)
                return bool(messagebox.askyesno("Could not save", f"{e}\n\nClose anyway?", parent=self))
        return True

    # --------------------------------------------------------------- welcome
    def _prefs(self):
        try:
            return json.loads(PREFS.read_text())
        except Exception:
            return {}

    def _maybe_welcome(self):
        if not self._prefs().get("hide_welcome"):
            self._show_welcome()

    def _show_welcome(self):
        w = tk.Toplevel(self)
        w.title("Guided labelling - quick start")
        w.configure(bg=self.T["panel"])
        w.transient(self.winfo_toplevel())
        txt = ("Teach CUBE what a behaviour looks like - in about a minute.\n\n"
               "1.  Open a video, scrub to a behaviour, press S (start) and E (end), then click its coloured behaviour button.\n"
               "2.  Mark at least 2 different behaviours (a 'Background' example helps).\n"
               "3.  Press Start questions: pick which of your behaviours each clip shows (or Neither) for clips CUBE is unsure of.\n\n"
               "Everything saves automatically. Nothing changes in CUBE's normal results unless you leave "
               "'Use my labels in the next CUBE run' ticked.")
        tk.Label(w, text=txt, justify="left", bg=self.T["panel"], fg=self.T["text"], font=("Segoe UI", 10),
                 padx=20, pady=16, wraplength=460).pack()
        var = tk.BooleanVar(master=w, value=False)
        tk.Checkbutton(w, text="Don't show again", variable=var, bg=self.T["panel"],
                       selectcolor=self.T["panel"]).pack(anchor="w", padx=20)

        def close():
            if var.get():
                try:
                    p = self._prefs()
                    p["hide_welcome"] = True
                    PREFS.write_text(json.dumps(p))
                except Exception:
                    pass
            w.destroy()
        self._btn(w, "Got it", close, self.T["accent"], "#ffffff").pack(pady=12)
        w.protocol("WM_DELETE_WINDOW", close)

    # ------------------------------------------------------------- lifecycle
    def flush(self):
        if self.ctrl is not None:
            self.ctrl.save()

    def undo(self):
        self.player.undo()

    def shutdown(self):
        self._shutdown = True
        self.flush()
        if self.ctrl is not None and self.ctrl.dirty:
            try:
                messagebox.showwarning("Labels not saved",
                                       f"Your latest labelling changes could not be saved:\n{self.ctrl.last_error}\n\n"
                                       f"Copy them elsewhere or fix the problem, then reopen this tab.", parent=self)
            except Exception:
                pass
        for p in (self.player, self.p_cand, self.p_ex):
            try:
                p.close()
            except Exception:
                pass
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except Exception:
                pass








