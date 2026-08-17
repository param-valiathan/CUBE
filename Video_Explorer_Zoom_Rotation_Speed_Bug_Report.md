# Video Explorer: Zoom / Rotation / Speed Controls Had No Effect — Investigation & Fix

**Date:** 2026-08-17
**File affected:** `cube_video_explorer.py` (CUBE Step 4 — Cluster Annotator)
**Reported symptom:** Changing Rotation, Speed, or Zoom in the annotator had no visible effect on the displayed videos, no matter which option was clicked.

---

## Root cause

`BSoidAnnotator` (the annotator window) is a `tk.Tk()` — a second, independent Tk **root** window — created *while `cube.py`'s own `tk.Tk()` root is already running*, via `cube.py`'s `_launch_annotate()`:

```python
app = _MOD_VIDEO.BSoidAnnotator(auto_open=False)   # a SECOND tk.Tk() root
...
app.mainloop()                                      # nested mainloop
```

Inside `BSoidAnnotator._build_ui()`, the rotation/speed/zoom controls were backed by plain Tk variables created without an explicit owner:

```python
self._rot_var   = tk.IntVar(value=0)
self._speed_var = tk.DoubleVar(value=1.0)
self._zoom_var  = tk.DoubleVar(value=1.0)
```

When a `tkinter.Variable` subclass (`IntVar`/`DoubleVar`/`StringVar`/`BooleanVar`) is constructed **without** a `master=` argument, it silently attaches itself to `tkinter._default_root`. Critically, `_default_root` is set **once**, by the *first* `Tk()` instance ever created in the process, and is **never updated** when a second `Tk()` is created later — CPython's `tkinter/__init__.py` only assigns it `if _default_root is None`.

Since `cube.py` creates its own root before ever launching Step 4, `_default_root` was already pointing at `cube.py`'s window by the time `BSoidAnnotator` (and its three control variables) were created. The result:

- `app._rot_var._root`, `app._speed_var._root`, `app._zoom_var._root` were all bound to **`cube.py`'s Tcl interpreter**, not the annotator's own.
- A real mouse click on a Rotation/Speed/Zoom control is handled by the **widget**, which lives in and updates a Tcl variable within the **annotator's own interpreter**.
- But `_apply_zoom()` / `_apply_rotation()` / `_apply_speed()` read the value back via `self._zoom_var.get()` (etc.), which reads from the **wrong interpreter** (`cube.py`'s) — a completely separate copy of that Tcl variable that was only ever set once, at creation (`value=1.0` / `value=0`), and never touched again by any click.

This is why:
- Every click, regardless of which option was visually selected, always logged `zoom -> 1.0` (confirmed directly via debug instrumentation — see below).
- Rotation and Speed exhibited the identical symptom (same root cause, not zoom-specific).
- The bug reproduced identically across **two structurally different widget types** (a `Radiobutton` row, and later a `Menubutton`+`Menu` dropdown) — ruling out widget-specific hit-testing/DPI causes, since both share the same underlying `tk.Variable` binding bug.
- Every scripted reproduction attempt during debugging (calling `.set()`/`.get()`/`_apply_zoom()` directly from Python, not through an actual widget click) **appeared to work correctly** — because those calls are self-consistent through the same (wrong) `Variable` object, and never exercise the cross-interpreter mismatch that only a genuine widget-generated Tcl event triggers.

### Direct confirmation

A diagnostic script (`test_root_mismatch.py`) reproduced `cube.py`'s exact launch pattern (an outer `tk.Tk()` created first, `BSoidAnnotator` created second) and inspected the bound interpreter directly:

```
_default_root after creating parent: .
app is _default_root: False
parent is _default_root still: True

app._zoom_var._root is app:     False   <-- WRONG interpreter
app._zoom_var._root is parent:  True    <-- bound to cube.py's root instead
```

A follow-up script (`test_nested_real_click.py`) simulated an actual Tk **widget** click (`menu.invoke(...)`, the same code path a real mouse click goes through) in the same nested-root setup:

```
zoom var BEFORE real click: 1.0
[CUBE DEBUG] zoom -> 1.0      <-- confirmed bug: real click still reads 1.0
```

After the fix (see below), the same real-click test produces:

```
zoom var BEFORE real click: 1.0
[CUBE DEBUG] zoom -> 3.0      <-- correct
sd.zoom after click: 3.0
```

---

## Fix

Explicitly pass `master=self` (the `BSoidAnnotator` instance itself) to every control variable created inside the class, so each binds to its own interpreter instead of falling back to `_default_root`:

```python
self.filter_var = tk.StringVar(master=self, value="all")
self._rot_var   = tk.IntVar(master=self, value=0)
self._speed_var = tk.DoubleVar(master=self, value=1.0)
self._zoom_var  = tk.DoubleVar(master=self, value=1.0)
```

and, inside the dropdown-menu helper `_make_dropdown()`, its internal display variable:

```python
display_var = tk.StringVar(master=parent, value=fmt(var.get()))
```

**Known related risk (not fixed as part of this report):** `cube_analyser.py`'s `BSOiDApp(ctk.CTk)` is launched the same way, as a second root nested inside `cube.py`'s already-running root (`_launch_analyse()`). Any `tk`/`ctk` control variable created there without an explicit `master=` is at risk of the identical bug. This was out of scope for the reported issue (which was specific to the Step 4 annotator) and was not audited or fixed here — flagging it for future attention if similar "control does nothing" reports surface in Step 5.

---

## Everything else tried first (in order), and why each was a reasonable but ultimately wrong turn

The investigation took a long path before landing on the actual cause. Documenting each step because each one *was* a real, legitimate bug — just not *the* bug — and each fix remains correct and in place.

### 1. Column layout didn't reorganize with zoom
**Symptom reasoning:** `MultiPlayer.load()` computed grid columns purely from the number of example clips (`1 if n==1 else 2 if n<=4 else 3`), never from zoom level or available width — so increasing zoom only grew tiles without ever changing the layout, and could overflow the window.
**Fix applied:** Column count is now computed from the canvas's real width and the zoomed tile size (`cols = avail_width // (target_w + pad)`), so zooming in reduces columns and pushes overflow into the existing vertical scrollbar.
**Verdict:** Real, legitimate bug; fix correct and kept. Not the reported issue's cause.

### 2. Canvas clipping content instead of resizing
**Symptom reasoning:** The scrollable canvas forced its embedded frame to always match the canvas's own viewport width on every `<Configure>` event, silently clipping any content wider than the window.
**Fix applied:** The canvas item now only expands to fill, never shrinks below the content's natural required size.
**Verdict:** Real bug; fix correct and kept.

### 3. Rotated tiles stayed landscape-shaped
**Symptom reasoning:** A 90°/270°-rotated (portrait) video was still being fit into a landscape-shaped tile box, heavily letterboxed.
**Fix applied:** Tile box dimensions swap (width↔height) when rotation is 90° or 270°.
**Verdict:** Real improvement; fix correct and kept.

### 4. `PIL.Image.thumbnail()` never upscales
**Symptom reasoning:** `Image.thumbnail()` only shrinks images larger than the target box — it never enlarges a smaller source image. Since BSOID example clips are often smaller than a zoomed tile target, increasing zoom had **zero visible effect on the video content itself** (only on the padding around it). Directly confirmed: a 160×120 test frame stayed 160×120 through `thumbnail()` even with a 960×720 target.
**Fix applied:** Added `VideoTile._fit_resize()`, an aspect-preserving resize that upscales via `Image.resize()` when the source is smaller than the target box. Used in both the GIF and MP4 decode paths.
**Verdict:** Real, significant bug — this alone would have caused exactly the reported symptom for small-resolution clips. Fix correct and kept. **This is very likely a second, independent contributing cause** on top of the root cause above (i.e., even after fixing the interpreter-binding bug, zoom would have had a muted/absent effect on any clip smaller than the zoomed target box without this fix).

### 5. Suspected stale process / not re-reading edited file
Multiple rounds of "did not fix it" were, on inspection via `Get-CimInstance Win32_Process`, actually caused by testing against Python processes that had been running for days (since 2026-08-13) and therefore never re-read the edited `.py` file — editing source on disk has no effect on an already-running interpreter. Two long-lived stale `cube.py` processes were found and terminated over the course of debugging.
**Verdict:** Real operational confound that repeatedly invalidated test results; not a code bug, but consumed significant debugging time and is worth remembering (always confirm `CreationDate` of the running `python.exe` postdates the last file edit before concluding a fix didn't work).

### 6. Suspected Tk/Windows repaint bug
**Symptom reasoning:** After a real (non-stale) relaunch, the user reported the display never updated after changing a setting, but a manual window resize made it "snap into place" instantly — a classic symptom of a widget being logically correct but not repainted.
**Fix applied:** Added `_force_player_redraw()`, which nudges the canvas window item's width and calls `update_idletasks()` after every reload, to force an immediate repaint rather than waiting on an unrelated event.
**Verdict:** Harmless defensive fix, kept, but **not the actual cause** — later diagnosis (console value tracing) showed the underlying *value* itself was never changing, so no amount of forced repainting could have helped. The earlier "resize snaps it into place" observation is now understood to have actually been the column/canvas-width layout (issues #1–#2 above) correcting itself for the *stuck-at-1.0* zoom level, not the zoom value changing.

### 7. Suspected stale `winfo_width()` cache
**Symptom reasoning:** `avail_width` for column computation was read via `self._player_outer.winfo_width()`, a queried/cached value, rather than the width delivered directly by `<Configure>` events.
**Fix applied:** Track canvas width from the `<Configure>` event's `event.width` directly into `self._player_canvas_width`, used instead of querying `winfo_width()`.
**Verdict:** A real (minor) robustness improvement; kept. Not the reported issue's cause.

### 8. Console value tracing added (the turning point)
Added temporary `[CUBE DEBUG]` print statements to `_apply_rotation`/`_apply_speed`/`_apply_zoom`, `VideoTile.__init__`, and the MP4 decode loop, and asked for the user's actual live-session terminal output rather than continuing to reason from screenshots.
**Result:** Every single click, regardless of which option was visually selected, logged `zoom -> 1.0`. This was the first piece of evidence that pointed away from rendering/layout entirely and toward "the value itself never changes" — which led directly to the interpreter-binding root cause.

### 9. Suspected Windows per-monitor DPI scaling / click-coordinate mismatch
**Symptom reasoning:** A 3-monitor setup with mismatched resolutions is a classic setup for per-monitor DPI scaling, which (for a non-DPI-aware app) can cause a visual click on one widget to be delivered to a different widget at the OS level.
**Fix applied:** Declared the process DPI-aware (`SetProcessDpiAwareness`) in both `cube.py` and `cube_video_explorer.py` before any Tk window is created.
**Verdict:** Good defensive practice for a Windows Tk app in general, kept, but **ruled out as the cause** here — the same `zoom -> 1.0` result persisted afterward.

### 10. Replaced Radiobutton row with a Menubutton+Menu dropdown
**Symptom reasoning:** If (9) didn't fix a hit-testing mismatch, switching to a completely different widget interaction model (single click target, explicit menu-item selection) would sidestep any remaining widget-specific coordinate bug.
**Fix applied:** `_make_dropdown()` helper; Rotation/Speed/Zoom rows now use dropdown buttons instead of inline radio buttons.
**Verdict:** The dropdown mechanism itself was verified correct in isolation (programmatically invoking each menu entry set the right value and fired the callback with it, every time). But the *same* `zoom -> 1.0` result persisted in the live app — which was the strongest single clue that the bug was never about widget type or hit-testing at all, and lived instead in how the backing **variable** was being read, independent of whatever widget wrote to it. Kept as a UI improvement regardless (unambiguous single-target clicks are good UX), but this is what finally pointed at the `Variable`/`_default_root` binding issue.

---

## Verification performed

- `py_compile` clean on both `cube.py` and `cube_video_explorer.py` after every change.
- Synthetic and real-data (`D:\CUBE_Pipeline\cube_results_20260812_164847_consensus\videos\example_clips`) reproductions confirming tile resize, column reflow, and rotation-aware tile shape all behave correctly.
- `VideoTile._fit_resize()` verified directly: upscales a 160×120 source to fill a 960×720 target, unlike the old `thumbnail()` call.
- Dropdown mechanism verified via direct `Menu.invoke()` calls: each of 1x/1.5x/2x/3x sets the correct value and fires the callback with it.
- **Root cause verified directly two ways:**
  1. `app._zoom_var._root is app` — `False` before the fix, `True` after.
  2. A real simulated widget click (`Menu.invoke()`, not a Python-level `.set()`) in `cube.py`'s exact nested-root launch pattern: logged `zoom -> 1.0` before the fix, `zoom -> 3.0` after, matching the clicked option.

## Confirmed fixed

The user confirmed live, after this fix and a fresh relaunch, that Rotation/Speed/Zoom now work correctly in the Step 4 annotator. The `[CUBE DEBUG]` print statements (in `_apply_rotation`/`_apply_speed`/`_apply_zoom`, `VideoTile.__init__`, and the MP4 decode loop) have been removed from `cube_video_explorer.py`.

---

## Follow-up: the same bug across `cube_analyser.py` (Step 5 — Behavioural Analyser)

The user reported a **past** issue in the Reclustering tab where the app "wasn't replotting when the user inputs new k values," and asked to check for the same root cause there. It was confirmed present, and turned out to be far more widespread than just the reclustering tab.

### Confirmation

`cube_analyser.py`'s `BSOiDApp(ctk.CTk)` is launched by `cube.py`'s `_launch_analyse()` the exact same way as the annotator — a second root created while `cube.py`'s own root is already running (`app = _MOD_ANALYSER.BSOiDApp()`, no `master=`). A diagnostic script (`test_analyser_root.py`) reproducing this exact launch order confirmed, for all 7 reclustering-tab control variables in `UnbiasedAnalyticsPanel` (`_maxk_var`, `_compare_k_var`, `_save_k_var`, `_pose_cap_var`, `_guided_cap_var`, `_guided_gap_var`, `_save_source_var`):

```
_save_k_var._root is app:     False   <-- wrong interpreter (before fix)
_save_k_var._root is parent:  True    <-- bound to cube.py's root instead
```

Note: the code already contained a `_force_repaint_reclustering_defaults()` workaround (called after `_run_reclustering()` and via `self.after(50, ...)` at panel build time) with a comment attributing the old symptom to `CTkEntry` widgets failing to paint their initial `textvariable` content while unmapped. That diagnosis was a reasonable read of the *symptom* at the time (`.get()` on the variable *did* look correct when checked directly — because reading and writing both went through the same, consistently-wrong-interpreter Python object) but not the true cause: the actual widget on screen lives in a different interpreter than the one the `.get()`/`.set()` calls were touching, so no amount of re-triggering a write-trace on the wrong interpreter could reliably repaint the right one.

### Scope

A file-wide search found **42 separate `tk.Var`/`ctk.Var` construction sites missing `master=`**, spanning:
- `UnbiasedAnalyticsPanel` (Top-N Bar / Volcano / Heatmap / **Reclustering** / Save Groups / plot mode) — 15 sites
- Group Editor row widgets (`GroupRow`, "Fill Selected" dialog) — 6 sites
- Model diagnostics / permutation testing panel — 9 sites
- Diff Heatmap / Sankey / transition-network panel — 7 sites
- `BSOiDApp` itself (theme toggle, FPS, ignore-groups checkbox, combined-analysis group-by) — 5 sites

All 42 were fixed by adding the appropriate `master=` (`master=self` in every case except the "Fill Selected" dialog's three fields, which use `master=dlg`, its enclosing `CTkToplevel` — Toplevels share their parent's interpreter, so either would have worked, but `dlg` is the more locally correct owner). This was applied via a small one-off patch script (`patch_analyser_vars.py`, not part of the shipped codebase) driven by the exact line numbers found, then verified line-by-line and re-diagnosed:

```
_save_k_var._root is app:     True    <-- fixed
_save_k_var._root is parent:  False
```

...and spot-checked additionally for `_theme_var`, `_fps_var`, and `_ignore_groups_var` (constructed directly in `BSOiDApp`, not a sub-panel) with the same result.

`py_compile` clean on `cube_analyser.py` after the patch.

### Not done

- This was a mechanical, pattern-matched fix (add `master=` where a `Variable` is constructed without one, inside a method where `self` is a legitimate already-parented widget). Each of the 42 sites' surrounding context was read to confirm `self` (or, for the one dialog case, `dlg`) is a valid in-interpreter widget reference before patching, but the resulting behavior of each affected tab beyond Reclustering (Group Editor, Model Diagnostics, Diff Heatmap/Sankey) was **not manually re-tested end-to-end in a live session** — only the variable-binding fix itself was verified programmatically. Recommend a general pass through each affected tab's controls next time any of them is touched, to confirm the reported "stuck value" symptom doesn't resurface elsewhere in a form not caught by this audit.
- Per `CLAUDE.md`, documentation updates (`README.md`, `CUBE_GUIDE.md`) and `md_to_docx.py` regeneration were not performed as part of this bugfix-only session; this report is a standalone investigation record, not user-facing docs. Worth a one-line changelog mention on next docs pass if this file is normally tracked there.
