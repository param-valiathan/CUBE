"""CUBE system check ("doctor").

Reports, in plain language, whether this computer and Python environment are ready to run each CUBE step,
and what to do when something is missing.

    python cube_doctor.py            print the report (exit code 1 if a required part is missing)
    python cube.py --check           same thing

Only the standard library is imported at module level, so the check itself still runs in a broken
environment. The GUI calls run_checks() / format_report() / group_ready() from a background thread.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"

REFERENCE_PYTHON = (3, 12)          # major, minor of the reference environment (3.12.13)
HERE = Path(__file__).resolve().parent
LOCK_FILE = HERE / "env" / "requirements-lock.txt"

INSTALL_HINT = "Double-click install_cube.bat in the CUBE folder to repair the installation."

# group -> [(label, distribution name, import name)]
GROUPS = {
    "Core (Steps 2-5)": [
        ("NumPy", "numpy", "numpy"),
        ("pandas", "pandas", "pandas"),
        ("SciPy", "scipy", "scipy"),
        ("scikit-learn", "scikit-learn", "sklearn"),
        ("HDBSCAN", "hdbscan", "hdbscan"),
        ("UMAP", "umap-learn", "umap"),
        ("hmmlearn", "hmmlearn", "hmmlearn"),
        ("Matplotlib", "matplotlib", "matplotlib"),
        ("Plotly", "plotly", "plotly"),
        ("Pillow", "pillow", "PIL"),
        ("OpenCV", "opencv-contrib-python", "cv2"),
        ("customtkinter", "customtkinter", "customtkinter"),
        ("PyTables", "tables", "tables"),
        ("numba", "numba", "numba"),
        ("NetworkX", "networkx", "networkx"),
        ("scikit-posthocs", "scikit-posthocs", "scikit_posthocs"),
        ("psutil", "psutil", "psutil"),
        ("joblib", "joblib", "joblib"),
        ("threadpoolctl", "threadpoolctl", "threadpoolctl"),
        ("FFmpeg for video (imageio-ffmpeg)", "imageio-ffmpeg", "imageio_ffmpeg"),
    ],
    "Step 0 (multi-camera synchronisation)": [
        ("h5py (reads Doric files)", "h5py", "h5py"),
    ],
    "Step 1a / 1b (pose tracking)": [
        ("DeepLabCut", "deeplabcut", "deeplabcut"),
        ("PyTorch", "torch", "torch"),
        ("torchvision", "torchvision", "torchvision"),
    ],
    "Step 1b (3D multi-camera)": [
        ("Anipose (aniposelib)", "aniposelib", "aniposelib"),
    ],
}

# Which groups are required for which kind of use. A missing "Core" part is a failure;
# a missing part of any other group only blocks the step(s) named by that group.
GROUP_FAIL_LEVEL = {
    "Core (Steps 2-5)": FAIL,
    "Step 0 (multi-camera synchronisation)": WARN,
    "Step 1a / 1b (pose tracking)": WARN,
    "Step 1b (3D multi-camera)": WARN,
}


def _check(group, name, status, detail, fix=""):
    return {"group": group, "name": name, "status": status, "detail": detail, "fix": fix}


def _lock_versions() -> dict:
    """Package versions pinned in requirements-lock.txt (empty dict if the file is missing)."""
    out = {}
    try:
        for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-")) or "==" not in line:
                continue
            name, ver = line.split("==", 1)
            out[name.strip().lower().replace("_", "-")] = ver.strip().split("+")[0]
    except OSError:
        pass
    return out


def _installed_version(dist: str):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _check_python():
    v = sys.version_info
    detail = f"Python {platform.python_version()} ({sys.executable})"
    if (v.major, v.minor) == REFERENCE_PYTHON:
        return _check("System", "Python", OK, detail)
    return _check("System", "Python", WARN, detail + f" - CUBE is tested with Python "
                  f"{REFERENCE_PYTHON[0]}.{REFERENCE_PYTHON[1]}",
                  "Run install_cube.bat to create the supported environment.")


def _check_system():
    res = [_check("System", "Operating system", OK if platform.system() == "Windows" else WARN,
                  f"{platform.system()} {platform.release()}"
                  + ("" if platform.system() == "Windows" else " - CUBE is only tested on Windows 10/11"))]
    res.append(_check_python())
    try:
        import tkinter  # noqa: F401
        res.append(_check("System", "Tk (window toolkit)", OK, "available"))
    except Exception as e:
        res.append(_check("System", "Tk (window toolkit)", FAIL, f"tkinter cannot be imported: {e}", INSTALL_HINT))
    # disk space on the drive holding CUBE
    try:
        free_gb = shutil.disk_usage(str(HERE.anchor or HERE)).free / 1e9
        st = OK if free_gb >= 50 else (WARN if free_gb >= 10 else FAIL)
        res.append(_check("System", "Free disk space", st, f"{free_gb:,.0f} GB free on {HERE.anchor or HERE}",
                          "" if st == OK else "Videos and results are large; free up space or choose another output drive."))
    except OSError:
        pass
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        res.append(_check("System", "Memory (RAM)", OK if ram >= 16 else WARN, f"{ram:,.0f} GB",
                          "" if ram >= 16 else "16 GB or more is recommended for clustering many videos."))
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
                enabled = winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
            res.append(_check("System", "Windows long paths", OK if enabled else INFO,
                              "enabled" if enabled else "not enabled (only matters for very deep folder names)",
                              "" if enabled else "If a step reports 'path too long', keep video folders close to the drive root."))
        except OSError:
            pass
    return res


def _check_packages():
    lock = _lock_versions()
    res = []
    for group, items in GROUPS.items():
        for label, dist, mod in items:
            ver = _installed_version(dist)
            if ver is None and mod == "cv2":                       # any OpenCV flavour provides cv2
                for alt in ("opencv-python", "opencv-python-headless"):
                    ver = _installed_version(alt)
                    if ver:
                        dist = alt
                        break
            present = importlib.util.find_spec(mod) is not None
            level = GROUP_FAIL_LEVEL[group]
            if ver is None and not present:
                res.append(_check(group, label, level, "not installed", INSTALL_HINT))
                continue
            want = lock.get(dist.lower().replace("_", "-"))
            detail = f"version {ver}" if ver else "installed"
            if want and ver and ver.split("+")[0] != want:
                res.append(_check(group, label, WARN, f"{detail} (tested with {want})",
                                  "Usually harmless. If you see odd errors, run install_cube.bat to restore the tested versions."))
            else:
                res.append(_check(group, label, OK, detail))
    # several OpenCV wheels overwrite each other's files
    cv = [d for d in ("opencv-python", "opencv-python-headless", "opencv-contrib-python") if _installed_version(d)]
    if len(cv) > 1:
        res.append(_check("Core (Steps 2-5)", "OpenCV packages", WARN, "several installed: " + ", ".join(cv),
                          "Run install_cube.bat to leave a single OpenCV (opencv-contrib-python)."))
    return res


def _check_gpu():
    group = "Step 1a / 1b (pose tracking)"
    if importlib.util.find_spec("torch") is None:
        return []
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return [_check(group, "GPU for pose tracking", OK,
                           f"{props.name}, {props.total_memory / 1e9:.0f} GB, CUDA {torch.version.cuda}")]
        built = "built without CUDA (CPU-only install)" if not getattr(torch.version, "cuda", None) else \
            f"CUDA {torch.version.cuda} build, but no usable GPU or driver found"
        return [_check(group, "GPU for pose tracking", WARN,
                       f"no GPU available - PyTorch is {built}. Pose tracking will work but be very slow; "
                       "Steps 2-5 are not affected.",
                       ("No NVIDIA GPU in this computer? Ignore this. Have one? Update the NVIDIA driver "
                        "(nvidia.com/drivers, version 580 or newer), then run install_cube.bat -Gpu."
                        if not getattr(torch.version, "cuda", None) else
                        "Update the NVIDIA driver (nvidia.com/drivers, version 580 or newer) and restart. "
                        "No NVIDIA GPU? Ignore this."))]
    except Exception as e:
        return [_check(group, "GPU for pose tracking", WARN, f"PyTorch could not be loaded: {e}", INSTALL_HINT)]


def _check_ffmpeg():
    if importlib.util.find_spec("imageio_ffmpeg") is None:
        return []
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return [_check("Core (Steps 2-5)", "FFmpeg program", OK if os.path.isfile(exe) else FAIL,
                       exe, "" if os.path.isfile(exe) else INSTALL_HINT)]
    except Exception as e:
        return [_check("Core (Steps 2-5)", "FFmpeg program", FAIL, str(e), INSTALL_HINT)]


def run_checks() -> list:
    """Run every check and return a list of result dicts (group, name, status, detail, fix)."""
    res = []
    for fn in (_check_system, _check_packages, _check_ffmpeg, _check_gpu):
        try:
            res.extend(fn())
        except Exception as e:                      # a broken check must never take the app down
            res.append(_check("System", fn.__name__, WARN, f"check could not run: {e}"))
    return res


def group_ready(results: list, group: str) -> tuple:
    """(ready, message) for one group. ready is False only when a required part of that group is missing."""
    bad = [r for r in results if r["group"] == group and r["status"] == FAIL]
    miss = [r for r in results if r["group"] == group and r["status"] == WARN and r["detail"] == "not installed"]
    if bad or miss:
        names = ", ".join(r["name"] for r in bad + miss)
        return False, f"Missing: {names}. {INSTALL_HINT}"
    return True, "ready"


def step1_blocker(three_d: bool = False):
    """Plain-language reason Step 1 (pose tracking) cannot run, or None when it can. Cheap: no heavy imports."""
    missing = []
    wanted = list(GROUPS["Step 1a / 1b (pose tracking)"])
    if three_d:
        wanted += GROUPS["Step 1b (3D multi-camera)"]
    for label, dist, mod in wanted:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(label)
    if missing:
        return (f"Pose tracking needs {', '.join(missing)}, which is not installed in this Python environment "
                f"({sys.executable}).\n\n{INSTALL_HINT}")
    return None


_ICON = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", INFO: "[ i  ]"}


def format_report(results: list) -> str:
    lines = ["CUBE system check", "=" * 60]
    for group in ["System"] + list(GROUPS):
        rows = [r for r in results if r["group"] == group]
        if not rows:
            continue
        lines.append("")
        lines.append(group)
        for r in rows:
            lines.append(f"  {_ICON[r['status']]} {r['name']}: {r['detail']}")
            if r["fix"] and r["status"] in (WARN, FAIL):
                lines.append(f"           -> {r['fix']}")
    n_fail = sum(r["status"] == FAIL for r in results)
    n_warn = sum(r["status"] == WARN for r in results)
    lines += ["", "=" * 60]
    if n_fail:
        lines.append(f"{n_fail} required item(s) missing. CUBE may not start or run correctly.")
    elif n_warn:
        lines.append(f"CUBE can run. {n_warn} item(s) need attention for some steps (see above).")
    else:
        lines.append("Everything is ready.")
    return "\n".join(lines)


def main() -> int:
    results = run_checks()
    print(format_report(results))
    return 1 if any(r["status"] == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
