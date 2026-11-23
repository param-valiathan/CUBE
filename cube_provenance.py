"""Run card: a per-run reproducibility record for CUBE.

Writes ``run_card.json`` (machine-readable) and ``run_card.md`` (human-readable,
including a DRAFT methods paragraph) next to ``bsoid_run_summary.json``.

What it records, and why (the three reviewer complaints it answers: "results
cannot be reproduced because defaults and code changed between sessions"):
  * environment      - Python, OS, CPU/RAM, versions of the packages that decide
                       numerical output (numpy, umap-learn, hdbscan, ...)
  * code identity    - git commit/branch (read from .git without invoking git)
                       AND a SHA-256 of every CUBE source module, because the
                       working tree is routinely ahead of the last commit
  * configuration    - canonical hash of the resolved cfg, the exact keys that
                       differ from BSoidEngine.DEFAULTS, and keys that are not
                       in DEFAULTS at all (a typo'd key is silently ignored by
                       the engine; a few real optional keys are read with a
                       built-in fallback and show up here as well)
  * seeds            - every seed that is a cfg value, the derived seed-sweep
                       seed list, and a count of hard-coded RNG seeds in the
                       source (so nothing is silently outside the cfg)
  * inputs           - size + SHA-256 of each pose file (fast head/tail hash by
                       default, full hash optional), plus an order-independent
                       combined hash
  * results          - the stability / benchmark / timing numbers the run
                       already wrote, so the card is a one-page summary

Design rules (shared with cube_guided*.py): stdlib only at module level, no
Tk, no ``cube_core`` import at module level (it is lazily imported only by the
command-line entry point to obtain DEFAULTS). Nothing here is allowed to raise
into the pipeline: every collector degrades to a recorded "unavailable" note.

Command line (works on OLD runs too, from their bsoid_run_summary.json)::

    python cube_provenance.py "<...>/CUBE 3 - Clusters" [--hash full|head|none]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

CARD_SCHEMA_VERSION = 1
CARD_JSON = "run_card.json"
CARD_MD = "run_card.md"

# Distributions whose versions decide numerical output or the supported env.
KEY_DISTRIBUTIONS = (
    "numpy", "scipy", "pandas", "scikit-learn", "umap-learn", "pynndescent",
    "hdbscan", "hmmlearn", "numba", "joblib", "h5py", "tables", "matplotlib",
    "opencv-contrib-python", "opencv-python", "torch", "aniposelib",
    "deeplabcut", "psutil",
)

_LARGE_VALUE_CHARS = 2000
_MIB = 1 << 20


# ──────────────────────────────────────────────────────────────────────────
#  JSON helpers
# ──────────────────────────────────────────────────────────────────────────

def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    return str(o)


def _normalize(obj):
    """Recursively make ``obj`` JSON-canonical: str dict keys, plain types."""
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    converted = _json_default(obj)
    return _normalize(converted) if converted is not obj else str(obj)


def canonical_json(obj) -> str:
    """Deterministic JSON text (sorted keys, no whitespace) of any cfg-like object."""
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=_json_default)


def canonical_cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────

def diff_from_defaults(cfg: dict, defaults: dict) -> dict:
    """``{key: {"value": v, "default": d}}`` for every cfg key whose value
    differs from ``defaults``. Keys absent from ``defaults`` are reported by
    :func:`unknown_keys`, not here. Very large values are summarised."""
    out = {}
    for k, v in (cfg or {}).items():
        if k not in defaults:
            continue
        cv, dv = canonical_json(v), canonical_json(defaults[k])
        if cv == dv:
            continue
        out[str(k)] = {"value": _summarise(v, cv), "default": _summarise(defaults[k], dv)}
    return dict(sorted(out.items()))


def _summarise(value, canon: str):
    if len(canon) <= _LARGE_VALUE_CHARS:
        return _normalize(value)
    return {"_large_value": True, "chars": len(canon),
            "sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest()}


def unknown_keys(cfg: dict, defaults: dict) -> list:
    """cfg keys that BSoidEngine.DEFAULTS does not define. The engine keeps
    such keys verbatim and never validates them. A misspelled name is
    therefore silently ignored; but a few genuine optional settings (e.g.
    umap_n_jobs, hdbscan_sweep_n_steps, pca_n_components) are read with a
    built-in ``cfg.get(key, fallback)`` instead of a DEFAULTS entry, so they
    appear here too. The list is a prompt to check each name, not an error."""
    return sorted(str(k) for k in (cfg or {}) if k not in defaults)


def seeds_from_cfg(cfg: dict) -> dict:
    cfg = cfg or {}
    base = cfg.get("umap_random_state", 42)
    n_sweep = cfg.get("seed_sweep_n", 0) or 0
    try:
        sweep_seeds = [int(base) + i for i in range(int(n_sweep))]
    except Exception:
        sweep_seeds = []
    return {
        "umap_random_state": base,
        "hmm_random_state": cfg.get("hmm_random_state"),
        "seed_sweep_n": n_sweep,
        "seed_sweep_seeds": sweep_seeds,
        "consensus_n_seeds": cfg.get("consensus_n_seeds"),
        "train_frac": cfg.get("train_frac"),
        "seed_sweep_train_frac": cfg.get("seed_sweep_train_frac"),
    }


_RNG_PATTERNS = (
    ("default_rng", re.compile(r"default_rng\(\s*(\d+)\s*\)")),
    ("RandomState", re.compile(r"RandomState\(\s*(\d+)\s*\)")),
    ("random_state=", re.compile(r"random_state\s*=\s*(\d+)")),
    ("np.random.seed", re.compile(r"random\.seed\(\s*(\d+)\s*\)")),
)


def scan_hardcoded_seeds(source_path) -> dict:
    """Count integer-literal RNG seeds in a source file (e.g. cube_core.py).
    These are NOT controlled by any cfg key, so a reproducibility record
    should say they exist. Returns ``{"default_rng(0)": 12, ...}``."""
    counts: dict = {}
    try:
        text = Path(source_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"_unavailable": str(e)}
    for label, rx in _RNG_PATTERNS:
        for m in rx.finditer(text):
            key = f"{label}({m.group(1)})" if label != "random_state=" else f"random_state={m.group(1)}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ──────────────────────────────────────────────────────────────────────────
#  Environment and code identity
# ──────────────────────────────────────────────────────────────────────────

def _dist_version(dist: str):
    try:
        from importlib import metadata as md
        return md.version(dist)
    except Exception:
        return None


def collect_environment() -> dict:
    env = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {d: _dist_version(d) for d in KEY_DISTRIBUTIONS},
        "numba_cpu_name_env": os.environ.get("NUMBA_CPU_NAME"),
    }
    try:
        import psutil
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        env["ram_total_gb"] = None
    env["packages"] = {k: v for k, v in env["packages"].items() if v is not None}
    return env


def read_git_state(repo_dir) -> dict:
    """Commit and branch read straight from ``.git`` (git need not be on PATH).
    Does NOT detect a dirty working tree -- see :func:`hash_source_files`."""
    info = {"commit": None, "branch": None, "note": None}
    try:
        git = Path(repo_dir) / ".git"
        if git.is_file():                       # worktree/submodule pointer
            txt = git.read_text().strip()
            if txt.startswith("gitdir:"):
                git = (git.parent / txt.split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            info["branch"] = ref.rsplit("/", 1)[-1]
            ref_file = git / ref
            if ref_file.exists():
                info["commit"] = ref_file.read_text().strip()
            else:
                packed = git / "packed-refs"
                if packed.exists():
                    for line in packed.read_text().splitlines():
                        if line.endswith(" " + ref) or line.endswith("\t" + ref):
                            info["commit"] = line.split()[0]
                            break
        else:
            info["commit"] = head
            info["branch"] = "(detached HEAD)"
    except Exception as e:
        info["note"] = f"unavailable: {e}"
    return info


def hash_source_files(repo_dir, patterns=("cube_*.py", "calibrate_cameras.py",
                                          "calibration_refine_poses.py",
                                          "stats_policy.py")) -> dict:
    """SHA-256 of each CUBE source module. This is the authoritative code
    identity when the working tree is ahead of the last commit (routine)."""
    out = {}
    root = Path(repo_dir)
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            try:
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(_MIB), b""):
                        h.update(chunk)
                out[p.name] = h.hexdigest()
            except Exception:
                continue
    return out


# ──────────────────────────────────────────────────────────────────────────
#  Input files
# ──────────────────────────────────────────────────────────────────────────

def hash_file(path, mode: str = "head") -> dict:
    """mode 'full' = SHA-256 of the whole file; 'head' (default) = SHA-256 of
    the first and last MiB plus the size (fast on multi-GB inputs and
    sensitive to appended/truncated/edited data in practice)."""
    p = Path(path)
    info = {"name": p.name, "path": str(p)}
    try:
        st = p.stat()
        info["size_bytes"] = st.st_size
        info["modified_utc"] = datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        h = hashlib.sha256()
        with p.open("rb") as f:
            if mode == "full" or st.st_size <= 2 * _MIB:
                for chunk in iter(lambda: f.read(_MIB), b""):
                    h.update(chunk)
                info["hash_scope"] = "full"
            else:
                h.update(f.read(_MIB))
                f.seek(-_MIB, os.SEEK_END)
                h.update(f.read(_MIB))
                h.update(str(st.st_size).encode())
                info["hash_scope"] = "first+last 1 MiB + size"
        info["sha256"] = h.hexdigest()
    except Exception as e:
        info["error"] = str(e)
    return info


def hash_inputs(paths, mode: str = "head", max_files: int = 2000) -> dict:
    """Hash a list of input files. ``combined_sha256`` is independent of the
    folder the files live in (names + per-file hashes, sorted), so the same
    data moved to another drive gives the same combined hash."""
    if mode == "none" or not paths:
        return {"mode": "none", "n_files": len(paths or []), "files": []}
    files = []
    for p in list(paths)[:max_files]:
        files.append(hash_file(p, mode))
    comb = hashlib.sha256()
    for f in sorted(files, key=lambda d: d.get("name", "")):
        comb.update(f"{f.get('name')}|{f.get('size_bytes')}|{f.get('sha256')}".encode())
    out = {"mode": mode, "n_files": len(paths), "files": files,
           "combined_sha256": comb.hexdigest()}
    if len(paths) > max_files:
        out["truncated_at"] = max_files
    return out


# ──────────────────────────────────────────────────────────────────────────
#  Results already written by the run
# ──────────────────────────────────────────────────────────────────────────

def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_key(obj, key, depth: int = 3):
    if depth < 0 or not isinstance(obj, dict):
        return None
    if key in obj:
        return obj[key]
    for v in obj.values():
        r = _find_key(v, key, depth - 1)
        if r is not None:
            return r
    return None


def read_run_outputs(output_dir) -> dict:
    """Collect the numbers the engine already wrote. Missing files are fine."""
    od = Path(output_dir)
    summary = _read_json(od / "bsoid_run_summary.json") or {}
    validation = _read_json(od / "validation_report.json") or {}
    bench = _read_json(od / "publication_metrics.json") or summary.get("benchmark") or {}
    timings = _read_json(od / "model" / "phase_timings.json") or {}
    featcfg = _read_json(od / "model" / "feature_config.json") or {}
    stability = _find_key(validation, "cluster_stability")
    res = {
        "n_sessions": summary.get("n_sessions"),
        "sessions": summary.get("sessions"),
        "fps": summary.get("fps"),
        "n_clusters": summary.get("n_clusters"),
        "cv_accuracy": summary.get("cv_accuracy"),
        "validation_status": summary.get("validation"),
        "benchmark": bench,
        "cluster_stability": stability,
        "phase_timings_total_s": timings.get("total_s"),
        "feature_config": {k: featcfg.get(k) for k in (
            "feature_version", "n_features", "bodyparts", "boxcar_win_sec",
            "pca_n_components") if k in featcfg},
        "cube_version_constant": summary.get("cube_version"),
        "created": summary.get("created"),
        "summary_cfg": summary.get("cfg"),
    }
    return res


# ──────────────────────────────────────────────────────────────────────────
#  Card assembly
# ──────────────────────────────────────────────────────────────────────────

RECONSTRUCTION_NOTE = (
    "RECONSTRUCTED AFTER THE RUN. The configuration and results below were read "
    "from the run's own output files, but the environment, package versions, git "
    "commit and source hashes describe the installation that wrote this card, "
    "which is not necessarily the one that produced the results. Do not cite "
    "them as the versions used for this run.")


def build_run_card(output_dir, cfg: dict | None = None, defaults: dict | None = None,
                   input_paths=None, input_hash_mode: str = "head",
                   repo_dir=None, extra: dict | None = None,
                   reconstructed: bool = False) -> dict:
    """Assemble the card. ``cfg`` defaults to the cfg stored in the run's own
    bsoid_run_summary.json; ``defaults`` is BSoidEngine.DEFAULTS (when absent,
    the diff / unknown-key sections are marked unavailable).

    ``reconstructed=True`` marks a card written AFTER the run (the command-line
    entry point): the environment / code sections then describe the current
    installation, not the one that produced the results, and the card says so
    (banner in run_card.md, no software versions in the methods paragraph)."""
    repo = Path(repo_dir) if repo_dir else Path(__file__).resolve().parent
    outputs = read_run_outputs(output_dir)
    cfg_used = cfg if cfg is not None else (outputs.pop("summary_cfg", None) or {})
    outputs.pop("summary_cfg", None)
    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reconstructed": bool(reconstructed),
        "reconstruction_note": RECONSTRUCTION_NOTE if reconstructed else None,
        "output_dir": str(output_dir),
        "environment": collect_environment(),
        "code": {
            "git": read_git_state(repo),
            "source_sha256": hash_source_files(repo),
            "pyproject_version": _pyproject_version(repo),
            "cube_version_constant_in_summary": outputs.get("cube_version_constant"),
            "note": ("cube_core.VERSION is a legacy constant and is not the "
                     "release version; identify the code by commit + source_sha256."),
        },
        "config": {
            "sha256": canonical_cfg_hash(cfg_used),
            "n_keys": len(cfg_used),
            "differs_from_defaults": (diff_from_defaults(cfg_used, defaults)
                                      if defaults else "unavailable (defaults not supplied)"),
            "unknown_keys": (unknown_keys(cfg_used, defaults)
                             if defaults else "unavailable (defaults not supplied)"),
        },
        "seeds": {
            **seeds_from_cfg(cfg_used),
            "hardcoded_in_cube_core": scan_hardcoded_seeds(repo / "cube_core.py"),
        },
        "inputs": hash_inputs(list(input_paths or []), input_hash_mode),
        "results": outputs,
    }
    if extra:
        card["extra"] = _normalize(extra)
    return card


def _pyproject_version(repo: Path):
    try:
        for line in (repo / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────
#  Human-readable output
# ──────────────────────────────────────────────────────────────────────────

def _fmt(v, nd=3):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return "n/a" if v is None else str(v)


def render_methods_paragraph(card: dict, cfg: dict | None = None) -> str:
    """A DRAFT methods paragraph built only from values the card holds. It is
    deliberately plain and conservative: the author must read it before use."""
    res = card.get("results", {}) or {}
    env = (card.get("environment", {}) or {}).get("packages", {}) or {}
    seeds = card.get("seeds", {}) or {}
    cfg = cfg or {}
    diff = (card.get("config", {}) or {}).get("differs_from_defaults")
    fv = (res.get("feature_config") or {}).get("feature_version")
    dim = "three-dimensional (triangulated) " if fv == "v3d" else ""
    recon = bool(card.get("reconstructed"))

    def _ver(name):          # versions only when they were recorded WITH the run
        v = env.get(name)
        return f" ({name} {v})" if (v and not recon) else ""

    parts = []
    parts.append(
        f"Behavioural segmentation used CUBE on {_fmt(res.get('n_sessions'))} "
        f"recording session(s) of {dim}pose data at {_fmt(res.get('fps'))} fps.")
    if (res.get("feature_config") or {}).get("n_features") is not None:
        parts.append(
            f"Per-bin kinematic features ({res['feature_config']['n_features']} dimensions) "
            f"were standardised and embedded with UMAP{_ver('umap-learn')} "
            f"(random state {_fmt(seeds.get('umap_random_state'))}), clustered with HDBSCAN"
            f"{_ver('hdbscan')}, and extended to all bins with a trained "
            f"classifier followed by hidden-Markov-model smoothing{_ver('hmmlearn')}.")
    if seeds.get("seed_sweep_n"):
        stab = res.get("cluster_stability") or {}
        ari = stab.get("mean_ari") if isinstance(stab, dict) else None
        parts.append(
            f"Partition stability was assessed over {seeds['seed_sweep_n']} embedding seeds "
            f"(mean pairwise adjusted Rand index {_fmt(ari)}).")
    if res.get("n_clusters") is not None:
        parts.append(f"The final partition contained {res['n_clusters']} clusters"
                     + (f"; classifier cross-validated accuracy was {_fmt(res.get('cv_accuracy'))}."
                        if res.get("cv_accuracy") is not None else "."))
    if isinstance(diff, dict):
        what = ("the full configuration (SHA-256 "
                f"{card['config']['sha256'][:12]}...) is" if recon else
                "the full configuration (SHA-256 "
                f"{card['config']['sha256'][:12]}...) and software versions are")
        parts.append(f"{len(diff)} setting(s) differed from the software defaults; "
                     f"{what} recorded in the accompanying run card.")
    git = (card.get("code", {}) or {}).get("git", {}) or {}
    if git.get("commit") and not recon:
        parts.append(f"Software version: git commit {git['commit'][:10]}"
                     f" (branch {git.get('branch')}); source hashes are in the run card.")
    elif recon:
        parts.append("Software versions were not recorded when this run was made.")
    return " ".join(parts)


def render_markdown(card: dict) -> str:
    env = card.get("environment", {}) or {}
    code = card.get("code", {}) or {}
    conf = card.get("config", {}) or {}
    seeds = card.get("seeds", {}) or {}
    inp = card.get("inputs", {}) or {}
    res = card.get("results", {}) or {}
    L = ["# CUBE run card", "",
         f"*Generated {card.get('created_utc')} (schema {card.get('schema_version')}).*", ""]
    if card.get("reconstructed"):
        L += [f"> **{card.get('reconstruction_note')}**", ""]
    L += ["## Draft methods paragraph", "",
         "> DRAFT built from recorded values only. Read and edit before using it.", "",
         render_methods_paragraph(card), "",
         "## Code", "",
         f"- git commit: `{(code.get('git') or {}).get('commit')}` "
         f"(branch `{(code.get('git') or {}).get('branch')}`)",
         f"- pyproject version: {code.get('pyproject_version')}",
         f"- {code.get('note')}",
         f"- source modules hashed: {len(code.get('source_sha256') or {})}", "",
         "## Environment", "",
         f"- Python {env.get('python')} on {env.get('platform')}, "
         f"{env.get('cpu_count')} logical CPUs, {env.get('ram_total_gb')} GB RAM",
         "- Packages: " + ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items()), "",
         "## Configuration", "",
         f"- cfg SHA-256: `{conf.get('sha256')}` ({conf.get('n_keys')} keys)"]
    unk = conf.get("unknown_keys")
    if isinstance(unk, list):
        L.append(f"- keys not in DEFAULTS (a misspelled name is ignored; a few optional "
                 f"settings are read with a built-in fallback and also appear here): "
                 f"{unk if unk else 'none'}")
    diff = conf.get("differs_from_defaults")
    if isinstance(diff, dict):
        L += ["", f"### Settings that differ from DEFAULTS ({len(diff)})", ""]
        L += [f"- `{k}` = `{canonical_json(v['value'])}` (default `{canonical_json(v['default'])}`)"
              for k, v in diff.items()] or ["- none"]
    L += ["", "## Seeds", "",
          f"- UMAP random state: {seeds.get('umap_random_state')}; "
          f"HMM random state: {seeds.get('hmm_random_state')}",
          f"- seed sweep: n={seeds.get('seed_sweep_n')}, seeds={seeds.get('seed_sweep_seeds')}",
          f"- hard-coded RNG seeds in cube_core.py (not controlled by cfg): "
          f"{sum(v for v in (seeds.get('hardcoded_in_cube_core') or {}).values() if isinstance(v, int))}"
          " occurrences", "",
          "## Inputs", "",
          f"- {inp.get('n_files')} file(s); hash mode `{inp.get('mode')}`; "
          f"combined SHA-256 `{inp.get('combined_sha256')}`", "",
          "## Results recorded by the run", "",
          f"- sessions: {res.get('n_sessions')}; fps: {res.get('fps')}; "
          f"clusters: {res.get('n_clusters')}; CV accuracy: {_fmt(res.get('cv_accuracy'))}",
          f"- validation status: {res.get('validation_status')}",
          f"- benchmark: `{canonical_json(res.get('benchmark'))}`"]
    stab = res.get("cluster_stability")
    if isinstance(stab, dict):
        L.append(f"- cluster stability: mean ARI {_fmt(stab.get('mean_ari'))}; "
                 f"cluster counts {stab.get('cluster_counts')}")
    L.append(f"- total phase time: {_fmt(res.get('phase_timings_total_s'), 1)} s")
    return "\n".join(L) + "\n"


def write_run_card(output_dir, cfg: dict | None = None, defaults: dict | None = None,
                   input_paths=None, input_hash_mode: str = "head",
                   repo_dir=None, extra: dict | None = None,
                   reconstructed: bool = False):
    """Build and write ``run_card.json`` + ``run_card.md`` into ``output_dir``.
    Returns ``(json_path, md_path)``. Raises only on a failed write."""
    out = Path(output_dir)
    card = build_run_card(out, cfg=cfg, defaults=defaults, input_paths=input_paths,
                          input_hash_mode=input_hash_mode, repo_dir=repo_dir, extra=extra,
                          reconstructed=reconstructed)
    jp, mp = out / CARD_JSON, out / CARD_MD
    for path, text in ((jp, json.dumps(card, indent=2, default=_json_default)),
                       (mp, render_markdown(card))):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    return jp, mp


# ──────────────────────────────────────────────────────────────────────────
#  Command line (works on old runs)
# ──────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write run_card.json/.md for a CUBE run folder.")
    ap.add_argument("output_dir", help="the 'CUBE 3 - Clusters' folder (holds bsoid_run_summary.json)")
    ap.add_argument("--hash", choices=("head", "full", "none"), default="head",
                    help="input-file hashing mode (default: head)")
    ap.add_argument("--inputs", nargs="*", default=[],
                    help="pose files / folders to hash (folders: *.h5 and *.csv inside)")
    args = ap.parse_args(argv)
    paths = []
    for item in args.inputs:
        p = Path(item)
        paths += (sorted(list(p.glob("*.h5")) + list(p.glob("*.csv"))) if p.is_dir() else [p])
    defaults = None
    try:                                           # lazy: heavy import, CLI only
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import cube_core
        defaults = dict(cube_core.BSoidEngine.DEFAULTS)
    except Exception as e:
        print(f"[warn] DEFAULTS unavailable ({e}); diff/unknown-key sections skipped")
    # A card written from the command line is, by definition, written after the
    # run: it must not pass the current installation off as the run's own.
    jp, mp = write_run_card(args.output_dir, defaults=defaults, input_paths=paths,
                            input_hash_mode=args.hash, reconstructed=True)
    print(f"wrote {jp}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
