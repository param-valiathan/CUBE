"""
Keep CUBE off CPU cores known to be faulty.

Why (2026-11-19, this workstation, Intel i9-14900K): WHEA-Logger event 19
("Processor Core, Corrected Machine Check, Internal parity error") was logged
repeatedly for ONE core -- x2APIC IDs 32/33 = logical processors 8 and 9 =
P-core 4 (seen 2026-09-01, 10-05, 10-20, 10-30).  Crashes that followed --
bugchecks 0x20001 / 0x3B / 0x1E, random access violations and illegal-
instruction faults in python, numpy, matplotlib's Agg module (always the same
offset), Bonsai, SearchIndexer, and hard freezes with no bugcheck -- are
consistent with threads landing on that core.  Two freezes happened right
after code was executed ON logical processors 8/9 (a render pinned to LP 8-11,
then a CPUID probe that ran a thread on every logical processor).

THIS MODULE NEVER EXECUTES CODE ON THE FAULTY CORE.  It only removes the
configured logical processors from the affinity mask of the current process;
child processes (multiprocessing spawn, joblib/loky, ffmpeg, DLC) inherit it.
It cannot stop the OS, drivers or the hypervisor from using the core, and it
is no substitute for replacing the CPU -- it lowers the odds that a CUBE run
hits it.

The logical-processor numbers were mapped to APIC IDs ONCE (2026-11-19, CPUID
on each processor): LP n and n+1 on this CPU share a physical P-core; APIC IDs
32/33 are LP 8/9.  Do not re-probe on this machine (see above).  If the BIOS
or CPU changes, re-derive the list deliberately (python cube_cpu_guard.py
--probe-warning) and edit bad_cpu_cores.json.

Configuration (first that applies):
  * env CUBE_CPU_GUARD=0                      -> do nothing
  * env CUBE_AVOID_LOGICAL_PROCESSORS="8,9"   -> avoid these logical processors
  * file bad_cpu_cores.json next to this module:
        {"avoid_logical_processors": [8, 9]}
  * default: (8, 9)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_AVOID = (8, 9)            # P-core 4 on the i9-14900K workstation (see above)
_info: dict = {"applied": False, "reason": "not run"}


def _configured_avoid() -> tuple:
    env = os.environ.get("CUBE_AVOID_LOGICAL_PROCESSORS")
    if env is not None:
        return tuple(int(x) for x in env.replace(";", ",").split(",") if x.strip())
    f = Path(__file__).with_name("bad_cpu_cores.json")
    if f.exists():
        try:
            return tuple(int(x) for x in json.loads(f.read_text()).get("avoid_logical_processors", []))
        except Exception:
            return ()
    return _DEFAULT_AVOID


def apply() -> dict:
    """Remove the faulty core's logical processors from this process's affinity.
    Never raises, never runs code on the avoided processors; returns (and
    caches) an info dict."""
    global _info
    if os.environ.get("CUBE_CPU_GUARD", "1") == "0":
        _info = {"applied": False, "reason": "disabled (CUBE_CPU_GUARD=0)"}
        return _info
    try:
        import psutil
        avoid = _configured_avoid()
        if not avoid:
            _info = {"applied": False, "reason": "no cores configured"}
            return _info
        p = psutil.Process()
        cur = p.cpu_affinity()
        bad = [lp for lp in avoid if lp in cur]
        keep = [lp for lp in cur if lp not in avoid]
        if not bad:
            _info = {"applied": False, "reason": f"logical processor(s) {list(avoid)} not in affinity"}
            return _info
        if len(keep) < 2:
            _info = {"applied": False, "reason": "would leave < 2 logical processors"}
            return _info
        p.cpu_affinity(keep)
        _info = {"applied": True, "avoided_logical_processors": bad, "usable": len(keep)}
    except Exception as e:                                 # a guard must never break the program
        _info = {"applied": False, "reason": f"error: {e}"}
    return _info


def usable_cpu_count() -> int:
    """Logical processors this process may run on (respects the guard)."""
    try:
        import psutil
        return len(psutil.Process().cpu_affinity()) or (os.cpu_count() or 1)
    except Exception:
        return os.cpu_count() or 1


def info() -> dict:
    return dict(_info)


if __name__ == "__main__":
    print(json.dumps(apply(), indent=2))
