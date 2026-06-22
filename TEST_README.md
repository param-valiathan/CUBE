# CUBE Test Suite

## Requirements

All tests must run inside the **CUBE conda environment**.  
Always use the full Python path — `conda activate` does not work in Claude terminals.

```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" <test_script.py>
```

### Optional: psutil (memory / CPU / process diagnostics)

`psutil` is not installed by default. Install it once to enable system-level diagnostics:

```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" -m pip install psutil
```

Without `psutil` the test still runs and reports accuracy, speed, and crash tracebacks.
With `psutil` it also reports peak RSS, CPU %, open file handles, and child process counts.

---

## test_group_predictor.py — Group Predictor headless test

Tests the `GroupPredictorPanel` computation backend (from `cube_analyser.py`)
without launching the GUI. Two analyses run back-to-back in **one Python process**
so shared state, loky pool reuse, and cleanup paths are exercised as they would
be in a real session.

### Run

```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" D:\CUBE\test_group_predictor.py
```

### What it does

1. Imports `cube_analyser.py` via `importlib` (source file is never modified).
2. Loads all `*_bout_lengths_hmm.csv` files from `D:\cube_results_20260613_100431\bout_lengths\`.
3. Randomly assigns animals to **GroupA** / **GroupB** (seed = 42, reproducible).
4. Runs three models — Frequency, Total Duration, Transition — twice:
   - **Run 1**: `max_contributors = 1`, `n_permutations = 9`
   - **Run 2**: `max_contributors = 2`, `n_permutations = 9`
5. Prints live progress as each model starts and finishes.
6. Shuts down all loky worker processes cleanly.
7. Prints a diagnostic analysis: accuracy, speed, memory, crashes, and suggested improvements.

### Configuration

Edit the constants near the top of `test_group_predictor.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `DATA_DIR` | `D:\cube_results_20260613_100431\bout_lengths` | Folder containing `*_bout_lengths_hmm.csv` files |
| `ANALYSER` | `D:\CUBE\cube_analyser.py` | Path to the analyser module |
| `GROUP_SEED` | `42` | Random seed for group assignment (change to test different splits) |
| `RUNS` | `[{max_k:1, n_perm:9}, {max_k:2, n_perm:9}]` | List of run configs; add/remove as needed |

Increase `n_perm` (e.g. to `99` or `199`) for a more rigorous permutation test.
`max_k` sets the maximum number of features selected per model step.

### Interpreting output

| Column | Meaning |
|--------|---------|
| `LOO` | Leave-One-Out accuracy (fraction of animals correctly classified) |
| `BAL` | Balanced accuracy — accounts for class imbalance |
| `p` | Permutation test p-value; <0.05 = significant |
| `feat` | Number of features in the raw feature matrix (before SelectKBest / PCA) |
| `Xs` | Wall-clock time for that model |
| `peak+XMB` | Peak RSS increase during that model *(psutil required)* |
| `cpu=X%` | CPU usage of the Python process *(psutil required)* |

**Expected behaviour with random groups:** accuracy ~0.5, p-value ~0.5–1.0.
High accuracy with random groups indicates a bug (label leakage, overfitting).

### Crash tracebacks

Any crash prints `✗ ModelName: CRASH` immediately followed by the full Python
traceback. The diagnostic section at the end classifies the crash type and
suggests a fix. Review the traceback for the exact file and line number.

---

## Adding new tests

Place new test scripts in `D:\CUBE\` with the prefix `test_`.
Each script should:

- Import cube source files via `importlib.util.spec_from_file_location` (never modify source).
- Run with `"C:\Users\param\anaconda3\envs\CUBE\python.exe" test_<name>.py`.
- Not save any results to disk (print only).
- Clean up child processes at the end (see `get_reusable_executor().shutdown()` pattern
  in `test_group_predictor.py`).
- Document itself with a brief docstring at the top and an entry in this file.
