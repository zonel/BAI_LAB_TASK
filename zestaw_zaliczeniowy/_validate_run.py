"""Validate the BAI notebook by executing only the solved-example cells in order.
Captures per-cell status, timing, and full traceback for any failure."""
import os
import sys
import time
import traceback
from pathlib import Path

# Disable matplotlib interactive backend, suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"
import matplotlib
matplotlib.use("Agg")

import nbformat

NOTEBOOK = Path(__file__).parent / "BAI_Zestaw_Zaliczeniowy.ipynb"
nb = nbformat.read(NOTEBOOK, as_version=4)

SOLVED_CODE_INDICES = [1, 3, 7, 11, 15, 19, 23, 27]

# Inject load_dataset import that lives only in TODO cell 5 — needed by solved cells 7/15/19
INJECTED_SETUP = "from datasets import load_dataset\n"

# Shared namespace so cells share state in order (just like a notebook kernel)
ns: dict = {"__name__": "__main__"}

# Pre-inject the missing import
exec(INJECTED_SETUP, ns)
print("[setup] injected: from datasets import load_dataset")
print()

results = []
for idx in SOLVED_CODE_INDICES:
    cell = nb.cells[idx]
    src = cell.source
    first_line = next((l for l in src.split("\n") if l.strip()), "")[:90]
    print(f"=== Cell {idx} ===")
    print(f"    First line: {first_line}")
    sys.stdout.flush()

    # Suppress plt.show blocking and silence verbose output where possible
    # plt.show() under Agg backend is a no-op so we are safe.

    t0 = time.time()
    status = "PASS"
    err_text = ""
    try:
        exec(src, ns)
    except Exception as exc:
        status = "FAIL"
        err_text = traceback.format_exc()
    dt = time.time() - t0

    print(f"    -> {status} in {dt:.1f}s")
    if status == "FAIL":
        print("    Traceback:")
        for line in err_text.splitlines()[-15:]:
            print(f"      {line}")
    print()
    sys.stdout.flush()
    results.append((idx, status, dt, err_text))

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for idx, status, dt, _ in results:
    print(f"Cell {idx:3d}: {status}  ({dt:.1f}s)")
passed = sum(1 for _, s, _, _ in results if s == "PASS")
failed = sum(1 for _, s, _, _ in results if s == "FAIL")
print(f"\nTotal: {passed} passed, {failed} failed (out of {len(results)})")
