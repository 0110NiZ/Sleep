"""Compatibility entry; implementation lives in run_e2_evaluation_resumable."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("run_e2_evaluation_resumable", run_name="__main__")
else:
    # Alias the module itself: private helpers and mutable globals stay shared.
    sys.modules[__name__] = importlib.import_module("run_e2_evaluation_resumable")
