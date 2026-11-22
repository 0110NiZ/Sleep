"""Compatibility entry; implementation lives in visualize_noise."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("visualize_noise", run_name="__main__")
else:
    # Alias the module itself: private helpers and mutable globals stay shared.
    sys.modules[__name__] = importlib.import_module("visualize_noise")
