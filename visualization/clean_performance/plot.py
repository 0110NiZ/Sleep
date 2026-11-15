"""Draw the clean figure from archived results; no model execution."""
import runpy, sys
from pathlib import Path
if __name__ == '__main__':
    sys.argv[1:1] = ['--figure', 'clean']
    if '--out' not in sys.argv:
        sys.argv += ['--out', str(Path(__file__).resolve().parents[2] / 'local/runs/clean_performance_figures')]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'comparison_plots.py'), run_name='__main__')
