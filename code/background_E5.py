"""Run verification, training and testing sequentially, recording failures."""

# Locate shared modules from either the new or legacy command path.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[1]
_layout_sys.path.insert(0, str(_layout_root / 'scripts'))
_layout_sys.path.insert(0, str(_layout_root / 'code'))
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from project_paths import ROOT, RESULT_ROOT
root=ROOT
out=RESULT_ROOT/'e5_dae';out.mkdir(parents=True,exist_ok=True)
os.chdir(root)
os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
try:
    for stage in ['verify','train','test']:
        (out/'pipeline_status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),started=time.time())))
        print('START STAGE',stage,flush=True)
        subprocess.run([sys.executable,'scripts/run_e5_dae.py','--stage',stage],check=True)
    (out/'pipeline_status.json').write_text(json.dumps(dict(stage='complete',finished=time.time())))
except Exception:
    (out/'pipeline_status.json').write_text(json.dumps(dict(stage='failed',error=traceback.format_exc())))
    raise
