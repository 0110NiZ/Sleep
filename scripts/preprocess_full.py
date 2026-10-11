from pathlib import Path
import subprocess,json,os,time,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]; S=R/'reports/preprocess_full_status.json'
def status(stage,**extra):
 d=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra)
 t=S.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(S)
try:
 status('verifying_raw')
 manifest=json.loads((R/'reports/sleep_edf20_manifest.json').read_text())['files']
 assert len(manifest)==78
 for i in manifest:
  p=R/'data/raw/sleep-edf-20'/i['name']
  assert p.stat().st_size==i['bytes']
  h=hashlib.sha1()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  assert h.hexdigest()==i['sha1'],i['name']
 out=R/'data/processed/sleep-edf-20'
 assert not out.exists(),'Output exists; inspect before retrying'
 status('preprocessing',verified_raw_files=78)
 subprocess.run([str(R/'.venv/bin/python'),'-u',str(R/'AttnSleep/prepare_datasets/prepare_physionet.py'),'--data_dir',str(R/'data/raw/sleep-edf-20'),'--output_dir',str(out),'--select_ch','EEG Fpz-Cz'],cwd=R/'AttnSleep',check=True)
 status('validating')
 expected={i['name'].replace('-PSG.edf','.npz') for i in manifest if i['name'].endswith('-PSG.edf')}
 assert {p.name for p in out.glob('*.npz')}==expected
 records=[]
 for p in sorted(out.glob('*.npz')):
  with np.load(p) as z:
   x,y=z['x'],z['y']; assert x.shape==(len(y),3000,1)
   assert float(z['fs'])==100 and str(z['ch_label'])=='EEG Fpz-Cz'
   assert np.isfinite(x).all() and np.isin(y,range(5)).all()
   records.append(dict(record=p.stem,subject=p.name[3:5],epochs=len(y),class_counts=np.bincount(y,minlength=5).tolist()))
 assert len(records)==39 and len({r['subject'] for r in records})==20
 (R/'reports/preprocess_full_summary.json').write_text(json.dumps(records,indent=2))
 status('completed',records=39,subjects=20,total_epochs=sum(r['epochs'] for r in records),output=str(out),note='Shape, label, finite-value and metadata checks passed; independent full alignment audit pending before training.')
except Exception as e:
 status('failed',error=repr(e));raise
