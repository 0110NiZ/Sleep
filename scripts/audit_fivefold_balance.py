from pathlib import Path
import json,csv,hashlib
import numpy as np,mne
R=Path(__file__).resolve().parents[1];p=R/'configs/fivefold_subject_splits.json';before=hashlib.sha256(p.read_bytes()).hexdigest();split=json.loads(p.read_text());D=Path(split['data_directory'])
meta={}
src=Path(mne.__file__).parent/'datasets/sleep_physionet/age_records.csv'
for r in csv.DictReader(src.open()):
 if int(r['subject'])<20:
  key=f"{int(r['subject']):02d}";v=dict(age=int(r['age']),sex=r['sex'])
  assert key not in meta or meta[key]==v
  meta[key]=v
subjects={}
for f in sorted(D.glob('*.npz')):
 with np.load(f,allow_pickle=False) as z:y=z['y'];counts=np.bincount(y,minlength=5)
 sid=f.name[3:5];s=subjects.setdefault(sid,dict(subject=sid,**meta[sid],nights=0,epochs=0,counts=np.zeros(5,dtype=int)))
 s['nights']+=1;s['epochs']+=len(y);s['counts']+=counts
def agg(ids):
 ss=[subjects[i] for i in ids];c=sum((s['counts'] for s in ss),np.zeros(5,dtype=int));n=int(c.sum())
 return dict(subjects=ids,nights=sum(s['nights'] for s in ss),epochs=n,hours=n/120,counts=c.tolist(),percent=(c/n*100).tolist(),age_mean=float(np.mean([s['age'] for s in ss])),age_min=min(s['age'] for s in ss),age_max=max(s['age'] for s in ss),female=sum(s['sex']=='female' for s in ss),male=sum(s['sex']=='male' for s in ss),subject_mean_percent=np.mean([s['counts']/s['epochs']*100 for s in ss],axis=0).tolist())
overall=agg(sorted(subjects));groups={chr(65+k):agg(g) for k,g in enumerate(split['groups'])}
folds=[]
for f in split['folds']:
 a={name:agg(f[name]['subjects']) for name in ['train','validation','test']}
 ids=[set(a[k]['subjects']) for k in a];assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
 assert set.union(*ids)==set(subjects)
 folds.append(dict(fold=f['fold']+1,**a,train_test_max_percentage_point_gap=float(np.max(np.abs(np.array(a['train']['percent'])-a['test']['percent'])))))
for s in subjects.values():s['counts']=s['counts'].tolist();s['percent']=(np.array(s['counts'])/s['epochs']*100).tolist()
report=dict(scope='Descriptive data-only audit; no model results used, no split changes and no retraining.',splits_sha256=before,demographics_source=str(src),class_order=['W','N1','N2','N3','REM'],overall=overall,groups=groups,folds=folds,subjects=list(subjects.values()))
(R/'reports/fivefold_balance_audit.json').write_text(json.dumps(report,indent=2))
with (R/'reports/fivefold_subject_distribution.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['subject','group','age','sex','nights','epochs','W','N1','N2','N3','REM'])
 for sid,s in subjects.items():w.writerow([sid,next(k for k,g in groups.items() if sid in g['subjects']),s['age'],s['sex'],s['nights'],s['epochs'],*s['counts']])
assert hashlib.sha256(p.read_bytes()).hexdigest()==before
print(json.dumps(dict(overall=overall,groups=groups,folds_summary=[dict(fold=f['fold'],train_n=f['train']['epochs'],val_n=f['validation']['epochs'],test_n=f['test']['epochs'],train_test_gap=f['train_test_max_percentage_point_gap']) for f in folds],subjects=report['subjects']),indent=2))
