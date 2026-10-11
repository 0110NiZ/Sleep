from pathlib import Path
import numpy as np
import mne,json,sys
root=Path('/home/nzd/Sleep/data')
mapping={'Sleep stage W':0,'Sleep stage 1':1,'Sleep stage 2':2,'Sleep stage 3':3,'Sleep stage 4':3,'Sleep stage R':4}
reports=[]
early='--early' in sys.argv
for name in [p.stem for p in sorted((root/'processed/sleep-edf-20').glob('*.npz'))]:
    raw=mne.io.read_raw_edf(root/'raw/sleep-edf-20'/f'{name}-PSG.edf',preload=False,verbose=False)
    annfile=next((root/'raw/sleep-edf-20').glob(name[:6]+'*-Hypnogram.edf'))
    ann=mne.read_annotations(annfile)
    data=raw.get_data(picks=['EEG Fpz-Cz'],units='uV')[0]
    samples=[]; labels=[]
    for onset,duration,desc in zip(ann.onset,ann.duration,ann.description):
        if desc not in mapping: continue
        assert duration%30==0
        for t in np.arange(onset,onset+duration,30):
            a=round(t*100); b=a+3000
            if b<=len(data): samples.append(data[a:b]); labels.append(mapping[desc])
    labels=np.array(labels); samples=np.array(samples,dtype=np.float32)
    nonwake=np.where(labels!=0)[0]; a=max(0,nonwake[0]-60); b=min(len(labels),nonwake[-1]+61)
    samples,labels=samples[a:b],labels[a:b]
    z=np.load(root/('processed/sleep-edf-20')/f'{name}.npz')
    actual=z['x'].reshape(-1,3000)
    np.testing.assert_array_equal(z['y'],labels)
    np.testing.assert_allclose(actual,samples,rtol=1e-5,atol=1e-4)
    print('ALIGNED',name,len(labels),flush=True)
    reports.append(dict(record=name,segments=len(labels),class_counts=np.bincount(labels,minlength=5).tolist(),
                        finite=bool(np.isfinite(actual).all()),units='uV',independent_mne_alignment_check='passed'))
(root.parent/'reports'/'full_alignment_validation.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
