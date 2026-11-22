"""Plot a genuine held-out Sleep-EDF epoch with the official Noise v2 generator.

No synthetic base-waveform fallback: missing study data raises an error.
Use make_figures.py with --data-dir pointing to the processed Sleep-EDF records.
"""
import _bootstrap
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plot_style import *


def main():
    project = ROOT.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=project/'data/processed/sleep-edf-20')
    p.add_argument('--project-dir', type=Path, default=project)
    p.add_argument('--splits', type=Path, default=project/'configs/fivefold_subject_splits.json')
    p.add_argument('--file', default='SC4021E0.npz')
    p.add_argument('--epoch-index', type=int, default=100)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    split = next(f for f in json.loads(args.splits.read_text())['folds'] if f['fold']==args.fold)
    assert args.file in split['test']['files'], 'Selected recording is not in the specified test fold'
    assert args.file not in split['train']['files'] + split['validation']['files']
    source = args.data_dir/args.file
    with np.load(source, allow_pickle=False) as z:
        clean = np.asarray(z['x'][args.epoch_index], dtype=np.float32).reshape(-1).copy()
        stage = ['W','N1','N2','N3','REM'][int(z['y'][args.epoch_index])]
        if 'fs' in z: assert float(z['fs'])==100
    assert clean.shape==(3000,) and np.isfinite(clean).all()
    cfg_path = args.project_dir/'configs/noise_conditions_v2.json'
    cfg = json.loads(cfg_path.read_text())
    sys.path.insert(0,str(args.project_dir/'experiments'))
    from noise_conditions import apply_noise_v2
    sample_id = f'{args.file}:{args.epoch_index}'
    conditions={c['name']:c for c in cfg['conditions']}
    arrays={'Clean':clean}
    for name in ['Gaussian_High','Drift_High','Burst_High']:
        arrays[name]=apply_noise_v2(clean.reshape(1,1,-1),conditions[name],sample_ids=[sample_id],seed=args.seed,fs=100)[0,0]
        assert np.isfinite(arrays[name]).all()
    names=['Clean reference (no added synthetic noise)','Gaussian white noise | High: 5 dB',
           'Baseline drift | High: 0.5 Hz, 40 μV',
           'Oscillatory burst | High: 7σ; sampled within 15–35 Hz and 0.5–2 s']
    fig,axes=plt.subplots(4,1,figsize=(16,6.8),sharex=True)
    t=np.arange(3000)/100
    for ax,(key,sig),label,color in zip(axes,arrays.items(),names,[BLUE,YELLOW,MINT,ROSE]):
        ax.plot(t,sig,color=color,linewidth=.8)
        ax.set_ylabel('μV',fontsize=10)
        ax.set_title(label,loc='left',fontsize=11,pad=3)
        ax.set_xlim(0,30)
        ax.grid(axis='x',alpha=.4)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'Real held-out EEG with three noise types | High-severity examples',fontsize=17,fontweight='bold',y=.995)
    fig.text(.5,.012,f'Sleep-EDF-20 · {sample_id} (zero-based epoch) · Stage {stage} · Test fold {args.fold} · Fpz–Cz · 100 Hz · Seed {args.seed}',ha='center',fontsize=10,color=SUBTLE)
    fig.tight_layout(rect=(0,.045,1,.955),h_pad=.65)
    out=save(fig,'02_eeg_waveforms')
    meta={'source_file':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'sample_id':sample_id,'epoch_index_zero_based':args.epoch_index,'stage':stage,'test_fold':args.fold,
          'splits_sha256':hashlib.sha256(args.splits.read_bytes()).hexdigest(),'config_file':'configs/noise_conditions_v2.json','seed':args.seed,
          'generator_file':'experiments/noise_conditions.py',
          'unit':'μV','sampling_rate':100,'conditions':[conditions[k] for k in arrays],
          'note':'Actual preprocessed study epoch. All three noisy traces reuse this same epoch. No model inference or retraining.'}
    (OUTPUT/'02_eeg_waveforms.provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    print(f'SAVED {out}\nSample: {sample_id}, stage {stage}, test fold {args.fold}')

if __name__=='__main__': main()
