"""P10: E6 from the user's own prediction NPZs. White background, pastel bars."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parent
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
NOISY = [f'{n}_{s}' for n in ['Gaussian','Drift','Burst'] for s in ['Low','Medium','High']]
CONDITIONS = ['Clean'] + NOISY
METHODS = ['E2','E5','E6']
PALETTE = {'E2':'#C7B7E2', 'E5':'#B7DDCA', 'E6':'#F1B8C5'}
LABELS = {'E2':'E2 Baseline', 'E5':'E5 DAE', 'E6':'E6 Augmentation'}
TEXT = '#384650'
MUTED = '#6D7780'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csvwrite(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def f1_from_cm(cm):
    assert cm.shape == (5,5) and (cm >= 0).all()
    den=cm.sum(0)+cm.sum(1)
    return np.divide(2.*cm.diagonal(),den,out=np.zeros(5),where=den!=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--e6-dir',type=Path,
        default=BASE.parent/'result/local/e6_noise_augmentation',
        help='Your original E6 output directory containing fold0..fold4 prediction NPZs')
    ap.add_argument('--project-root',type=Path,default=BASE.parent,
        help='Prefer E2/E5 formal test records in this project; use bundled group reference records if absent')
    ap.add_argument('--out',type=Path,default=BASE/'output')
    ap.add_argument('--no-labels',action='store_true',help='Hide numeric labels to match the original label-free chart')
    args=ap.parse_args()
    e6dir=args.e6_dir.resolve()
    if not e6dir.is_dir():
        raise FileNotFoundError(f'Your E6 directory does not exist: {e6dir}. '
            'Place p10 inside F:\\sleep\\Sleep, or supply --e6-dir.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    provenance=dict(e6_source='Actual y_true/y_pred from your 50 formal prediction NPZs; no PPT values.',
        e6_directory=str(e6dir),e6_prediction_sha256={},reference_sources={},checks={})
    f1vals={m:np.empty((5,9,5)) for m in METHODS}
    supports={};condition_rows=[];total_predictions=0
    # Read actual user predictions, including Clean for audit; Clean is excluded from the plotted mean.
    for fold in range(5):
        ids_ref=y_ref=None
        metric_path=e6dir/f'fold{fold}/test_metrics.json'
        recorded=read(metric_path) if metric_path.is_file() else None
        for condition in CONDITIONS:
            path=e6dir/f'fold{fold}/predictions_{condition}.npz'
            if not path.is_file():raise FileNotFoundError(f'Missing your E6 predictions: {path}')
            with np.load(path,allow_pickle=False) as z:
                y=np.asarray(z['y_true']);p=np.asarray(z['y_pred']);ids=np.asarray(z['sample_ids'])
            assert y.ndim==p.ndim==ids.ndim==1 and len(y)==len(p)==len(ids)>0
            assert np.issubdtype(y.dtype,np.integer) and np.issubdtype(p.dtype,np.integer)
            assert np.isin(y,range(5)).all() and np.isin(p,range(5)).all()
            assert len(set(ids.tolist()))==len(ids), 'Duplicate sample IDs'
            if ids_ref is None:ids_ref,y_ref=ids.copy(),y.copy()
            else:
                np.testing.assert_array_equal(ids,ids_ref)
                np.testing.assert_array_equal(y,y_ref)
            cm=np.zeros((5,5),dtype=np.int64)
            np.add.at(cm,(y,p),1)
            f1=f1_from_cm(cm)
            if recorded is not None:
                np.testing.assert_array_equal(cm,np.asarray(recorded[condition]['confusion_matrix']))
                np.testing.assert_allclose(f1,[recorded[condition]['stage_metrics'][s]['f1'] for s in STAGES],atol=1e-12,rtol=0)
                np.testing.assert_allclose(f1.mean(),recorded[condition]['macro_f1'],atol=1e-12,rtol=0)
            supports[fold,condition]=cm.sum(1)
            if condition in NOISY:f1vals['E6'][fold,NOISY.index(condition),:]=100*f1
            for s,v in zip(STAGES,100*f1):
                condition_rows.append(dict(method='E6',fold=fold,condition=condition,stage=s,f1_pct=float(v)))
            total_predictions+=len(y)
            provenance['e6_prediction_sha256'][path.relative_to(e6dir).as_posix()]=sha(path)
    assert len(provenance['e6_prediction_sha256'])==50
    assert total_predictions==423080, 'Unexpected record count for this formal experiment'
    # E2/E5 require their own formal outputs, never inferred from E6 or a PPT.
    e2local=args.project_root/'result/e2_v2/e2_details.json'
    e2path=e2local if e2local.is_file() else BASE/'inputs/E2/e2_details.json'
    e2=read(e2path)
    provenance['reference_sources']['E2']=dict(path=str(e2path.resolve()),sha256=sha(e2path),
        origin='local project' if e2path==e2local else 'bundled group formal test records')
    local_e5=args.project_root/'result/e5_dae'
    use_local=all((local_e5/f'fold{f}/test_metrics.json').is_file() for f in range(5))
    for method in ['E2','E5']:
        method_sources=[]
        for fold in range(5):
            if method=='E2':detail={c:e2[f'fold{fold}_{c}'] for c in NOISY}
            else:
                path=(local_e5 if use_local else BASE/'inputs/E5')/f'fold{fold}/test_metrics.json'
                detail=read(path);method_sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
            for j,c in enumerate(NOISY):
                m=detail[c];cm=np.asarray(m['confusion_matrix'],dtype=np.int64)
                f1=f1_from_cm(cm)
                np.testing.assert_array_equal(cm.sum(1),supports[fold,c])
                np.testing.assert_allclose(f1,[m['stage_metrics'][s]['f1'] for s in STAGES],atol=1e-12,rtol=0)
                np.testing.assert_allclose(f1.mean(),m['macro_f1'],atol=1e-12,rtol=0)
                f1vals[method][fold,j,:]=100*f1
                for s,v in zip(STAGES,100*f1):
                    condition_rows.append(dict(method=method,fold=fold,condition=c,stage=s,f1_pct=float(v)))
        if method=='E5':provenance['reference_sources']['E5']=dict(files=method_sources,
            origin='local project' if use_local else 'bundled group formal test records')
    means={m:f1vals[m].mean(axis=1).mean(axis=0) for m in METHODS}
    sd={m:f1vals[m].mean(axis=1).std(axis=0,ddof=1) for m in METHODS}
    rows=[]
    for i,s in enumerate(STAGES):
        r=dict(stage=s)
        for m in METHODS:r[m+'_f1_pct']=float(means[m][i]);r[m+'_fold_sd_pp']=float(sd[m][i])
        rows.append(r)
    csvwrite(out/'p10_stage_f1_values.csv',rows)
    csvwrite(out/'p10_stage_f1_per_condition.csv',condition_rows)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Calibri','Carlito','Arial','DejaVu Sans'] if f in fonts),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':12,'svg.fonttype':'none','axes.unicode_minus':False})
    fig=plt.figure(figsize=(10.8,4.8),facecolor='white')
    ax=fig.add_axes([0.068,0.205,0.92,0.755],facecolor='white')
    x=np.arange(5);width=0.235
    for j,m in enumerate(METHODS):
        bars=ax.bar(x+(j-1)*width,means[m],width,color=PALETTE[m],edgecolor='none',
                    label=LABELS[m],zorder=3)
        if not args.no_labels:ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=8.7,color=TEXT)
    ax.set_xlim(-.5,4.5);ax.set_ylim(0,100)
    ax.set_xticks(x,STAGES);ax.set_yticks(np.arange(0,101,20))
    ax.set_ylabel('F1 (%)',color=MUTED,fontsize=11,labelpad=9)
    ax.tick_params(axis='x',length=0,pad=11,colors=TEXT)
    ax.tick_params(axis='y',length=0,pad=10,colors=MUTED,labelsize=10)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E7EB',linewidth=.7)
    for k,s in ax.spines.items():
        s.set_visible(k in ['left','bottom']);s.set_color('#D5DBE0');s.set_linewidth(.7)
    legend=ax.legend(loc='upper center',bbox_to_anchor=(.5,-.115),ncol=3,frameon=False,
        fontsize=11,handlelength=.9,handleheight=.8,handletextpad=.5,columnspacing=1.4,borderaxespad=0)
    for t in legend.get_texts():t.set_color(TEXT)
    fig.savefig(out/'p10_stage_f1_white_macaron.png',dpi=300,facecolor='white')
    fig.savefig(out/'p10_stage_f1_white_macaron.svg',facecolor='white')
    plt.close(fig)
    provenance.update(aggregation='Per-stage F1 from confusion matrices. Equal average of nine noisy '
        'conditions in each fold, then equal average of five folds. Clean excluded from the graph.',
        e6_prediction_files=50,e6_prediction_records=total_predictions,
        mean_stage_f1_pct={m:means[m].tolist() for m in METHODS},
        noisy_macro_f1_pct={m:float(means[m].mean()) for m in METHODS},
        colors=PALETTE,background='#FFFFFF',font=font)
    provenance['checks']=dict(e6_predictions_match_saved_metrics=all((e6dir/f'fold{f}/test_metrics.json').exists() for f in range(5)),
                              same_ids_and_labels_across_conditions=True,same_stage_support_across_methods=True)
    (out/'p10_data_check.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('PASS: recomputed your E6 predictions: 50 NPZs, 423080 records')
    for m in METHODS:print(m,' '.join(f'{s}={v:.2f}' for s,v in zip(STAGES,means[m])))
    print('Saved:',out/'p10_stage_f1_white_macaron.png')
    print('E6 N1 F1: %.2f%%' % means['E6'][1])


if __name__=='__main__':main()
