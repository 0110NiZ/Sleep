"""E4: fixed candidates, validation selection, then frozen-classifier test evaluation."""
import argparse, csv, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pywt
import torch
import run_e2_evaluation as e2
from noise_conditions import apply_noise, apply_noise_v2
from wavelet_denoising import denoise
from project_paths import E2_DIR, ROOT, BASELINE_DIR, RESULT_ROOT, baseline_fold


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def model_for(fold):
    folder=baseline_fold(fold)
    cfg=load(folder/'config.json'); assert cfg['fold']==fold
    state=torch.load(folder/'best_validation_macro_f1.pth',weights_only=True,map_location='cpu')
    model=e2.AttnSleep().eval();model.load_state_dict(state['state_dict'],strict=True)
    return model,int(cfg['batch_size'])


def evaluate(model, batch_size, x, y, ids, condition, candidate, config, noise):
    preds=[];before=after=0.;duration=0.
    for start in range(0,len(y),batch_size*4):
        clean=x[start:start+batch_size*4]
        noisy=apply_noise(clean,condition,sample_ids=ids[start:start+len(clean)],seed=noise['seed'],fs=noise['sampling_rate'])
        tick=time.perf_counter()
        filtered=denoise(noisy,candidate,config)
        preds.append(e2.predict(model,filtered,batch_size))
        duration+=time.perf_counter()-tick
        before+=float(np.square(noisy.astype('float64')-clean).sum())
        after+=float(np.square(filtered.astype('float64')-clean).sum())
    pred=np.concatenate(preds)
    score=e2.calculate_metrics(y,pred)
    score.update(rmse_noisy_uv=float(np.sqrt(before/x.size)),rmse_denoised_uv=float(np.sqrt(after/x.size)),
                 snr_improvement_db=None if condition['noise_type']=='clean' else float(10*np.log10(max(before,1e-30)/max(after,1e-30))),
                 denoise_and_inference_seconds=duration,denoise_and_inference_ms_per_epoch=duration*1000/len(y))
    return score,pred


def run(stage,out,config,noise,splits,folds=range(5)):
    selections=load(out/'selection_frozen.json') if stage=='test' else None
    for fold in folds:
        candidates=config['candidates'] if stage=='validation' else [next(c for c in config['candidates'] if c['id']==selections['folds'][str(fold)]['candidate'])]
        jobs=[(c,n) for c in candidates for n in noise['conditions'] if not (out/stage/f"fold{fold}_{c['id']}_{n['name']}.json").exists()]
        if not jobs:continue
        x,y,ids=e2.load_test(splits['folds'][fold][stage])
        model,batch=model_for(fold)
        # Untimed warm-up on this stage's own data.
        e2.predict(model,x[:batch],batch)
        for candidate,condition in jobs:
            path=out/stage/f"fold{fold}_{candidate['id']}_{condition['name']}.json"
            tick=time.perf_counter()
            score,pred=evaluate(model,batch,x,y,ids,condition,candidate,config,noise)
            if stage=='test':
                pp=out/'predictions'/f"fold{fold}_{condition['name']}.npz";pp.parent.mkdir(exist_ok=True)
                np.savez_compressed(pp,y_true=y,y_pred=pred,sample_ids=np.asarray(ids),candidate=candidate['id'])
            save(path,dict(fold=fold,candidate=candidate['id'],condition=condition['name'],**score))
            print(stage,fold,candidate['id'],condition['name'],'Macro-F1',round(score['macro_f1']*100,4),'seconds',round(time.perf_counter()-tick,1),flush=True)
        del x,y,model


def freeze(out,config,noise):
    scores={};rows=[]
    for fold in range(5):
        candidate_scores={}
        for c in config['candidates']:
            values=[load(out/'validation'/f"fold{fold}_{c['id']}_{n['name']}.json")['macro_f1'] for n in noise['conditions']]
            candidate_scores[c['id']]=float(np.mean(values))
            rows.append(dict(fold=fold,candidate=c['id'],validation_mean_macro_f1_pct=100*np.mean(values)))
        best=max(candidate_scores,key=candidate_scores.get)
        scores[str(fold)]=dict(candidate=best,scores=candidate_scores)
    path=out/'selection_frozen.json'
    if path.exists(): assert load(path)['folds']==scores,'Existing selection differs'
    else:save(path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),rule=config['selection'],folds=scores))
    table(out/'validation_candidate_scores.csv',rows)
    print('SELECTION FROZEN', {f:v['candidate'] for f,v in scores.items()},flush=True)


def summarize(out,noise):
    selections=load(out/'selection_frozen.json')['folds'];rows=[];summary=[];stages=[];pooled=[];cms={}
    with (E2_DIR/'e2_per_fold.csv').open(encoding='utf-8-sig') as f:
        baseline={(int(r['fold']),r['condition']):r for r in csv.DictReader(f)}
    for condition in noise['conditions']:
        name=condition['name'];foldscores=[];ys=[];ps=[]
        for fold in range(5):
            c=selections[str(fold)]['candidate'];score=load(out/'test'/f'fold{fold}_{c}_{name}.json');foldscores.append(score)
            with np.load(out/'predictions'/f'fold{fold}_{name}.npz') as z:ys.append(z['y_true']);ps.append(z['y_pred'])
            rows.append(dict(fold=fold,candidate=c,condition=name,accuracy_pct=score['accuracy']*100,macro_f1_pct=score['macro_f1']*100,kappa=score['kappa'],delta_macro_f1_vs_e2_pp=score['macro_f1']*100-float(baseline[fold,name]['macro_f1_pct']),rmse_denoised_uv=score['rmse_denoised_uv'],snr_improvement_db=score['snr_improvement_db'],denoise_and_inference_ms_per_epoch=score['denoise_and_inference_ms_per_epoch']))
        line={'condition':name}
        for key in ['accuracy','macro_f1','kappa','rmse_denoised_uv','denoise_and_inference_ms_per_epoch']:
            values=[s[key] for s in foldscores];line['mean_'+key]=float(np.mean(values));line['sd_'+key]=float(np.std(values,ddof=1))
        line['mean_snr_improvement_db']=None if name=='Clean' else float(np.mean([s['snr_improvement_db'] for s in foldscores]))
        summary.append(line)
        for stage in e2.LABELS:
            line={'condition':name,'stage':stage}
            for key in ['precision','recall','f1']:
                values=[s['stage_metrics'][stage][key] for s in foldscores];line['mean_'+key]=float(np.mean(values));line['sd_'+key]=float(np.std(values,ddof=1))
            stages.append(line)
        pscore=e2.calculate_metrics(np.concatenate(ys),np.concatenate(ps));pooled.append(dict(condition=name,accuracy=pscore['accuracy'],macro_f1=pscore['macro_f1'],kappa=pscore['kappa']));cms[name]=pscore
    table(out/'test_per_fold.csv',rows);table(out/'test_summary.csv',summary);table(out/'stage_summary.csv',stages);table(out/'pooled_summary.csv',pooled);save(out/'pooled_details.json',cms)
    fold_noisy=[np.mean([r['macro_f1_pct'] for r in rows if r['fold']==f and r['condition']!='Clean']) for f in range(5)]
    save(out/'overall.json',dict(selected={f:s['candidate'] for f,s in selections.items()},noisy_macro_f1_mean_pct=float(np.mean(fold_noisy)),noisy_macro_f1_sd_pp=float(np.std(fold_noisy,ddof=1)),clean_macro_f1_mean_pct=float(np.mean([r['macro_f1_pct'] for r in rows if r['condition']=='Clean']))))
    print('COMPLETE',load(out/'overall.json'),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['init','select','validation','test','summary','all'],default='all');ap.add_argument('--fold',choices=['all','0','1','2','3','4'],default='all');ap.add_argument('--output-dir',type=Path,default=RESULT_ROOT/'e4_wavelet');args=ap.parse_args()
    if args.fold!='all' and args.stage!='validation':ap.error('--fold is supported only for independent validation jobs')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    config=load(ROOT/'configs/e4_wavelet.json');noise=load(ROOT/'configs/noise_conditions_v2.json');splits=load(ROOT/'configs/fivefold_subject_splits.json')
    files=['configs/e4_wavelet.json','configs/noise_conditions_v2.json','configs/fivefold_subject_splits.json','scripts/noise_conditions.py','scripts/wavelet_denoising.py','scripts/run_e4_wavelet.py','scripts/run_e2_evaluation.py','AttnSleep/model/model.py']
    hashes={f:digest(ROOT/f) for f in files}
    hashes.update({f'checkpoint_fold{f}':digest(baseline_fold(f)/'best_validation_macro_f1.pth') for f in range(5)})
    datafiles=sorted({name for fold in splits['folds'] for name in fold['test']['files']})
    assert len(datafiles)==39
    hashes.update({f'data/{name}':digest(e2.DATA_DIR/name) for name in datafiles})
    info=out/'experiment_info.json'
    if info.exists():assert load(info)['hashes']==hashes,'Inputs or code changed: use a new output directory'
    else:save(info,dict(created_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,config=config,python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,pywavelets=pywt.__version__,torch=torch.__version__,threads=4,device='cpu',timing='untimed warm-up; includes denoising plus CPU classification, excludes noise generation and disk IO'))
    torch.set_num_threads(4)
    if args.stage in ['all','validation']:
        run('validation',out,config,noise,splits,range(5) if args.fold=='all' else [int(args.fold)])
        if args.fold=='all':freeze(out,config,noise)
    if args.stage=='select':freeze(out,config,noise)
    if args.stage in ['all','test']:run('test',out,config,noise,splits)
    if args.stage in ['all','summary']:summarize(out,noise)

if __name__=='__main__':main()
