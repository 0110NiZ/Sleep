"""E5: train on training subjects, select on validation, then test frozen selections."""
import argparse,csv,hashlib,json,os,platform,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from dae_model import DAE
import run_e2_evaluation as e2
from noise_conditions import apply_noise, apply_noise_v2

ROOT=Path(__file__).resolve().parents[1]
from project_paths import BASELINE_DIR, RESULT_ROOT, configured, baseline_fold
BASE=configured('E5_BASELINE_ROOT', BASELINE_DIR)
OUT=RESULT_ROOT/'e5_dae'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def csvsave(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def baseline(fold,device):
    folder=baseline_fold(fold, BASE)
    cfg=load(folder/'config.json');assert cfg['fold']==fold
    state=torch.load(folder/'best_validation_macro_f1.pth',map_location='cpu',weights_only=True)
    m=e2.AttnSleep();m.load_state_dict(state['state_dict'],strict=True)
    m.eval().to(device);m.requires_grad_(False)
    return m,int(cfg['batch_size'])
def noisy(x,ids,c,seed):
    return apply_noise(x,c,sample_ids=ids,seed=int(seed),fs=100)
def read_group(split):return e2.load_test(split)
@torch.inference_mode()
def evaluate(dae,clf,x,y,ids,conditions,batch,device):
    dae.eval();clf.eval();scores={};predictions={}
    # Warm-up is outside timed inference; no updates in either model.
    clf(dae(torch.from_numpy(x[:min(batch,len(x))]).to(device)))
    for c in conditions:
        preds=[];before=after=seconds=0.
        for j in range(0,len(x),batch):
            clean=x[j:j+batch];z=noisy(clean,ids[j:j+batch],c,123)
            t=torch.from_numpy(z).to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();restored=dae(t);logits=clf(restored)
            if device.type=='cuda':torch.cuda.synchronize()
            seconds+=time.perf_counter()-start
            preds.append(logits.argmax(1).cpu().numpy());r=restored.cpu().numpy()
            if not np.isfinite(r).all():raise ValueError('Non-finite reconstruction')
            before+=np.square(z.astype('float64')-clean).sum()
            after+=np.square(r.astype('float64')-clean).sum()
        pr=np.concatenate(preds);m=e2.calculate_metrics(y,pr)
        m.update(rmse_denoised_uv=float(np.sqrt(after/x.size)),rmse_noisy_uv=float(np.sqrt(before/x.size)),
            snr_improvement_db=None if c['name']=='Clean' else float(10*np.log10(max(before,1e-30)/max(after,1e-30))),
            denoise_and_inference_ms_per_epoch=seconds*1000/len(x))
        scores[c['name']]=m;predictions[c['name']]=pr
    return scores,predictions

def audit(cfg,splits,conditions):
    recorded=load(ROOT/'result/e5_dae/manifest.json')['code']
    assert sha(ROOT/'configs/fivefold_subject_splits.json')==recorded['configs/fivefold_subject_splits.json'],'Baseline split content mismatch'
    reference=load(ROOT/'result/e4_wavelet/experiment_info.json')['hashes']
    for fold in splits['folds']:
        sets=[set(fold[k]['subjects']) for k in ('train','validation','test')]
        assert [len(s) for s in sets]==[12,4,4]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert len(set(s for f in splits['folds'] for s in f['test']['subjects']))==20
    weights={f'fold{i}':sha(baseline_fold(i, BASE)/'best_validation_macro_f1.pth') for i in range(5)}
    assert all(weights[f'fold{i}']==reference[f'checkpoint_fold{i}'] for i in range(5))
    report=load(ROOT/'reports/full_alignment_validation.json');hashes={};total=0
    for row in report:
        path=e2.DATA_DIR/(row['record']+'.npz');hashes[path.name]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            x,y=z['x'],z['y'];assert x.shape==(row['segments'],3000,1)
            assert np.isfinite(x).all();assert np.bincount(y,minlength=5).tolist()==row['class_counts']
            total+=len(y)
    assert total==42308
    source=['scripts/dae_model.py','scripts/run_e5_dae.py','scripts/noise_conditions.py','scripts/run_e2_evaluation.py',
        'AttnSleep/model/model.py','configs/e5_dae.json','configs/noise_conditions_v2.json','configs/fivefold_subject_splits.json']
    manifest=dict(code={p:sha(ROOT/p) for p in source},data=hashes,weights=weights,config=cfg,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        note='NPZ file hashes differ from regenerated E4 files; shape/label counts match. Full original E2 CPU prediction reproduction is required before final E5 test.')
    path=OUT/'manifest.json'
    if path.exists():assert load(path)==manifest,'Run inputs changed; choose a new output directory'
    else:save(path,manifest)
    save(OUT/'data_audit.json',dict(samples=total,records=39,subject_splits='matched',baseline_weight_hashes='matched',labels_and_shapes='matched',npz_hashes_match_e4=all(h==reference['data/'+n] for n,h in hashes.items())))

def train_fold(fold,cfg,split,conditions,device,smoke=False):
    dest=OUT/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selection.json').exists():return
    seed_all(cfg['seed']+fold)
    x,y,ids=read_group(split['train']);vx,vy,vids=read_group(split['validation'])
    if smoke:x,y,ids=x[:256],y[:256],ids[:256];vx,vy,vids=vx[:128],vy[:128],vids[:128]
    scale=max(float(np.sqrt(np.mean(x.astype('float64')**2))),1e-6)
    dae=DAE(scale).to(device);clf,basebatch=baseline(fold,device)
    opt=torch.optim.Adam(dae.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=-1.;stale=0;history=[];start_epoch=1;elapsed=0.
    last=dest/'latest.pth'
    if last.exists():
        ck=torch.load(last,map_location=device,weights_only=True);dae.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best=ck['best'];stale=ck['stale'];start_epoch=ck['epoch']+1;elapsed=ck['elapsed']
        history=load(dest/'history.json')[:ck['epoch']]
    epochs=1 if smoke else cfg['epochs']
    for epoch in range(start_epoch,epochs+1):
        tick=time.perf_counter();dae.train()
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],fold,epoch]))
        order=rng.permutation(len(x));assign=rng.integers(0,9,len(x));counts=np.bincount(assign,minlength=9)
        train_seed=int(np.random.SeedSequence([cfg['seed'],771,fold,epoch]).generate_state(1)[0])
        loss_sum=0.
        for start in range(0,len(x),cfg['batch_size']):
            ix=order[start:start+cfg['batch_size']];clean=x[ix];z=np.empty_like(clean)
            for k,c in enumerate(conditions[1:]):
                local=np.flatnonzero(assign[ix]==k)
                if len(local):z[local]=noisy(clean[local],[ids[ix[n]] for n in local],c,train_seed)
            inp=torch.from_numpy(z).to(device);target=torch.from_numpy(clean).to(device)
            opt.zero_grad(set_to_none=True);pred=dae(inp)
            loss=torch.mean(((pred-target)/dae.scale)**2)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(dae.parameters(),cfg['gradient_clip_norm']);opt.step()
            loss_sum+=float(loss.detach())*len(ix)
        metrics,_=evaluate(dae,clf,vx,vy,vids,conditions,basebatch,device)
        score=float(np.mean([m['macro_f1'] for m in metrics.values()]));elapsed+=time.perf_counter()-tick
        improved=score>best
        if improved:
            best=score;stale=0
            torch.save(dict(model=dae.state_dict(),scale=scale,epoch=epoch,validation_score=score),dest/'best.pth')
        else:stale+=1
        history.append(dict(epoch=epoch,train_mse_scaled=loss_sum/len(x),validation_macro_f1=score,training_condition_counts=counts.tolist(),seconds=time.perf_counter()-tick))
        save(dest/'history.json',history);save(dest/f'validation/epoch{epoch:03d}.json',metrics)
        torch.save(dict(model=dae.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,stale=stale,elapsed=elapsed),last)
        save(OUT/'status.json',dict(stage='training',fold=fold,epoch=epoch,best_validation_macro_f1=best))
        print(f'fold={fold} epoch={epoch} train_mse={loss_sum/len(x):.5f} val_macro_f1={score:.5f} best={best:.5f} seconds={time.perf_counter()-tick:.1f}',flush=True)
        if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
    ck=torch.load(dest/'best.pth',map_location='cpu',weights_only=True)
    save(dest/'selection.json',dict(fold=fold,selected_epoch=ck['epoch'],validation_macro_f1=ck['validation_score'],epochs_run=len(history),training_and_validation_seconds=elapsed,checkpoint_sha256=sha(dest/'best.pth'),parameter_count=sum(t.numel() for t in dae.parameters())))
    del x,vx,dae,clf,opt
    if device.type=='cuda':torch.cuda.empty_cache()

def verify_e2(splits,conditions):
    """Reproduce fixed E2 outputs, never use them to choose DAE settings."""
    import io
    reference=os.environ.get('E2_REFERENCE_CSV')
    if reference:
        raw=configured('E2_REFERENCE_CSV', reference).read_text(encoding='utf-8-sig')
    else:
        reference='c2547fb:result/e2_v2/predictions_with_ids.csv'
        try:
            raw=subprocess.check_output(['git','show',reference],cwd=ROOT).decode('utf-8-sig')
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Provide E2_REFERENCE_CSV or fetch the historical E3 commit c2547fb; no reference is regenerated automatically.') from exc
    rows=list(csv.DictReader(io.StringIO(raw)))
    print('Reference prediction fields:',list(rows[0]),flush=True)
    groups={}
    for row in rows:groups.setdefault((int(row['fold']),row['condition']),[]).append(row)
    differences=0;total=0
    for f in range(5):
        x,y,ids=read_group(splits['folds'][f]['test']);clf,batch=baseline(f,torch.device('cpu'))
        for c in conditions:
            ref=groups[f,c['name']];refmap={r['sample_id']:r for r in ref};assert len(refmap)==len(y)
            expected=np.array([int(refmap[s]['y_pred']) for s in ids]);labels=np.array([int(refmap[s]['y_true']) for s in ids]);assert np.array_equal(labels,y)
            z=noisy(x,ids,c,123);pred=e2.predict(clf,z,batch)
            count=int(np.count_nonzero(pred!=expected));differences+=count;total+=len(y)
            print('E2 CHECK',f,c['name'],'differences',count,flush=True)
        del x,clf
    save(OUT/'e2_reproduction.json',dict(predictions=total,differences=differences,reference=reference))
    assert differences==0,'E2 reproduction differs: stop before official E5 testing'

def test_and_summary(splits,conditions):
    assert load(OUT/'e2_reproduction.json')['differences']==0
    selections=[load(OUT/f'fold{f}/selection.json') for f in range(5)]
    save(OUT/'all_selections_frozen.json',selections)
    rows=[];allmetrics={};pooled={c['name']:[[],[]] for c in conditions}
    reference=list(csv.DictReader((ROOT/'result/e4_wavelet/e2_reference_per_fold.csv').open(encoding='utf-8-sig')))
    refs={(int(r['fold']),r['condition']):float(r['macro_f1_pct']) for r in reference}
    for f in range(5):
        dest=OUT/f'fold{f}';assert sha(dest/'best.pth')==selections[f]['checkpoint_sha256']
        ck=torch.load(dest/'best.pth',map_location='cpu',weights_only=True)
        dae=DAE(ck['scale']);dae.load_state_dict(ck['model']);clf,batch=baseline(f,torch.device('cpu'))
        x,y,ids=read_group(splits['folds'][f]['test'])
        metrics,preds=evaluate(dae,clf,x,y,ids,conditions,batch,torch.device('cpu'))
        save(dest/'test_metrics.json',metrics)
        for c in conditions:
            n=c['name'];m=metrics[n];allmetrics[f,n]=m
            np.savez_compressed(dest/f'predictions_{n}.npz',sample_ids=np.asarray(ids),y_true=y,y_pred=preds[n])
            pooled[n][0].append(y);pooled[n][1].append(preds[n])
            rows.append(dict(fold=f,condition=n,accuracy_pct=m['accuracy']*100,macro_f1_pct=m['macro_f1']*100,kappa=m['kappa'],delta_macro_f1_vs_e2_pp=m['macro_f1']*100-refs[f,n],rmse_denoised_uv=m['rmse_denoised_uv'],snr_improvement_db=m['snr_improvement_db'],denoise_and_inference_ms_per_epoch=m['denoise_and_inference_ms_per_epoch']))
        print('E5 TEST fold',f,'noisy F1',np.mean([m['macro_f1'] for n,m in metrics.items() if n!='Clean'])*100,flush=True)
        save(OUT/'status.json',dict(stage='test',fold=f));del x,dae,clf
    csvsave(OUT/'test_per_fold.csv',rows);summary=[];stages=[]
    for c in conditions:
        n=c['name'];line={'condition':n}
        for key in ['accuracy_pct','macro_f1_pct','kappa','delta_macro_f1_vs_e2_pp','rmse_denoised_uv','denoise_and_inference_ms_per_epoch']:
            vals=[r[key] for r in rows if r['condition']==n];line['mean_'+key]=float(np.mean(vals));line['sd_'+key]=float(np.std(vals,ddof=1))
        line['mean_snr_improvement_db']=None if n=='Clean' else float(np.mean([r['snr_improvement_db'] for r in rows if r['condition']==n]))
        summary.append(line)
        for stage in e2.LABELS:
            row={'condition':n,'stage':stage}
            for k in ['precision','recall','f1']:
                vals=[allmetrics[f,n]['stage_metrics'][stage][k] for f in range(5)]
                row['mean_'+k]=float(np.mean(vals));row['sd_'+k]=float(np.std(vals,ddof=1))
            stages.append(row)
    csvsave(OUT/'test_summary.csv',summary);csvsave(OUT/'stage_summary.csv',stages)
    save(OUT/'pooled_details.json',{n:e2.calculate_metrics(np.concatenate(a),np.concatenate(b)) for n,(a,b) in pooled.items()})
    vals=[np.mean([r['macro_f1_pct'] for r in rows if r['fold']==f and r['condition']!='Clean']) for f in range(5)]
    save(OUT/'overall.json',dict(noisy_macro_f1_mean_pct=float(np.mean(vals)),noisy_macro_f1_sd_pp=float(np.std(vals,ddof=1)),clean_macro_f1_mean_pct=float(np.mean([r['macro_f1_pct'] for r in rows if r['condition']=='Clean'])),delta_noisy_vs_e2_pp=float(np.mean(vals)-58.14146253053502)))
    save(OUT/'status.json',dict(stage='complete',results=load(OUT/'overall.json')))
    print('COMPLETE',load(OUT/'overall.json'),flush=True)

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--stage',choices=['all','train','verify','test'],default='all');args=ap.parse_args()
    if args.smoke:OUT=ROOT/'result/local/e5_smoke'
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=load(ROOT/'configs/e5_dae.json');splits=load(ROOT/'configs/fivefold_subject_splits.json');conditions=load(ROOT/'configs/noise_conditions_v2.json')['conditions']
    assert conditions[0]['name']=='Clean' and len(conditions)==10
    audit(cfg,splits,conditions)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage in ['all','train']:
        for f in range(1 if args.smoke else 5):train_fold(f,cfg,splits['folds'][f],conditions,device,args.smoke)
    if args.smoke:
        print('SMOKE PASS: train/validation completed without accessing test inputs',flush=True);return
    if args.stage in ['all','verify']:verify_e2(splits,conditions)
    if args.stage in ['all','test']:test_and_summary(splits,conditions)
if __name__=='__main__':main()
