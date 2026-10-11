"""Train exactly one configured fold; test labels are read only after best epoch is frozen."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,random,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'AttnSleep'))
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from model.model import AttnSleep
from train_Kfold_CV import weights_init_normal
from utils.util import calc_class_weight
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,cohen_kappa_score
cfg=json.loads(Path(sys.argv[1]).read_text());fold=cfg['fold'];run=Path(cfg['run_directory'])
S=R/f'reports/clean_baseline_5fold_fold{fold}_status.json';start=time.time()
lock=(R/'reports/clean_baseline_5fold.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 d=dict(stage=stage,fold=fold,pid=os.getpid(),run_dir=str(run),elapsed_seconds=round(time.time()-start,2),updated_unix=time.time(),**kw)
 tmp=S.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(S)
def save_checkpoint(path,epoch):
 obj=dict(epoch=epoch,state_dict=model.state_dict(),optimizer=opt.state_dict(),config=cfg,best_validation_macro_f1=best,best_epoch=best_epoch)
 tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def load_split(name):
 xs=[];ys=[]
 for filename in split[name]['files']:
  with np.load(D/filename,allow_pickle=False) as z:
   x=z['x'];y=z['y'];assert x.shape==(len(y),3000,1) and np.isfinite(x).all()
   assert np.isin(y,range(5)).all()
   xs.append(x.transpose(0,2,1));ys.append(y)
 return TensorDataset(torch.from_numpy(np.concatenate(xs).copy()).float(),torch.from_numpy(np.concatenate(ys)).long())
def evaluate(loader):
 model.eval();ys=[];preds=[];loss_sum=0.;weight_sum=0.
 with torch.no_grad():
  for x,y in loader:
   x,y=x.cuda(),y.cuda();out=model(x)
   assert torch.isfinite(out).all()
   loss_sum+=torch.nn.functional.cross_entropy(out,y,weight=weights,reduction='sum').item()
   weight_sum+=weights[y].sum().item()
   ys.extend(y.cpu().tolist());preds.extend(out.argmax(1).cpu().tolist())
 return dict(loss=loss_sum/weight_sum,accuracy=accuracy_score(ys,preds),macro_f1=f1_score(ys,preds,labels=list(range(5)),average='macro',zero_division=0)),np.array(ys),np.array(preds)
try:
 assert not run.exists(),'Output already exists; do not overwrite or repeat training'
 run.mkdir(parents=True)
 status('loading_train_validation')
 raw=Path(cfg['splits_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==cfg['splits_sha256']
 manifest=json.loads(raw);split=manifest['folds'][fold];D=Path(manifest['data_directory'])
 ids=[set(split[k]['subjects']) for k in ['train','validation','test']]
 assert list(map(len,ids))==[12,4,4] and not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
 file_sets=[set(split[k]['files']) for k in ['train','validation','test']]
 assert not(file_sets[0]&file_sets[1] or file_sets[0]&file_sets[2] or file_sets[1]&file_sets[2])
 for name in ['train','validation','test']:
  assert all(f[3:5] in split[name]['subjects'] for f in split[name]['files'])
 assert len(file_sets[0]|file_sets[1]|file_sets[2])==39
 (run/'splits.json').write_bytes(raw);(run/'config.json').write_text(json.dumps(cfg,indent=2))
 (run/'runner.py').write_text(Path(__file__).read_text())
 (run/'requirements-lock.txt').write_bytes((R/'requirements-lock.txt').read_bytes())
 (run/'author_code_diff.patch').write_text((R/'patches/author_compatibility.patch').read_text())
 assert torch.cuda.is_available();torch.set_num_threads(4)
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 train=load_split('train');val=load_split('validation')
 counts=np.bincount(train.tensors[1].numpy(),minlength=5).tolist()
 weights=torch.tensor(calc_class_weight(counts),dtype=torch.float32,device='cuda')
 info=dict(train_samples=len(train),validation_samples=len(val),train_class_counts=counts,class_weights=weights.cpu().tolist(),weight_source='train only',test_data_loaded=False,torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
 (run/'training_data_audit.json').write_text(json.dumps(info,indent=2))
 tl=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=0)
 vl=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
 model=AttnSleep();model.apply(weights_init_normal);model=model.cuda()
 opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']);best=-1.;best_epoch=0
 for epoch in range(1,cfg['epochs']+1):
  tick=time.time();model.train();loss_sum=0.;weight_sum=0.;correct=0;n=0
  for x,y in tl:
   x,y=x.cuda(),y.cuda();opt.zero_grad();out=model(x)
   loss=torch.nn.functional.cross_entropy(out,y,weight=weights);assert torch.isfinite(loss)
   loss.backward();opt.step()
   w=weights[y].sum().item();loss_sum+=loss.item()*w;weight_sum+=w
   correct+=(out.argmax(1)==y).sum().item();n+=len(y)
  metrics,_,_=evaluate(vl)
  if metrics['macro_f1']>best:
   best=metrics['macro_f1'];best_epoch=epoch;save_checkpoint(run/'best_validation_macro_f1.pth',epoch)
  if epoch==10:
   for group in opt.param_groups:group['lr']=cfg['lr_after_epoch10']
  save_checkpoint(run/'latest.pth',epoch)
  entry=dict(epoch=epoch,train_loss=loss_sum/weight_sum,train_accuracy=correct/n,validation=metrics,best_epoch=best_epoch,best_validation_macro_f1=best,seconds=round(time.time()-tick,3))
  with (run/'history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
  print(json.dumps(entry),flush=True)
  status('training',epoch=epoch,total_epochs=cfg['epochs'],metrics=entry)
 # Freeze model selection BEFORE opening any test NPZ file.
 (run/'selection_frozen.json').write_text(json.dumps(dict(best_epoch=best_epoch,validation_macro_f1=best,frozen_unix=time.time()),indent=2))
 state=torch.load(run/'best_validation_macro_f1.pth',weights_only=True,map_location='cuda');model.load_state_dict(state['state_dict'])
 check,_,_=evaluate(vl);assert abs(check['macro_f1']-best)<1e-12
 status('evaluating_test',best_epoch=best_epoch)
 test=load_split('test');metrics,y,pred=evaluate(DataLoader(test,batch_size=cfg['batch_size'],shuffle=False))
 names=['W','N1','N2','N3','REM'];cm=confusion_matrix(y,pred,labels=range(5))
 report=dict(fold=fold,selected_epoch=best_epoch,selection='validation macro-F1',test_subjects=split['test']['subjects'],test_samples=len(test),**metrics,kappa=cohen_kappa_score(y,pred),classification_report=classification_report(y,pred,labels=list(range(5)),target_names=names,output_dict=True,zero_division=0),confusion_matrix=cm.tolist(),scope='One fold of fixed fivefold subject split; train, validation, test disjoint. No other folds trained.')
 np.savez(run/'test_predictions.npz',y_true=y,y_pred=pred)
 (run/'test_metrics.json').write_text(json.dumps(report,indent=2))
 status('completed',epochs=cfg['epochs'],metrics=report)
except Exception as e:
 status('failed',error=repr(e));raise
