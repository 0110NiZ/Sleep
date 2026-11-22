"""Export frozen models and a stage-stratified, prediction-independent demo subset."""
from pathlib import Path
import argparse,csv,sys,json,hashlib
from functools import lru_cache
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Rebuild browser assets from the formal experiment inputs and checkpoints.')
parser.add_argument('--data-dir',type=Path,required=True,help='Directory containing the 39 original clean NPZ files')
parser.add_argument('--baseline-dir',type=Path,required=True,help='Contains fold0..fold4/best_validation_macro_f1.pth')
parser.add_argument('--e5-dir',type=Path,required=True,help='Contains fold0..fold4/best.pth and selection.json')
parser.add_argument('--e6-dir',type=Path,required=True,help='Contains fold0..fold4/best.pth and selection.json')
parser.add_argument('--check-inputs',action='store_true',help='Check all recordings and load all 15 models without exporting')
args=parser.parse_args()
import numpy as np
import torch
import onnxruntime as ort
from scipy.signal import butter,sosfiltfilt
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'AttnSleep'))
from model.model import AttnSleep
from experiments.e5_dae.dae_model import DAE
from experiments.support.shared.noise_conditions import apply_noise
from experiments.e4_wavelet.wavelet_denoising import denoise
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
splits=read(PROJECT/'configs/fivefold_subject_splits.json')
files={name:f for f,fold in enumerate(splits['folds']) for name in fold['test']['files']}
noise=read(PROJECT/'configs/noise_conditions_v2.json')
wave_config=read(PROJECT/'configs/e4_wavelet.json')
with (PROJECT/'results/e3_bandpass/comparison/parameter_selection.csv').open(encoding='utf-8-sig') as handle:
    e3={int(row['fold']):row['selected_filter'] for row in csv.DictReader(handle)}
selection=read(PROJECT/'results/e4_wavelet/selection_frozen.json')['folds']
e4={f:selection[str(f)]['candidate'] for f in range(5)}
for f in range(5):
    for path in [args.baseline_dir/f'fold{f}'/'best_validation_macro_f1.pth',args.e5_dir/f'fold{f}'/'best.pth',args.e5_dir/f'fold{f}'/'selection.json',args.e6_dir/f'fold{f}'/'best.pth',args.e6_dir/f'fold{f}'/'selection.json']:
        if not path.is_file():parser.error(f'Missing input: {path}')
for name in files:
    if not (args.data_dir/name).is_file():parser.error(f'Missing clean recording: {name}')
torch.set_num_threads(4)
@lru_cache(maxsize=2)
def recording(name):
    with np.load(args.data_dir/name,allow_pickle=False) as z:
        return z['x'].transpose(0,2,1).astype('float32'),z['y']
def baseline(fold):
    m=AttnSleep().eval()
    ck=torch.load(args.baseline_dir/f'fold{fold}'/'best_validation_macro_f1.pth',map_location='cpu',weights_only=True)
    m.load_state_dict(ck['state_dict'],strict=True)
    return m
def trained(fold,kind):
    folder=(args.e5_dir if kind=='dae' else args.e6_dir)/f'fold{fold}'
    path=folder/'best.pth'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=read(folder/'selection.json')['checkpoint_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {kind}, fold {fold}')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=DAE(ck['scale']) if kind=='dae' else AttnSleep()
    m.load_state_dict(ck['model'] if kind=='dae' else ck['state_dict'],strict=True)
    return m.eval()
source=SimpleNamespace(LABELS=['W','N1','N2','N3','REM'],NOISE=noise,WAVE=wave_config,FILES=files,E3=e3,E4=e4,recording=recording,model=baseline,apply_noise_v2=apply_noise,denoise=denoise)
e5_adapter=SimpleNamespace(load_model=lambda f:trained(f,'dae'))
e6_adapter=SimpleNamespace(load_model=lambda f:trained(f,'augmented'))
if args.check_inputs:
    for name in files:
        x,y=recording(name)
        if x.ndim!=3 or x.shape[1:]!=(1,3000) or len(x)!=len(y) or not np.isfinite(x).all():
            raise ValueError(f'Invalid input recording: {name}')
    with torch.inference_mode():
        for fold in range(5):
            for model in [baseline(fold),trained(fold,'dae'),trained(fold,'augmented')]:
                output=model(torch.zeros(1,1,3000))
                if not torch.isfinite(output).all():raise ValueError('Invalid model output')
    print('PASS: 39 clean recordings, repository imports, formal selections and 15 PyTorch models')
    raise SystemExit(0)
DIST=ROOT/'dist'
for d in ['models','data','vendor']: (DIST/d).mkdir(exist_ok=True)
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
opt=ort.SessionOptions();opt.intra_op_num_threads=2
rng=np.random.default_rng(4330)
manifest={'labels':source.LABELS,'conditions':[c['name'] for c in source.NOISE['conditions']], 'samples':[], 'models':{}, 'noise_seed':source.NOISE['seed'],'subset_seed':4330,'preprocessing':'Noise v2 and E3/E4 prepared with original Python code; DAE and all classifiers run live in browser.'}
checks=[]
for fold in range(5):
    models={'baseline':source.model(fold),'dae':e5_adapter.load_model(fold),'augmented':e6_adapter.load_model(fold)}
    sessions={};manifest['models'][str(fold)]={}
    for name,m in models.items():
        target=DIST/'models'/f'{name}-{fold}.onnx'
        torch.onnx.export(m,torch.zeros(1,1,3000),str(target),input_names=['eeg'],output_names=['output'],opset_version=17,dynamo=False)
        sessions[name]=ort.InferenceSession(str(target),sess_options=opt,providers=['CPUExecutionProvider'])
        manifest['models'][str(fold)][name]={'url':f'models/{target.name}','sha256':sha(target),'bytes':target.stat().st_size}
    available={stage:[] for stage in range(5)}
    for record,f in source.FILES.items():
        if f!=fold:continue
        x,y=source.recording(record)
        for stage in available:
            available[stage].extend((record,int(i)) for i in np.flatnonzero(y==stage))
    for stage,choices in available.items():
        for pick in rng.choice(len(choices),size=min(2,len(choices)),replace=False):
            record,index=choices[int(pick)];x,y=source.recording(record);clean=x[index:index+1].copy()
            sid=f'{record}:{index}';key=f's{len(manifest["samples"]):03d}'
            sample={'key':key,'record':record,'index':index,'id':sid,'fold':fold,'truth':source.LABELS[stage], 'signals':{}}
            for condition in source.NOISE['conditions']:
                noisy=source.apply_noise_v2(clean,condition,sample_ids=[sid],seed=source.NOISE['seed'],fs=100)
                a=source.E3[fold];lo,hi=(.3,35) if a=='A' else (.5,30)
                bp=sosfiltfilt(butter(4,[lo,hi],btype='bandpass',fs=100,output='sos'),noisy.astype('float64'),axis=-1).astype('float32')
                candidate=next(c for c in source.WAVE['candidates'] if c['id']==source.E4[fold])
                wave=source.denoise(noisy,candidate,source.WAVE)
                signals=[clean,noisy,bp,wave]
                path=DIST/'data'/f'{key}-{condition["name"]}.bin'
                np.concatenate([v.flatten() for v in signals]).astype('<f4').tofile(path)
                sample['signals'][condition['name']]=f'data/{path.name}'
                with torch.inference_mode():
                    dae=models['dae'](torch.from_numpy(noisy)).numpy()
                    ref=[models['baseline'](torch.from_numpy(v)).numpy()[0] for v in signals+[dae]]
                    ref.append(models['augmented'](torch.from_numpy(noisy)).numpy()[0])
                outdae=sessions['dae'].run(None,{'eeg':noisy})[0]
                converted=[sessions['baseline'].run(None,{'eeg':v})[0][0] for v in signals+[outdae]]
                converted.append(sessions['augmented'].run(None,{'eeg':noisy})[0][0])
                assert [v.argmax() for v in ref]==[v.argmax() for v in converted],(sid,condition['name'])
                checks.append({'sample':key,'condition':condition['name'],'predictions':[int(v.argmax()) for v in ref], 'max_logit_difference':float(max(np.max(np.abs(a-b)) for a,b in zip(ref,converted))), 'dae_max_difference':float(np.max(np.abs(dae-outdae)))})
            sample['e3_parameter']=f'{a} · {lo}–{hi} Hz · 4阶'
            sample['e4_parameter']=f"{candidate['id']} · {candidate['wavelet']} · 阈值 {candidate['scale']}"
            manifest['samples'].append(sample)
    print(f'PASS fold {fold+1}: 3 models exported and 100 sample/condition comparisons',flush=True)
dump(DIST/'manifest.json',manifest)
dump(ROOT/'verification.json',{'all_labels_match':True,'comparisons':len(checks)*6,'sample_conditions':len(checks),'max_logit_difference':max(c['max_logit_difference'] for c in checks),'max_dae_difference':max(c['dae_max_difference'] for c in checks),'checks':checks})
dump(DIST/'verification-summary.json',{'sample_conditions':len(checks),'predictions_checked':len(checks)*6,'all_labels_match':True,'max_logit_difference':max(c['max_logit_difference'] for c in checks)})
print('PASS browser assets complete',flush=True)
