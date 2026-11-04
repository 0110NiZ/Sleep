"""Small contract checks, using training examples only."""
import numpy as np
import torch
from run_e5_dae import ROOT,load,read_group,baseline,noisy
from dae_model import DAE
torch.set_num_threads(4)
s=load(ROOT/'configs/fivefold_subject_splits.json')['folds'][0]
x,_,ids=read_group(s['train']);x=x[:8];ids=ids[:8];original=x.copy()
conditions=load(ROOT/'configs/noise_conditions_v2.json')['conditions']
for c in conditions:
    whole=noisy(x,ids,c,123)
    split=np.concatenate([noisy(x[:3],ids[:3],c,123),noisy(x[3:],ids[3:],c,123)])
    assert np.array_equal(whole,split),'Noise depends on batching'
    assert np.array_equal(x,original),'Noise mutated clean reconstruction targets'
dae=DAE(25);t=torch.from_numpy(x)
assert dae(t).shape==t.shape and torch.allclose(dae(t),t,atol=1e-5),'Identity initialization/shape failed'
state=dae.state_dict();copy=DAE(1);copy.load_state_dict(state);assert torch.equal(dae(t),copy(t))
model,_=baseline(0,torch.device('cpu'));assert not model.training
assert all(not p.requires_grad for p in model.parameters())
for _ in range(2):
    loss=((dae(t)-t)/25).square().mean();loss.backward()
assert all(p.grad is None for p in model.parameters())
print('PASS: sample-specific noise, unchanged targets, shape/scale, checkpoint roundtrip, frozen classifier.')
