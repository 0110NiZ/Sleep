"""E6: independent noise-augmented AttnSleep training and held-out evaluation.

Training never opens current-fold test NPZs. Official test requires five frozen
selections. Epoch checkpoints include RNG state for reproducible continuation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AttnSleep'))
from noise_conditions import apply_noise, apply_noise_v2


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment(x, ids, assignment, conditions, seed):
    """0 is clean, 1..9 index the shared conditions; labels are unchanged."""
    z = x.copy()
    for k, condition in enumerate(conditions[1:], start=1):
        ix = np.flatnonzero(assignment == k)
        if len(ix):
            z[ix] = apply_noise(x[ix], condition,
                sample_ids=[ids[i] for i in ix], seed=seed, fs=100)
    return z


def validate_protocol(cfg, splits, noise):
    reference = read(ROOT / 'result/e5_dae/manifest.json')['code']
    for name in ['configs/fivefold_subject_splits.json',
                 'configs/noise_conditions_v2.json',
                 'AttnSleep/model/model.py']:
        if sha(ROOT / name) != reference[name]:
            raise ValueError(f'Changed shared protocol/model: {name}')
    generator = (ROOT / 'scripts/noise_conditions.py').read_bytes()
    alias = b'\n# Public version-neutral name; protocol remains Noise v2.\napply_noise = apply_noise_v2\n'
    if not generator.endswith(alias) or hashlib.sha256(generator[:-len(alias)]).hexdigest() != reference['scripts/noise_conditions_v2.py']:
        raise ValueError('Changed Noise v2 implementation: historical content mismatch')
    assert cfg['epochs'] == 100 and cfg['batch_size'] == 128
    base = read(ROOT / 'configs/clean_baseline_fold0.json')
    for key in ['seed', 'optimizer', 'lr_after_epoch10']:
        assert cfg[key] == base[key], key
    assert 0 < cfg['clean_probability'] < 1
    assert cfg['test_device'] == 'cpu' and cfg['inference_batch_size'] == 128
    assert noise['seed'] == 123 and noise['sampling_rate'] == 100
    assert noise['signal_unit'] == 'uV'
    assert len(noise['conditions']) == 10 and noise['conditions'][0]['name'] == 'Clean'
    all_tests = []
    for i, fold in enumerate(splits['folds']):
        assert fold['fold'] == i
        groups = [fold[k] for k in ['train', 'validation', 'test']]
        subjects = [set(g['subjects']) for g in groups]
        files = [set(g['files']) for g in groups]
        assert [len(s) for s in subjects] == [12, 4, 4]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            assert not subjects[a] & subjects[b] and not files[a] & files[b]
        assert len(set.union(*files)) == 39
        for g in groups:
            assert all(f[3:5] in g['subjects'] for f in g['files'])
        all_tests += list(files[2])
    assert len(all_tests) == len(set(all_tests)) == 39


def audit_group(group, data_dir, strict_hashes=True):
    """Audit ONLY the requested group. No test file opening during fold training."""
    expected = {r['record'] + '.npz': r for r in
                read(ROOT / 'reports/full_alignment_validation.json')}
    hashes = read(ROOT / 'result/e5_dae/manifest.json')['data']
    for name in group['files']:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing clean NPZ: {path}')
        digest = sha(path)
        if strict_hashes and digest != hashes[name]:
            raise ValueError(f'{name}: differs from E5 clean NPZ bytes. Obtain shared originals; '
                'or use --allow-repacked-data ONLY after verifying equivalent arrays.')
        with np.load(path, allow_pickle=False) as z:
            x, y = z['x'], z['y']
            row = expected[name]
            assert x.shape == (row['segments'], 3000, 1)
            assert np.isfinite(x).all() and np.isin(y, range(5)).all()
            assert np.bincount(y, minlength=5).tolist() == row['class_counts']
    return {name: sha(data_dir / name) for name in group['files']}


def bind_manifest(out, cfg, splits, noise, smoke, device, allow_repacked):
    paths = ['scripts/run_e6_noise_augmentation.py', 'scripts/run_e2_evaluation.py',
             'scripts/noise_conditions.py', 'scripts/project_paths.py',
             'AttnSleep/model/model.py', 'AttnSleep/utils/util.py']
    manifest = dict(config=cfg, splits=splits, noise=noise, smoke=smoke,
        allow_repacked_data=allow_repacked, train_device=str(device),
        code={p: sha(ROOT / p) for p in paths},
        versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                      numpy=np.__version__, sklearn=sklearn.__version__),
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)
    path = out / 'manifest.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Run configuration/code/environment changed. Use a new --out directory.')
    else:
        save(path, manifest)


def evaluate(model, x, y, ids, noise, device, batch=128, timing=False):
    model.eval()
    scores, predictions = {}, {}
    with torch.inference_mode():
        if timing:
            model(torch.from_numpy(x[:min(batch, len(x))]).to(device))
        for c in noise['conditions']:
            parts, seconds = [], 0.
            for j in range(0, len(x), batch):
                z = apply_noise(x[j:j+batch], c, sample_ids=ids[j:j+batch],
                                  seed=noise['seed'], fs=noise['sampling_rate'])
                t = torch.from_numpy(z).to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                logits = model(t)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                seconds += time.perf_counter() - tick
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite logits')
                parts.append(logits.argmax(1).cpu().numpy())
            pred = np.concatenate(parts)
            scores[c['name']] = e2.calculate_metrics(y, pred)
            if timing:
                scores[c['name']]['inference_ms_per_epoch'] = seconds * 1000 / len(y)
            predictions[c['name']] = pred
    return scores, predictions


def init_weights(m):
    # Identical to E1 train_Kfold_CV.weights_init_normal without import side effects.
    if type(m) in (torch.nn.Conv1d, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0., 0.02)
    elif type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1., 0.02)
        torch.nn.init.constant_(m.bias.data, 0.)


def checkpoint(path, state):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def train_fold(f, cfg, split, noise, out, device, smoke, allow_repacked):
    dest = out / f'fold{f}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'selection.json').exists():
        selected = read(dest / 'selection.json')
        assert sha(dest / 'best.pth') == selected['checkpoint_sha256']
        print(f'Fold {f}: frozen selection exists, skipping training.', flush=True)
        return
    data_hashes = {}
    for group in ['train', 'validation']:
        data_hashes.update(audit_group(split[group], e2.DATA_DIR, not allow_repacked))
    audit_path = dest / 'training_data_audit.json'
    audit = dict(hashes=data_hashes, test_data_loaded=False)
    if audit_path.exists():
        assert read(audit_path) == audit, 'Training data changed during continuation'
    save(audit_path, audit)
    x, y, ids = e2.load_test(split['train'])
    vx, vy, vids = e2.load_test(split['validation'])
    if smoke:
        ix = np.concatenate([np.flatnonzero(y == k)[:8] for k in range(5)])
        x, y, ids = x[ix], y[ix], [ids[i] for i in ix]
        ix = np.concatenate([np.flatnonzero(vy == k)[:4] for k in range(5)])
        vx, vy, vids = vx[ix], vy[ix], [vids[i] for i in ix]
    seed_all(cfg['seed'])  # E1 uses seed 123 separately for each fold.
    model = e2.AttnSleep()
    model.apply(init_weights)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), **cfg['optimizer'])
    counts = np.bincount(y, minlength=5).tolist()
    weights = torch.tensor(calc_class_weight(counts), dtype=torch.float32, device=device)
    save(dest / 'class_weights.json', dict(source='training labels only', counts=counts,
                                          weights=weights.cpu().tolist()))
    best, best_epoch, start, elapsed, history = -1., 0, 1, 0., []
    last = dest / 'latest.pth'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['state_dict'])
        opt.load_state_dict(ck['optimizer'])
        best, best_epoch = ck['best'], ck['best_epoch']
        start, elapsed, history = ck['epoch'] + 1, ck['elapsed'], ck['history']
        torch.set_rng_state(ck['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        # Latest and best are committed together. Recover best after interrupted writes.
        checkpoint(dest / 'best.pth', ck['best_checkpoint'])
        save(dest / 'history.json', history)
        save(dest / f"validation/epoch{ck['epoch']:03d}.json", ck['validation_metrics'])
        best_checkpoint = ck['best_checkpoint']
    else:
        best_checkpoint = None
    probabilities = [cfg['clean_probability']] + [(1-cfg['clean_probability'])/9]*9
    for epoch in range(start, (1 if smoke else cfg['epochs']) + 1):
        tick = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], f, epoch, 66]))
        order = rng.permutation(len(x))
        assignment = rng.choice(10, len(x), p=probabilities)
        training_seed = int(np.random.SeedSequence([cfg['seed'],
            cfg['training_noise_seed_namespace'], f, epoch]).generate_state(1)[0])
        model.train()
        loss_sum, denom, correct = 0., 0., 0
        for j in range(0, len(x), cfg['batch_size']):
            ix = order[j:j+cfg['batch_size']]
            z = augment(x[ix], [ids[i] for i in ix], assignment[ix],
                        noise['conditions'], training_seed)
            t = torch.from_numpy(z).to(device)
            labels = torch.from_numpy(y[ix]).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(t)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            opt.step()
            w = weights[labels].sum().item()
            loss_sum += loss.item()*w
            denom += w
            correct += (logits.argmax(1) == labels).sum().item()
        train_seconds = time.perf_counter() - tick
        scores, _ = evaluate(model, vx, vy, vids, noise, device,
                             cfg['inference_batch_size'])
        score = float(np.mean([s['macro_f1'] for s in scores.values()]))
        if score > best:
            best, best_epoch = score, epoch
            best_checkpoint = dict(state_dict={k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()}, epoch=epoch,
                validation_score=score, config=cfg)
        # E1 changes LR after completing epoch 10.
        if epoch == 10:
            for group in opt.param_groups:
                group['lr'] = cfg['lr_after_epoch10']
        seconds = time.perf_counter() - tick
        elapsed += seconds
        entry = dict(epoch=epoch, train_loss=loss_sum/denom,
            train_accuracy=correct/len(y), validation_macro_f1=score,
            best_epoch=best_epoch, training_condition_counts=np.bincount(
                assignment, minlength=10).tolist(), training_noise_seed=training_seed,
            train_seconds=train_seconds, validation_seconds=seconds-train_seconds,
            seconds=seconds)
        history.append(entry)
        # Commit authoritative latest first. Resume recovers best/history from latest.
        checkpoint(last, dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
            epoch=epoch, best=best, best_epoch=best_epoch, elapsed=elapsed,
            history=history, validation_metrics=scores, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            best_checkpoint=best_checkpoint))
        checkpoint(dest / 'best.pth', best_checkpoint)
        save(dest / 'history.json', history)
        save(dest / f'validation/epoch{epoch:03d}.json', scores)
        save(out / 'status.json', dict(stage='training', fold=f, **entry))
        print(f'fold={f} epoch={epoch} val10={score:.5f} best={best:.5f} '
              f'best_epoch={best_epoch} seconds={seconds:.1f}', flush=True)
    save(dest / 'selection.json', dict(fold=f, selected_epoch=best_epoch,
        validation_macro_f1=best, epochs_run=len(history),
        training_and_validation_seconds=elapsed,
        train_seconds=sum(h['train_seconds'] for h in history),
        checkpoint_sha256=sha(dest / 'best.pth'), smoke=smoke))
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def official_test(cfg, splits, noise, out, allow_repacked):
    selections = [read(out / f'fold{f}/selection.json') for f in range(5)]
    assert all(not s['smoke'] and s['epochs_run'] == cfg['epochs'] for s in selections)
    # All five selections are frozen before any test group is loaded here.
    save(out / 'all_selections_frozen.json', selections)
    device = torch.device('cpu')
    pooled = {c['name']: [[], []] for c in noise['conditions']}
    rows, stage_rows = [], []
    for f in range(5):
        dest = out / f'fold{f}'
        assert sha(dest / 'best.pth') == selections[f]['checkpoint_sha256']
        audit_group(splits['folds'][f]['test'], e2.DATA_DIR, not allow_repacked)
        # Ensure this fold's test arrays are the same bytes already used in the run.
        known = {}
        for k in range(5):
            known.update(read(out / f'fold{k}/training_data_audit.json')['hashes'])
        assert all(sha(e2.DATA_DIR / n) == known[n]
                   for n in splits['folds'][f]['test']['files'])
        ck = torch.load(dest / 'best.pth', map_location='cpu', weights_only=True)
        model = e2.AttnSleep()
        model.load_state_dict(ck['state_dict'], strict=True)
        x, y, ids = e2.load_test(splits['folds'][f]['test'])
        scores, predictions = evaluate(model, x, y, ids, noise, device,
                                       cfg['inference_batch_size'], timing=True)
        save(dest / 'test_metrics.json', scores)
        for name, m in scores.items():
            np.savez_compressed(dest / f'predictions_{name}.npz',
                sample_ids=np.asarray(ids), y_true=y, y_pred=predictions[name])
            pooled[name][0].append(y)
            pooled[name][1].append(predictions[name])
            rows.append(dict(fold=f, condition=name, samples=len(y),
                accuracy_pct=100*m['accuracy'], macro_f1_pct=100*m['macro_f1'],
                kappa=m['kappa'], inference_ms_per_epoch=m['inference_ms_per_epoch']))
            for stage, sm in m['stage_metrics'].items():
                stage_rows.append(dict(fold=f, condition=name, stage=stage, **sm))
        print(f'Official test fold {f} complete', flush=True)
    write_csv(out / 'test_per_fold.csv', rows)
    write_csv(out / 'stage_per_fold.csv', stage_rows)
    stage_summary = []
    for c in noise['conditions']:
        for stage in e2.LABELS:
            subset = [r for r in stage_rows if r['condition'] == c['name'] and r['stage'] == stage]
            line = dict(condition=c['name'], stage=stage)
            for key in ['precision', 'recall', 'f1']:
                values = [r[key] for r in subset]
                line['mean_'+key] = float(np.mean(values))
                line['sd_'+key] = float(np.std(values, ddof=1))
            stage_summary.append(line)
    write_csv(out / 'stage_summary.csv', stage_summary)
    summaries = []
    for c in noise['conditions']:
        subset = [r for r in rows if r['condition'] == c['name']]
        line = dict(condition=c['name'])
        for key in ['accuracy_pct', 'macro_f1_pct', 'kappa', 'inference_ms_per_epoch']:
            v = [r[key] for r in subset]
            line['mean_'+key], line['sd_'+key] = float(np.mean(v)), float(np.std(v, ddof=1))
        summaries.append(line)
    write_csv(out / 'test_summary.csv', summaries)
    save(out / 'pooled_details.json', {n:e2.calculate_metrics(np.concatenate(a),
        np.concatenate(b)) for n,(a,b) in pooled.items()})
    assert sum(r['samples'] for r in rows) == 423080
    noisy = [float(np.mean([r['macro_f1_pct'] for r in rows
             if r['fold'] == f and r['condition'] != 'Clean'])) for f in range(5)]
    clean = [r['macro_f1_pct'] for r in rows if r['condition'] == 'Clean']
    overall = dict(noisy_macro_f1_per_fold_pct=noisy,
        noisy_macro_f1_mean_pct=float(np.mean(noisy)),
        noisy_macro_f1_sd_pp=float(np.std(noisy, ddof=1)),
        clean_macro_f1_mean_pct=float(np.mean(clean)),
        clean_macro_f1_sd_pp=float(np.std(clean, ddof=1)),
        prediction_files=50, prediction_records=423080,
        timing_scope='CPU classifier forward only; noise generation, transfers and warm-up excluded')
    save(out / 'overall.json', overall)
    save(out / 'status.json', dict(stage='complete', **overall))
    print(json.dumps(overall, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['check','smoke','train','test','all'], default='check')
    ap.add_argument('--fold', choices=['0','1','2','3','4','all'], default='all')
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    ap.add_argument('--allow-repacked-data', action='store_true',
        help='Permit different NPZ container hashes; equivalence of signal arrays must be verified separately')
    args = ap.parse_args()
    if args.data_dir:
        os.environ['SLEEP_DATA_DIR'] = str(args.data_dir.resolve())
    global torch, e2, calc_class_weight, sklearn
    import torch
    import sklearn
    import run_e2_evaluation as e2
    from utils.util import calc_class_weight
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = read(ROOT / 'configs/e6_noise_augmentation.json')
    splits = read(ROOT / 'configs/fivefold_subject_splits.json')
    noise = read(ROOT / 'configs/noise_conditions_v2.json')
    validate_protocol(cfg, splits, noise)
    if args.stage == 'check':
        names = sorted({n for f in splits['folds'] for n in f['train']['files']})
        missing = [n for n in names if not (e2.DATA_DIR / n).is_file()]
        print(json.dumps(dict(data_directory=str(e2.DATA_DIR), expected_files=39,
            missing_files=missing, cuda=torch.cuda.is_available(),
            torch=str(torch.__version__), protocol='PASS'), indent=2))
        if missing:
            raise SystemExit('Place the 39 shared Clean NPZ files in --data-dir first.')
        return
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    smoke = args.stage == 'smoke'
    out = (args.out or ROOT / ('result/local/e6_smoke' if smoke else
                               'result/local/e6_noise_augmentation')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # OS lock is released automatically on exit, including interruption.
    with (out / '.run.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.tell() == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Test uses the stored training environment and does not alter it.
        if args.stage == 'test':
            old = read(out / 'manifest.json')
            bind_manifest(out, cfg, splits, noise, False,
                torch.device(old['train_device']), args.allow_repacked_data)
        else:
            bind_manifest(out, cfg, splits, noise, smoke, device, args.allow_repacked_data)
        if args.stage in ['smoke','train','all']:
            folds = ([0] if smoke and args.fold == 'all' else range(5)
                     if args.fold == 'all' else [int(args.fold)])
            for f in folds:
                train_fold(f, cfg, splits['folds'][f], noise, out, device, smoke,
                           args.allow_repacked_data)
        if args.stage in ['test','all']:
            official_test(cfg, splits, noise, out, args.allow_repacked_data)
        if smoke:
            print('SMOKE complete: subset only, no official test accessed.', flush=True)


if __name__ == '__main__':
    main()
