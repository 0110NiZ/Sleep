"""Check portable demo assets; optionally rerun all 3,000 native ONNX predictions."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / 'dist'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def asset(relative):
    path = (DIST / relative).resolve()
    if DIST.resolve() not in path.parents or not path.is_file():
        raise ValueError(f'Missing or unsafe runtime asset: {relative}')
    return path

def validate_assets():
    for relative, expected in read(ROOT / 'assets.sha256.json').items():
        path = (ROOT / relative).resolve()
        if DIST.resolve() not in path.parents or not path.is_file():
            raise ValueError(f'Missing asset: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Asset hash mismatch: {relative}')
    meta = read(DIST / 'manifest.json')
    if len(meta['samples']) != 50 or len(meta['conditions']) != 10:
        raise ValueError('Expected 50 samples and 10 conditions')
    if meta['labels'] != ['W', 'N1', 'N2', 'N3', 'REM']:
        raise ValueError('Unexpected class order')
    for fold in range(5):
        models = meta['models'][str(fold)]
        if set(models) != {'baseline', 'dae', 'augmented'}:
            raise ValueError('Incomplete fold model mapping')
        for spec in models.values():
            path = asset(spec['url'])
            if path.stat().st_size != spec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError(f'Model hash mismatch: {path.name}')
        samples = [s for s in meta['samples'] if s['fold'] == fold]
        if len(samples) != 10 or any(sum(s['truth'] == label for s in samples) != 2 for label in meta['labels']):
            raise ValueError('Unexpected sample distribution')
    for sample in meta['samples']:
        if set(sample['signals']) != set(meta['conditions']):
            raise ValueError(f'Incomplete conditions: {sample["id"]}')
        for relative in sample['signals'].values():
            if asset(relative).stat().st_size != 4 * 3000 * 4:
                raise ValueError(f'Wrong signal length: {relative}')
    for name in ['index.html', 'inference.js', 'vendor/ort.wasm.min.js',
                 'vendor/ort-wasm-simd-threaded.wasm', 'vendor/ort-wasm-simd-threaded.mjs']:
        asset(name)
    if (ROOT / 'page_template.html').read_bytes() != (DIST / 'index.html').read_bytes():
        raise ValueError('Page template differs from dist/index.html; run prepare_page.py')
    return meta

def validate_inference(meta):
    import numpy as np
    import onnxruntime as ort
    reference = {(x['sample'], x['condition']): x['predictions'] for x in read(ROOT / 'verification.json')['checks']}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    count = 0
    for fold in range(5):
        sessions = {name: ort.InferenceSession(str(asset(spec['url'])), sess_options=options,
                    providers=['CPUExecutionProvider']) for name, spec in meta['models'][str(fold)].items()}
        for sample in [s for s in meta['samples'] if s['fold'] == fold]:
            for condition, relative in sample['signals'].items():
                signals = np.fromfile(asset(relative), dtype='<f4').reshape(4, 1, 1, 3000)
                clean, noisy, bandpass, wavelet = signals
                reconstructed = sessions['dae'].run(None, {'eeg': noisy})[0]
                if not np.isfinite(signals).all() or not np.isfinite(reconstructed).all():
                    raise ValueError('Non-finite waveform')
                predicted = [int(sessions['baseline'].run(None, {'eeg': value})[0].argmax())
                             for value in [clean, noisy, bandpass, wavelet, reconstructed]]
                predicted.append(int(sessions['augmented'].run(None, {'eeg': noisy})[0].argmax()))
                if predicted != reference[sample['key'], condition]:
                    raise ValueError(f'Prediction mismatch: {sample["id"]}, {condition}')
                count += 6
        print(f'PASS fold {fold + 1}: {count} predictions checked', flush=True)
    print(f'PASS: all {count} predictions match the archived PyTorch conversion checks')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', action='store_true', help='Requires numpy and onnxruntime; not needed to launch the demo')
    args = parser.parse_args()
    try:
        meta = validate_assets()
        print('PASS: asset hashes, 15 models, 50 samples, 500 signal files and class/fold mappings')
        if args.inference:
            validate_inference(meta)
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.exit(1, f'Validation failed: {error}\n')
