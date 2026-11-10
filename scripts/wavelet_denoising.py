"""E4 per-epoch denoising. Only the supplied input is used to estimate thresholds."""
import numpy as np
import pywt


def denoise(x, candidate, config):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (1, config['epoch_samples']):
        raise ValueError('Expected (N, 1, 3000) EEG epochs')
    if not np.isfinite(x).all():
        raise ValueError('Non-finite EEG')
    wavelet = candidate['wavelet']
    level = config['level']
    if level > pywt.dwt_max_level(x.shape[-1], pywt.Wavelet(wavelet).dec_len):
        raise ValueError('Decomposition level exceeds useful maximum')
    coefficients = pywt.wavedec(x.astype(np.float64), wavelet,
                               mode=config['extension_mode'], level=level, axis=-1)
    finest = coefficients[-1]
    sigma = np.median(np.abs(finest - np.median(finest, axis=-1, keepdims=True)),
                      axis=-1, keepdims=True) / 0.6744897501960817
    threshold = candidate['scale'] * sigma * np.sqrt(2 * np.log(x.shape[-1]))
    # Explicit soft shrinkage handles zero thresholds and zero coefficients without 0/0.
    filtered = [coefficients[0]] + [np.sign(c) * np.maximum(np.abs(c)-threshold, 0)
                                  for c in coefficients[1:]]
    output = pywt.waverec(filtered, wavelet, mode=config['extension_mode'], axis=-1)
    output = np.ascontiguousarray(output[..., :x.shape[-1]], dtype=np.float32)
    if output.shape != x.shape or not np.isfinite(output).all():
        raise ValueError('Invalid reconstructed EEG')
    return output
