"""Test storage hypotheses against a native pre/downsample pair; no layout proof."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
analysis = json.loads((args.run / 'arena-analysis.json').read_text())
if analysis['pre_dimensions'] != [320, 320] or analysis['down_dimensions'] != [160, 160]:
    raise ValueError('This hypothesis experiment is specialized to the recorded dimensions')

def decode(raw):
    code = raw.astype(np.int32)
    exponent, mantissa = (code >> 3) & 15, code & 7
    value = np.where(exponent == 0, mantissa * 2.**-9, (1 + mantissa / 8.) * np.exp2(exponent - 7))
    if np.any((exponent == 15) & (mantissa == 7)):
        raise ValueError('Unexpected E4M3 NaN codes')
    return np.where(code & 128, -value, value).astype(np.float32)

def layouts(raw, size):
    yield 'plain_hwc', raw.reshape(size, size, 32)
    yield 'plain_chw', raw.reshape(32, size, size).transpose(1, 2, 0)
    physical = raw.reshape(size, 2, size // 4, 16, 4)
    yield 'public_x_byte_c_half_word', physical.transpose(0, 2, 4, 1, 3).reshape(size, size, 32)
    # Split a 16-word tile into explicit factors. These candidates only test
    # which byte/word factors belong to X; a shared channel permutation remains
    # unidentifiable from average pooling alone.
    tiled = physical.reshape(size, 2, size // 4, 4, 4, 4)
    yield 'x_word_low_c_half_high_byte', tiled.transpose(0, 2, 4, 1, 3, 5).reshape(size, size, 32)
    yield 'x_word_high_c_half_low_byte', tiled.transpose(0, 2, 3, 1, 4, 5).reshape(size, size, 32)

skip_raw = np.fromfile(args.run / 'pre_skip.raw', dtype=np.uint8)
down_raw = np.fromfile(args.run / 'pre_down_candidate.raw', dtype=np.uint8)
results = []
for (name, skip), (other, down) in zip(layouts(skip_raw, 320), layouts(down_raw, 160)):
    if name != other:
        raise ValueError('Candidate mismatch')
    full, reference = decode(skip), decode(down)
    pooled = full.reshape(160, 2, 160, 2, 32).mean(axis=(1, 3))
    quantized = torch.from_numpy(pooled).to(torch.float8_e4m3fn).float().numpy()
    error = quantized - reference
    results.append({'candidate': name, 'mae': float(np.abs(error).mean()),
                    'rmse': float(np.sqrt(np.square(error).mean())),
                    'exact_value_fraction': float((quantized == reference).mean()),
                    'max_error': float(np.abs(error).max())})
report = {'results': results, 'layout_validated': False,
          'limitations': 'Pooling quantized skip differs from pooling native accumulators; channel order cannot be identified by this constraint.'}
(args.run / 'pre-pool-layout-probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
