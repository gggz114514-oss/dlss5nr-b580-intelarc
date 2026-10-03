"""Locate raw pre producer ranges; leave the physical-to-logical layout unproven."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stats(data):
    counts = Counter(data)
    pairs = []
    for code, count in counts.items():
        exp, mantissa = (code >> 3) & 15, code & 7
        if exp == 15 and mantissa == 7:
            continue
        value = mantissa * 2**-9 if exp == 0 else (1 + mantissa / 8) * 2**(exp - 7)
        if code & 128:
            value = -value
        pairs.append((value, count))
    finite = sum(count for _, count in pairs)
    return {'sha256': sha(data), 'bytes': len(data), 'nonzero_bytes': len(data) - counts[0],
            'e4m3_assumption': {'nan_codes': counts[127] + counts[255],
                              'min': min(value for value, _ in pairs), 'max': max(value for value, _ in pairs),
                              'mean': sum(value * count for value, count in pairs) / finite,
                              'abs_mean': sum(abs(value) * count for value, count in pairs) / finite}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    trace = args.run / 'output/nvapi-trace'
    run = json.loads((args.run / 'run.json').read_text(encoding='utf-8-sig'))
    if not run['completedAllInputs']:
        raise ValueError('Trace did not complete')
    for item in run['traceFiles']:
        if sha((trace / item['name']).read_bytes()) != item['sha256'].lower():
            raise ValueError('Trace hash mismatch')
    events = [json.loads(line) for line in (trace / 'events.jsonl').read_text().splitlines()]
    pre_launches = [e for e in events if e['event'] == 'launch' and e['name'] == 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8']
    if len(pre_launches) != 1:
        raise ValueError('This experiment requires exactly one pre launch')
    payload = (trace / pre_launches[0]['file']).read_bytes()
    inp_w, inp_h = struct.unpack_from('<II', payload, 0xd0)
    skip_va = struct.unpack_from('<Q', payload, 0xd8)[0]
    pad_w, pad_h, down_va, down_w, down_h = struct.unpack_from('<IIQII', payload, 0xf0)
    snapshots = {e['label']: e for e in events if e['event'] == 'buffer_scheduled'}
    after, before = snapshots['after-pre'], snapshots['before-post']
    if after['gpu_va'] != before['gpu_va'] or after['bytes'] != before['bytes']:
        raise ValueError('Different arenas')
    buffers = [(trace / e['file']).read_bytes() for e in (after, before)]
    if any(len(data) != after['bytes'] for data in buffers):
        raise ValueError('Wrong arena size')
    skip_offset, down_offset = skip_va - after['gpu_va'], down_va - after['gpu_va']
    skip_size = down_offset - skip_offset
    # This second size is a shape-based candidate. Retain that distinction.
    down_size = down_w * down_h * 32
    if not 0 <= skip_offset < down_offset < down_offset + down_size <= after['bytes']:
        raise ValueError('Regions out of arena')
    regions = []
    for name, offset, size, evidence in [('pre_skip', skip_offset, skip_size, 'bounded by two captured output addresses'),
                                          ('pre_down_candidate', down_offset, down_size, 'captured dimensions times candidate 32 byte channels')]:
        a, b = [data[offset:offset + size] for data in buffers]
        (args.run / (name + '.raw')).write_bytes(a)
        regions.append({'name': name, 'offset': offset, 'bytes': size, 'range_evidence': evidence,
                        'after_pre': stats(a), 'before_post': stats(b), 'bytes_equal': a == b,
                        'changed_bytes': sum(x != y for x, y in zip(a, b)),
                        'logical_layout_validated': False})
    result = {'input_dimensions': [inp_w, inp_h], 'pre_dimensions': [pad_w, pad_h],
              'down_dimensions': [down_w, down_h], 'arena_gpu_va': after['gpu_va'], 'arena_bytes': after['bytes'],
              'skip_byte_count_matches_padded_32_channels': skip_size == pad_w * pad_h * 32,
              'regions': regions, 'copy_completion': 'host D3D12 queue fence completed before CPU mapping',
              'private_kernel_visibility_validated_independently': False,
              'logical_layout_validated': False, 'native_equivalence_validated': False}
    (args.run / 'arena-analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
