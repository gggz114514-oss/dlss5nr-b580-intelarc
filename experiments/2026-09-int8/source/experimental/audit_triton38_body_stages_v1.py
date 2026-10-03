"""Authenticate completed graph-segment report and reread saved temporal inputs."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import compressed_arrays_v1 as arrays

OUT = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/triton38-body-stages-v1')
path, target = OUT/'validation.json', OUT/'saved-audit-v1.json'
assert not target.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(path)=='619714ceafb097336fcd8e8a69b7d8de1e560d0bac42ff947da5af0076895ff2'
report = json.loads(path.read_text(encoding='utf-8'))
assert report['stage_inputs_unchanged'] and not report['tail_fusion_installed']
verified={}
lease = json.loads(OUT.with_suffix('.log.lease.json').read_text(encoding='utf-8'))
assert lease['returncode'] == 0 and report['passed'] and report['assembled_byte_equal_prior']
assert report['lut_unchanged'] and report['layout_calls']['c32_chunk_pack']>0 and report['vit_projection_calls']>0
assert report['history_unchanged'] and report['vit_tokens'] == 64
assert all(sha(p) == digest for p,digest in report['sources'].items())
assert len(report['stages']) == 13
names = ['pre', *[f'encoder_C{c}' for c in (32,64,128,256,512)], 'ViT_8_blocks',
         'decoder_C512_with_input', *[f'decoder_C{c}' for c in (256,128,64,32)], 'post']
assert [r['name'] for r in report['stages']] == names
for row in report['stages']:
    assert row['capture_byte_equal'] and row['persistent_io_outside_pool'] and len(row['samples_seconds']) == 7
    for m in row['inputs']+row['outputs']:
        if m is not None:
            a=arrays.load(m);assert np.isfinite(a).all();verified[m['path']]=m['stored_bytes']
    assert [m['raw_sha256'] for m in row['outputs']]==row['output_sha256']
    assert all(0 < t < 1 for t in row['samples_seconds'])
    assert row['median_seconds'] == statistics.median(row['samples_seconds'])
assert len(report['full_body_samples_seconds']) == 7
assert report['full_body_median_seconds'] == statistics.median(report['full_body_samples_seconds'])
assert report['segment_median_sum_seconds'] == sum(r['median_seconds'] for r in report['stages'])
count = 0
for name, meta in report['body_inputs'].items():
    assert meta is not None, 'Expected a real temporal body'
    value = arrays.load(meta)
    assert np.isfinite(value).all()
    count += 1
expected = arrays.load(report['expected_output'])
assert expected.shape == (256,256,3) and expected.dtype == np.dtype('f2')
assert hashlib.sha256(expected.tobytes()).hexdigest() == report['stages'][-1]['output_sha256'][0]
summary = dict(passed=True, report_sha256=sha(path), audit_source_sha256=sha(__file__),
               lease_sha256=sha(OUT.with_suffix('.log.lease.json')), saved_arrays_reread=count+1, stage_unique_arrays_reread=len(verified), stage_array_stored_bytes=sum(verified.values()),
               whole_body_median_ms=report['full_body_median_seconds']*1000,
               segment_median_sum_ms=report['segment_median_sum_seconds']*1000,
               limitation='Intermediate graph/eager byte checks occurred in the GPU process; '
                          'CPU audit authenticates source and saved inputs/output, not a CPU NR recomputation.')
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
