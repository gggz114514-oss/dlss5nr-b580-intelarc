"""CPU reread of every complete1080 output/motion and independently decoded input."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import numpy as np
import compressed_arrays_v1 as arrays

DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', required=True, choices=['baseline', 'fp16_xmx', 'int8_fused_v2'])
a = p.parse_args()
out = DREF / f'results/long-precision-1080-{a.mode}-v1'
path, audit_path = out / 'validation.json', out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
rawsha = lambda value: hashlib.sha256(value.tobytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
r = js(path)
assert r['passed'] and r['mode'] == a.mode and r['frames_completed'] == len(r['frames']) == 390
assert r['dimension'] == '1920x1080' and r['fps'] == '60000/1001'
assert not r['native_capture_for_this_sequence'] and not r['complete_migration']
lease_path = Path(str(out) + '.log.lease.json')
assert js(lease_path)['returncode'] == 0
for source, digest in r['sources'].items():
    assert sha(source) == digest
teacher = None if a.mode == 'baseline' else js(DREF / 'results/long-precision-1080-baseline-v1/validation.json')
ffmpeg = next(path for path in r['sources'] if Path(path).name == 'ffmpeg.exe')
decoder = subprocess.Popen([ffmpeg, '-v', 'error', '-threads', '2', '-i', r['source'], '-vf', r['decode_filter'],
    '-frames:v', '390', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=0x08000000)
verified = {}
mses = []

def read_frame():
    chunks, remaining = [], 1080 * 1920 * 3
    while remaining:
        data = decoder.stdout.read(remaining)
        if not data:
            raise EOFError('Incomplete independent decode')
        chunks.append(data)
        remaining -= len(data)
    return b''.join(chunks)

try:
    for i, row in enumerate(r['frames']):
        assert row['frame'] == row['source_frame'] == i and row['reset'] == (i == 0)
        assert row['next_seed'] == i + 1 and row['graph_replays'] == i + 3
        assert hashlib.sha256(read_frame()).hexdigest() == row['input_rgb8_sha256']
        value, flow = arrays.load(row['output']), arrays.load(row['motion'])
        assert value.dtype == flow.dtype == np.dtype('f2')
        assert value.shape == (1080, 1920, 3) and flow.shape == (1080, 1920, 2)
        assert np.isfinite(value).all() and np.isfinite(flow).all()
        assert row['private'] == row['output'] and row['private_byte_equal_output'] and row['finite']
        assert row['runtime_dispatch']['xpu_graph_replay'] == 1
        if i < 2:
            assert row['warmup_repeat_equal']
        if i == 0:
            assert flow.tobytes() == np.zeros_like(flow).tobytes()
        if teacher is not None:
            old = teacher['frames'][i]
            assert row['input_rgb8_sha256'] == old['input_rgb8_sha256']
            assert flow.tobytes() == arrays.load(old['motion']).tobytes()
            expected = arrays.load(old['output'])
            difference = value.astype('f4') - expected.astype('f4')
            mse = float(np.square(difference.astype('f8')).mean())
            assert mse == row['mse']
            assert row['psnr_db'] == (None if mse == 0 else float(-10 * np.log10(mse)))
            assert row['mae'] == float(np.abs(difference).mean())
            assert row['max_abs'] == float(np.abs(difference).max())
            mses.append(mse)
        for meta in (row['output'], row['motion']):
            verified[meta['path']] = meta['stored_bytes']
    assert decoder.stdout.read() == b''
    assert decoder.wait(timeout=30) == 0, decoder.stderr.read().decode(errors='replace')
finally:
    if decoder.poll() is None:
        decoder.kill()
        decoder.wait()
assert len(r['graphs']) == 2 and len({g['pool'] for g in r['graphs']}) == 1
assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'])
assert r['reset_reproduces_first_frame'] and r['held_outputs_survive_replay']
assert r['independent_history'] and r['caller_ownership_guards_passed']
assert sum(verified.values()) == r['unique_used_artifact_bytes']
assert statistics.mean(row['seconds'] for row in r['frames']) == r['diagnostic_mean_model_seconds']
audit = dict(passed=True, mode=a.mode, report_sha256=sha(path), auditor_sha256=sha(Path(__file__)),
    lease_sha256=sha(lease_path), process_exit_code=0, full_frames_verified=390,
    complete_output_and_motion_arrays_verified=780, independently_decoded_input_frames=390,
    unique_used_artifact_bytes=sum(verified.values()), complete_migration=False,
    native_capture_for_this_sequence=False,
    private_scope='Runtime compared full private RGB against returned RGB. This audit rereads its shared full artifact, not device memory.')
if mses:
    aggregate_mse = statistics.mean(mses)
    audit.update(aggregate_psnr_db=None if aggregate_mse == 0 else float(-10 * np.log10(aggregate_mse)),
        worst_psnr_db=min(row['psnr_db'] for row in r['frames'] if row['psnr_db'] is not None),
        max_abs=max(row['max_abs'] for row in r['frames']),
        complete_output_byte_identical_frames=sum(row['mse'] == 0 for row in r['frames']))
with audit_path.open('x', encoding='utf-8', newline='\n') as f:
    f.write(json.dumps(audit, indent=2, allow_nan=False) + '\n')
print(json.dumps(audit), flush=True)
