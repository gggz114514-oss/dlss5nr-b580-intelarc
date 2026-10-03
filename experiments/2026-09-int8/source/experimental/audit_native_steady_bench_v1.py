"""Audit native GPU timestamp samples, staged sources, and complete reset outputs.

This is a fixed input/zero-motion microbenchmark of the original DLL on one
4060 Laptop, not a game FPS measurement or a moving-video quality comparison.
"""
import hashlib,json,math,statistics,zipfile
from pathlib import Path,PureWindowsPath
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE.parent.parent/'nr-b580/reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/native-steady-bench-v1')
target=D/'validation.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
source=js(D/'sources.json');transport=js(D/'resume-transport.json');r=js(D/'native/run.json')
assert r['passed'] and r['sessionId']>0 and transport['returncode']==0
assert sha(HERE/'resume_native_steady_bench_v1.py')==transport['runner_sha256']
assert sha(D/'remote-command.ps1')==transport['script_sha256']
assert all(sha(p)==h for p,h in source['sources'].items())
assert sha(D/'stage.zip')==source['archive_sha256']
with zipfile.ZipFile(D/'stage.zip') as z:
    for name,h in source['stage_files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
assert r['runtimeSha256'].lower()=='6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
frozen={**source['sources'],str(Path(__file__)):sha(__file__)}
for p in [D/'sources.json',D/'resume-transport.json',D/'native/run.json',D/'resume-stdout.log']:
    frozen[str(p)]=sha(p)
for item in r['files']:
    p=D/'native'/item['relative'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'].lower()
    frozen[str(p)]=sha(p)
summaries={}
for case,version in zip(r['cases'],[3,2,3]):
    dimension=case['dimension'];width,height=map(int,dimension.split('x'))
    p=D/'native'/dimension/'native-timing.json';timing=js(p)
    assert case['exitCode']==0 and all('NVIDIA GeForce RTX 4060 Laptop GPU, 616.86' in text for text in case['before']+case['after'])
    assert timing['passed'] and (timing['width'],timing['height'])==(width,height)
    assert timing['warmup_per_block']==20 and timing['measured_per_block']==60 and timing['rounds']==3
    assert not timing['capture'] and not timing['kernel_trace']
    assert timing['resident_inputs'] and timing['fixed_image_zero_motion'] and timing['reset_byte_equal'] and timing['input_unchanged']
    assert timing['mean_abs_change']>0 and timing['output_max']>timing['output_min']
    fixture=R/f'inputs/flow-full-{dimension}-v{version}'
    m=js(fixture/'manifest.json');input_path=fixture/m['frames'][0]['file']
    assert case['inputSha256'].lower()==sha(input_path)
    old_path=R/f'4060-real-flow-{dimension}-sequence-v{version}.json';old=js(old_path)
    assert old['completedAllInputs'] and old['exitCode']==0
    old_file=next(f for f in old['outputs'] if f['name']=='frame00.png_output.rgba32f.bin')
    native_ref=R/'results'/PureWindowsPath(old['runDirectory']).name/'output'/old_file['name']
    assert sha(native_ref)==old_file['sha256'].lower()
    new_path=p.with_name('reset-output.rgba32f.bin')
    a=np.fromfile(new_path,'<f4').reshape(height,width,4);b=np.fromfile(native_ref,'<f4').reshape(height,width,4)
    assert np.isfinite(a).all() and a.tobytes()==b.tobytes()
    assert sha(new_path)==old_file['sha256'].lower()
    frozen[str(old_path)]=sha(old_path);frozen[str(native_ref)]=sha(native_ref)
    samples=timing['samples'];assert len(samples)==360
    expected=[(r,('reset','temporal')[(j+r)%2],i) for r in range(3) for j in range(2) for i in range(60)]
    assert [(s['round'],s['mode'],s['sample']) for s in samples]==expected
    assert all(math.isfinite(s[k]) and 0<s[k]<10000 for s in samples for k in ('gpu_ms','host_ms'))
    summary={}
    for mode in ('reset','temporal'):
        values=[s for s in samples if s['mode']==mode]
        summary[mode]={}
        for key in ('gpu_ms','host_ms'):
            v=[s[key] for s in values]
            summary[mode][key]=dict(mean=statistics.mean(v),median=statistics.median(v),
                p95_nearest_rank=sorted(v)[math.ceil(.95*len(v))-1],minimum=min(v),maximum=max(v),
                round_means=[statistics.mean(s[key] for s in values if s['round']==round) for round in range(3)])
    summaries[dimension]=dict(samples=360,reset_matches_previous_native_rgba_bytes=True,statistics=summary)
report=dict(passed=True,scope=__doc__,sources=frozen,cases=summaries,measured_calls=1080,
    gpu_timing_scope='D3D12 timestamps directly around original NR evaluate command recording.',
    host_timing_scope='Begin/record/submit, GPU completion fence and timestamp readback; inputs already resident.',
    excluded='Model creation; image load/color conversion/upload; flow estimation; capture; output readback/file IO; composition; presentation; game contention.',
    limitation='One fixed source image per dimension, zero motion, default SDR controls;180 reset and180 temporal samples each. Repeated same-frame history, not390-frame real video.',
    current_b580_same_scope_comparison=False,new_b580_quality_change=False,complete_migration=False)
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,measured_calls=1080,summary=summaries,report_sha256=sha(target)),indent=2),flush=True)
