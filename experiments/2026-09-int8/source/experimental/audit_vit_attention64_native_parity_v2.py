"""Audit the saved same-scope NR256 timing and complete-output receipts.

Reread the saved reset tensor and native input/motion/output arrays. Every timed
candidate output was compared in full during the GPU run; this CPU audit checks
all resulting hashes and state records, not an independent model arithmetic
replay. Device body event intervals include scheduling and are not busy time or
a complete NR GPU timing suitable for a cross-device speed ratio.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
out=D/'results/vit-attention64-native-parity-v2'
path=out/'validation.json';target=out/'saved-audit-v1.json'
assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
parser=argparse.ArgumentParser();parser.add_argument('report_sha256');args=parser.parse_args()
assert sha(path)==args.report_sha256
r=js(path);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
for flag in ('inputs_unchanged','reset_reproduces_first','all_candidate_outputs_byte_equal',
             'held_outputs_unchanged','lut_bytes_unchanged','body_event_timing_is_not_complete_nr_gpu_time'):
    assert r[flag]
native_path=D/'experimental/native-steady-bench-v1/validation.json'
assert sha(native_path)=='70a86900dbde71d285f187cedc42a123bc7706cf27f21f0231e515b4d964659d'
native=js(native_path)
assert native['passed'] and native['cases']['256x256']['samples']==360
assert native['cases']['256x256']['reset_matches_previous_native_rgba_bytes']
assert r['native_4060_reference']==native['cases']['256x256']['statistics']
paired_path=D/'results/vit-attention64-residual256-v2/validation.json'
assert sha(paired_path)=='8ff85699fdf7c35e7273787e2f56fd3917c83cbbc1694011354b44f7a33b69b3'
paired_audit=js(paired_path.with_name('saved-audit-v1.json'))
assert paired_audit['passed'] and paired_audit['report_sha256']==sha(paired_path)

files={Path(p).name:Path(p) for p in r['sources'] if p.endswith('.bin')
       and Path(p).parent.parent.name=='real-flow-256x256-v3-sequence-20260908-180043'}
pixels=np.fromfile(files['frame00.png_input.rgba32f.bin'],'<f4').reshape(256,256,4)[...,:3].copy()
motion=np.fromfile(files['frame00.png_motion.rg32f.bin'],'<f4').reshape(256,256,2)
reference=np.fromfile(files['frame00.png_output.rgba32f.bin'],'<f4').reshape(256,256,4)[...,:3].copy()
assert digest(pixels)==r['input_rgb_sha256'] and digest(motion)==r['input_motion_sha256']
assert np.isfinite(pixels).all() and np.isfinite(reference).all() and not motion.any()
assert pixels.astype('f2').astype('f4').tobytes()==pixels.tobytes()
reset=arrays.load(r['reset_output'])
assert reset.shape==(256,256,3) and reset.dtype.str=='<f2' and np.isfinite(reset).all()
error=np.abs(reset.astype('f4')-reference)
assert r['existing_fast_difference_from_native_reset']==dict(mean_abs=float(error.mean()),
    max_abs=float(error.max()),byte_equal=reset.astype('f4').tobytes()==reference.tobytes())

expected=[(j,('reset','temporal')[(position+j)%2],i)
          for j in range(3) for position in range(2) for i in range(60)]
assert set(r['measured'])=={'previous','vit_attention'}
summary={};pools=set()
for name,rows in r['measured'].items():
    assert len(rows)==360
    for row,(j,mode,i) in zip(rows,expected):
        assert (row['round'],row['mode'],row['sample'])==(j,mode,i)
        assert row['seed']==(1 if mode=='reset' else 22+i)
        assert row['private_equal'] and row['caller_independent']
        assert math.isfinite(row['host_ms']) and row['host_ms']>0
        assert len(row['output_raw_sha256'])==64
        if mode=='reset':assert row['output_raw_sha256']==digest(reset)
    summary[name]={mode:dict(mean_host_ms=statistics.mean(s['host_ms'] for s in rows if s['mode']==mode),
        median_host_ms=statistics.median(s['host_ms'] for s in rows if s['mode']==mode),
        round_mean_host_ms=[statistics.mean(s['host_ms'] for s in rows if s['mode']==mode and s['round']==j)
                           for j in range(3)]) for mode in ('reset','temporal')}
    graphs=r['graphs'][name];assert len(graphs)==2
    assert sorted(g['replays'] for g in graphs)==[241,248]
    assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in graphs)
    pools.update(g['pool'] for g in graphs)
    assert [g['captured_dispatch'] for g in graphs]==[g['captured_dispatch'] for g in r['graphs']['previous']]
assert summary==r['summary'] and len(pools)==1 and r['complete_outputs_compared']==720
assert r['all_outputs_match_frozen_372'] and r['fused_history_builds']==1 and r['vit_qkv_calls']>0
assert len(r['dense_tiles_calls'])==17 and sum(r['dense_tiles_calls'].values())==498
assert r['native_cubic_calls']==dict(c32={'102400':6,'25600':12,'28224':12,'26880':24,'107584':6},batched={'576x256':96},split={'144x512':96})
assert r['nr_input']==[256,256]
assert r['fused_vit_attention_calls']==r['vit_qkv_calls']==48
legacy_path=D/'results/nr256-native-parity-v1/validation.json'
assert sha(legacy_path)=='5008dfcf5316aabe984e1e30d1db1c933bb3536c5e4a54a3020390c4845ff007'
legacy=js(legacy_path)
for name,rows in r['measured'].items():
    assert [q['output_raw_sha256'] for q in rows]==[q['output_raw_sha256'] for q in legacy['measured']['native_half']]
for old,new in zip(r['measured']['previous'],r['measured']['vit_attention']):
    assert old['output_raw_sha256']==new['output_raw_sha256']

body={}
for name,entries in r['static_body_diagnostic'].items():
    assert len(entries)==2 and {e['temporal'] for e in entries}=={False,True}
    body[name]={}
    for entry in entries:
        samples=entry['samples'];assert len(samples)==10 and entry['output_unchanged']
        assert all(math.isfinite(s[k]) and 0<s[k]<10000 for s in samples for k in ('event_interval_ms','host_ms'))
        body[name]['temporal' if entry['temporal'] else 'reset']={
            k:statistics.mean(s[k] for s in samples) for k in ('event_interval_ms','host_ms')}
assert set(body)=={'previous','vit_attention'}
audit=dict(passed=True,report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    sources_authenticated=len(r['sources']),complete_output_receipts=720,
    native_input_motion_output_and_fast_reset_reread=True,
    summary=summary,body_diagnostic=body,
    b580_over_4060_complete_host_ratio=summary['vit_attention']['temporal']['mean_host_ms']/r['native_4060_reference']['temporal']['host_ms']['mean'],
    native_equivalence_claim_for_fast_mode=False,limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2),flush=True)
