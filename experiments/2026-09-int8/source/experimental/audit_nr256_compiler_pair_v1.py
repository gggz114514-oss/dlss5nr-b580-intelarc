"""Audit four serial compiler runs and their complete-output hash receipts.

This CPU audit authenticates source, lease, timing and saved GPU comparisons.
It does not independently execute the NR model. ABBA mitigates run-order drift;
the runtimes occupy separate processes, not simultaneous device execution.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('report_sha256',nargs=4);args=p.parse_args()
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/results')
OUT=D/'nr256-compiler-abba-v1';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
legacy_path=D/'nr256-native-parity-v1/validation.json'
assert sha(legacy_path)=='5008dfcf5316aabe984e1e30d1db1c933bb3536c5e4a54a3020390c4845ff007'
legacy=js(legacy_path);old=legacy['measured']['native_half']
sources={str(legacy_path):sha(legacy_path),str(Path(__file__)):sha(__file__)}
cases=[('372',0),('380',0),('380',1),('372',1)];runs=[];groups={'372':[],'380':[]};last_end=0
for (runtime,index),expected_digest in zip(cases,args.report_sha256):
    folder=D/f'nr256-compiler-{runtime}-r{index}-v1'
    path=folder/'validation.json';lease_path=folder.with_suffix('.log.lease.json')
    assert sha(path)==expected_digest
    r=js(path);lease=js(lease_path)
    assert r['passed'] and not r['complete_migration'] and lease['returncode']==0
    assert lease['started_unix']>=last_end;last_end=lease['finished_unix']
    assert all(sha(p)==h for p,h in r['sources'].items())
    sources.update(r['sources']);sources[str(path)]=sha(path);sources[str(lease_path)]=sha(lease_path)
    assert r['runtime']['selector']==runtime and r['runtime']['run']==index
    for flag in ('all_outputs_match_frozen_372','inputs_and_lut_unchanged','held_outputs_unchanged','reset_reproduces_first'):
        assert r[flag]
    rows=r['measured'];assert len(rows)==r['full_outputs_verified']==360
    assert r['input_rgb_sha256']==legacy['input_rgb_sha256'] and r['input_motion_sha256']==legacy['input_motion_sha256']
    assert r['native_4060_reference']==legacy['native_4060_reference']
    for row,target in zip(rows,old):
        for key in ('round','mode','sample','output_raw_sha256','seed'):assert row[key]==target[key]
        assert row['private_equal'] and row['caller_independent']
        assert math.isfinite(row['host_ms']) and row['host_ms']>0
    for mode in ('reset','temporal'):
        values=[s['host_ms'] for s in rows if s['mode']==mode];assert len(values)==180
        assert r['summary'][mode]==dict(mean_host_ms=statistics.mean(values),median_host_ms=statistics.median(values),
            round_mean_host_ms=[statistics.mean(s['host_ms'] for s in rows if s['mode']==mode and s['round']==j) for j in range(3)])
    assert len(r['graphs'])==2 and all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'])
    body=r['static_body_diagnostic'];assert len(body)==2 and {b['temporal'] for b in body}=={False,True}
    for b in body:
        assert len(b['samples_host_ms'])==b['complete_output_readbacks']==5 and b['graph_replays']==50
        assert all(math.isfinite(s) and s>0 for s in b['samples_host_ms'])
        assert b['median_host_ms']==statistics.median(b['samples_host_ms'])
    runs.append(dict(runtime=runtime,run=index,report_sha256=sha(path),lease_sha256=sha(lease_path),
        summary=r['summary'],static_body_diagnostic=body,started_unix=lease['started_unix'],finished_unix=lease['finished_unix']))
    groups[runtime].extend(rows)
summary={runtime:{mode:dict(mean_host_ms=statistics.mean(s['host_ms'] for s in rows if s['mode']==mode),
    median_host_ms=statistics.median(s['host_ms'] for s in rows if s['mode']==mode)) for mode in ('reset','temporal')}
    for runtime,rows in groups.items()}
old_ms=summary['372']['temporal']['mean_host_ms'];new_ms=summary['380']['temporal']['mean_host_ms']
native_ms=legacy['native_4060_reference']['temporal']['host_ms']['mean']
report=dict(passed=True,scope=__doc__,sources=sources,ordered_runs=runs,summary=summary,
    compiler_temporal_speedup=old_ms/new_ms,compiler_temporal_reduction_percent=(1-new_ms/old_ms)*100,
    complete_output_hash_receipts=1440,outputs_match_frozen_372=True,
    b580_over_4060_complete_host_ratio=new_ms/native_ms,native_4060_complete_host_ms=native_ms,
    no_tail_fusion=True,installed_runtime_unmodified=True,complete_migration=False)
OUT.mkdir();(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
