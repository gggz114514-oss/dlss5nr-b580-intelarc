"""Recalculate published timing statistics. No model or device imports."""
from pathlib import Path
import hashlib,json,math,statistics

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def mean_check(events):
    if not isinstance(events,dict) or 'samples_ms' not in events:return 0
    xs=events['samples_ms']
    assert xs and all(type(n) in (int,float) and math.isfinite(n) and n>0 for n in xs)
    assert len(xs)==events['count']
    assert math.isclose(statistics.fmean(xs),events['mean_ms'],abs_tol=1e-8)
    ordered=sorted(xs)
    if 'p50_ms' in events:
        assert math.isclose(statistics.median(xs),events['p50_ms'],abs_tol=1e-8)
    if 'p95_ms' in events:
        assert math.isclose(ordered[math.ceil(.95*len(xs))-1],events['p95_ms'],abs_tol=1e-8)
    return len(xs)
def main():
    index=read(ROOT/'evidence/2026-10-03/performance/index.json')
    measurements=samples=0
    for attempt in index['attempts']:
        result=ROOT/attempt['result_path']
        assert hashlib.sha256(result.read_bytes()).hexdigest()==attempt['result_sha256']
        for row in attempt['measurements']:
            p=ROOT/row['path'];assert p.resolve().is_relative_to(ROOT)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
            data=read(p)
            samples+=mean_check(data.get('whole_modes_timing',{}).get('warm_events',{}))
            samples+=mean_check(data.get('raw',{}))
            measurements+=1
    print(json.dumps(dict(passed=True,measurements=measurements,timing_samples=samples,GPU_executed=False,
        scope='Published timing arithmetic and identities; no new speed, quality or game validation'),indent=2))
if __name__=='__main__':main()
