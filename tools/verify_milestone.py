"""CPU-only source/evidence checks; never substitutes for GPU reproduction."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def main():
    rows=read(ROOT/'evidence/source-manifest-2026-09-14.json')['files']
    for row in rows:
        p=(ROOT/row['path']).resolve()
        assert p.is_relative_to(ROOT) and p.is_file()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        if row['byte_identical']:assert row['source_sha256']==row['sha256']
        else:assert row['transformation']
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
    data=read(ROOT/'evidence/fullsize-2026-09-14.json')
    assert all(v['passed'] and v['excerpt'] for v in data.values())
    full=data['fullsize'];assert full['frames']==243
    assert [v['index'] for v in full['rmse_by_frame']]==list(range(243))
    rmse=math.sqrt(sum(v['rmse']**2 for v in full['rmse_by_frame'])/243)
    assert math.isclose(rmse,2.1609184573,abs_tol=1e-8)
    b=data['optimized_benchmark']
    assert math.isclose(b['summary']['exact']/b['summary']['fast'],b['exact_over_fast'])
    assert len(b['worker_results'])==4
    for w in b['worker_results']:
        assert len(w['frames'])==32 and all(f['byte_equal'] for f in w['frames'])
        if w['mode']=='exact':
            assert w['backend']=='branched_exact_v1.session.Session'
            assert 'graph' in w['enabled'] and w['graph_entries']==2 and w['graph_calls']>0
    print(json.dumps(dict(passed=True,source_files=len(rows),fullsize_rmse=rmse,
        exact_over_fast=b['exact_over_fast'],scope='Source identity, syntax and evidence arithmetic; no GPU execution'),indent=2))


if __name__=='__main__':main()
