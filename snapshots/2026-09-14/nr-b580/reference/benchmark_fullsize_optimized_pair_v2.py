"""Same ABBA benchmark, with the validated optimized exact branch selected.

Keep v1 evidence intact. Require optimized scopes and actual graph use; verify
every output against frozen exact frames. No new precision or product change.
"""
import json
from pathlib import Path


def prepared_source():
    here=Path(__file__).resolve().parent
    text=(here/'benchmark_fullsize_pair_v1.py').read_text(encoding='utf-8')
    old='            from nr_exact_runtime_v1 import Session'
    new='''            from branched_exact_v1.session import Session
            evidence=read(DATA/'exact-branched-dimensions-v1/480x864-01/dimensions-validation.json')
            assert evidence['passed'] and evidence['geometry']==[480,864] and evidence['features']==['branched']
            Session.diagnostics=False
            assert {'graph','shortfp8','cubic','decoder','front','history','k8','layout','compact'} <= Session.enabled'''
    assert text.count(old)==1
    text=text.replace(old,new)
    old="                result=measure(lambda rgb,flow,reset:session.process(rgb,flow,reset=reset).color,'exact')"
    new=old+'''
                graph=session.adapters['graph']
                assert graph.calls>0 and len(graph.entries)>=2, 'Optimized graph path not active'
                result.update(backend='branched_exact_v1.session.Session',enabled=sorted(session.enabled),
                    diagnostics=session.diagnostics,graph_calls=graph.calls,graph_entries=len(graph.entries),
                    implementation_sha256=sha(PRODUCT/'branched_exact_v1/session.py'))'''
    assert text.count(old)==1
    text=text.replace(old,new)
    old="report=dict(passed=False,runs=[],order=['fast','exact','exact','fast'])"
    new="report=dict(passed=False,runs=[],order=['fast','exact','exact','fast'],exact_baseline='branched_exact_v1 optimized; not nr_exact_runtime_v1',supersedes_v1_comparison=True)"
    assert text.count(old)==1
    return text.replace(old,new)


if __name__=='__main__':
    exec(compile(prepared_source(),__file__,'exec'),dict(__name__='__main__',__file__=__file__))
