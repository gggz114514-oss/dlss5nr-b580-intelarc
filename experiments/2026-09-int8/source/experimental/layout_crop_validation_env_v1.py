"""Authenticate the selected runtime and all three independent layout/crop screens."""
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import urllib.error

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXACT=ROOT.parent/'nr-b580'
R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
ap=D/'experimental/short-fp8-checkpoint-v1/saved-audit-v1.json'
assert sha(ap)=='1413f26f3269ed8b7379763077314f3090de1467b76309d1af0ff063bb4db085'
audit=js(ap)
assert audit['passed']
sources=dict(audit['sources'])
for p,h in audit['new_files'].items():sources[str(ROOT/p)]=h
sources[str(ap)]=sha(ap)


def receipt(path,expected):
    path=Path(path)
    assert sha(path)==expected
    result=js(path)
    assert result['passed'] and not result.get('error') and not result.get('finalization_error')
    for p,h in result.get('sources',{}).items():
        assert sha(p)==h
        if p in sources:assert sources[p]==h
        sources[p]=h
    sources[str(path)]=expected
    return result


for folder,digest in (
    ('post-region-body-v1','37d793cfbae681344dda80185e0955065fc2612270e51a0bb155a096f7349086'),
    ('vit-head-layout-body-v1','95faf6bb6818627d30f1aa5cf53fd0b30c01050d0130e6705dce96af206a2c26'),
    ('c512-window-layout-body-v1','d538d84f920fd39b31403d49996b6eceb02f4c0576a33cd670309a5d11b66268')):
    report=receipt(D/'experimental'/folder/'validation.json',digest)
    assert report['all_outputs_byte_equal']
    lease=D/'experimental'/(folder+'.log.lease.json')
    assert js(lease)['returncode']==0
    sources[str(lease)]=sha(lease)
for p in (Path(__file__),HERE/'layout_crop_stack_v1.py'):
    sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
provision=TOOLCHAIN/'provision-v1.json'
assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision)['files'].items())
try:
    with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=3) as response:queue=json.load(response)
    assert not queue['queue_running'] and not queue['queue_pending']
    queue_check='running_and_empty'
except urllib.error.URLError as error:
    assert getattr(error.reason,'winerror',None)==10061 or getattr(error.reason,'errno',None)==10061
    queue_check='connection_refused_service_not_running'


def candidate_gates(stack):
    meta=stack.rewrite.metadata()
    assert len(stack.rewrite.builds)==meta['capture_scopes']==6
    assert meta['resources'] and all(v['spills']==0 for v in meta['resources'].values())
    route=meta['layout_crop']
    assert (route['c512_calls'],route['vit_calls'],route['post_calls'])==(96,48,6)
    assert len(route['c512_blocks'])==16 and set(route['c512_blocks'].values())=={6}
    assert len(route['vit_blocks'])==8 and set(route['vit_blocks'].values())=={6}
    for key in ('c512_resources','vit_resources'):
        assert route[key] and all(v['spills']==0 for v in route[key].values())
    assert len(route['post_geometries'])==6
    assert all(g['padded']==[264,264] and g['head']==[256,256,8] for g in route['post_geometries'])
    for build in stack.rewrite.builds:
        assert (build['triton_calls'],build['standalone_fp8'],build['quantization_calls'],build['elided_fp8'])==(651,180,395,215)
    stack.rewrite.verify_restored()
    return meta


def finalize_sources():
    assert all(sha(p)==h for p,h in sources.items())
    authenticate_main()
