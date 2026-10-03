"""Finite, prewritten Luna audit of stored color-cause diagnostics; no reruns."""
import hashlib,json,math,sys,traceback
from datetime import datetime
from pathlib import Path
M=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/int8-face-color-cause-v1-monitor-luna-v1/request-from-main.json')
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
m=js(M);assert Path(m['log']).is_file(),'Known-launched log missing'
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+chr(10),encoding='utf-8')
if sys.argv[1:]==['--startup']:
    assert not Path(m['startup']).exists()
    write(m['startup'],dict(read_at=datetime.now().astimezone().isoformat(),manifest=str(M),log=m['log'],log_bytes=Path(m['log']).stat().st_size))
    print('Startup recorded');sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
a=dict(passed=False,phase='auditing',no_model_execution=True,no_rerun=True)
cache={}
def file(path,expected=None,size=None):
    p=Path(path).resolve();key=str(p).casefold()
    if key not in cache:
        h=hashlib.sha256();n=0
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
        cache[key]=dict(path=str(p),sha256=h.hexdigest(),bytes=n)
    v=cache[key];assert expected is None or v['sha256']==expected,p
    assert size is None or v['bytes']==size,p
    return v
try:
    r=js(m['result']);l=js(m['lease'])
    assert l['owner']==m['owner'] and l['returncode']==0
    assert r['passed'] and r['phase']=='completed' and not r.get('error') and not r.get('finalization_error')
    assert r['targets']==[96,192] and len(r['cases'])==2 and len(r['saved_output_analysis'])==6
    for key in ('all_reviewed_outputs_reproduced','all16_gpu_ffns_match_cpu','both_cpu_body_replacements_match_gpu','no_default_changes','no_recalibration'):assert r[key]
    assert r['candidate_promoted'] is False and r['performance_benchmark'] is False and r['new_video'] is False
    for p,h in m['critical_sources'].items():file(p,h)
    for key in ('sources','exact_gate'):
        assert r[key]
        for p,h in r[key].items():file(p,h)
    def arrays(x):
        if isinstance(x,dict):
            if {'path','sha256','stored_bytes'}<=x.keys():file(x['path'],x['sha256'],x['stored_bytes'])
            for v in x.values():arrays(v)
        elif isinstance(x,list):
            for v in x:arrays(v)
    arrays(r['cases']);summary=[]
    for expected,c in zip([96,192],r['cases']):
        assert c['frame']==expected and c['diagnostic_inputs_histories_seed_unchanged'] and c['cpu_replacement_reproduces_gpu']
        assert len(c['factorial'])==4 and len(c['interventions'])==13
        assert [x['block'] for x in c['ffn_clip_statistics']]==list(range(8))
        for x in c['ffn_clip_statistics']:
            assert x['cpu_matches_gpu'] and 0<=x['clip_fraction']<=1
        baseline=c['factorial']['B_int8_body_fp16_history']['versus_fp16']['rgb_rmse_8bit']
        for name,v in c['interventions'].items():
            error=v['versus_fp16']['rgb_rmse_8bit'];assert math.isfinite(error) and error>=0
            if baseline:assert math.isclose(v['roi_rgb_rmse_change_percent_vs_int8_same_history'],(error/baseline-1)*100,abs_tol=1e-9)
        assert c['interventions']['fp16_all']['versus_fp16']['rgb_rmse_8bit']==0
        ranked=sorted(((name,v['versus_fp16']['rgb_rmse_8bit']) for name,v in c['interventions'].items() if name!='fp16_all'),key=lambda x:x[1])
        summary.append(dict(frame=expected,factorial=c['factorial_decomposition'],clip_fraction_by_block=[x['clip_fraction'] for x in c['ffn_clip_statistics']],intervention_roi_rgb_rmse_8bit_ranked=ranked))
    final=json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final['passed'] and final['frames']==[96,192]
    a.update(passed=True,phase='completed',primary={k:file(m[k]) for k in ('result','log','lease')},
        source_counts={k:len(r[k]) for k in ('sources','exact_gate')},unique_files_checked=len(cache),summary=summary,
        scope='Stored two-frame interventions, fixed-history comparisons and 16 GPU/CPU FFN checks; not performance or general visual approval')
except BaseException:a.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:write(m['handoff'],a)
print(json.dumps(dict(passed=True,handoff=m['handoff'],frames=[96,192]),ensure_ascii=False))
