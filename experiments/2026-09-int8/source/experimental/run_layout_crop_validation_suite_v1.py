"""Main-prepared serial validation; Luna observes and never launches its stages.

Run residual/guard validation, then 390-frame equivalence only if stage one
passes. Both children stay inside the caller's bounded Windows GPU Job lease.
No automatic fixes, retries, promotion, video encoding or new frame dumps.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'results/layout-crop-validation-suite-v1'
STAGES=(
    ('residual','benchmark_layout_crop_residual256_v1.py','layout-crop-residual256-v1'),
    ('long','validate_layout_crop_long1080_v1.py','layout-crop-long1080-v1'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert not OUT.exists()
assert all(not (D/'results'/folder).exists() for _,_,folder in STAGES)
sources={str(p):sha(p) for p in (Path(__file__),HERE/'Run-LayoutCropValidationSuiteV1.cmd',
    HERE/'benchmark_layout_crop_residual256_v1.py',HERE/'validate_layout_crop_long1080_v1.py',
    HERE/'layout_crop_stack_v1.py',HERE/'layout_crop_validation_env_v1.py')}
OUT.mkdir()
report=dict(scope='serial_layout_crop_validation',description=__doc__,sources=sources,
    started_unix=time.time(),passed=False,complete_migration=False,candidate_promoted=False,
    phases=[],active_phase=None,new_videos=False,new_frame_arrays=False)
save=lambda:(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
save()
try:
    for name,script,folder in STAGES:
        assert all(sha(p)==h for p,h in sources.items())
        command=[sys.executable,'-B','-X','utf8',str(HERE/script)]
        report['active_phase']=name
        save()
        phase=dict(name=name,command=command,started_unix=time.time())
        print('Starting validation phase: '+name,flush=True)
        process=subprocess.Popen(command,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        phase['pid']=process.pid
        phase['returncode']=process.wait()
        phase['finished_unix']=time.time()
        result_path=D/'results'/folder/'validation.json'
        if result_path.exists():
            result=js(result_path)
            phase['result']=dict(path=str(result_path),sha256=sha(result_path),bytes=result_path.stat().st_size)
            phase['passed']=bool(result.get('passed')) and not result.get('error') and not result.get('finalization_error')
        else:
            phase['passed']=False
        report['phases'].append(phase)
        report['active_phase']=None
        save()
        assert phase['returncode']==0 and phase['passed'],('Phase failed; no retry or next stage',name)
        if name=='residual':
            assert result['full_outputs_verified']==78 and len(result['progress_fallback'])==2
            assert len(result['guards'])==2 and result['lut_graph_guards']
            report['residual_summary']=result['summary']
            report['residual_change_percent']=result['change_percent']
        else:
            assert result['frames_completed']==390 and result['all_outputs_match_approved_sequence']
            assert result['reset_reproduces_first_frame'] and result['independent_uninterrupted_history']
            report['long_frames_verified']=390
        save()
        print('Completed validation phase: '+name,flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    report['passed']=True
except BaseException:
    report['error']=traceback.format_exc()
    raise
finally:
    report['finished_unix']=time.time()
    save()
print(json.dumps(dict(passed=report['passed'],residual_change_percent=report['residual_change_percent'],
    long_frames=report['long_frames_verified'],candidate_promoted=False)),flush=True)
