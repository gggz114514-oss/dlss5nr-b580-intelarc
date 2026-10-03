"""Relocated archived constructors using the public serial reproduction harness.

The immutable original PHASE2 runner is a reference, not a source of new results.
This adapter binds relative roots through JSON and retains its source/receipt
references. Missing original media never becomes a claimed historical replay.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import release_contracts as c
import run_current

def bindings(path):
    path=c.plain_path(path);value=c.read_json(path)
    c.require(value.get('schema')=='nr-release-historical-bindings-v1','wrong historical bindings schema')
    result=dict(value)
    for key in ('repo_root','runtime_root','assets','source_root','manifest','inputs','legacy_root','original_fixture'):
        if key in value and value[key] is not None:
            p=Path(value[key]);result[key]=str(c.plain_path(p if p.is_absolute() else path.parent/p))
    return result

def reference(value):
    result=dict(schema='nr-release-historical-reference-v1',original_runner_bytes_rewritten=False,
        execution_harness='tests/run_current.py; same frozen model/constructor; portable initialization',
        historical_result_claims_inherited=False,private_video_qualification=False,game_FPS=None)
    if value.get('legacy_root'):
        root=c.plain_path(value['legacy_root'])
        c.require(root.is_dir(),'legacy runner not extracted yet: '+str(root))
        result['original_runner_files']={name:c.sha(path) for name,path in c.all_files(root).items()
                                         if path.suffix in ('.py','.json','.md') and '__pycache__' not in path.parts}
    fixture=value.get('original_fixture')
    if fixture:
        result['original_fixture_reference']=c.file_record(fixture)
        result['original_fixture_replayed']=False
    else:
        result['original_private_fixed13']='not supplied; no historical quality/result equality claim'
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate',help='r18 arm/alias; matching frozen baseline runs first')
    p.add_argument('--arm',help='one archived arm')
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--plan',action='store_true')
    args=p.parse_args(argv)
    try:
        value=bindings(args.bindings);ref=reference(value)
        command=['--output',str(args.output),'--python',args.python]
        for key,flag in (('repo_root','--repo-root'),('runtime_root','--runtime-root'),('assets','--assets'),
                         ('source_root','--source-root'),('manifest','--manifest'),('inputs','--inputs')):
            if value.get(key):command += [flag,value[key]]
        selected=args.candidate or value.get('candidate')
        arm=args.arm or value.get('arm')
        c.require(not (selected and arm),'choose candidate pair or single arm')
        c.require(selected or arm,'historical candidate/arm required')
        command += ['--candidate',selected] if selected else ['--arm',arm]
        parsed=run_current.parser().parse_args(command)
        plan,selections=run_current.prepare_plan(parsed)
        plan['historical_reference']=ref
        plan['bindings_manifest']=c.file_record(args.bindings)
        if args.plan:
            print(json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False));return 0
        return run_current.execute(parsed,plan,selections)
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(completed=False,GPU_executed=False,error=str(exc)),ensure_ascii=False),file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())
