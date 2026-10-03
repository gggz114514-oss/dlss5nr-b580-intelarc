"""Portable serial offline-precompile -> fresh DiskOnly XPU reproduction.

--plan is CPU/stdlib only. GPU and NumPy imports are confined to explicitly
launched children. No game/native bridge, global environment or source writes.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_contracts as c
from portable_runtime import (write_json, validate_asset_bundle, runtime_dependencies,
                              source_selection, build_view)

ALIASES = dict(fdp='audit_impl_fdp_all', native_qkv='audit_impl_c512_native_qkv_720_v2',
    rowonce_vit='audit_impl_vit_ffn_rowonce_bk64_full_720_v1',
    swin_serial='audit_impl_swin_c64_serial_720_v3', swin_split='audit_impl_swin_c256_split_tail_720_v3',
    swin_serial_split='audit_impl_swin_c64_serial_c256_split_720_v3')
RUNNER_FILES = ('run_current.py','gpu_reproduction.py','release_contracts.py','portable_runtime.py',
                'reproduction_cache.py','runtime_support/fast_cached_runtime_v1.py',
                'runtime_support/portable_cache.py','runtime_support/runtime_environment.py',
                'runtime_support/SOURCE_PINS.json')

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=c.REPO)
    p.add_argument('--runtime-root', type=Path, help='read-only source runtime/dependency root; defaults to current/runtime')
    p.add_argument('--assets', type=Path, help='external assets.bundle directory containing assets.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--height720', dest='height', action='store_const', const=720)
    p.add_argument('--frames', type=int, default=13)
    p.add_argument('--frames13', dest='frames', action='store_const', const=13)
    p.add_argument('--repeats', type=int, default=3, help='measured repeated13 cycles after two warm cycles')
    p.add_argument('--repeats3', dest='repeats', action='store_const', const=3)
    p.add_argument('--python', default=sys.executable, help='dedicated Torch-XPU + NumPy Python executable')
    p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--inputs', type=Path, help='external nr-release-inputs-v1 NPY manifest; otherwise fixed-seed synthetic')
    p.add_argument('--source-manifest', type=Path, help='installed current source inventory')
    p.add_argument('--source-root', type=Path, help='frozen full source tree for archived arm reproduction')
    p.add_argument('--manifest', type=Path, help='frozen arm registry/source inventory')
    p.add_argument('--arm', help='one archived arm, optionally --baseline paired results')
    p.add_argument('--candidate', help='r18 arm or fdp/native_qkv/rowonce_vit/swin_serial/swin_split aliases; runs matching baseline first')
    p.add_argument('--baseline', type=Path, help='optional matching complete readonly RESULT.json')
    p.add_argument('--plan', action='store_true', help='CPU validation/preparation plan only; no GPU/NumPy imports or writes')
    p.add_argument('--gpu-lock', type=Path, help='shared exclusive serial lock; defaults to OS temp nr-release-xpu0.lock')
    p.add_argument('--timeout', type=int, default=14400, help='per child seconds; failure stops every later launch')
    p.add_argument('--_child', choices=('precompile','readonly'), help=argparse.SUPPRESS)
    p.add_argument('--_compare', nargs=3, metavar=('LEFT','RIGHT','REPORT'), help=argparse.SUPPRESS)
    p.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--config-sha256', help=argparse.SUPPRESS)
    return p

def matching_baseline(manifest, candidate):
    arms = manifest['arms']
    c.require(candidate in arms, 'unknown candidate: '+candidate)
    controls = c.validate_controls(arms[candidate].get('controls',{}))
    matches = [name for name,row in arms.items() if row.get('baseline') is True and
               c.validate_controls(row.get('controls',{})) == controls]
    c.require(matches, 'no frozen baseline with exactly matching controls')
    return 'accepted_baseline' if 'accepted_baseline' in matches else matches[0]

def prepare_plan(args):
    c.require(args.height == 720 and args.frames == 13, 'this entry is exact current720 fixed13 only')
    c.require(type(args.repeats) is int and 1 <= args.repeats <= 20, 'repeats must be 1..20')
    c.require(0 <= args.seed <= 0xffffffff and args.timeout > 0, 'invalid seed/timeout')
    c.require(not (args.candidate and args.arm), '--candidate and --arm are mutually exclusive')
    c.require(args.output is not None and args.assets is not None, '--output and external --assets are required (see export_assets.py)')
    repo = c.plain_path(args.repo_root)
    runtime = c.plain_path(args.runtime_root or repo/'current/runtime')
    assets, output = c.plain_path(args.assets), c.plain_path(args.output)
    c.require(not output.exists(), 'choose a fresh output directory')
    c.require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in (repo,assets,runtime)),
              '--output must be outside published source, runtime and assets roots')
    source_manifest = c.plain_path(args.source_manifest or repo/'evidence/2026-10-03/current-source-manifest.json')
    source_root, manifest = args.source_root, args.manifest
    if args.candidate:
        source_root = c.plain_path(source_root or repo/'experiments/2026-10-03/r18/source')
        manifest = c.plain_path(manifest or c.HERE/'registry/r18.json')
        selected = ALIASES.get(args.candidate,args.candidate)
        baseline = matching_baseline(c.read_json(manifest),selected)
        arms = [baseline,selected] if selected != baseline else [baseline]
    else:
        arms = [args.arm]
    asset_report = validate_asset_bundle(assets)
    dependencies, missing = runtime_dependencies(runtime,assets)
    c.require(not missing, 'runtime dependencies absent: '+', '.join(missing))
    selections, receipts = [], []
    for arm in arms:
        sources, receipt = source_selection(repo,source_manifest,source_root,manifest,arm)
        selections.append((arm or 'installed_current', sources, receipt)); receipts.append(receipt)
    if args.inputs: c.validate_inputs(args.inputs)
    return dict(schema='nr-release-reproduction-plan-v1', GPU_executed=False,
        source_receipts=receipts, assets=asset_report, runtime_root=str(runtime), output=str(output),
        runtime_dependency_file_count=len(dependencies), arms=[s[0] for s in selections],
        input=dict(height=720,width=1280,frames=13,seed=args.seed,external=None if args.inputs is None else c.file_record(args.inputs)),
        warm_cycles=2, measured_cycles=args.repeats, raw_sameframe_events=50,
        launch_order=[dict(arm=s[0],phase=phase) for s in selections for phase in ('precompile','readonly')],
        scenario='external NPY' if args.inputs else 'synthetic fixed-seed smoke',
        private_quality_qualification=False, game_FPS=None), selections

@contextmanager
def serial_lock(path, output):
    path = c.plain_path(path)
    c.require(path.parent.is_dir(), 'GPU lock parent must exist')
    try:
        fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise c.ContractError('another serial GPU owner or stale lock exists: '+str(path)+'; inspect it, never kill unrelated processes')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(dict(schema='nr-release-serial-lock-v1',pid=os.getpid(),output=str(output)),stream)
        yield
    finally:
        # Remove only the exact lock created/owned by this supervisor.
        try:
            owner = c.read_json(path)
            if owner.get('pid') == os.getpid() and owner.get('output') == str(output): path.unlink()
        except OSError:
            pass

def child_command(python, config, phase):
    return [str(python),'-I','-X','utf8','-B',str(c.HERE/'run_current.py'),
            '--_child',phase,'--config',str(config),'--config-sha256',c.sha(config)]

def checked_launch(command, output, tag, *, timeout, env=None, launcher=None):
    """Fail-stop is enforced here for subprocess, report, source and numeric failures."""
    started = time.perf_counter()
    stdout, stderr = c.under(output,tag+'-stdout.log'),c.under(output,tag+'-stderr.log')
    failure = None
    with stdout.open('x',encoding='utf-8') as out,stderr.open('x',encoding='utf-8') as err:
        try:
            proc = (launcher or subprocess.run)(command,cwd=str(output),env=env,
                    stdout=out,stderr=err,timeout=timeout,check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = None; failure = 'CHILD_TIMEOUT'
        except OSError:
            code = None; failure = 'CHILD_LAUNCH_FAILURE'; err.write(traceback.format_exc())
    audit = dict(command=command,returncode=code,failure=failure,
                 wall_seconds=time.perf_counter()-started,stdout=c.file_record(stdout),stderr=c.file_record(stderr))
    write_json(c.under(output,tag+'-PROCESS.json'),audit)
    c.require(code == 0 and failure is None, 'child failure: '+tag+'; later GPU launches are stopped')
    return audit

def gpu_phases(python, config, output, *, timeout, env, launch=checked_launch):
    phases = []
    for phase in ('precompile','readonly'):
        audit = launch(child_command(python,config,phase),output,phase,timeout=timeout,env=env)
        result = c.read_json(output/phase/'RESULT.json')
        c.require(result.get('completed') is True and result.get('phase') == phase and
                  result.get('GPU_executed') is True and not result.get('failures'), 'invalid/incomplete child evidence; stop all later launches')
        phases.append(dict(phase=phase,process=audit,result=c.file_record(output/phase/'RESULT.json')))
    return phases

def compare_process(python, left, right, report, output, timeout):
    command = [str(python),'-I','-X','utf8','-B',str(c.HERE/'run_current.py'),'--_compare',str(left),str(right),str(report)]
    checked_launch(command,output,report.stem,timeout=timeout)
    return c.read_json(report)

def execute(args, plan, selections):
    output = c.plain_path(args.output); output.mkdir(parents=True)
    write_json(output/'PLAN.json',plan)
    result = dict(schema='nr-release-reproduction-run-v1',completed=False,GPU_executed=False,
                  plan=c.file_record(output/'PLAN.json'),arms=[],comparison=None,game_FPS=None)
    lock = args.gpu_lock or Path(tempfile.gettempdir())/'nr-release-xpu0.lock'
    try:
        with serial_lock(lock,output):
            for name,sources,receipt in selections:
                arm_out = c.under(output,name);arm_out.mkdir()
                view,view_receipt = build_view(arm_out,sources,Path(plan['runtime_root']),c.plain_path(args.assets),plan['assets'])
                cache = arm_out/'cache';cache.mkdir()
                write_json(arm_out/'RUNTIME_VIEW.json',view_receipt)
                config = dict(view=str(view),output=str(arm_out),cache=str(cache),view_files=view_receipt['files'],
                    profile=view_receipt['profile'],constructor=receipt['constructor'],controls=receipt['controls'],
                    source=receipt,numeric_identity=receipt['numeric_identity'],seed=args.seed,repeats=args.repeats,
                    inputs=None if args.inputs is None else str(c.plain_path(args.inputs)),
                    runner_files={name:c.sha(c.HERE/name) for name in RUNNER_FILES})
                config_path=arm_out/'CONFIG.json';write_json(config_path,config)
                env=dict(os.environ)
                env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
                tmp=arm_out/'tmp';tmp.mkdir()
                env.update(TRITON_CACHE_DIR=str(cache),TMP=str(tmp),TEMP=str(tmp),TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE='1')
                phases=gpu_phases(args.python,config_path,arm_out,timeout=args.timeout,env=env)
                result['GPU_executed']=True
                pair=compare_process(args.python,arm_out/'precompile/RESULT.json',arm_out/'readonly/RESULT.json',
                    arm_out/'PRECOMPILE_READONLY_PAIR.json',arm_out,args.timeout)
                c.require(pair['output_all_byte_identical'] and pair['history_all_byte_identical'], 'same-arm readonly differs from its real precompile outputs/history')
                result['arms'].append(dict(name=name,source=receipt,phases=phases,same_arm_pair=c.file_record(arm_out/'PRECOMPILE_READONLY_PAIR.json')))
            if len(selections)==2 or args.baseline:
                left = c.plain_path(args.baseline) if args.baseline else output/selections[0][0]/'readonly/RESULT.json'
                right = output/selections[-1][0]/'readonly/RESULT.json'
                result['comparison']=compare_process(args.python,left,right,output/'BASELINE_CANDIDATE_PAIR.json',output,args.timeout)
            result['completed']=True
    except BaseException:
        result['failure_stack']=traceback.format_exc()
    result['status']='OFFLINE_PAIR_COMPLETE_QUALITY_GAME_UNQUALIFIED' if result['completed'] else 'FAILED_NO_FURTHER_LAUNCH'
    write_json(output/'RESULT.json',result)
    return 0 if result['completed'] else 1

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args._child:
            c.require(args.config is not None and c.sha(args.config)==args.config_sha256, 'child config SHA mismatch')
            from gpu_reproduction import run_child
            return run_child(args.config,args._child)
        if args._compare:
            from gpu_reproduction import compare_saved
            left,right,out=args._compare
            write_json(out,compare_saved(left,right));return 0
        plan,selections=prepare_plan(args)
        if args.plan:
            print(json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False));return 0
        return execute(args,plan,selections)
    except (ValueError,OSError,KeyError,StopIteration,SyntaxError) as exc:
        print(json.dumps(dict(schema='nr-release-preparation-failure-v1',completed=False,
            GPU_executed=False,error_type=type(exc).__name__,error=str(exc)),ensure_ascii=False),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
