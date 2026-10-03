"""Independently recompute saved module/boundary/matrix timings; stdlib only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

DATA = Path('D:/Codex-NR-Experiments/cyberpunk-opt/int8-c128-cost-graph-v2-20261002')
STAGE = Path(__file__).resolve().parents[1] / 'artifacts/int8-c128-cost-graph-v2-20261002'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))

def require(value,message):
    if not value:
        raise RuntimeError(message)

def percentile(values,fraction):
    values=sorted(values)
    at=(len(values)-1)*fraction
    lo=math.floor(at);hi=math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)

def timing(row,key):
    saved=row[key];n=saved['repetitions_per_sample']
    values=saved['block_raw_ms']
    require(len(values)==50 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values),'Incomplete/invalid raw timing')
    require(n==(16 if key=='fixed_buffer_graph' else 1),'Unexpected repetitions')
    units=[v/n for v in values]
    require(all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(units,saved['per_repetition_raw_ms'])),'Raw normalization differs')
    require(math.isclose(statistics.median(units),saved['per_repetition_median_ms'],rel_tol=1e-10),'Saved P50 differs')
    return dict(mean_ms=statistics.mean(units),p50_ms=statistics.median(units),p95_ms=percentile(units,.95),samples=50,repetitions=n)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve(strict=True);out=args.out.absolute()
    require(run.is_relative_to(DATA) and out.is_relative_to(DATA) and not out.exists(),'Use owned data root and fresh review output')
    for q in (out,*out.parents):
        require(not q.is_symlink() and not (hasattr(q,'is_junction') and q.is_junction()),'Review output crosses reparse point')
    supervisor=read(run/'RESULT.json')
    require(supervisor['completed'] and not supervisor['accepted'] and supervisor['cache_copied_bytes']==0,'Incomplete supervisor result')
    children={}
    for phase in ('prepare','measure'):
        pin=supervisor['children'][phase];path=Path(pin['path']).resolve(strict=True)
        require(path.is_relative_to(run) and sha(path)==pin['sha256'],'Child result pin changed')
        child=read(path);process=read(path.parent/'PROCESS.json')
        require(process['exit_code']==0 and not process['timed_out'] and child['completed'] and child['finite_cleanup'],'Child did not complete')
        require(child['source_freeze']==supervisor['source_freeze'],'Child source freeze differs')
        children[phase]=child
    freeze=read(STAGE/'FREEZE.json')
    require(sha(STAGE/'FREEZE.json')==supervisor['source_freeze']['freeze_sha256'],'Source freeze drift')
    for relative,pin in freeze['source_files'].items():
        require(sha(STAGE/relative)==pin,'Frozen probe changed: '+relative)
    prepared,measured=children['prepare'],children['measure']
    pk=prepared['cache_gate']['actual_compiler_keys'];mk=measured['cache_gate']['actual_compiler_keys']
    require(len(pk)==10 and set(pk)==set(mk) and measured['cache_gate']['readonly_write_attempts']==0,'Actual key/read-only gate incomplete')
    for key in pk:
        require(pk[key]['loaded_binary_sha256']==mk[key]['loaded_binary_sha256'],'Loaded binary changed')
    rows={};reviews=[]
    for row in measured['raw']:
        ident=(row['tile'],row['arm'],row['operation'])
        require(ident not in rows and row['tile'] in (16,32) and row['arm'] in ('B1','C1','C2','B2'),'Duplicate/unknown raw row')
        proof=row['sentinel_replay_proof']
        require(proof['restore_matches_eager'] and proof['protected_inputs_unchanged'] and proof['performed_outside_raw_events'],'Captured graph work not proven')
        require(proof['eager_output_sha256']==proof['replay_output_sha256'] and proof['eager_output_sha256']!=proof['poisoned_output_sha256'],'Poison/replay SHA proof invalid')
        require(proof['protected_input_sha256_before']==proof['protected_input_sha256_after'],'Graph changed protected inputs')
        require(row['compiler_counter_unchanged_during_replay'] and row['full_output_finite'],'Raw replay contract failed')
        record=dict(tile=ident[0],arm=ident[1],operation=ident[2],eager=timing(row,'single_launch_or_segment_event'),graph=timing(row,'fixed_buffer_graph'),roles=row['roles'])
        rows[ident]=record;reviews.append(record)
    require(len(rows)==32,'Expected all 32 graph/Eager measurement rows')
    comparisons=[]
    for tile in (16,32):
        def average(arms,operation,kind):
            sample=[rows[(tile,arm,operation)][kind] for arm in arms]
            return {key:statistics.mean(r[key] for r in sample) for key in ('mean_ms','p50_ms','p95_ms')}
        for kind in ('graph','eager'):
            baseline=average(('B1','B2'),'fp16_full_mlp',kind)
            complete=average(('C1','C2'),'int8_full_mlp',kind)
            prequant=average(('C1','C2'),'int8_prequantized_segment',kind)
            bare_fp16=average(('B1','B2'),'bare_expand_f16',kind)
            bare_int8=average(('C1','C2'),'bare_expand_i8',kind)
            comparisons.append(dict(tile=tile,kind=kind,baseline_full=baseline,int8_full=complete,
                int8_prequantized=prequant,bare_fp16=bare_fp16,bare_int8=bare_int8,
                full_delta_ms={k:complete[k]-baseline[k] for k in baseline},
                bare_delta_ms={k:bare_int8[k]-bare_fp16[k] for k in bare_fp16},
                baseline_first_to_last_ms=rows[(tile,'B2','fp16_full_mlp')][kind]['p50_ms']-rows[(tile,'B1','fp16_full_mlp')][kind]['p50_ms'],
                candidate_first_to_last_ms=rows[(tile,'C2','int8_full_mlp')][kind]['p50_ms']-rows[(tile,'C1','int8_full_mlp')][kind]['p50_ms'],
                entry_omission_contrast_ms=complete['p50_ms']-prequant['p50_ms'],
                entry_contrast_is_not_pure_transfer_time=True,components_not_summed=True))
    result=dict(schema='main-int8-C128-graph-raw-review-v1',review_passed=True,rows=reviews,comparisons=comparisons,
        correctness=measured['correctness'],actual_compiler_keys=mk,
        supervisor=dict(path=str(run/'RESULT.json'),sha256=sha(run/'RESULT.json')),
        main_CPU_receipt=dict(path=str(DATA/'cpu-review-main-01/CPU_DRY_CHECK.json'),sha256=sha(DATA/'cpu-review-main-01/CPU_DRY_CHECK.json')),
        loaded_binary_identity_stable=True,sentinel_graph_proofs=32,raw_samples_per_row_and_kind=50,
        mathematical_scope='One authentic offline encoder128.0 C128 MLP at M15360; fixed buffers; baseline=current pairwise plus FP32 projection; original integer Q12/cubic LUT and FP16 residual exit',
        bare_scope='One branch expand only; same M/K/N, 32bit output; no activation/reduce/project/requant; not deployable or whole-module math',
        measurement_scope='Device Event interval and amortized repeated graph; neither isolates physical ALU or DRAM transfer time',
        arm_summary='Arithmetic mean of first/last arm summaries; P95 is mean of arm P95s, not a pooled or paired-sample P95',
        G_writes=0,GPU_executed_by_main=False,frame_speed_or_quality_accepted=False,FPS_claim=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),comparisons=comparisons),ensure_ascii=False))

if __name__=='__main__':
    main()
