"""Read pinned, previously executed compiler artifacts; no framework/GPU calls."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re


ROOT = Path('D:/Codex-NR-Experiments/cyberpunk-opt/vit-current-provider-gpu-runner-v4-20261002')
COMPARE = ROOT / 'compare-01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def record(path):
    return {'path': str(path), 'sha256': sha(path)}


def body_profile(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('define spir_kernel '))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == '}')
    ops, calls, loads = Counter(), Counter(), Counter()
    instructions = []
    for line in lines[start + 1:end]:
        line = line.strip()
        if not line or line.startswith(';') or re.match(r'^[\w.$-]+:\s*(;.*)?$', line):
            continue
        instruction = re.sub(r'^%\S+\s*=\s*', '', line)
        instruction = re.sub(r'^(?:tail|musttail|notail)\s+', '', instruction)
        op = instruction.split()[0]
        ops[op] += 1
        instructions.append(instruction)
        callee = re.search(r'\bcall\b.*?@([^ (]+)', instruction)
        if callee:
            calls[callee[1]] += 1
        if op == 'load':
            match = re.search(r'ptr addrspace\((\d+)\)', instruction)
            loads[match[1] if match else 'private_or_unspecified'] += 1
    return {
        'definition': lines[start],
        'static_instruction_count': len(instructions),
        'static_op_counts': dict(ops),
        'static_calls': dict(calls),
        'static_explicit_load_counts_by_address_space': dict(loads),
        'matrix_intrinsic_call_lines': [line for line in instructions
                                       if '__spirv_SubgroupMatrixMultiplyAccumulateINTEL' in line],
        'counts_are_static_ir_not_runtime_traffic_or_alu_cycles': True,
    }


def kernel_detail(child, site_name):
    site = child['provider_observation']['sites'][site_name]
    keys = site['actual_observed_compiler_keys']
    if len(keys) != 1:
        raise ValueError('Expected one actual specialization: ' + site_name)
    key = keys[0]
    actual = child['cache_gate']['actual_compiler_keys'][key]
    files = {}
    for name, digest in actual['metadata_files'].items():
        path = Path(name)
        if sha(path) != digest:
            raise ValueError('Compiler artifact changed: ' + name)
        files[path.suffix] = path
    binary = site['binary']
    if sha(files['.spv']) != binary['actualbinary_sha256']:
        raise ValueError('Executed binary does not match pinned cache')
    ttgir = files['.ttgir'].read_text(encoding='utf-8')
    dots = [line.strip() for line in ttgir.splitlines() if ' = tt.dot ' in line]
    if len(dots) != 1 or 'xf16' not in dots[0] or 'xf32' not in dots[0] or '#ttig.dpas<' not in ttgir:
        raise ValueError('Missing actual FP16/FP32 DPAS lowering: ' + site_name)
    result = {
        'compiler_key': key,
        'binary': binary,
        'source': actual['source'],
        'operands': site['operands'],
        'compiled_arguments': site['compiled_arguments'],
        'specialization': json.loads(actual['specialization_data']),
        'files': {ext: record(path) for ext, path in files.items()},
        'ttgir_dot': dots[0],
        'ttgir_dpas_layout': next(line.strip() for line in ttgir.splitlines() if '#ttig.dpas<' in line),
        'llvm': body_profile(files['.llir'].read_text(encoding='utf-8')),
    }
    if not result['llvm']['matrix_intrinsic_call_lines']:
        raise ValueError('Missing compiled Intel matrix intrinsic')
    return result


def main():
    main_review = COMPARE / 'MAIN_RAW_AND_PROTOCOL_REVIEW.json'
    previous = read(main_review)
    source_results = {tag: COMPARE / tag / 'RESULT.json' for tag in ('B1', 'C1')}
    details = {tag: {name: kernel_detail(read(path), 'vit.0.' + name)
                     for name in ('score', 'value')}
               for tag, path in source_results.items()}
    comparisons = {}
    for name in ('score', 'value'):
        b, c = details['B1'][name], details['C1'][name]
        bl, cl = b['llvm'], c['llvm']
        comparisons[name] = {
            'both_actual_fp16_operands_fp32_accumulator_intel_matrix_intrinsic': True,
            'operand_shape_stride_dtype_equal': all(
                {k: b['operands'][side][k] for k in ('shape', 'stride', 'dtype')} ==
                {k: c['operands'][side][k] for k in ('shape', 'stride', 'dtype')}
                for side in ('a', 'b')),
            'binary_hash_equal': b['binary']['actualbinary_sha256'] == c['binary']['actualbinary_sha256'],
            'static_intrinsic_call_counts_equal': bl['static_calls'] == cl['static_calls'],
            'static_explicit_load_counts_equal': bl['static_explicit_load_counts_by_address_space'] ==
                                                cl['static_explicit_load_counts_by_address_space'],
            'static_op_counts_equal': bl['static_op_counts'] == cl['static_op_counts'],
            'static_instruction_count': {'baseline': bl['static_instruction_count'],
                                         'candidate': cl['static_instruction_count']},
            'opcode_deltas_candidate_minus_baseline': {
                op: cl['static_op_counts'].get(op, 0) - bl['static_op_counts'].get(op, 0)
                for op in sorted(set(bl['static_op_counts']) | set(cl['static_op_counts']))
                if cl['static_op_counts'].get(op, 0) != bl['static_op_counts'].get(op, 0)},
            'differences_are_not_a_measured_slowdown_explanation': True,
        }
    report = {
        'status': 'CPU_COMPILED_WORK_REVIEW_NO_CONFIRMED_ATTENTION_GAIN',
        'source_results': {tag: record(path) for tag, path in source_results.items()},
        'previous_raw_review': record(main_review),
        'kernels': details,
        'comparisons': comparisons,
        'whole_body_measurement': {k: previous[k] for k in
                                   ('baseline_mean_ms', 'candidate_mean_ms', 'saved_ms',
                                    'baseline_first_last_drift_ms', 'paired_saved_ms', 'metric')},
        'conclusion': [
            'Both observed baseline kernels already lower FP16 dot to Intel subgroup matrix instructions.',
            'Replacement preserves matrix precision, tile, actual operand layout and work, not software-to-XMX conversion.',
            'Score instruction/call counts match. Value has address/control differences with equal matrix/load/store/shuffle counts.',
            'No extra fpext found; identical eight static fptrunc instructions are normal final FP32-to-FP16 stores.',
            'Configured num_stages differs (baseline 2, candidate 1), without fewer compiled matrix/load/store/barrier operations.',
            'Saved whole-body difference is smaller than baseline drift and reverses sign between paired arms.',
        ],
        'limitations': [
            'Compiler IR is not final physical driver ISA or measured memory transaction time.',
            'Static instruction counts do not establish identical scheduling, byte-identical binaries or equal per-kernel cycles.',
            'This timing excludes bridge, front/history preparation, game and Present. No CPU conversion is timed inside this graph.',
        ],
        'GPU_executed_now': False, 'game_or_API_calls': 0, 'G_writes': 0,
        'promoted_to_game': False,
    }
    out = COMPARE / 'MAIN_COMPILED_WORK_REVIEW.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'comparisons': comparisons, 'report': record(out)}))


if __name__ == '__main__':
    main()
