"""Delete only qualified, byte-confirmed duplicate arrays in exact child dirs."""
from __future__ import annotations
import traceback
from pathlib import Path
import re
import runner_common as c

def array_rows(result):
    rows = []
    for frame in result['frames']:
        rows.extend((frame['output'], frame['history']))
    for name in ('body_output', 'final_history'):
        row = result.get(name)
        if isinstance(row,dict) and 'path' in row: rows.append(row)
    if isinstance(result.get('raw'),dict) and result['raw'].get('final_body'):
        rows.append(result['raw']['final_body'])
    return rows

def plan_arrays(result, result_path, *, keep_outputs, keep_history):
    directory = c.output_path(Path(result_path).parent)
    planned = []
    seen = set()
    for row in array_rows(result):
        path = c.output_path(row['path'])
        c.require(path.parent == directory and re.fullmatch(
            r'(?:output-\d{2}|history-\d{2}|body-before-raw|raw-final-body|final-history)\.npy', path.name),
            'Cleanup array is outside its exact child directory/expected names')
        c.require(str(path) not in seen, 'Duplicate cleanup array path')
        seen.add(str(path))
        if path.name.startswith('output-') and keep_outputs: continue
        if path.name.startswith('history-') and keep_history: continue
        c.require(row['finite'] is True, 'Unknown/nonfinite array cannot be cleaned')
        c.checked(row)
        planned.append(dict(path=str(path), sha256=row['sha256'], bytes=path.stat().st_size,
                            raw_sha256=row['raw_sha256'], reason='qualified duplicate or already compared private diagnostic'))
    return planned

def cleanup(directory, precompile_result, readonly_result, *, qualification, comparison,
            same_arm_byte_proof, keep_outputs=True, keep_history=False, clean_precompile=True):
    """Cleanup errors are reported and never overwrite successful model evidence."""
    directory = c.output_path(directory)
    receipt = dict(schema='b580-phase2-array-cleanup-v1', status='PENDING', deleted=[], failures=[],
                   qualification=qualification, comparison=comparison, same_arm_byte_proof=same_arm_byte_proof,
                   bytes_freed=0, GPU_access=False, unrelated_paths_touched=False)
    try:
        c.require(c.read(c.checked(qualification))['completed'] is True, 'No successful qualification')
        proof = c.read(c.checked(same_arm_byte_proof))
        c.require(proof['same_arm_initial13_byte_identity'] is True, 'Duplicate bytes not confirmed')
        compared = c.read(c.checked(comparison))
        c.require(compared['baseline_error_comparison_passed'] is True, 'No successful baseline comparison')
        paths = []
        if clean_precompile:
            value = c.read(c.checked(precompile_result))
            c.require(value['completed'] is True and value['cleanup']['verified'] is True, 'Preserving failed/unknown precompile owner')
            paths.extend(plan_arrays(value, precompile_result['path'], keep_outputs=False, keep_history=False))
        value = c.read(c.checked(readonly_result))
        c.require(value['completed'] is True and value['cleanup']['verified'] is True, 'Preserving failed/unknown readonly owner')
        paths.extend(plan_arrays(value, readonly_result['path'], keep_outputs=keep_outputs, keep_history=keep_history))
        # Validate EVERY target before any deletion; each actual deletion repeats
        # absolute path/link/SHA verification. No rglob deletion or recursive rm.
        receipt['planned_arrays'] = paths
        c.write(directory/'CLEANUP_PLAN.json', receipt)
        for row in paths:
            try:
                path = c.output_path(c.checked(row))
                c.require(path.stat().st_size == row['bytes'], 'Array size drift before cleanup')
                path.unlink()
                receipt['deleted'].append(row); receipt['bytes_freed'] += row['bytes']
            except BaseException:
                receipt['failures'].append(dict(array=row,stack=traceback.format_exc()))
                break
        receipt['status'] = 'CLEANED' if not receipt['failures'] else 'CLEANUP_PARTIAL_PRESERVED'
    except BaseException:
        receipt['status'] = 'CLEANUP_FAILED_PRESERVED'
        receipt['failures'].append(dict(stack=traceback.format_exc()))
    c.write(directory/'CLEANUP.json', receipt)
    return receipt
