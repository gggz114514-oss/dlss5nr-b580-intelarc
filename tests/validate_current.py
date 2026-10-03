"""CLI source/data validation; stdlib only, no model imports or GPU queries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_contracts as c

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=c.REPO)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--source-root', type=Path, help='relative_path root for a frozen PHASE2 source manifest')
    p.add_argument('--exact-inventory', action='store_true')
    p.add_argument('--inputs', type=Path, help='optional external fixed13 .NPY manifest')
    p.add_argument('--assets', type=Path, help='optional external asset bundle directory')
    p.add_argument('--report', type=Path)
    return p

def check(args):
    root = args.source_root or args.repo_root
    manifest = args.manifest or args.repo_root/'evidence/2026-10-03/current-source-manifest.json'
    report, trees = c.validate_sources(root, manifest, exact=args.exact_inventory or args.source_root is not None)
    candidates = [t for rel, t in trees.items() if rel in ('game/nr_game_fullsize.py', 'current/runtime/game/nr_game_fullsize.py')]
    c.require(len(candidates) == 1, 'one pinned nr_game_fullsize.py required')
    report['contract'] = c.validate_constructor(c.OPTIONS, candidates[0])
    actual_root = root if args.source_root else root/'current/runtime'
    report['numeric_identity'] = c.numeric_identity(actual_root,c.OPTIONS)
    report['controls'] = c.validate_controls(c.CONTROLS)
    # Inventory validation authenticates every local dependency, even modules
    # only reached by conditional/capture routes. Required entry modules must
    # also be present independently of whatever a supplied manifest omits.
    prefix = '' if args.source_root else 'current/runtime/'
    for rel in ('game/fullsize_session_v1.py', 'game/re4_session_v1.py', 'game/rows_paths_v1.py',
                'game/rows_scopes_v1.py', 'game/nr_game_history_fused.py',
                'experimental/fp8_unround_overlay/bootstrap.py',
                'experimental/fp8_unround_overlay/nr_backend/controlled_temporal.py',
                'experimental/fp8_unround_overlay/modules/graph_front_v6.py',
                'experimental/fp8_unround_overlay/modules/nr_runtime_v1.py'):
        c.require(prefix+rel in trees, 'required local dependency omitted: ' + rel)
    if args.inputs:
        report['inputs'] = c.validate_inputs(args.inputs)
    if args.assets:
        from portable_runtime import validate_asset_bundle
        report['assets'] = validate_asset_bundle(args.assets)
    report['passed'] = True
    return report

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = check(args)
        code = 0
    except (ValueError, OSError, SyntaxError, StopIteration, KeyError) as exc:
        report = dict(schema='nr-release-source-check-v1', passed=False, GPU_executed=False,
                      error_type=type(exc).__name__, error=str(exc))
        code = 1
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.report:
        out = c.plain_path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    print(payload, end='')
    return code

if __name__ == '__main__':
    raise SystemExit(main())
