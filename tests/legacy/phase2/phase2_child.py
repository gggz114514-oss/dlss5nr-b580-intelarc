"""Isolated GPU entry point, launched only by Luna's serial supervisor."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--tag', choices=('PRECOMPILE_PARENT', 'PARENT_OFF'), required=True)
    parser.add_argument('--phase', choices=('precompile', 'readonly'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--prepared')
    parser.add_argument('--lock', required=True)
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    if not args.execute_gpu or args.phase == 'readonly' and not args.prepared:
        parser.error('explicit Luna --execute-gpu and same-arm prepared receipt required')
    from gpu_worker import run
    print(json.dumps(run(args), indent=2, ensure_ascii=False, allow_nan=False), flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
