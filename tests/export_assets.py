"""Export a user's existing runtime assets to an external, portable bundle (CPU)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_contracts as c
from portable_runtime import REQUIRED_ASSETS, SUPPORT, write_json, validate_asset_bundle

def collect(runtime):
    runtime = c.plain_path(runtime)
    profile_path = runtime/'data/product-v1/profile-v1.json'
    cfg_path = runtime/'data/product-v1/local-runtime-v1.json'
    profile, cfg = c.read_json(profile_path), c.read_json(cfg_path)
    c.require(c.sha(profile_path) == cfg['profile_sha256'], 'installed cfg/profile SHA disagreement')
    files = {}
    for prefix in ('exact/model-assets', 'toolchain', 'host-helpers'):
        directory = c.under(runtime, prefix)
        c.require(directory.is_dir(), 'required runtime dependency missing: ' + prefix)
        for name, path in c.all_files(directory).items():
            if '__pycache__' not in path.parts and path.suffix != '.pyc':
                files[prefix+'/'+name] = path
    for name in REQUIRED_ASSETS:
        if name != 'templates/profile-v1.json':
            files[name] = c.under(runtime, name)
    def relocate(v):
        if isinstance(v, list):
            return [relocate(x) for x in v]
        if not isinstance(v, dict):
            return v
        out = {k: relocate(x) for k, x in v.items()}
        if 'path' in v and 'sha256' in v:
            text = v['path']
            if text.startswith('${ROOT}/'):
                path = c.under(runtime, text[8:])
            elif Path(text).is_absolute():
                path = c.plain_path(text)
            else:
                path = c.under(runtime, text)
            c.require(path.is_relative_to(runtime), 'profile points outside the selected runtime: ' + str(path))
            c.require(path.is_file() and c.sha(path) == v['sha256'], 'profile asset missing/changed: ' + str(path))
            name = path.relative_to(runtime).as_posix()
            files[name] = path
            out['path'] = '${ROOT}/'+name
        return out
    template = relocate(profile)
    receipts = {name:dict(relative_path=name, sha256=c.sha(path), bytes=path.stat().st_size)
                for name,path in sorted(files.items())}
    encoded = (json.dumps(template, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    receipts['templates/profile-v1.json'] = dict(relative_path='templates/profile-v1.json',
        sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded))
    provenance = dict(source_profile=c.file_record(profile_path), source_cfg=c.file_record(cfg_path),
                      derived_template_sha256=receipts['templates/profile-v1.json']['sha256'],
                      original_results_inherited=False, model_array_bytes_changed=False)
    return files, template, receipts, provenance

def export(runtime, output, *, plan=False):
    runtime, output = c.plain_path(runtime), c.plain_path(output)
    # A public checkout (including publish) must never receive private assets.
    c.require(not output.is_relative_to(c.REPO.parent) and not output.is_relative_to(runtime)
              and not runtime.is_relative_to(output), 'bundle output must be external to source/archive and runtime')
    c.require(not output.exists(), 'choose a fresh external bundle directory')
    files, template, receipts, provenance = collect(runtime)
    result = dict(schema='nr-release-assets-v1', files=list(receipts.values()),
        profile_template='templates/profile-v1.json', provenance=provenance,
        driver='record actual driver version during GPU reproduction', GPU_executed=False,
        total_bytes=sum(r['bytes'] for r in receipts.values()), scenario='external user-owned runtime assets')
    if plan:
        return dict(result, planned_output=str(output), exported=False)
    output.mkdir(parents=True)
    for name, path in sorted(files.items()):
        target = c.under(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        c.require(c.sha(target) == receipts[name]['sha256'], 'export changed bytes: ' + name)
    write_json(output/'templates/profile-v1.json', template)
    write_json(output/'assets.json', result)
    validation = validate_asset_bundle(output)
    return dict(schema='nr-release-assets-export-result-v1', completed=True,
                manifest=c.file_record(output/'assets.json'), validation=validation, GPU_executed=False)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', action='store_true', help='inventory/hashes only; no files written')
    args = p.parse_args(argv)
    try:
        print(json.dumps(export(args.source_runtime, args.output, plan=args.plan), indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps(dict(completed=False, GPU_executed=False, error=str(exc))), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
