"""CPU copy of only hash-validated PHASE1 compiler groups. Never rekey binaries."""
from __future__ import annotations
import copy
from pathlib import Path
import re
import runner_common as c

def validate_key(key, receipt, source, allowed_sources):
    source = c.no_reparse(source)
    c.require(re.fullmatch('[a-f0-9]{64}', key) and receipt['compiler_key'] == key, 'Malformed compiler key')
    owner = receipt['source']
    path = c.checked(owner)
    c.require(allowed_sources.get(str(path)) == owner['sha256'], 'Compiler source is not pinned')
    c.require(receipt['target']['backend'] == 'xpu' and 'B580' in str(receipt['target']['arch']).upper(), 'Foreign seed target')
    files = receipt['metadata_files']
    c.require(isinstance(files, dict) and files, 'Empty seed group')
    paths = [c.checked(dict(path=name, sha256=sha)) for name, sha in files.items()]
    c.require(all(p.is_relative_to(source) for p in paths) and len({p.parent for p in paths}) == 1,
              'Compiler group metadata escapes source cache')
    binary = [p for p in paths if p.suffix in ('.spv', '.zebin', '.cubin')]
    c.require(len(binary) == 1 and c.sha(binary[0]) == receipt['loaded_binary_sha256'], 'Seed loaded binary SHA differs')
    group = paths[0].parent / ('__grp__' + receipt['src_name'][:150] + '.json')
    c.require(group.is_file(), 'Seed group receipt missing')
    group_data = c.read(group)
    children = group_data.get('child_paths')
    c.require(isinstance(children, dict) and children and
              {str(c.no_reparse(p)) for p in children.values()} == {str(p) for p in paths},
              'Seed group child paths differ from actual metadata files')
    c.require(all(Path(name).name == name and Path(path).name == name for name, path in children.items()),
              'Seed group child basename mismatch')
    return paths, group, group_data

def collect_phase1_keys(source):
    c.require(c.no_reparse(source) == c.SEED_ROOT.resolve(), 'Only the requested PHASE1 shared cache can seed')
    c.require(c.sha(c.PHASE1_MANIFEST) == c.PHASE1_SHA256, 'PHASE1 source parent drift')
    phase1 = c.read(c.PHASE1_MANIFEST)
    allowed = {str(Path(row['path']).resolve()): row['sha256'] for row in phase1['source_pins']}
    old = c.runtime_pins()
    allowed.update({str(Path(row['path']).resolve()): row['sha256'] for row in old['current_runtime_sources'] + old['runtime_assets']})
    keys, receipts = {}, []
    # A failed run may supply verified compiler BYTES; it never supplies model,
    # graph, timing, numerical, or cleanup acceptance to PHASE2.
    for path in sorted(c.EVIDENCE_DATA.parent.joinpath('phase1').rglob('RESULT.json')):
        c.no_reparse(path)
        value = c.read(path)
        if value.get('schema') != 'b580-phase1-fullmodel-child-v1':
            continue
        if value.get('sources', {}).get('phase1_manifest', {}).get('sha256') != c.PHASE1_SHA256:
            continue
        selected = value.get('cache_gate', {}).get('actual_compiler_keys', {})
        contributed = []
        for key, row in selected.items():
            if not row.get('metadata_files') or not all(Path(p).resolve().is_relative_to(Path(source).resolve())
                                                       for p in row['metadata_files']):
                continue
            validate_key(key, row, source, allowed)
            if key in keys:
                earlier = keys[key]
                c.require(earlier['metadata_files'] == row['metadata_files'] and
                          earlier['loaded_binary_sha256'] == row['loaded_binary_sha256']
                          and earlier['target'] == row['target'], 'Conflicting seed key receipts')
            else:
                keys[key] = copy.deepcopy(row)
            contributed.append(key)
        if contributed:
            receipts.append(dict(result=c.record(path), compiler_keys=sorted(contributed),
                                 status=value.get('status'), acceptance_inherited=False))
    c.require(keys, 'No actual PHASE1 compiler-key receipts for requested seed cache')
    return keys, receipts, allowed

def copy_validated(source, destination, keys, allowed_sources):
    source, destination = c.no_reparse(source), c.output_path(destination)
    c.require(source != destination and destination.is_dir() and not any(destination.iterdir()), 'Seed destination must be empty PHASE2 cache')
    before = c.cache_inventory(source)
    copied, groups, key_rows = {}, {}, {}
    for key, receipt in keys.items():
        paths, group, group_data = validate_key(key, receipt, source, allowed_sources)
        for path in paths:
            relative = path.relative_to(source).as_posix()
            digest = receipt['metadata_files'][str(path)]
            c.require(before.get(relative) == digest, 'Seed inventory and key disagree')
            if relative not in copied:
                target = c.output_path(destination / relative); target.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as src, target.open('xb') as dst:
                    while block := src.read(1024*1024): dst.write(block)
                c.require(c.sha(target) == digest, 'Copied metadata/binary SHA mismatch')
                copied[relative] = digest
        rel_group = group.relative_to(source).as_posix()
        rewritten = copy.deepcopy(group_data)
        rewritten['child_paths'] = {name:str((destination / Path(path).relative_to(source)).resolve())
                                    for name,path in group_data['child_paths'].items()}
        if rel_group not in groups:
            target = c.output_path(destination / rel_group)
            c.write(target, rewritten)
            groups[rel_group] = dict(source_sha256=c.sha(group), copied_sha256=c.sha(target),
                only_change='child_paths absolute cache location; key/binary/metadata bytes unchanged')
            c.require({k:v for k,v in c.read(target).items() if k != 'child_paths'} ==
                      {k:v for k,v in group_data.items() if k != 'child_paths'}, 'Group metadata changed')
        key_rows[key] = dict(source_receipt=receipt, unchanged_compiler_key=key,
                            metadata_relatives=[p.relative_to(source).as_posix() for p in paths], group_relative=rel_group)
    c.require(c.cache_inventory(source) == before, 'PHASE1 cache changed concurrently; reject copied seed')
    return dict(schema='b580-phase2-validated-cache-copy-v1', source=str(source), destination=str(destination),
                actual_compiler_key_count=len(keys), compiler_keys=key_rows,
                exact_copied_metadata=copied, relocated_group_paths=groups,
                seed_inventory_sha256=c.digest_obj(before), destination_inventory=c.cache_inventory(destination),
                copied_bytes=sum((destination / p).stat().st_size for p in copied),
                measurements_reused=False, validation_acceptance_inherited=False, source_writes=0)

def initialize(cache, receipt_path):
    cache = c.output_path(cache)
    existing = cache.parent / 'CACHE_SEED.json'
    if cache.exists() and any(cache.iterdir()):
        c.require(existing.is_file(), 'Existing shared cache has no PHASE2 validated seed receipt')
        value = c.read(existing)
        c.require(value['destination'] == str(cache.resolve()) and value['source_writes'] == 0,
                  'Existing cache belongs to another run')
        for relative, expected in value['exact_copied_metadata'].items():
            c.require(c.sha(cache / relative) == expected, 'Existing seed binary/metadata drift')
        return c.record(existing)
    cache.mkdir(parents=True, exist_ok=True)
    keys, receipts, allowed = collect_phase1_keys(c.SEED_ROOT)
    value = copy_validated(c.SEED_ROOT, cache, keys, allowed)
    value['phase1_compiler_receipts'] = receipts
    c.write(existing, value)
    return c.record(existing)
