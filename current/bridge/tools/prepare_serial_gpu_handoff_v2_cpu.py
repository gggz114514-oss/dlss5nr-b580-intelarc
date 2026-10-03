"""Freeze the prior product plus the installed owner-transfer adapter; CPU only."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parents[1]
    prior = project / 'artifacts/gpu-handoff-product-v1-20261002'
    stage = project / 'artifacts/gpu-handoff-serial-owner-v2-20261002'
    installation = Path('D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-owner-transfer-v1-20261002/install-01/installed.json')
    install = json.loads(installation.read_text())
    source_pins = json.loads((prior / 'SOURCE_PINS.json').read_text())
    for name, digest in source_pins.items():
        if sha(prior / 'payload' / name) != digest:
            raise ValueError('Frozen product input changed: ' + name)
    for row in install['files']:
        if sha(Path(row['target'])) != row['new_sha256']:
            raise ValueError('Installed input changed: ' + row['target'])
    if stage.exists():
        raise ValueError('Use a fresh stage; do not overwrite prior work')
    inputs = {str(prior / 'payload' / name): digest for name, digest in source_pins.items()}
    inputs[str(installation)] = sha(installation)
    shutil.copytree(prior / 'payload', stage / 'payload')
    adapter = Path(install['files'][2]['source'])
    host = Path(install['files'][5]['source'])
    for source, name in ((adapter, 'cyberpunk_nr_adapter.py'), (host, 'nr_gpu_handoff_host_v1.py')):
        inputs[str(source)] = sha(source)
        (stage / 'payload/game' / name).write_bytes(source.read_bytes())
    (stage / 'tests').mkdir()
    (stage / 'INPUT_PINS.json').write_text(json.dumps({'inputs': inputs,
        'parent_installation': str(installation), 'GPU_or_API_executed': False,
        'math_changed': False}, indent=2) + '\n')
    print(json.dumps({'status': 'serial_stage_created_CPU_only', 'stage': str(stage),
                      'inputs': len(inputs)}))


if __name__ == '__main__':
    main()
