"""Cold extension pins, verified bytes; this never disables a source guard."""
import hashlib
import importlib
import json
from pathlib import Path
import sys


class SourceAdmission720:
    def __init__(self):
        self.saved = []
        self.receipts = {}

    def install(self):
        manifest = json.loads(Path(__file__).with_name('HISTORY_SOURCE_ADMISSION.json').read_text(encoding='utf-8'))
        rows = manifest['roles']
        if self.saved or self.receipts:
            raise RuntimeError('Source extension admission must run once per scope')
        for role, row in rows.items():
            module = sys.modules.get(row['module'])
            if module is None:
                raise RuntimeError('Source extension actual role is not loaded: ' + role)
            path = Path(module.__file__).resolve(strict=True)
            raw = hashlib.sha256(path.read_bytes()).hexdigest()
            if raw != row['sha256']:
                raise RuntimeError('Source extension byte pin mismatch: ' + role)
            self.receipts[role] = dict(path=str(path), sha256=raw, purpose=row['purpose'])
        # Both guards are stdlib-only until their candidate constructors run.
        decoder = importlib.import_module('decoder_input_full_k_720_v1')
        post = importlib.import_module('post_numeric_suite_720_v1')
        history = importlib.import_module('history_numeric_suite_720_v1')
        guard_maps = [(decoder._KNOWN, 'nr_backend.temporal', 'temporal'),
                      (decoder._KNOWN, 'nr_game_controlled_model', 'model'),
                      (post.KNOWN_SHA256, 'temporal', 'temporal'),
                      (post.KNOWN_SHA256, 'controlled', 'model'),
                      (post.KNOWN_SHA256, 'graph_v1', 'graph_v1'),
                      (history.REFERENCE_SHA256, 'temporal', 'temporal')]
        for mapping, key, role in guard_maps:
            # Validate every destination before mutating any source guard.
            if key not in mapping:
                raise RuntimeError('Source extension guard destination is missing: ' + key)
        for mapping, key, role in guard_maps:
            raw, old = self.receipts[role]['sha256'], mapping[key]
            self.saved.append((mapping, key, old))
            # Preserve exact baseline alternatives when the guard supports
            # a tuple; a scalar guard gets exactly the admitted current bytes.
            mapping[key] = tuple(dict.fromkeys((*old, raw))) if isinstance(old, tuple) else raw
        return dict(self.receipts)

    def restore(self):
        for mapping, key, old in reversed(self.saved):
            mapping[key] = old
        self.saved.clear()
