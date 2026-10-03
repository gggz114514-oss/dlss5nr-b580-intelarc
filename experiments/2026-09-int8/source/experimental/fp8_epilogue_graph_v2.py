"""Retain the 84 previously byte-validated matrix epilogues without compiler spills.

The all-92 candidate failed the complete-host performance gate. Eight MLP
matrices reported spills=192; leave those on the existing two-kernel path.
This is a performance candidate, not an assumption that zero spills means faster.
"""
from pathlib import Path
import hashlib,json
from fp8_epilogue_graph_v1 import EpilogueGraphRewrite as AllEpilogues

VALIDATION=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/fp8-epilogues-v1/validation.json')
VALIDATION_SHA='be2ace2418b866f76490994dba3da332516b4fa21412ad3c606800c6a81f11b3'

class EpilogueGraphRewrite(AllEpilogues):
    def __init__(self,model,provider):
        super().__init__(model,provider)
        assert hashlib.sha256(VALIDATION.read_bytes()).hexdigest()==VALIDATION_SHA
        record=json.loads(VALIDATION.read_text(encoding='utf-8'));assert record['passed']
        first,second=[b['epilogues']['proofs'] for b in record['matrix_validation_builds']]
        assert all(a['ordinal']==b['ordinal'] and a['spills']==b['spills'] and a['all_bytes_equal'] and b['all_bytes_equal'] for a,b in zip(first,second))
        accepted={p['ordinal'] for p in first if p['spills']==0}
        assert len(first)==len(second)==92 and len(accepted)==84
        self.plan=dict(self.plan,selected=[r for r in self.plan['selected'] if r['kernel_ordinal'] in accepted])
        assert len(self.plan['selected'])==84
