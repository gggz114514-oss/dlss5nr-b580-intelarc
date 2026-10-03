"""Derive explicit attention kernel versions without changing frozen source files."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/native-half-attention-derivation-v1')
assert not D.exists();D.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
changes=[
    ('fused_c32_projection_pack_v1.py','fused_c32_projection_native_half_v1.py',
     'from nr_backend.triton_cubic_fp8 import _half_fma_value',
     'from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value'),
    ('fused_qkv_pack_v1.py','fused_qkv_pack_native_half_v1.py',
     'from nr_backend.triton_cubic_fp8 import _half_fma_value',
     'from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value'),
    ('fused_swin_core_v2.py','fused_swin_core_native_half_v1.py',
     'from nr_backend.triton_cubic_fp8 import _half_fma_value',
     'from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value'),
    ('fused_swin_heads_v1.py','fused_swin_heads_native_half_v1.py',
     'from fused_swin_core_v2 import _exp, _weights_pair',
     'from fused_swin_core_native_half_v1 import _exp, _weights_pair'),
    ('fused_swin_scheduling_v1.py','fused_swin_native_half_v1.py',
     'from fused_swin_core_v2 import forward','from fused_swin_core_native_half_v1 import forward'),
]
rows=[]
for source,name,old,new in changes:
    p=HERE/source;out=HERE/name;assert not out.exists()
    text=p.read_text();assert text.count(old)==1
    note='# Explicit native-half attention version; see native_half_attention_fma_v1.py.\n'
    out.write_text(note+text.replace(old,new),encoding='utf-8',newline='\n')
    rows.append(dict(source=str(p),source_sha256=sha(p),output=str(out),output_sha256=sha(out),replace=[old,new],prefix=note))
(D/'source-derivation.json').write_text(json.dumps(dict(generator_sha256=sha(__file__),changes=rows),indent=2)+'\n')
print(json.dumps(dict(derived_files=len(rows),manifest=str(D/'source-derivation.json'))))
