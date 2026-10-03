"""CPU regression for the actual runner's JSON failure; imports no GPU runtime."""
import ast,json,hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
runner=HERE/'benchmark_fp8_partial_int8_projection_v3.py'
tree=ast.parse(runner.read_text(encoding='utf-8'))
definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('difference','json_scalar','save')]
assert len(definitions)==3
scope={'np':np,'json':json}
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(runner),'exec'),scope)


class CpuArray:
    def __init__(self,value):self.value=np.asarray(value,dtype=np.float16)
    def cpu(self):return self
    def numpy(self):return self.value


rows=[]
for a,b,equal,count in (
    ([0,1,-2],[0,1,-2],True,0),
    ([0,1,-2],[0,2,-2],False,1),
    ([0.0,1],[-0.0,1],False,1),
):
    row=scope['difference'](CpuArray(a),CpuArray(b))
    assert row['byte_equal'] is equal and row['different_half_words']==count
    assert json.loads(json.dumps(row))==row
    rows.append(row)
assert rows[2]['max_absolute_error']==0 and rows[2]['rmse']==0

scalars=dict(flag=np.bool_(True),count=np.int64(7),ms=np.float32(0.25))
assert json.loads(json.dumps(scalars,default=scope['json_scalar']))==dict(flag=True,count=7,ms=0.25)
try:json.dumps(object(),default=scope['json_scalar'])
except TypeError:pass
else:raise AssertionError('Unknown report types must not be silently stringified')
try:json.dumps(np.float32(float('nan')),default=scope['json_scalar'],allow_nan=False)
except ValueError:pass
else:raise AssertionError('Nonfinite report numbers must not be accepted')

out=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/fp8-partial-int8-report-cpu-v3')
out.mkdir(exist_ok=False)
scope.update(OUT=out,report=dict(passed=True,no_gpu_execution=True,cases=rows,numpy_scalars=scalars,
    runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
    check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
scope['save']()
result=json.loads((out/'validation.json').read_text(encoding='utf-8'))
assert result['passed'] and len(result['cases'])==3 and result['numpy_scalars']['flag'] is True
assert result['cases'][2]['byte_equal'] is False
print(json.dumps(dict(passed=True,no_gpu_execution=True,result=str(out/'validation.json'),
    sha256=hashlib.sha256((out/'validation.json').read_bytes()).hexdigest())),flush=True)
