"""Use the authenticated complete-use plan to fuse 92 FP16 matrix FP8 stores.

Only owned NR256 graph construction is intercepted. Matrix ordinals and full
operand geometry are checked against both reset and temporal plans. Public
outputs/history still follow the serial GraphFront ownership rules.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json
import torch
import fast_matrices_v3 as matrices
import dense_tiles_v1
import fp8_epilogue_matmul_v1 as fused
import nr_backend.triton_fp8 as fp8
from immutable_artifacts_v1 import put
from nr_backend.execution import current_arithmetic_backend
from fp8_graph_rewrite_v1 import FP8GraphRewrite,RewritingDataflow
from quantization_dataflow_v1 import CONTRACTS

PLAN=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/fp8-epilogue-plan-v1/plan.json')
PLAN_SHA='aa5b8db36edba548bb061fd91339e20df8f1723ab384c0ffc18d876d1f06a267'
FUSED='fp8_epilogue_matmul_v1._matmul'

def descriptor(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),logical_bytes=t.numel()*t.element_size())


class MatrixEpilogues:
    def __init__(self,plan,*,validate_outputs=False):
        self.plan={row['kernel_ordinal']:row for row in plan['selected']}
        self.expected_calls=plan['matrix_calls'];self.calls=0;self.fused_calls=[]
        self.validate_outputs=validate_outputs;self.proofs=[]

    @contextmanager
    def installed(self):
        target=matrices._matmul
        assert dense_tiles_v1._matmul is target
        original=target.run;had='run' in target.__dict__;prior=target.__dict__.get('run')
        assert FUSED not in CONTRACTS
        CONTRACTS[FUSED]=(('OUT',),('OUT',))
        def run(*args,**kwargs):
            if kwargs.get('warmup'):return original(*args,**kwargs)
            ordinal=self.calls;self.calls+=1
            row=self.plan.get(ordinal)
            if row is None:return original(*args,**kwargs)
            assert len(args)==len(target.arg_names)
            bound=dict(zip(target.arg_names,args))
            params={n:v for n,v in bound.items() if not isinstance(v,torch.Tensor)}
            assert params==row['constants'],('Matrix plan parameters changed',ordinal)
            assert list(kwargs['grid'])==row['grid'] and kwargs['enable_fp_fusion'] is False
            assert kwargs['num_warps']==row['launch_options']['num_warps']
            if 'num_stages' in kwargs:assert kwargs['num_stages']==row['launch_options']['num_stages']
            assert not params['INT8'] and not params['BATCHED'] and params['BK']==32
            for name,expected in row['operands'].items():
                actual=bound[name]
                assert actual.device.type=='xpu' and actual.device==bound['OUT'].device
                # A newly eliminated contiguous slice may carry a different
                # storage offset. Triton receives its already-offset data_ptr;
                # shape/strides/value bytes, not allocation offset, define A.
                assert descriptor(actual)=={k:v for k,v in expected.items() if k!='offset'},(ordinal,name)
            output=bound['OUT']
            assert descriptor(output)=={k:v for k,v in row['output'].items() if k!='offset'}
            assert output.is_contiguous() and output.storage_offset()==0 and output.numel()*output.element_size()==output.untyped_storage().nbytes()
            expected=None
            if self.validate_outputs:
                reference=torch.empty_like(output)
                reference_args=list(args);reference_args[target.arg_names.index('OUT')]=reference
                original(*reference_args,**kwargs)
                expected=fp8.quantize_fp8(reference)
            kernel=fused._matmul.run(*args,**kwargs)
            if expected is not None:
                expected_bytes=expected.cpu().numpy().tobytes();actual_bytes=output.cpu().numpy().tobytes()
                assert actual_bytes==expected_bytes,('Matrix epilogue output mismatch',ordinal)
                self.proofs.append(dict(ordinal=ordinal,shape=list(output.shape),all_bytes_equal=True,
                    raw_sha256=hashlib.sha256(actual_bytes).hexdigest(),bytes=len(actual_bytes),
                    llir=put(kernel.asm['llir'].encode('utf-8'),'llir'),spills=getattr(kernel,'n_spills',None)))
            self.fused_calls.append(ordinal)
            return kernel
        target.run=run
        try:yield self
        finally:
            assert target.run is run and CONTRACTS[FUSED]==(('OUT',),('OUT',))
            if had:target.run=prior
            else:del target.run
            del CONTRACTS[FUSED]

    def finish(self):
        assert self.calls==self.expected_calls
        assert self.fused_calls==sorted(self.plan)
        if self.validate_outputs:assert len(self.proofs)==len(self.plan)
        return dict(matrix_calls=self.calls,fused_ordinals=self.fused_calls,proofs=self.proofs)


class EpilogueGraphRewrite(FP8GraphRewrite):
    def __init__(self,model,provider):
        super().__init__(model,provider)
        assert hashlib.sha256(PLAN.read_bytes()).hexdigest()==PLAN_SHA
        record=json.loads(PLAN.read_text(encoding='utf-8'));assert record['passed']
        reset,temporal=record['plans']['reset'],record['plans']['temporal']
        for k in ('kernel_ordinal','constants','launch_options','operands','output'):
            assert [r[k] for r in reset['selected']]==[r[k] for r in temporal['selected']]
        assert reset['matrix_calls']==temporal['matrix_calls']==167 and len(temporal['selected'])==92
        self.plan=temporal;self.validate_outputs=False

    def apply(self,module,rgb,front,**options):
        if (module is not self.model or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton'
                or rgb.device.type!='xpu' or tuple(rgb.shape)!=(256,256,3)
                or tuple(front.shape)!=(320,320,16) or options.get('return_float32',False)):
            return self.original(module,rgb,front,**options)
        analysis=RewritingDataflow();epilogues=MatrixEpilogues(self.plan,validate_outputs=self.validate_outputs)
        with epilogues.installed(),analysis.installed():result=self.original(module,rgb,front,**options)
        summary=analysis.rewrite_summary();summary['epilogues']=epilogues.finish()
        self.builds.append(summary)
        return result
