"""Process-local diagnostic compiler pipeline; no installed file modifications.

Reproduced from the pinned installed make_ttgir, omitting exactly three optional
RemoveLayoutConversions calls. This is NOT a fix or a production recommendation.
The public inspection hook changes the disk-cache key and is always restored.
Only uniquely named candidate kernels enter this context; baselines use stock.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from triton import knobs
from triton._C.libtriton import ir,passes,intel
from triton.backends.compiler import Language
COMPILER=Path('E:\\ComfyUI-aki-v3-IntelArc_20260722\\ComfyUI-aki-v3-IntelArc\\python\\Lib\\site-packages\\triton\\backends\\intel\\compiler.py')
COMPILER_SHA256='bb48bc9c95215733092e407ce34f975a3a2717cfff64308f378ba4702ab9548c'

def make_ttgir(cls, mod, metadata, opt, properties):
        # Annotate module with information required by subsequent transformations.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        module_opts = intel.passes.ttgpuir.AnnotateModuleOptions()
        cls.annotate_module(module_opts, properties, opt)
        module_opts.is_lts = cls.is_lts(metadata["target"].arch.get("driver_version"))
        module_opts.use_cl_rounded_divide_sqrt = (module_opts.is_lts and intel.has_precise_divide_sqrt(mod))
        intel.passes.ttgpuir.add_triton_annotate_module(pm, module_opts)
        pm.run(mod, 'annotate_module')

        # Overwrite the warp_size option with the module annotation.
        opt.warp_size = intel.get_threads_per_warp(mod)
        cls.validate_options(opt, properties)

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, "xpu", opt.num_warps, opt.warp_size, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        if properties["has_256b_load_store"]:
            intel.passes.ttgpuir.add_widen_load_store_encoding(pm)
        # Diagnostic: omit the crashing optional layout-rematerialization pass.

        intel.passes.ttgpuir.add_accelerate_matmul(pm)
        intel.passes.ttgpuir.add_materialize_block_pointer(pm)
        # Diagnostic: omit the crashing optional layout-rematerialization pass.
        intel.passes.ttgpuir.add_optimize_dot_operands(pm)
        intel.passes.ttgpuir.add_hoist_layout_conversions(pm, opt.grf_mode)
        intel.passes.ttgpuir.add_pipeline(pm, opt.num_stages, opt.use_barrier)

        if (opt.reduce_variable_liveness):
            intel.passes.ttgpuir.add_reduce_variable_liveness(pm)

        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_fuse_nested_loops(pm)

        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.common.add_cse(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        # Diagnostic: omit the crashing optional layout-rematerialization pass.
        if not knobs.intel.disable_annotate_cache_control:
            intel.passes.ttgpuir.add_annotate_cache_control(pm)
        intel.passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_canonicalizer(pm)
        if knobs.intel.opt_reduction_locality:
            intel.passes.ttgpuir.add_optimize_reduction_locality(pm)
        intel.passes.arith.add_arith_emulate_unsupported_floats(pm, ["bf16"], "f32")
        if opt.instrumentation_mode == "fpsan":
            passes.ttgpuir.add_fp_sanitizer(pm)
        pm.run(mod, 'make_ttgir')
        return mod


@contextmanager
def diagnostic_pipeline():
    assert hashlib.sha256(COMPILER.read_bytes()).hexdigest()==COMPILER_SHA256
    assert knobs.runtime.add_stages_inspection_hook is None
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def hook(backend=None,stages=None,options=None,language=None,capability=None):
        if stages is None:return 'nr-c32-no-layout-v1-'+digest,digest
        assert language==Language.TRITON and backend.binary_ext=='spv'
        stages['ttgir']=lambda src,metadata:make_ttgir(type(backend),src,metadata,options,backend.properties)
    knobs.runtime.add_stages_inspection_hook=hook
    try:yield
    finally:
        assert knobs.runtime.add_stages_inspection_hook is hook
        knobs.runtime.add_stages_inspection_hook=None
