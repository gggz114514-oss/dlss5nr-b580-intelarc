"""Local compiler diagnostic: trace three layout passes or omit exactly one.

The source of make_ttgir is pinned; no installed files change. Selection zero
traces boundaries without omitting a pass. Selection1..3 keeps the original
pipeline grouping but omits that one optional pass. Not a production fix.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from triton import knobs
from triton._C.libtriton import ir,passes,intel
from triton.backends.compiler import Language
COMPILER=Path('E:\\ComfyUI-aki-v3-IntelArc_20260722\\ComfyUI-aki-v3-IntelArc\\python\\Lib\\site-packages\\triton\\backends\\intel\\compiler.py')
COMPILER_SHA256='bb48bc9c95215733092e407ce34f975a3a2717cfff64308f378ba4702ab9548c'

def make_ttgir(cls, mod, metadata, opt, properties, selected):
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
        if selected==0:
            pm.run(mod,'nr-before-layout-1')
            pm=ir.pass_manager(mod.context);pm.enable_debug()
            print('NR_COMPILER_ENTER_LAYOUT_1',flush=True)
        if selected!=1:
            intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        if selected==0:
            pm.run(mod,'nr-layout-1')
            pm=ir.pass_manager(mod.context);pm.enable_debug()
            print('NR_COMPILER_EXIT_LAYOUT_1',flush=True)

        intel.passes.ttgpuir.add_accelerate_matmul(pm)
        intel.passes.ttgpuir.add_materialize_block_pointer(pm)
        if selected==0:
            pm.run(mod,'nr-before-layout-2')
            pm=ir.pass_manager(mod.context);pm.enable_debug()
            print('NR_COMPILER_ENTER_LAYOUT_2',flush=True)
        if selected!=2:
            intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        if selected==0:
            pm.run(mod,'nr-layout-2')
            pm=ir.pass_manager(mod.context);pm.enable_debug()
            print('NR_COMPILER_EXIT_LAYOUT_2',flush=True)
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
        if selected==0:
            pm.run(mod,'nr-before-layout-3')
            pm=ir.pass_manager(mod.context);pm.enable_debug()
            print('NR_COMPILER_ENTER_LAYOUT_3',flush=True)
        if selected!=3:
            intel.passes.ttgpuir.add_remove_layout_conversions(pm)
        if selected==0:
            pm.run(mod,'nr-layout-3')
            pm=ir.pass_manager(mod.context);pm.enable_debug()
            print('NR_COMPILER_EXIT_LAYOUT_3',flush=True)
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
def selected_pipeline(selected):
    assert selected in (0,1,2,3)
    assert hashlib.sha256(COMPILER.read_bytes()).hexdigest()==COMPILER_SHA256
    assert knobs.runtime.add_stages_inspection_hook is None
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def hook(backend=None,stages=None,options=None,language=None,capability=None):
        if stages is None:return f'nr-c32-select-layout-{selected}-'+digest,digest
        assert language==Language.TRITON and backend.binary_ext=='spv'
        stages['ttgir']=lambda src,metadata:make_ttgir(type(backend),src,metadata,options,backend.properties,selected)
    knobs.runtime.add_stages_inspection_hook=hook
    try:yield
    finally:
        assert knobs.runtime.add_stages_inspection_hook is hook
        knobs.runtime.add_stages_inspection_hook=None
