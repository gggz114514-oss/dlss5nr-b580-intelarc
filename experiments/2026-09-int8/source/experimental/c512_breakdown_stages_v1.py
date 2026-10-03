"""Trace-only boundaries around the unchanged C512 layout computations.

The caller proves the complete physical sequence against the actual stack.
Instance hooks exist only for that trace, are restored before isolated captures,
and never change a public model class. Pooling retains unquantized full output.
"""
import torch
import nr_backend.split_block as blocks
from nr_backend.execution import record_arithmetic_dispatch
from c512_window_projection_v1 import forward as project
from selected_body_stage_graphs_v1 import Recorder as BaseRecorder, own_inputs, run_seeded, signature
from repaired_body_stage_graphs_v1 import forward_staged as broad_forward


class Recorder(BaseRecorder):
    def __init__(self, analysis):
        super().__init__(analysis)
        self.c512_details = []


def split_staged(layout, module, features, record):
    assert id(module) in layout.modules and tuple(features.shape) == (12, 12, 512)
    assert layout.probe is None
    name = layout.modules[id(module)]
    height, width = features.shape[:2]
    sy, sx = module.window_shift

    def ffn(x):
        ffwd = module.ffwd(x)
        return module.ffwd_projection(ffwd, x)

    mlp = record(name + '.ffn', ffn, features)

    def attention(x):
        padded = torch.nn.functional.pad(x, (0, 0, sx, (-width-sx) % 8, sy, (-height-sy) % 8))
        return layout.stack.window_blocks.windows(module.attention, padded)

    packed = record(name + '.attention', attention, mlp)

    def projection(packed, mlp):
        full, kernel, selection = project(packed, module.projection.weight, blocks.q(mlp),
            module.projection.skip_scale, module.attention.pixel_inverse, shift=(sy, sx))
        layout.resources[kernel.hash] = dict(spills=kernel.n_spills, registers=kernel.n_regs,
                                            shared_bytes=kernel.metadata.shared)
        layout.selections[name] = selection
        layout.provider.record('fp16_dense')
        record_arithmetic_dispatch('dense')
        for _ in range(2):
            record_arithmetic_dispatch('fp8')
        output = blocks.q(full)
        final = None
        if module.final_weight is not None:
            top = (full[0::2, 0::2] + full[0::2, 1::2]).half()
            bottom = (full[1::2, 0::2] + full[1::2, 1::2]).half()
            pool = blocks.q(((top + bottom).half() * .25).half())
            pool = torch.nn.functional.pad(pool, (0, 0, 0, (-pool.shape[1]) % 4, 0, (-pool.shape[0]) % 4))
            final = blocks.q(blocks.dot(pool, module.final_weight, chunk_k=16))
        layout.calls += 1
        layout.blocks[name] = layout.blocks.get(name, 0) + 1
        return output, output if final is None else final

    return record(name + '.projection_pool', projection, packed, mlp)


def forward_staged(layout, model, rgb, front, stage, **options):
    assert isinstance(stage, Recorder) and not stage.c512_details
    assert 'split' not in layout.__dict__ and 'forward' not in model.decoder_input.__dict__
    old_split, old_input = layout.split, model.decoder_input.forward
    detail = BaseRecorder(stage.analysis)
    split_replacement = lambda module, features: split_staged(layout, module, features, detail)
    input_replacement = lambda features, skip: detail('decoder_input', old_input, features, skip)
    layout.split, model.decoder_input.forward = split_replacement, input_replacement
    try:
        result = broad_forward(layout, model, rgb, front, stage, **options)
        assert len(detail.stages) == 49
        stage.c512_details = detail.stages
        return result
    finally:
        valid = layout.split is split_replacement and model.decoder_input.forward is input_replacement
        del layout.split
        del model.decoder_input.forward
        assert valid and layout.split == old_split and model.decoder_input.forward == old_input
