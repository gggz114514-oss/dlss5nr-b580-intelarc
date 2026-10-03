"""Diagnostic stage boundaries with full-body provenance and stride-preserved inputs.

The caller records the complete selected body first, verifies its output and
physical operation sequence, then captures isolated stage replays. These stage
timings change scheduling/cache boundaries and are not additive frame costs.
"""
import torch
from full_body_dataflow_v1 import DetailedDataflow
from quantization_dataflow_v1 import FP8


def forward_staged(model, rgb, front, stage, *, previous=None, sigmoid=None,
                   blend_scale=None, history_reciprocal=None, return_float32=False):
    pre_skip, x = stage('pre', model.pre.forward_features_outputs, front)
    skips = []

    def encoder(group, features):
        for block in group:
            skip, down = block.forward_outputs(features)
            features = down if down is not None else skip
        return skip, features

    for channels, group in zip((32, 64, 128, 256), model.encoder):
        skip, x = stage(f'encoder_C{channels}', lambda a, g=group: encoder(g, a), x)
        skips.append(skip)

    def encoder512(features):
        for i, block in enumerate(model.encoder512):
            values = block.forward_boundaries(features)
            features = values[-1]
            if i == 7:
                skip = values[3]
        return skip, features

    skip512, x = stage('encoder_C512', encoder512, x)
    vit_shape = x.shape

    def vit(features):
        features = features.reshape(-1, 1024)
        for block in model.vit:
            features = block(features)
        return features

    x = stage('ViT_8_blocks', vit, x)

    def decoder512(features, skip):
        features = model.decoder_input(features.reshape(vit_shape), skip)
        for block in model.decoder512:
            features = block(features)
        return features

    x = stage('decoder_C512_with_input', decoder512, x, skip512)

    def decoder(group, features, skip):
        features = group[0](features, skip)
        for block in group[1:]:
            features = block(features)
        return features

    for channels, group, skip in zip((256, 128, 64, 32), model.decoder, reversed(skips)):
        x = stage(f'decoder_C{channels}', lambda a, b, g=group: decoder(g, a, b), x, skip)

    def post(features, pre, color, history, reciprocal):
        return model.post(features, pre, color, previous=history, sigmoid=sigmoid,
            blend_scale=blend_scale, history_reciprocal=reciprocal, return_float32=return_float32)

    return stage('post', post, x, pre_skip, rgb, previous, history_reciprocal)


def signature(events):
    """Physical Triton sequence, launch policy and tensor descriptors, no addresses."""
    descriptor = lambda r: {k: r[k] for k in ('shape', 'stride', 'dtype', 'offset')}
    return [dict(name=e['name'], grid=e['grid'], constants=e['constants'],
                 launch_options=e['launch_options'],
                 reads={n: descriptor(r) for n, r in e['reads'].items()},
                 writes={n: descriptor(r) for n, r in e['writes'].items()})
            for e in events if e['kind'] == 'triton']


class Recorder:
    def __init__(self, analysis):
        self.analysis = analysis
        self.stages = []

    def __call__(self, name, fn, *args):
        a = self.analysis
        start, q, elided = len(a.events), a.quantization_calls, len(a.elisions)
        references = [None if t is None else a.ref(t) for t in args]
        result = fn(*args)
        outputs = result if isinstance(result, tuple) else (result,)
        self.stages.append(dict(name=name, fn=fn, args=args, outputs=outputs,
            input_provenance=references, events=a.events[start:],
            quantization_calls=a.quantization_calls - q, elided_fp8=len(a.elisions) - elided))
        return result


def own_inputs(item):
    """Copy whole backing stores once per alias group, retaining offset/stride."""
    roots, args = {}, []
    for t, ref in zip(item['args'], item['input_provenance']):
        if t is None:
            args.append(None)
            continue
        key = ref['storage']
        assert t.untyped_storage().nbytes() == ref['storage_bytes']
        if key not in roots:
            flat = t.as_strided((ref['storage_bytes'] // t.element_size(),), (1,), 0)
            roots[key] = dict(tensor=flat.clone(), domain=ref['domain'], reference=ref)
        root = roots[key]
        assert root['tensor'].dtype == t.dtype and root['domain'] == ref['domain']
        owned = root['tensor'].as_strided(t.shape, t.stride(), t.storage_offset())
        assert owned.shape == t.shape and owned.stride() == t.stride()
        assert owned.storage_offset() == t.storage_offset()
        assert torch.equal(owned.view(torch.int16), t.view(torch.int16)) if t.dtype == torch.float16 else torch.equal(owned, t)
        args.append(owned)
    return tuple(args), list(roots.values())


def run_seeded(item, args, roots):
    a = DetailedDataflow()
    # This proof is restricted to copies of full-body traced backing stores.
    # It is not inferred merely from the range of observed numeric values.
    for root in roots:
        domain = root['domain']
        assert domain in (None, FP8)
        if domain is not None:
            index = len(a.events)
            event = dict(id=index, kind='input_proof', name='full_body_storage_copy',
                         original=root['reference'])
            a.events.append(event)
            event['output'] = a.write(root['tensor'], index, domain)
            assert a.ref(root['tensor'])['domain'] == domain
    with a.installed():
        result = item['fn'](*args)
    assert signature(a.events) == signature(item['events']), item['name']
    assert a.quantization_calls == item['quantization_calls'], item['name']
    assert len(a.elisions) == item['elided_fp8'], item['name']
    return result, a
