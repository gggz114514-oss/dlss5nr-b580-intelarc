"""Stage boundaries for the existing C512-layout body, without changing kernels.

Uses the existing layout.split directly, including its unquantized pooling path.
Recorder, storage ownership and provenance seeding are the frozen helpers.
The runner must prove the full physical sequence and output before timing.
"""
from selected_body_stage_graphs_v1 import Recorder, own_inputs, run_seeded, signature


def forward_staged(layout, model, rgb, front, stage, **options):
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
            output, features = layout.split(block, features)
            if i == 7:
                skip = output
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
            output, features = layout.split(block, features)
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
        return model.post(features, pre, color, previous=history,
                          history_reciprocal=reciprocal, **post_options)

    post_options = {k: v for k, v in options.items() if k not in ('previous', 'history_reciprocal')}
    return stage('post', post, x, pre_skip, rgb,
                 options.get('previous'), options.get('history_reciprocal'))
