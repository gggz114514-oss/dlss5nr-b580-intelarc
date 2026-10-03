"""One transient pool for serialized experimental adapters with external outputs.

The benchmark owns all adapters, completes every replay before the next one,
and keeps every model, static input and output outside this shared pool. This
helper does not implement a production concurrent-session or queue interface.
"""
import torch
from graph_front_v6 import GraphFront


def share_before_capture(adapters):
    adapters = list(adapters)
    if len(adapters) < 2 or len({id(adapter) for adapter in adapters}) != len(adapters):
        raise ValueError('Expected distinct serialized adapters')
    for adapter in adapters:
        if type(adapter) is not GraphFront or adapter.entries or adapter._graph_cache is not None or adapter.replays:
            raise ValueError('Only fresh v6 adapters with pool-external buffers may join')
    stream, pool = torch.xpu.Stream(), torch.xpu.graph_pool_handle()
    for adapter in adapters:
        adapter.capture_stream = stream
        adapter.capture_pool = pool
    return dict(adapters=len(adapters), pool=str(pool), serialized_only=True,
                completion_before_each_replay=True, persistent_buffers_shared=False)
