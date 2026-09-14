"""Explicit context-local arithmetic selection; reference remains the default.

The yielded counters record successful Python dispatches. GPU completion and
numerical correctness require synchronization and output validation separately.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_backend = ContextVar('nr_arithmetic_backend', default='reference')
_counters = ContextVar('nr_arithmetic_dispatches', default=None)

def current_arithmetic_backend():
    return _backend.get()

def record_arithmetic_dispatch(kind):
    counters = _counters.get()
    if counters is not None:
        counters[kind] = counters.get(kind, 0) + 1

@contextmanager
def use_arithmetic_backend(backend):
    if type(backend) is not str or backend not in ('reference', 'triton'):
        raise ValueError('Expected arithmetic backend reference or triton')
    counters = {'backend': backend, 'dense': 0, 'batched': 0}
    backend_token = _backend.set(backend)
    counters_token = _counters.set(counters)
    try:
        yield counters
    finally:
        _counters.reset(counters_token)
        _backend.reset(backend_token)
