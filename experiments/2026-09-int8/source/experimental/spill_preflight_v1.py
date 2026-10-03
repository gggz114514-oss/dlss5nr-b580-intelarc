"""Compile/load reviewed launch choices before dispatch and reject spilled variants.

Uses the pinned Intel Triton runtime's warmup and handle initialization APIs.
Loading a kernel exposes driver-reported GRF/scratch metadata without launching
it. This is construction-time selection, not per-frame GPU replay logic.
Zero spills is a resource gate for this experiment, not a speed guarantee.

Memoization
-----------
The reviewed answer is a pure function of ``(jit, configs, options, argument
signature)``: ``jit.warmup`` returns the compiled kernel already held by the
Triton kernel cache, and spills / registers / shared bytes are properties of
that compiled artifact. A frame loop repeats the same call sites every frame, so
the answer is memoized under a key that refines Triton's own specialization
(non-constexpr tensors contribute dtype plus 16-byte alignment, everything else
contributes its full value, so shapes carried as constexpr are part of the key).

A hit returns the *identical* ``(config, kernel, choice)`` triple the uncached
path would have returned, which is why it cannot change a byte. A sampled
self-check re-runs the real path and compares kernel identity, so a stale key
survives at most ``SAMPLE - 1`` calls and every failure is counted in
``select_stats()['verify_fail']`` -- never silent. ``NR_SELECT_CACHE=off``
bypasses the memo entirely.
"""
import os

SAMPLE = int(os.environ.get('NR_SELECT_SAMPLE', '32'))

_ENABLED = os.environ.get('NR_SELECT_CACHE', 'on').strip().lower() not in ('', 'off', 'none', '0')
_CACHE = {}
_PLANS = {}
_STATS = dict(calls=0, hits=0, misses=0, fallback=0, verify=0, verify_ok=0,
              verify_fail=0, retired=0)


def _plan_for(jit):
    """Per-jit argument index plan, built once.

    `jit.params` is rebuilt on every access, and this runs ~189 times per frame,
    so the index split is computed once per JITFunction instead of per call.
    """
    entry = _PLANS.get(id(jit))
    if entry is not None and entry[0] is jit:
        return entry[1]
    params = list(jit.params)
    plan = (tuple(i for i, p in enumerate(params) if not p.is_constexpr),
            tuple(i for i, p in enumerate(params) if p.is_constexpr))
    _PLANS[id(jit)] = (jit, plan)
    return plan


def _signature(jit, args):
    """Refine Triton's specialization into a hashable per-argument signature."""
    tx, cx = _plan_for(jit)
    parts = []
    for index in tx:
        value = args[index]
        if hasattr(value, 'data_ptr') and hasattr(value, 'dtype'):
            parts.append((value.dtype, value.data_ptr() & 15))
        else:
            hash(value); parts.append(value)
    for index in cx:
        value = args[index]
        hash(value); parts.append(value)
    return tuple(parts)


def _select_uncached(jit,configs,args_for,grid_for,**options):
    attempts=[]
    for config in configs:
        args=args_for(config);grid=grid_for(config)
        kernel=jit.warmup(*args,grid=grid,**options)
        kernel._init_handles()
        spills=getattr(kernel,'n_spills',None);regs=getattr(kernel,'n_regs',None)
        if not isinstance(spills,int) or spills<0:raise RuntimeError('Unavailable compiler spill metadata')
        attempts.append(dict(config=list(config),spills=spills,registers=regs,shared_bytes=kernel.metadata.shared))
        if spills==0:return config,kernel,dict(selected=list(config),attempts=attempts)
    raise RuntimeError(f'All reviewed launch configurations spill: {attempts}')


def select(jit,configs,args_for,grid_for,**options):
    if not _ENABLED:
        return _select_uncached(jit,configs,args_for,grid_for,**options)
    _STATS['calls']+=1
    try:
        key=(id(jit),tuple(tuple(config) for config in configs),
             tuple(sorted((name,str(value)) for name,value in options.items())),
             _signature(jit,args_for(configs[0])))
        hash(key)
    except Exception:
        _STATS['fallback']+=1
        return _select_uncached(jit,configs,args_for,grid_for,**options)
    hit=_CACHE.get(key)
    if hit is not None and hit[0] is jit:
        cached=hit[1]
        _STATS['hits']+=1
        if _STATS['hits']%SAMPLE==0:
            _STATS['verify']+=1
            real=_select_uncached(jit,configs,args_for,grid_for,**options)
            # Both `cached` and `real` are (config, kernel, choice) triples; the
            # identity that matters is the kernel object they point at.
            if real[1] is cached[1]:
                _STATS['verify_ok']+=1
            else:
                _STATS['verify_fail']+=1;_STATS['retired']+=1
                _CACHE.pop(key,None)
                return real
        return cached
    _STATS['misses']+=1
    value=_select_uncached(jit,configs,args_for,grid_for,**options)
    _CACHE[key]=(jit,value)
    return value


def select_stats():
    """Counters for the frame report; `verify_fail == 0` is the correctness gate."""
    return dict(_STATS,enabled=_ENABLED,entries=len(_CACHE),sample=SAMPLE)
