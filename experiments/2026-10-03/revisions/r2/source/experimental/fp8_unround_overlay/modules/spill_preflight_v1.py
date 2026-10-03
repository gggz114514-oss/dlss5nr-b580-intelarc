"""Compile/load reviewed launch choices before dispatch and reject spilled variants.

Uses the pinned Intel Triton runtime's warmup and handle initialization APIs.
Loading a kernel exposes driver-reported GRF/scratch metadata without launching
it. This is construction-time selection, not per-frame GPU replay logic.
Zero spills is a resource gate for this experiment, not a speed guarantee.
"""
def select(jit,configs,args_for,grid_for,**options):
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
