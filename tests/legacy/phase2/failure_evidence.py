"""Persist a mismatching warm frame after timing, keeping strict rejection."""
def persist_mismatch(np, current, expected, destination, checked, output_path, record):
    source = checked(expected)
    cold = np.load(source, allow_pickle=False)
    if cold.shape != current.shape or cold.dtype != current.dtype:
        raise RuntimeError('Mismatching warm diagnostic ABI changed')
    if not np.isfinite(cold).all() or not np.isfinite(current).all():
        raise RuntimeError('Nonfinite numeric diagnostic input')
    destination = output_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        np.save(stream, current, allow_pickle=False)
    delta = np.abs(current.astype(np.float64) - cold.astype(np.float64))
    return dict(cold_file=expected, warm_file=record(destination),
        shape=list(current.shape), dtype=current.dtype.str, finite=True,
        unequal_elements=int(np.count_nonzero(current != cold)),
        unequal_bytes=int(np.count_nonzero(current.view(np.uint8) != cold.view(np.uint8))),
        mae=float(delta.mean()), max_abs=float(delta.max()),
        acceptance='FAILED_UNACCEPTED; diagnostic persistence never relaxes equality',
        inside_measured_event_boundary=False)
