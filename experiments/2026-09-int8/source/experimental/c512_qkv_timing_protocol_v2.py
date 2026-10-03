"""Host-completion timing rechecks; no changes to arithmetic or graph outputs."""
import statistics
import time


def measure(workloads, *, name, warmup_replays, replays_per_sample,
            verify_per_workload, synchronize, digest, checkpoint, destination):
    assert workloads and warmup_replays >= 0 and replays_per_sample > 0
    block = dict(name=name, rounds=12, warmup_replays=warmup_replays,
                 replays_per_sample=replays_per_sample,
                 verify_per_workload=verify_per_workload,
                 clock='perf_counter_ns_with_device_completion',
                 orders=[], validated_orders=[], workloads=[], comparisons=[])
    destination.append(block)
    rows = {}
    for work in workloads:
        row = dict(name=work['stage'], samples_ms=[], elapsed_ns=[], median_ms=None,
                   output_hashes=work['expected'])
        rows[work['stage']] = row
        block['workloads'].append(row)

    def verify(work):
        assert [digest(t) for t in work['outputs']] == work['expected'], work['stage']

    for repeat in range(12):
        offset = repeat % len(workloads)
        order = workloads[offset:] + workloads[:offset]
        if repeat % 2:
            order = order[::-1]
        block['orders'].append([w['stage'] for w in order])
        checked = []
        for work in order:
            # Warm each route immediately before its own sample. It cannot borrow
            # timing credit from a different route's compilation or CPU checks.
            for _ in range(warmup_replays):
                work['graph'].replay()
            synchronize()
            started = time.perf_counter_ns()
            for _ in range(replays_per_sample):
                work['graph'].replay()
            synchronize()
            elapsed = time.perf_counter_ns() - started
            assert elapsed > 0
            rows[work['stage']]['elapsed_ns'].append(elapsed)
            rows[work['stage']]['samples_ms'].append(elapsed / (replays_per_sample * 1e6))
            if verify_per_workload:
                verify(work)
                checked.append(work['stage'])
        if not verify_per_workload:
            for work in order:
                verify(work)
                checked.append(work['stage'])
        block['validated_orders'].append(checked)
        checkpoint()
    for row in block['workloads']:
        row['median_ms'] = statistics.median(row['samples_ms'])
    base = block['workloads'][0]
    for row in block['workloads'][1:]:
        a, b = base['median_ms'], row['median_ms']
        block['comparisons'].append(dict(route=row['name'], baseline_ms=a, candidate_ms=b,
            saving_ms=a-b, saving_percent=(a-b)/a*100,
            faster_rounds=sum(y < x for x, y in zip(base['samples_ms'], row['samples_ms']))))
    checkpoint()
    return block
