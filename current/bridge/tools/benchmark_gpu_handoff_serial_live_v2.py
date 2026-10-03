"""Luna-only live v2 lane: require actual serial capability, retain v1 evidence."""
from pathlib import Path
import benchmark_gpu_handoff_live_v1 as base

_old_applied = base.applied
_old_pins = base.source_pins


def applied(state, pool, handoff):
    if not _old_applied(state, pool, handoff):
        return False
    cap = state.get('gpu_handoff') or {}
    host = cap.get('host') or {}
    if handoff:
        return (cap.get('serial_owner_transfer_armed') is True and
                host.get('serial_owner_transfer') is True)
    return (cap.get('serial_owner_transfer_armed', False) is False and
            host.get('serial_owner_transfer', False) is False)


def pins(receipt):
    result = _old_pins(receipt)
    result[str(Path(__file__).resolve())] = base.sha(Path(__file__))
    return result


def main():
    # No live calls or GPU imports: these CPU examples guard the strict ON gate.
    example = {'bridge_resource_pool': {'enabled': False}, 'gpu_handoff': {
        'requested': True, 'armed': True, 'cap_healthy': True, 'pending': False,
        'host': {'applied': True, 'cap_healthy': True}}}
    assert _old_applied(example, False, True)
    assert not applied(example, False, True)
    example['gpu_handoff']['serial_owner_transfer_armed'] = True
    assert not applied(example, False, True)
    example['gpu_handoff']['host']['serial_owner_transfer'] = True
    assert applied(example, False, True)
    example['gpu_handoff']['armed'] = False
    assert not applied(example, False, True)
    base.applied, base.source_pins = applied, pins
    base.main()


if __name__ == '__main__':
    main()
