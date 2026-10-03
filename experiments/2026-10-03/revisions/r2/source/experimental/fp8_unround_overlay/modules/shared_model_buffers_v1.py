"""Share verified immutable constants between fresh NR sessions, never their history."""
import torch


def share_identical_buffers(model, template):
    if type(model) is not type(template) or model is template:
        raise ValueError('Expected distinct sessions of the same model type')
    if any(m._previous is not None or m.next_seed != 0 for m in (model, template)):
        raise ValueError('Only fresh sessions may share constants')
    own = dict(model.named_buffers(remove_duplicate=False))
    source = dict(template.named_buffers(remove_duplicate=False))
    if own.keys() != source.keys() or '_previous' in own:
        raise ValueError('Different buffer topology or initialized history')
    old_storage = {}
    with torch.no_grad():
        for name, value in own.items():
            other = source[name]
            if (value.shape, value.dtype, value.device, value.stride()) != (other.shape, other.dtype, other.device, other.stride()):
                raise ValueError('Buffer descriptor mismatch: ' + name)
            # Compare raw bytes; equal NaN payloads must not fail a value equality test.
            if not torch.equal(value.contiguous().reshape(-1).view(torch.uint8), other.contiguous().reshape(-1).view(torch.uint8)):
                raise ValueError('Buffer contents differ: ' + name)
            storage = value.untyped_storage()
            old_storage[(str(value.device), storage.data_ptr())] = storage.nbytes()
        # All checks complete before any assignment. The template stays unchanged.
        for name in own:
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, leaf, source[name])
    return dict(buffers_shared=len(own), replaced_unique_storage_bytes=sum(old_storage.values()),
                all_contents_verified_as_raw_bytes=True, history_shared=False)
