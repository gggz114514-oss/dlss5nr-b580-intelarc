"""Authenticate temporary model.forward layers from this numerical session.

This is host metadata only. A registered parent chain proves which observers
will execute; it does not relax the model's mathematical or GPU contracts.
"""
from __future__ import annotations


REGISTRY = "_numeric_model_forward_registry_720"


def _same(left, right):
    return (getattr(left, "__self__", None) is getattr(right, "__self__", None)
            and getattr(left, "__func__", left) is getattr(right, "__func__", right))


def _model(counter):
    result = getattr(counter, "model", None)
    return result if result is not None else counter.stack.model


def _owner(counter, model, session):
    if counter.session is not session or _model(counter) is not model:
        raise RuntimeError("Model forward wrapper belongs to a foreign numerical session")
    if not any(session.__dict__.get(name) is counter for name in
               ("_vit_numeric_suite_720", "_history_numeric_suite_720")):
        raise RuntimeError("Model forward observer is not an installed ViT/history child")
    modes = counter.modes
    if modes.session is not session:
        raise RuntimeError("Model forward observer's active session changed")
    suite = getattr(modes, "_numeric_cleanup_owner", None)
    if suite is not None and not any(child is counter for child in suite.children.values()):
        raise RuntimeError("Model forward observer is absent from the fixed suite")


def register_forward(wrapper, parent, counter):
    """Register before assigning the replacement bound method to the model."""
    session, model = counter.session, _model(counter)
    try:
        _owner(counter, model, session)
        if (getattr(wrapper, "__self__", None) is not model
                or getattr(parent, "__self__", None) is not model):
            raise RuntimeError("Expected bound model methods for numerical wrapping")
        function = wrapper.__func__
        registry = session.__dict__.setdefault(REGISTRY, {})
        if not isinstance(registry, dict) or function in registry:
            raise RuntimeError("Model forward wrapper already has an owner")
        function.__nr_numeric_model_parent__ = parent
        function.__nr_numeric_model_owner__ = counter
        registry[function] = (function.__code__, parent, counter)
        return wrapper
    except BaseException:
        session._failed = True
        raise


def unregister_forward(wrapper, counter):
    """Remove only this layer, after its original model method was restored."""
    session = counter.session
    registry = session.__dict__.get(REGISTRY)
    function = wrapper.__func__
    row = registry.get(function) if isinstance(registry, dict) else None
    if row is None or row[2] is not counter:
        session._failed = True
        raise RuntimeError("Model forward wrapper ownership changed before restoration")
    del registry[function]
    if not registry:
        session.__dict__.pop(REGISTRY, None)


def require_forward(current, base, model, session):
    """Accept the exact base or a registered, same-session chain ending there."""
    try:
        if getattr(base, "__self__", None) is not model:
            raise RuntimeError("Frozen model forward base belongs to another model")
        registry = session.__dict__.get(REGISTRY, {})
        if not isinstance(registry, dict):
            raise RuntimeError("Model forward registry was replaced")
        seen = set()
        while not _same(current, base):
            if getattr(current, "__self__", None) is not model:
                raise RuntimeError("Model forward chain left its owned model")
            function = current.__func__
            if function in seen:
                raise RuntimeError("Cyclic model forward chain")
            seen.add(function)
            row = registry.get(function)
            if row is None:
                raise RuntimeError("Unregistered model forward replacement")
            code, parent, counter = row
            _owner(counter, model, session)
            if (function.__code__ is not code
                    or getattr(function, "__nr_numeric_model_parent__", None) is not parent
                    or getattr(function, "__nr_numeric_model_owner__", None) is not counter):
                raise RuntimeError("Registered model forward parent or callable changed")
            current = parent
        return len(seen)
    except BaseException:
        session._failed = True
        raise
