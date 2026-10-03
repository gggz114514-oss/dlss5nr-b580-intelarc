"""Fixed four-family audits at load/build; live gates under the owned frame.

The flag/session/entry authority lives in replay_lifecycle_audit_base_720_v1.
No GPU import, compiler pin, cache key or math implementation is changed here.
Supported resource maps revoke on mutation. Debug mutation/restore while the
adapter serial lock is held is outside this interface's product inputs.
"""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
import sys
from threading import get_ident
from types import MethodType
from pathlib import Path
import hashlib

LABELS = ('history', 'vit', 'post', 'front')


def _binding(child):
    return getattr(child, '_lifecycle_rest_720', None)


@contextmanager
def fixed_audit(child):
    binding = _binding(child)
    if binding is None:
        yield
        return
    binding.full_depth += 1
    try:
        yield
    except BaseException:
        binding.state_fail('Explicit/static family audit failed')
        raise
    finally:
        binding.full_depth -= 1


def hot_active(child):
    binding = _binding(child)
    if binding is None or binding.full_depth:
        return False
    from replay_lifecycle_audit_base_720_v1 import state_for
    state = state_for(getattr(child.modes, '_numeric_cleanup_owner', None))
    if state is None or not state.armed:
        return False
    state.require_frame()
    if state.children.get(binding.label) is not child:
        state.fail('Lifecycle four-family child owner changed')
    # Pending means actual descriptor miss: every original warmup/capture gate.
    return state.pending is None


def live_guard(child, label, *, check_thread=False):
    if not hot_active(child):
        return False
    _binding(child).live(check_thread=check_thread)
    return True


def history_ready(child):
    if hot_active(child):
        _binding(child).resources_current()
        return True
    return set(child._compiled) == child._specializations()


def history_binary(child, label, kernel, *, launched=False):
    expected = child._compiled[label]
    if hot_active(child):
        binding = _binding(child)
        binding.resources_current()
        if kernel is not expected or kernel.kernel is not binding.payloads.get(label):
            raise RuntimeError('History dispatch lost the actual screened owner: ' + label)
    elif launched:
        # Preserve the default-off/cold launch gate and its original exception.
        if kernel is not expected or kernel.hash != child.resources[label]['kernel_hash']:
            raise RuntimeError(f'History launch missed its screened kernel: {label}')
    elif (kernel is not expected or kernel.hash != child.resources[label]['kernel_hash'] or
            kernel.kernel is not child._binary_objects[label] or
            type(kernel.n_spills) is not int or kernel.n_spills != 0):
        raise RuntimeError(f'History dispatch changed its screened actual binary: {label}')


def history_binding(child):
    if not hot_active(child):
        return False
    fused, _ = child._history_binding
    counter = child._history_counter
    if (child.modules['history'].warp_history_fused is not fused or
            getattr(child.session, '_history_numeric_suite_720', None) is not counter):
        raise RuntimeError('Frozen fused/history child binding changed')
    if counter is not None:
        suite = child._history_suite
        if (counter.session is not child.session or counter.model is not child.model or
                counter.modes is not child.modes or counter.reference_game is not fused or
                not counter.active or counter.retired or
                getattr(child.modes, '_numeric_cleanup_owner', None) is not suite or
                suite.session is not child.session or suite.children.get('history') is not counter or
                suite.children.get('vit') is not child):
            raise RuntimeError('Owned history dispatcher/session/model changed')
    return True


class _ResourceMap(dict):
    """Mutations through the supported resource table revoke the ready session."""
    def __init__(self, binding, name, rows):
        super().__init__(rows)
        self.binding, self.name = binding, name
    def revoke(self):
        self.binding.state_fail('Fixed resource table changed: ' + self.name)
    def __setitem__(self, key, value):
        self.revoke()
        return super().__setitem__(key, value)
    def __delitem__(self, key):
        self.revoke()
        return super().__delitem__(key)
    def clear(self):
        self.revoke()
        return super().clear()
    def pop(self, *args):
        self.revoke()
        return super().pop(*args)
    def popitem(self):
        self.revoke()
        return super().popitem()
    def update(self, *args, **kw):
        self.revoke()
        return super().update(*args, **kw)
    def setdefault(self, key, default=None):
        if key not in self: self.revoke()
        return super().setdefault(key, default)
    def __ior__(self, other):
        self.update(other)
        return self
    def __copy__(self):
        return dict(self)
    def __deepcopy__(self, memo):
        # Diagnostics copy receipt rows only; never copy the Binding/session,
        # or route the detached copy through mutation/retirement callbacks.
        from copy import deepcopy
        result = {}
        memo[id(self)] = result
        for key, value in self.items():
            result[deepcopy(key, memo)] = deepcopy(value, memo)
        return result


class Binding:
    def __init__(self, child, label):
        self.child, self.label, self.full_depth = child, label, 0
        self.resource_maps = {}
        self.payloads = {}
        self.front_reference = None
        self.geometry, self.provider = child.modes.geometry, child.stack.provider

    def state_fail(self, reason):
        from replay_lifecycle_audit_base_720_v1 import state_for
        state = state_for(getattr(self.child.modes, '_numeric_cleanup_owner', None))
        if state is not None:
            state.invalidate(reason)
            self.child.session._failed = True

    def seal_resources(self):
        c = self.child
        for name in ('_compiled', 'resources'):
            rows = getattr(c, name)
            if name in self.resource_maps:
                if rows is not self.resource_maps[name]:
                    raise RuntimeError('Fixed resource map owner replaced: ' + name)
                continue
            wrapped = _ResourceMap(self, name, rows)
            self.resource_maps[name] = wrapped
            setattr(c, name, wrapped)
        self.payloads = {n: k.kernel for n, k in c._compiled.items()}
        if self.label in ('post', 'front'):
            self.front_reference = MethodType(c.modules['fused_front'].FusedFront.apply, c.front_owner)

    def resources_current(self):
        c = self.child
        if any(getattr(c, n) is not rows for n, rows in self.resource_maps.items()):
            self.state_fail('Fixed resource map owner replaced')
            raise RuntimeError('Fixed resource map owner replaced')

    def live(self, *, check_thread=True):
        c = self.child
        self.resources_current()
        c.session._ready()
        if check_thread and hasattr(c, 'thread') and c.thread != get_ident():
            raise RuntimeError('Lifecycle family CPU thread owner changed')
        if (c.modes.session is not c.session or c.session._stack is not c.stack or
                c.stack.model is not c.model or c.stack.graph is not c.graph or
                c.graph.closed or c.graph.model is not c.model or
                c.graph.arithmetic is not c.stack.provider or
                c.stack.provider is not self.provider or self.provider.mode != 'fp16_xmx' or
                c.modes.geometry is not self.geometry or
                tuple(self.geometry.source) != (720, 1280) or
                self.geometry.mode.model != (720, 1280) or self.geometry.mode.active != (720, 1280) or
                self.geometry.mode.internal != (768, 1280) or self.geometry.mode.inset != (0, 0) or
                c.modes.height != 720 or tuple(c.modes.source) != (720, 1280) or
                c.modes.variant != 'unrounded' or not c.modes.controlled or
                not c.modes.c512_qkv_library_720 or not c.modes.native_k8_720):
            raise RuntimeError('Lifecycle family left the owned actual720 session')
        if self.label == 'history':
            self.history_live()
        elif self.label == 'vit':
            self.vit_live()
        else:
            self.post_front_live()

    def history_live(self):
        c = self.child
        module = sys.modules[type(c).__module__]
        if (c.implementation is not c.implementation_identity or
                (hasattr(c.modes, '_audit_history_host_options_720') and
                 c.modes._audit_history_host_options_720 is not c.implementation) or
                not c.active or c.retired or c.session.__dict__.get('_history_numeric_suite_720') is not c or
                c.stack.provider is not c.provider or c.provider.__dict__.get('dense') is not c.dense or
                c.post._contiguous_cropped_head is not c.post_head or
                c.modes._c512_library_scope is not c.joint_scope or
                c.modes.c512_library_calls is not c.c512_calls or c.modes.native_k8_calls is not c.k8_calls or
                c.model.pre is not c.pre_module or c.model.post is not c.post_module or
                c.model.reciprocal is not c.reciprocal or c.model.dimension_reciprocal is not c.dimensions or
                c.reciprocal.values is not c.values or c.dimensions.values is not c.dimension_values or
                c.native_dimensions is not c.native_dimension_owner or
                c.prepared_dimensions is not c.prepared_dimension_owner or
                c.wrapper.__nr_numeric_original_installed__ is not c.previous_installed or
                not module._installed_contains(c.session._installed, c.wrapper)):
            raise RuntimeError('History current resource/installed owner changed')
        from numeric_model_forward_720_v1 import require_forward
        require_forward(c.model.forward, MethodType(c.model_forward_fn, c.model), c.model, c.session)
        if c.in_frame:
            if module._IN_FLIGHT is not c or c.temporal.warp_history_normalized is not c:
                raise RuntimeError('History temporary sampler/flight owner changed')
            contracts = sys.modules['quantization_dataflow_v1'].CONTRACTS
            expected = c.output_contracts()
            if any(contracts.get(c.kernels.__name__ + '.' + n) != row for n, row in expected.items()):
                raise RuntimeError('History temporary JIT output owner changed')

    def vit_live(self):
        c = self.child
        module = sys.modules[type(c).__module__]
        if (not c._live or getattr(c, '_retired', False) or not c._preflight_complete or
                c.session.__dict__.get('_vit_numeric_suite_720') is not c or
                c.stack.int8_vit is not c._lifecycle_int8_owner or c.stack.int8_vit.ffn is not c.ffn or
                not any(owner is c.projection_owner for owner in c.stack.components) or
                c.projection_owner.apply.__self__ is not c.projection_owner or
                c.projection_owner.apply.__func__ is not c._projection_method):
            raise RuntimeError('ViT current resource/session owner changed')
        module.require_forward(c.model.forward, c._model_forward, c.model, c.session)
        reference, chain = module._installed_chain(c.session)
        if reference is not c.reference_vforward or c._wrapper not in chain:
            raise RuntimeError('Layered ViT scope lost its actual parent')
        if (c.stack.provider.__dict__.get('dense') is not c.baseline_dense or
                c.modules['joint'].active_post._contiguous_cropped_head is not c.baseline_post or
                any(module.CONTRACTS.get(n) != value for n, value in c._contracts.items())):
            raise RuntimeError('ViT actual matrix callee/JIT owner changed')
        if c._history_binding is not None:
            history_binding(c)
        if c._in_scope:
            c._active_callees()
            replacements = c._lifecycle_block_hooks
            if (len(replacements) != 8 or any(c.blocks[i].__dict__.get('forward') is not hook
                                            for i, hook in replacements.items())):
                raise RuntimeError('ViT temporary block forward owner changed')

    def post_front_live(self):
        c = self.child
        common = sys.modules['post_numeric_suite_720_v1']
        if (not c._live or getattr(c, '_retired', False) or not c._preflight_complete or
                c.stack.provider is not c.provider or c.model.post is not c.post or
                c.model.sigmoid is not c.sigmoid or c.model.noise is not c.noise or
                c.front_owner.noise is not c.noise or
                c.provider.__dict__.get('dense') is not c.dense or
                c.modules['post_head']._contiguous_cropped_head is not c.post_k8 or
                c.session.__dict__.get(c.marker) is not c or
                not common._contains_wrapper(c.session._installed, c._scope_wrapper) or
                c._scope_wrapper.__nr_numeric_original_installed__ is not c._original_installed or
                any(c._contract_store.get(n) != value for n, value in c._contracts.items())):
            raise RuntimeError('Post/front current resource/installed owner changed')
        from numeric_model_forward_720_v1 import require_forward
        require_forward(c.model.forward, c.model_forward, c.model, c.session)
        # Permanent owner/local checks also run during outer scope unwind,
        # after the inner arithmetic/policy context has correctly restored.
        # The existing _guard(dispatch=True) and actual entry consumer below
        # retain the temporary backend requirement at dispatch sites.
        self.local()

    def local(self):
        c = self.child
        if self.label == 'post':
            if (c.sigmoid.values is not c.values or c.model.blend_scale is not c.blend_scale):
                raise RuntimeError('Post table/blend resource owner changed')
            if c._in_frame:
                if (c.post.__dict__.get('forward') is not c._post_replacement or
                        c.options['post_sigmoid'] == 'native' and
                        c.sigmoid.__dict__.get('forward') is not c._sigmoid_replacement):
                    raise RuntimeError('Owned post/sigmoid frame hooks changed')
            elif 'forward' in c.post.__dict__ or 'forward' in c.sigmoid.__dict__:
                raise RuntimeError('Persistent post/sigmoid override changed')
            if c.options['post_store'] == 'native_rtz':
                if (c._rtz_approval is not c._rtz_approval_frozen or
                        c._rtz_kwargs is not c._rtz_kwargs_frozen):
                    raise RuntimeError('Native RTZ approved evidence owner changed')
                for n in ('store.reset', 'store.history'):
                    k = c._compiled[n]
                    if k is not c._rtz_compiled[n] or k.kernel is not c._rtz_payloads[n]:
                        raise RuntimeError('Native RTZ production resource owner changed')
        elif self.label == 'front':
            if any(getattr(c.noise, n) is not t for n, t in c.tables.items()):
                raise RuntimeError('Complete noise table resource owner changed')
            sys.modules['front_route_dispatch_720_v1'].require_selected(c)


def local_guard(child):
    if not hot_active(child):
        return False
    _binding(child).local()
    return True


def local_after_guard(child):
    """Only immediately after _guard: its hot Binding.live did local already."""
    if not hot_active(child):
        return child._validate_local()


def keep_frame_diagnostics(child):
    """Receipt policy only; never grants live or entry authority."""
    binding = _binding(child)
    if binding is None or binding.full_depth:
        return True
    from replay_lifecycle_audit_base_720_v1 import state_for
    state = state_for(getattr(child.modes, '_numeric_cleanup_owner', None))
    return not (state is not None and state.armed and state.static_ready and
                state.frame is not None and state.pending is None and
                state.consumed is not None and not state.frame_cold)


def launch_owner(child, label, kernel):
    if not hot_active(child):
        return False
    binding = _binding(child)
    binding.resources_current()
    if kernel is not child._compiled.get(label) or kernel.kernel is not binding.payloads.get(label):
        raise RuntimeError('Actual launch lost the sealed executable owner: ' + label)
    return True


def front_reference(counter):
    binding = _binding(counter)
    # This returns only an already admitted immutable parent callable, never
    # entry/launch authority. Restoration must still compare the actual hooks
    # after failure, without replacing the original exception by a ready gate.
    if binding is None or binding.full_depth or binding.front_reference is None:
        return None
    owner = counter.front_owner
    if (counter.session._stack is not counter.stack or counter.stack.model is not counter.model or
            not any(component is owner for component in counter.stack.components) or
            owner.noise is not counter.noise or counter.model.noise is not counter.noise):
        raise RuntimeError('Front dispatcher current resource owner changed')
    return binding.front_reference


def full_audit(child, label, *, require_ready=False):
    """Execute the real complete guard plus sealed source and binary receipts."""
    with fixed_audit(child):
        binding = _binding(child)
        value = binding.original_validate() if binding is not None else child.validate()
        module = sys.modules[type(child).__module__]
        if label == 'history':
            if module._manifest(child.sources) != child.source_manifest:
                raise RuntimeError('History fixed source manifest changed')
            expected = child._specializations()
            compiled = set(child._compiled)
            # Original Suite.preflight first validates owners, then compiles.
            # A cold, unsealed diagnostic may inspect an empty/partial table.
            # A lifecycle seal/capture audit always requires the complete set.
            if (not compiled.issubset(expected) or
                    ((require_ready or (binding is not None and binding.resource_maps))
                     and compiled != expected)):
                raise RuntimeError('History fixed specialization table changed')
            for n, k in child._compiled.items():
                if k.kernel is not child._binary_objects[n]:
                    raise RuntimeError('History compiled payload owner changed')
                row = child._binary_receipt(k)
                if any(row[field] != child.resources[n][field]
                       for field in ('kernel_hash', 'actualbinary_sha256', 'spills')):
                    raise RuntimeError('History fixed binary receipt changed')
        elif label == 'vit':
            for row in dict(child.sources, **child.candidate_sources).values():
                if module._source(row['path']) != {n: row[n] for n in ('path', 'sha256')}:
                    raise RuntimeError('ViT fixed source manifest changed')
            for n, k in child._compiled.items():
                row = child._binary(k)
                if row['actualbinary_sha256'] != child.resources[n]['actualbinary_sha256']:
                    raise RuntimeError('ViT fixed binary receipt changed')
        else:
            common = sys.modules['post_numeric_suite_720_v1']
            child._guard(sources=True)
            for n, k in child._compiled.items():
                row = common._binary_row(k)
                if any(row[f] != child.resources[n][f] for f in
                       ('actualkernelhash', 'actualbinary_sha256', 'spills')):
                    raise RuntimeError('Post/front fixed binary receipt changed')
        return value


def ready(child, label):
    return (set(child._compiled) == child._specializations() if label == 'history'
            else child._preflight_complete)


def bind(child, label, install):
    """Load-time controlled instance interfaces; OFF retains the legacy path."""
    from replay_lifecycle_audit_base_720_v1 import trial_requested
    if not trial_requested():
        install('_validate_replay', child.validate)
        return
    binding = Binding(child, label)
    install('_lifecycle_rest_720', binding)
    if label == 'vit':
        install('_lifecycle_int8_owner', child.stack.int8_vit)
    original_validate = child.validate
    # The existing Frame wrapper retains the legacy validate memo only for OFF.
    original_validate = getattr(original_validate, '__wrapped__', original_validate)
    binding.original_validate = original_validate

    @wraps(original_validate)
    def diagnostic(*args, **kwargs):
        if args or kwargs:
            with fixed_audit(child):
                return original_validate(*args, **kwargs)
        return full_audit(child, label)
    install('validate', diagnostic)
    if label in ('history', 'vit'):
        @wraps(original_validate)
        def replay(*args, **kwargs):
            delegated = False
            try:
                if args or kwargs or not hot_active(child):
                    delegated = True
                    return original_validate(*args, **kwargs)
                if label == 'history':
                    # _validate_owner retains the actual thread gate, while its
                    # hot Binding.history_live already checks active/chain owners.
                    child._validate_owner()
                    child.validate_calls += 1
                else:
                    # Binding.vit_live already checks installed/history/active
                    # hook owners. No caller consumes this replay-only result.
                    child._guard()
                    child.validation_calls += 1
                return None
            except BaseException as error:
                if not delegated:
                    if label == 'vit':
                        child.failures.append({'phase': 'validate', 'type': type(error).__name__,
                                               'message': str(error)})
                    child.session._failed = True
                raise
        install('_validate_replay', replay)
    else:
        install('_validate_replay', original_validate)
    install('_validate_live_owner', binding.live)
    install('_require_live_front', lambda: require_entry_front(child, label))
    install('_require_front_after_live_owner', lambda: entry_front_after_live_owner(child, label))
    original_snapshot = child.snapshot
    @wraps(original_snapshot)
    def snapshot(*args, **kwargs):
        # Retired/failed snapshots remain readable, as in the original interface.
        if child.session._failed or getattr(child.session, '_closed', False):
            return original_snapshot(*args, **kwargs)
        full_audit(child, label)
        return original_snapshot(*args, **kwargs)
    install('snapshot', snapshot)


def require_entry_front(child, label):
    c = child
    binding = _binding(c)
    binding.live()
    return entry_front_after_live_owner(child, label)


def entry_front_after_live_owner(child, label):
    """Frontend-only half consumed by lookup immediately after live_all."""
    c = child
    if label == 'history':
        if not c.in_frame: raise RuntimeError('History entry outside actual installed frame')
        c._require_front()
    elif label == 'vit':
        if not c._in_scope or c._frame is None or c._frame.get('frontend_calls') != 1:
            raise RuntimeError('ViT entry outside actual frontend frame')
        c._active_callees(require_arithmetic=True)
    else:
        if not c._in_frame or c._front_event is None:
            raise RuntimeError('Post/front entry outside actual observed frontend frame')
        if (c.modules['policy'].ENABLED != c.modules['policy'].FAMILIES or
                c.modules['execution'].current_arithmetic_backend() != 'triton'):
            raise RuntimeError('Post/front entry lost the unrounded frame backend')
    if label != 'history':
        # The actual controlled front is already checked by original route code.
        # At entry consumption it must still be the graph's current owner.
        from decoder_input_full_k_720_v1 import _closure
        front = c.model._forward_front
        original = _closure(front, 'original') if getattr(front, '__self__', None) is not c.graph else front
        if getattr(original, '__self__', None) is not c.graph:
            raise RuntimeError('Actual entry frontend graph owner changed')
