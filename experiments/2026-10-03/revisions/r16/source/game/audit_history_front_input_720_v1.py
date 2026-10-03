"""HWC3 numerator/reciprocal -> contiguous native-front consumer interface.

The front worker owns apply_history_components math and observer routing. This
module authenticates its owner and keeps original normalization as a complete
fallback until that provider is installed. No placeholder/empty front is used.
"""


class FrontComponentConsumer720:
    def __init__(self, child, operation, broker=None):
        self.child, self.model, self.operation = child, child.model, operation
        self.function = getattr(operation, '__func__', operation)
        self.bound_owner = getattr(operation, '__self__', None)
        self.calls = 0
        self.broker = broker

    def apply(self, rgb, numerator, reciprocal, options):
        c = self.child
        operation = getattr(c, 'apply_history_components', None)
        suite = getattr(c.modes, '_numeric_cleanup_owner', None)
        if (c.model is not self.model or c.modes.session is not c.session
                or c.stack.model is not self.model or c.model._previous is None
                or suite is None or suite.session is not c.session or suite.children.get('front') is not c
                or c.options.get('front_noise') != 'native_both'
                or not getattr(c, 'active', getattr(c, '_live', False))
                or getattr(c, 'retired', getattr(c, '_retired', False))
                or getattr(operation, '__func__', operation) is not self.function
                or getattr(operation, '__self__', None) is not self.bound_owner):
            raise RuntimeError('Native front component consumer lost its actual owner')
        validate = getattr(c, 'validate_frame_context', None)
        if not callable(validate):
            validate = getattr(c, 'validate', None)
        if not callable(validate):
            raise RuntimeError('Native front components require an actual provider validation interface')
        validate()
        if (tuple(numerator.shape) != (720, 1280, 3) or tuple(reciprocal.shape) != (720, 1280)
                or numerator.dtype != c.torch.float32 or reciprocal.dtype != c.torch.float32
                or numerator.device != rgb.device or reciprocal.device != rgb.device
                or not numerator.is_contiguous() or not reciprocal.is_contiguous()):
            raise ValueError('Native front components need contiguous FP32 HWC3 numerator / HW reciprocal')
        if self.broker is not None:
            self.broker.stage_history_components_720(numerator, reciprocal)
        try:
            result = self.operation(rgb, numerator, reciprocal, **options)
        finally:
            if self.broker is not None:
                self.broker.end_history_components_720(numerator, reciprocal)
        if (tuple(result.shape) != (768, 1280, 16) or result.dtype != c.torch.float16
                or result.device != rgb.device or not result.is_contiguous()):
            raise RuntimeError('Native front returned an invalid complete HWC16 context')
        self.calls += 1
        return result


def bind_front_component_consumer(modes, front_child, broker=None):
    """Deferred hook after both history and the front provider are constructed."""
    operation = getattr(front_child, 'apply_history_components', None)
    if operation is None:
        return None
    session = modes.session
    suite = getattr(modes, '_numeric_cleanup_owner', None)
    if (front_child.model is not session._stack.model or front_child.session is not session
            or suite is None or suite.session is not session or suite.children.get('front') is not front_child
            or front_child.options.get('front_noise') != 'native_both'):
        raise RuntimeError('Front component consumer has a foreign session/model')
    slot = '_audit_history_front_components_720'
    if slot in front_child.model.__dict__:
        raise RuntimeError('Front component consumer is already bound')
    owner = FrontComponentConsumer720(front_child, operation, broker)
    front_child.model.__dict__[slot] = owner
    return owner


def prepare_history_front(model, rgb, numerator, reciprocal, options, reference):
    owner = model.__dict__.get('_audit_history_front_components_720')
    if owner is None:
        return reference(rgb, numerator * reciprocal[..., None], **options)
    return owner.apply(rgb, numerator, reciprocal, options)
