"""Opt-in v2 candidates layered over the validated all-v1 configuration."""
from nr_exact_all_candidate_v1 import Session as Previous
from .layout import ReuseLayout


class Session(Previous):
    variant = frozenset({'dataflow','compact'})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.variant <= {'dataflow','compact','tile'}:
            raise ValueError('Unknown v2 variant')
        layout = ReuseLayout(self._model)
        layout.dataflow = 'dataflow' in self.variant
        layout.compact_output = 'compact' in self.variant
        layout.tile = 'tile' in self.variant
        self.adapters['layout'] = layout
