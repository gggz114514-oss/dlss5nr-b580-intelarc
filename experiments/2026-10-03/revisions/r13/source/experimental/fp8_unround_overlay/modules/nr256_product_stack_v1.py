"""Consolidated NR256 product candidate; arithmetic frozen to accepted floor16.

Existing native-query layout plus the byte-preserving five decoder gathers.
Old selected factories remain frozen so historical receipts stay reproducible.
"""
from decoder_gather_scope_v1 import DecoderGather
from c512_quad_query_stack_v1 import Stack as ReviewedStack


class Stack(ReviewedStack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_gather = DecoderGather(self.model)
        self.components.append(self.decoder_gather)

    def metadata(self):
        return dict(super().metadata(), product_profile='nr256-reviewed-v1',
            decoder_gather=dict(calls=dict(self.decoder_gather.calls),
                resources=dict(self.decoder_gather.resources),
                arithmetic_changed=self.decoder_gather.unround_activations))

    def close(self):
        self.decoder_gather.verify_restored()
        super().close()
