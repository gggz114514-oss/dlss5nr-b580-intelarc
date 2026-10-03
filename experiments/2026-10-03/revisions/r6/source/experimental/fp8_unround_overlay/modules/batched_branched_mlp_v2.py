"""Select measured branch launch configurations for reduced and full inputs.

Selection receipt: batched-branches-v1, SHA256
62c7e54e919302d398ff6cd37255e68ac66a6c4a91a4c7c86f4106b810d938ab.
Only scheduling changes; v1 keeps every original FP16 branch boundary.
"""
from batched_branched_mlp_v1 import FusedBatched as Base


def configurations(workload):
    if workload not in ('small', 'full'):
        raise ValueError('Choose small or full from the measured workloads')
    return {
        64: dict(pair_bm=32, pair_stages=2 if workload == 'small' else 1,
                 project_bm=16, project_bn=64),
        128: dict(pair_bm=32, pair_stages=2, project_bm=16, project_bn=64),
        256: dict(pair_bm=16, pair_stages=1, project_bm=16, project_bn=64),
    }


class FusedBatched(Base):
    def __init__(self, model, provider, *, workload):
        super().__init__(model, provider, configurations=configurations(workload))
