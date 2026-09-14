"""Fixed controls candidate; not installed or advertised by the product yet."""
from pathlib import Path
import sys

from nr_exact_runtime_v1 import Session as DefaultSession, _SERIAL


def parse_controls(value=None):
    from nr_backend.controlled_temporal import NRControls
    if value is None:
        value = {}
    if type(value) is not dict:
        raise ValueError('nr_controls must be an object')
    allowed = {'style', 'intensity', 'local_tone', 'local_structure',
               'auto_mask', 'skin_structure'}
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown NR controls: {sorted(unknown)}')
    # JSON null means inherit. Do not silently translate invalid negative values.
    return NRControls(**value)


class Session(DefaultSession):
    @classmethod
    def create(cls, exact_root, *, controls=None):
        import torch
        from nr_backend.controlled_temporal import ControlledMotionNR
        from nr_backend.execution import use_arithmetic_backend
        backend = (Path(exact_root).resolve() / 'backend').resolve()
        for name, module in tuple(sys.modules.items()):
            if name == 'nr_backend' or name.startswith('nr_backend.'):
                path = getattr(module, '__file__', None)
                if path is None or not Path(path).resolve().is_relative_to(backend):
                    raise RuntimeError('Controls candidate requires isolated nr/exact imports')
        parsed = parse_controls(controls)
        with _SERIAL:
            root = Path(exact_root).resolve() / 'model-assets'
            model = ControlledMotionNR.from_assets(
                root / 'sf-v2/WEIGHTS_HT.bin', root / 'noise-sm89-v2',
                root / 'sigmoid-sm89-v1', controls=parsed,
                style_directory=root / 'style-sm89-v1').to('xpu').eval()
            return cls(model, torch, use_arithmetic_backend)
