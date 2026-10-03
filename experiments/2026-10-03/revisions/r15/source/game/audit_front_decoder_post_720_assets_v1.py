"""Current720 native-only lazy assets; exact constructors stay unchanged.

construction_assets() must surround model construction (before .to('xpu')).
It is deliberately separate from the before_numeric runtime scope. Placeholder
noise pointers are valid only when both native constexpr branches are enabled.
"""
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
import hashlib
import json


STYLE_CURVES = {
    1: (184332, "8078b564d0838fd68aa6680f990f8e60a872f1e881c57130600d5666118c1a5c"),
    2: (184332, "7c373336c6d34c7d4343c365cd4b364b1c5a9be6c23d1e8a244996330d056b7d"),
}
LAZY_NOISE_SCHEMA = "current720-native-both-placeholder-v1"
_CONSTRUCTION = False


def noise_manifest(path):
    path = Path(path).resolve()
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != 1 or manifest.get("domain_count") != 1 << 24:
        raise ValueError("Unknown native noise manifest")
    names = ("radius.f32.bin", "sin.f32.bin", "cos.f32.bin")
    files = {name: manifest["files"][name] for name in names}
    if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in files.values()):
        raise ValueError("Invalid native noise source digest")
    if any((path / name).stat().st_size != 1 << 26 for name in names):
        raise ValueError("Native noise asset inventory changed")
    return {"schema": LAZY_NOISE_SCHEMA, "directory": str(path),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": files,
            "native_radius": True, "native_trig": True, "retained_bytes": 12,
            "payloads_read": False, "table_payloads_authenticated": False,
            "reason": "No table value is consumed by native_both; immutable manifest/file sizes define provenance only"}


def is_lazy_native_noise(noise):
    contract = getattr(noise, "_audit_fdp_noise_contract", None)
    return isinstance(contract, dict) and contract.get("schema") == LAZY_NOISE_SCHEMA


def native_noise_admission(noise, *, front_noise, device):
    """Cold validator input for main's numeric/noise constant contract.

    Return None for ordinary owners, which retain their full-table validator.
    A lazy owner is admitted ONLY for the actual native_both child. This is
    metadata/provenance, not a claim that unused table payload SHA was read.
    Main still freezes the returned real tensors/pointers/versions and the
    selected FrontNoiseCounters native flags/resources/source identity.
    """
    if not is_lazy_native_noise(noise):
        return None
    if front_noise != "native_both":
        raise RuntimeError("Placeholder noise requires an actual native_both numeric child")
    torch = import_module("torch")
    contract = json.loads(json.dumps(noise._audit_fdp_noise_contract))
    if (contract.get("native_radius") is not True or contract.get("native_trig") is not True
            or contract.get("retained_bytes") != 12 or contract.get("payloads_read") is not False
            or contract.get("table_payloads_authenticated") is not False
            or set(contract.get("files", {})) != {"radius.f32.bin", "sin.f32.bin", "cos.f32.bin"}):
        raise RuntimeError("Invalid native placeholder provenance")
    tables, constants = {}, {}
    for name in ("radius", "sine", "cosine"):
        value = getattr(noise, name)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1,)
                or tuple(value.stride()) != (1,) or value.dtype != torch.float32
                or value.device != device or device.type != "xpu"):
            raise RuntimeError("Changed native placeholder buffer contract")
        tables[name] = value
        constants[name] = (id(value), value.data_ptr(), tuple(value.shape),
                           tuple(value.stride()), value.dtype, value.device, value._version)
    return {"schema": LAZY_NOISE_SCHEMA, "front_noise": "native_both", "table_shape": (1,),
            "native_radius": True, "native_trig": True, "tables": tables,
            "constants": constants, "provenance": contract,
            "table_payloads_authenticated": False, "reference_fallback_allowed": False}


def make_noise(path):
    torch = import_module("torch")
    contract = noise_manifest(path)
    class NativeBothNoise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.inference_mode(False):
                for name in ("radius", "sine", "cosine"):
                    self.register_buffer(name, torch.zeros((1,), dtype=torch.float32))
            self._audit_fdp_noise_contract = contract
        def forward(self, *args, **kwargs):
            raise RuntimeError("Tiny noise owner cannot execute reference/table noise; use current720 native_both")
    return NativeBothNoise()


class LazyNativeStyle:
    """Owned outside the frozen body's registered buffers; style0 holds no LUT."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.device = None
        self.curves = {}
        self.curve_pins = {}
        self.dispatch = None
        self.reference_global_preserved = False
        self._audit_fdp_style_owner = True

    def to(self, device):
        if self.curves and str(device) != str(self.device):
            raise RuntimeError("Cannot move active style assets across devices")
        self.device = str(device)
        return self

    def eval(self):
        return self

    def prepare(self, style, device):
        if style not in (1, 2):
            raise ValueError("Only style1/2 have grade assets")
        torch = import_module("torch")
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Cold style loading must run outside graph capture")
        key = (style, str(device))
        if key in self.curves:
            return self.curves[key]
        source = self.directory / f"curve-style{style}.rgb32f.bin"
        raw = source.read_bytes()
        size, expected = STYLE_CURVES[style]
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != size or digest != expected:
            raise ValueError(f"Style{style} authored tone curve changed")
        with torch.inference_mode(False):
            value = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().to(device)
        self.curves[key] = value
        self.curve_pins[key] = {"path": str(source), "sha256": digest, "bytes": size}
        return value

    def forward(self, neural, original, **kwargs):
        if self.dispatch is None:
            raise RuntimeError("Lazy style must be attached to the owned720 runtime before style processing")
        return self.dispatch(neural, original, **kwargs)

    __call__ = forward

    def snapshot(self):
        return {"owner": "LazyNativeStyle", "default_style0_asset_bytes": 0,
                "preserved_reference_global_owner": self.reference_global_preserved,
                "residency_scope": "new model only; a restored existing reference global may still retain its old tables",
                "loaded_curves": list(self.curve_pins.values()),
                "retained_bytes": sum(row["bytes"] for row in self.curve_pins.values()),
                "reciprocal_mantissa_table": False, "saturation_roundtrip_table": False,
                "exposure_reference_table": False}


@contextmanager
def construction_assets(*, front_noise="native_both", native_style=True,
                        retired_session=None, retired_graph=None):
    """Main construction hook; enter before GameLiveControlledNR.from_assets.

    Only the game fast model is supported; exact model creation must happen
    outside this context. For a checkbox cold switch main retains the OLD graph
    handle before close and passes it with the fully closed old session. An old
    global style owner is temporarily isolated, never permanently replaced;
    created models retain their own new owners even after globals are restored.
    """
    global _CONSTRUCTION
    if _CONSTRUCTION or front_noise != "native_both" or type(native_style) is not bool:
        raise RuntimeError("Invalid/nested current720 asset construction")
    noise_module = import_module("nr_backend.noise")
    game = import_module("nr_game_controlled_model")
    cls = noise_module.NativeNoiseTable
    original_loader = cls.__dict__["from_directory"]
    old_style, old_directory = game._STYLE_POST, game._STYLE_DIRECTORY
    switching = retired_session is not None or retired_graph is not None
    if switching and (retired_session is None or retired_graph is None
            or getattr(retired_session, "_closed", False) is not True
            or getattr(retired_session, "_stack", None) is not None
            or getattr(retired_graph, "closed", False) is not True
            or getattr(retired_graph, "entries", None) != {}
            or getattr(retired_graph, "last_entry", None) is not None
            or getattr(retired_session, "_implementation_failed_retirement_anchor", None) is not None
            or getattr(retired_session, "_numeric_cleanup_retired_owner", None) is not None):
        raise RuntimeError("Asset cold switch requires closed session and fully retired graph/owners")
    if old_style is not None and not switching:
        raise RuntimeError("Existing style global requires a retired cold-switch session/graph receipt")
    original_game_loader = game.GameLiveControlledNR.__dict__["from_assets"]
    new_style = None
    constructed = []
    def load_noise(owner, path):
        return make_noise(path)
    def from_assets(owner, *args, **kwargs):
        nonlocal new_style
        if constructed:
            raise RuntimeError("Asset scope constructs exactly one fresh game model")
        sigmoid = args[2] if len(args) > 2 else kwargs["sigmoid_directory"]
        directory = Path(kwargs.get("style_directory") or Path(sigmoid).parent / "style-sm89-v1").resolve()
        if native_style:
            new_style = LazyNativeStyle(directory).to("xpu").eval()
            new_style.reference_global_preserved = old_style is not None
            game._STYLE_POST = new_style
            game._STYLE_DIRECTORY = directory
        model = original_game_loader.__func__(owner, *args, **kwargs)
        if native_style and model.__dict__.get("style_post") is not new_style:
            raise RuntimeError("Fresh model lost its isolated lazy style owner")
        constructed.append(model)
        return model
    loader = classmethod(load_noise)
    game_loader = classmethod(from_assets)
    _CONSTRUCTION = True
    game._STYLE_POST, game._STYLE_DIRECTORY = None, None
    cls.from_directory = loader
    game.GameLiveControlledNR.from_assets = game_loader
    try:
        yield {"noise": "native_both placeholders", "style": "cold lazy native" if native_style else "reference",
               "cold_switch": switching, "restores_exact_style_global": True}
    finally:
        valid = (cls.__dict__["from_directory"] is loader
                 and game.GameLiveControlledNR.__dict__["from_assets"] is game_loader
                 and (not native_style or game._STYLE_POST is new_style))
        cls.from_directory = original_loader
        game.GameLiveControlledNR.from_assets = original_game_loader
        game._STYLE_POST, game._STYLE_DIRECTORY = old_style, old_directory
        _CONSTRUCTION = False
        if not valid:
            raise RuntimeError("Asset loader ownership changed during construction")
