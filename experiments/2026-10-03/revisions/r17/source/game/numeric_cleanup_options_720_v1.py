"""Immutable opt-ins for the active C512+K8 numerical-cleanup experiments.

This module deliberately imports no model, tensor library or GPU runtime.
An option describes a candidate, not a measured improvement or a deployment.
Existing Gather/C32/probability/merge/post-FMA flags stay on FullsizeGameModes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from collections.abc import Mapping


@dataclass(frozen=True)
class NumericCleanupOptions:
    decoder_input_full_k: bool = False
    branch_accum_families: tuple[str, ...] = ()
    vit_qkv_full_k: bool = False
    vit_projection_full_k: bool = False
    vit_exp_zero_constant: bool = False
    vit_denominator: str = "reference"
    vit_norm_fma: bool = False
    vit_exp_fma: bool = False
    history_value: str = "reference"
    history_reciprocal: str = "table"
    history_dimension_rcp: str = "table"
    history_coord: str = "reference"
    post_sigmoid: str = "table"
    post_store: str = "reference"
    front_noise: str = "table"

    def __post_init__(self):
        bool_fields = ("decoder_input_full_k", "vit_qkv_full_k",
                       "vit_projection_full_k", "vit_exp_zero_constant",
                       "vit_norm_fma", "vit_exp_fma")
        for name in bool_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.branch_accum_families, (tuple, list)):
            raise TypeError("branch_accum_families must be a sequence of family names")
        branches = tuple(self.branch_accum_families)
        if (len(set(branches)) != len(branches) or
                any(name not in ("c64", "c128", "c256") for name in branches)):
            raise ValueError("Expected unique branch families c64/c128/c256")
        # Store a canonical immutable value; callers cannot mutate an active graph.
        object.__setattr__(self, "branch_accum_families", tuple(sorted(branches)))
        enums = {
            "vit_denominator": ("reference", "ordered_fused", "fp32_reduction"),
            "history_value": ("reference", "fp32_fractional", "fp32_all_paths"),
            "history_reciprocal": ("table", "native"),
            "history_dimension_rcp": ("table", "native"),
            "history_coord": ("reference", "direct_pixel"),
            "post_sigmoid": ("table", "native"),
            "post_store": ("reference", "native_rtz", "rne"),
            "front_noise": ("table", "native_radius", "native_trig", "native_both"),
        }
        for name, values in enums.items():
            if getattr(self, name) not in values:
                raise ValueError(f"Unknown {name}: {getattr(self, name)!r}")

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("numeric_cleanup_720 must be a mapping or NumericCleanupOptions")
        allowed = {field.name for field in fields(cls)}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"Unknown numerical-cleanup options: {sorted(extra)}")
        return cls(**dict(value))

    @property
    def active(self):
        return self != type(self)()

    def to_dict(self):
        result = asdict(self)
        result["branch_accum_families"] = list(self.branch_accum_families)
        return result

    @property
    def identity(self):
        value = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "numeric-cleanup-720-v1:" + hashlib.sha256(value.encode()).hexdigest()

    def selected_scopes(self):
        """Return exact module names and independent installed() keyword arguments."""
        result = []
        if self.decoder_input_full_k:
            result.append(("decoder_input", "decoder_input_full_k_720_v1",
                           {"decoder_input_full_k": True}))
        if self.branch_accum_families:
            result.append(("branch_accum", "branch_accum_native_720_v1",
                           {"branch_accum": {name: name in self.branch_accum_families
                                             for name in ("c64", "c128", "c256")}}))
        vit = {name: getattr(self, name) for name in (
            "vit_qkv_full_k", "vit_projection_full_k", "vit_exp_zero_constant",
            "vit_denominator", "vit_norm_fma", "vit_exp_fma")}
        if any(vit[name] for name in vit if name != "vit_denominator") or \
                self.vit_denominator != "reference":
            result.append(("vit", "vit_numeric_suite_720_v1", vit))
        history = {name: getattr(self, name) for name in (
            "history_value", "history_reciprocal", "history_dimension_rcp", "history_coord")}
        if (self.history_value != "reference" or self.history_reciprocal != "table" or
                self.history_dimension_rcp != "table" or self.history_coord != "reference"):
            result.append(("history", "history_numeric_suite_720_v1", history))
        if self.post_sigmoid != "table" or self.post_store != "reference":
            result.append(("post", "post_numeric_suite_720_v1",
                           {"post_sigmoid": self.post_sigmoid, "post_store": self.post_store}))
        if self.front_noise != "table":
            result.append(("front", "front_noise_native_720_v1",
                           {"front_noise": self.front_noise}))
        return tuple(result)


# These are independent experiments. No candidate automatically becomes a default.
INDEPENDENT_CANDIDATES = {
    "decoder_input_full_k": {"decoder_input_full_k": True},
    "branch_c64": {"branch_accum_families": ("c64",)},
    "branch_c128": {"branch_accum_families": ("c128",)},
    "branch_c256": {"branch_accum_families": ("c256",)},
    "vit_qkv_full_k": {"vit_qkv_full_k": True},
    "vit_projection_full_k": {"vit_projection_full_k": True},
    "vit_exp_zero_constant": {"vit_exp_zero_constant": True},
    "vit_denominator_ordered": {"vit_denominator": "ordered_fused"},
    "vit_denominator_fp32": {"vit_denominator": "fp32_reduction"},
    "vit_norm_fma": {"vit_norm_fma": True},
    "vit_exp_fma": {"vit_exp_fma": True},
    "vit_norm_exp_fma": {"vit_norm_fma": True, "vit_exp_fma": True},
    "history_fractional": {"history_value": "fp32_fractional"},
    "history_all_paths": {"history_value": "fp32_all_paths"},
    "history_reciprocal": {"history_reciprocal": "native"},
    "history_dimension_rcp": {"history_dimension_rcp": "native"},
    "history_direct_pixel": {"history_coord": "direct_pixel"},
    "post_sigmoid": {"post_sigmoid": "native"},
    "post_rtz": {"post_store": "native_rtz"},
    "post_rne": {"post_store": "rne"},
    "front_radius": {"front_noise": "native_radius"},
    "front_trig": {"front_noise": "native_trig"},
    "front_both": {"front_noise": "native_both"},
}


def candidate_options(name):
    try:
        value = INDEPENDENT_CANDIDATES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown numerical-cleanup candidate: {name}") from exc
    return NumericCleanupOptions.parse(value)
