"""Setup-only receipt validation for the actual 720p native RTZ store.

The independent cast probe cannot authorize a different production binary.
Only explicitly supplied receipts for both production specializations qualify.
This module is stdlib-only and performs no GPU work or frame-path disk reads.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType


SCHEMA = "native-rtz-production-720-v1"
ROOT = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt")
SPECIFICATIONS = {
    "store.reset": {"n": 2764800, "unit": True, "rounding": "rtz", "block": 256,
                    "num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
    "store.history": {"n": 2764800, "unit": False, "rounding": "rtz", "block": 256,
                      "num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
}
GRID = (10800,)
DOMAINS = frozenset(("signed_zero", "half_neighborhoods", "midpoint_nextafter",
                     "subnormal", "normal", "overflow", "signed_overshoot",
                     "random_float32_bits", "nonfinite_classification"))
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PendingRTZEvidence(RuntimeError):
    """Missing evidence is a pending experiment, not a hardware verdict."""


def _hash(value, label):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA256 for {label}")
    return value


def _same_values(actual, expected, label):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(f"{label} does not contain the exact approved fields")
    if any(type(actual[key]) is not type(value) or actual[key] != value
           for key, value in expected.items()):
        raise ValueError(f"{label} differs from the actual specialization")


def _codegen_options(options, binary_format, label):
    if (not isinstance(options, dict) or set(options) != {"generate_native_code"}
            or type(options["generate_native_code"]) is not bool):
        raise ValueError(f"{label} needs the exact boolean native codegen option")
    if binary_format not in ("spv", "zebin") or options["generate_native_code"] != (
            binary_format == "zebin"):
        raise ValueError(f"{label} native codegen option differs from its binary format")


def _immutable(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    return value


def runtime_device_identity(torch, device):
    """Read current XPU name and installed Intel driver metadata at setup only."""
    import winreg

    name = str(torch.xpu.get_device_properties(device).name)
    if "B580" not in name.upper():
        raise RuntimeError("This RTZ production evidence is restricted to the tested B580")
    candidates = []
    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as parent:
        index = 0
        while True:
            try:
                child_name = winreg.EnumKey(parent, index)
            except OSError:
                break
            index += 1
            if len(child_name) != 4 or not child_name.isdecimal():
                continue
            with winreg.OpenKey(parent, child_name) as child:
                try:
                    description = str(winreg.QueryValueEx(child, "DriverDesc")[0])
                    driver = str(winreg.QueryValueEx(child, "DriverVersion")[0])
                    hardware = str(winreg.QueryValueEx(child, "MatchingDeviceId")[0]).lower()
                except FileNotFoundError:
                    continue
            if "B580" in description.upper() and "ven_8086" in hardware:
                candidates.append({"name": name, "driver": driver,
                                   "matching_device_id": hardware,
                                   "driver_class_key": child_name})
    if len(candidates) != 1 or not candidates[0]["driver"]:
        raise RuntimeError("Could not identify exactly one current B580 driver receipt")
    return candidates[0]


@dataclass(frozen=True)
class RTZEvidence:
    path: str
    sha256: str
    production: object
    device: object

    def launch_kwargs(self, label):
        """Only the reviewed per-store codegen choice can enter the launcher."""
        if label not in SPECIFICATIONS:
            raise ValueError(f"Unknown production RTZ specialization: {label}")
        expected = SPECIFICATIONS[label]
        return MappingProxyType({
            "num_warps": expected["num_warps"],
            "num_stages": expected["num_stages"],
            "enable_fp_fusion": expected["enable_fp_fusion"],
            "generate_native_code": self.production[label]["compile_options"]["generate_native_code"],
        })

    def require_binary(self, label, resource, *, specialization, grid,
                       input_dtype, output_dtype, contiguous):
        """Match a freshly screened actual store; call once during preflight."""
        if label not in SPECIFICATIONS:
            raise ValueError(f"Unknown production RTZ specialization: {label}")
        approved = self.production[label]
        _same_values(specialization, SPECIFICATIONS[label], label + "/actual specialization")
        if (tuple(grid) != GRID or any(type(item) is not int for item in grid)
                or input_dtype != "torch.float32"
                or output_dtype != "torch.float16" or contiguous is not True):
            raise RuntimeError(f"Actual RTZ grid or tensor contract differs from receipt: {label}")
        for key in ("actualkernelhash", "actualbinary_sha256"):
            if resource.get(key) != approved[key]:
                raise RuntimeError(f"Actual native RTZ binary differs from receipt: {label}/{key}")
        if type(resource.get("spills")) is not int or resource["spills"] != 0:
            raise RuntimeError(f"Actual native RTZ store spills: {label}")
        if resource.get("binary_format") != approved["binary_format"]:
            raise RuntimeError(f"Actual native RTZ binary format differs from receipt: {label}")
        for name in ("compile_options", "parsed_compile_options"):
            _codegen_options(resource.get(name), resource["binary_format"], label + "/" + name)
            _same_values(resource[name], dict(approved[name]), label + "/actual " + name)
        expected = SPECIFICATIONS[label]
        options = {key: expected[key] for key in
                   ("block", "num_warps", "num_stages", "enable_fp_fusion")}
        _same_values(resource.get("launch_options"), options, label + "/launch_options")
        return {"receipt_path": self.path, "receipt_sha256": self.sha256,
                "status": "production_evidence_verified", "specialization": label,
                "actualkernelhash": approved["actualkernelhash"],
                "actualbinary_sha256": approved["actualbinary_sha256"],
                "compile_options": dict(approved["compile_options"]),
                "parsed_compile_options": dict(approved["parsed_compile_options"]),
                "isa_sha256": approved["isa"]["sha256"],
                "device_isa_verified": True, "byte_identity_verified": True}


def load_evidence(path, *, source_sha256, toolchain_sha256, device, evidence_root=None):
    """Read frozen local evidence and verify both real conversion domains.

    ``device`` must come from the current runtime, including its driver. The
    caller supplies source/toolchain SHA values it just checked during setup.
    ISA text is an explicitly reviewed artifact, never executed as code.
    """
    if path is None:
        raise PendingRTZEvidence("Native RTZ production needs the completed store evidence receipt")
    try:
        path = Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise PendingRTZEvidence("Native RTZ production receipt has not been produced yet") from exc
    if not path.is_relative_to(ROOT.resolve()) or path.suffix.casefold() != ".json":
        raise ValueError("Native RTZ evidence must be a task-owned D: JSON receipt")
    task_root = path.parent if evidence_root is None else Path(evidence_root).resolve(strict=True)
    if not task_root.is_relative_to(ROOT.resolve()) or not path.is_relative_to(task_root):
        raise ValueError("Native RTZ receipt and artifacts must share one task evidence directory")
    raw = path.read_bytes()
    if len(raw) > 1048576:
        raise ValueError("Native RTZ evidence exceeds the bounded receipt size")
    record = json.loads(raw.decode("utf-8-sig"))
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA
            or record.get("status") != "production_domain_verified"):
        raise PendingRTZEvidence("Native RTZ production evidence is not complete")
    if _hash(record.get("source_sha256"), "source") != _hash(source_sha256, "actual source"):
        raise RuntimeError("RTZ kernel source changed after the production domain test")
    if set(toolchain_sha256) != {"core", "semantic", "intel"}:
        raise ValueError("Expected the three actual Triton conversion/compiler sources")
    for key, value in toolchain_sha256.items():
        _hash(value, key)
    _same_values(record.get("toolchain_sha256"), toolchain_sha256, "toolchain")
    if not isinstance(device, dict) or not device.get("name") or not device.get("driver"):
        raise ValueError("The current device and driver fingerprint is required")
    _same_values(record.get("device"), device, "device/driver")
    production = record.get("production")
    if not isinstance(production, dict) or set(production) != set(SPECIFICATIONS):
        raise PendingRTZEvidence("Both reset and signed-history production stores are required")
    for label, specification in SPECIFICATIONS.items():
        row = production[label]
        if not isinstance(row, dict):
            raise ValueError(f"Invalid production receipt: {label}")
        _same_values(row.get("specialization"), specification, label + "/specialization")
        if (row.get("grid") != list(GRID) or any(type(item) is not int for item in row["grid"])
                or row.get("input_dtype") != "torch.float32"
                or row.get("output_dtype") != "torch.float16" or row.get("contiguous") is not True):
            raise ValueError(f"Incomplete RTZ production tensor/grid metadata: {label}")
        if not isinstance(row.get("actualkernelhash"), str) or not row["actualkernelhash"]:
            raise ValueError(f"Actual loaded kernel hash missing: {label}")
        _hash(row.get("actualbinary_sha256"), label + "/binary")
        if row.get("binary_format") not in ("spv", "zebin"):
            raise ValueError(f"Actual loaded RTZ binary format missing: {label}")
        for name in ("compile_options", "parsed_compile_options"):
            _codegen_options(row.get(name), row["binary_format"], label + "/" + name)
        _same_values(row["parsed_compile_options"], row["compile_options"],
                     label + "/requested versus parsed codegen")
        if type(row.get("spills")) is not int or row["spills"] != 0:
            raise ValueError(f"Production store resource gate failed: {label}")
        numeric = row.get("numeric", {})
        counts = numeric.get("domain_sample_counts", {})
        if (not isinstance(counts, dict) or not DOMAINS.issubset(counts)
                or any(type(counts[name]) is not int or counts[name] <= 0 for name in DOMAINS)
                or numeric.get("passed") is not True
                or type(numeric.get("finite_bit_mismatches")) is not int
                or numeric["finite_bit_mismatches"] != 0
                or type(numeric.get("finite_sample_count")) is not int
                or numeric["finite_sample_count"] <= 0
                or type(numeric.get("nonfinite_class_mismatches")) is not int
                or numeric["nonfinite_class_mismatches"] != 0
                or type(numeric.get("nonfinite_sample_count")) is not int
                or numeric["nonfinite_sample_count"] <= 0
                or label == "store.reset" and numeric.get("clamp_domain_passed") is not True
                or label == "store.history" and numeric.get("signed_domain_passed") is not True):
            raise PendingRTZEvidence(f"Production store numerical domains incomplete: {label}")
        if (numeric.get("label") != label or numeric.get("source_sha256") != source_sha256
                or numeric.get("actualbinary_sha256") != row["actualbinary_sha256"]
                or numeric.get("reference") != ("store_unit_half_rz" if specification["unit"]
                                               else "store_half_rz")):
            raise RuntimeError(f"RTZ numerical report is not bound to this production store: {label}")
        generator = numeric.get("generator", {})
        generator_path = Path(generator.get("path", "")).resolve(strict=True)
        if not generator_path.is_relative_to(task_root):
            raise ValueError(f"Numeric generator is outside this frozen RTZ task: {label}")
        if hashlib.sha256(generator_path.read_bytes()).hexdigest() != _hash(
                generator.get("sha256"), label + "/numeric generator"):
            raise RuntimeError(f"Frozen RTZ numerical generator changed: {label}")
        isa = row.get("isa", {})
        if (isa.get("reviewed_native_fp32_to_fp16_rtz") is not True
                or isa.get("actualbinary_sha256") != row["actualbinary_sha256"]
                or isa.get("label") != label
                or not isinstance(isa.get("conversion_instruction"), str)
                or not isa["conversion_instruction"].strip()):
            raise PendingRTZEvidence(f"Actual native Intel conversion has not been reviewed: {label}")
        isa_path = Path(isa.get("path", "")).resolve(strict=True)
        if not isa_path.is_relative_to(task_root):
            raise ValueError(f"ISA artifact is outside the frozen experiment tree: {label}")
        digest = hashlib.sha256(isa_path.read_bytes()).hexdigest()
        if digest != _hash(isa.get("sha256"), label + "/isa"):
            raise RuntimeError(f"Reviewed actual ISA artifact changed: {label}")
    return RTZEvidence(str(path), hashlib.sha256(raw).hexdigest(),
                       _immutable(production), _immutable(device))
