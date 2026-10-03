"""CPU-only registry for live 720p comparisons against the C512+K8 baseline.

Availability is installed separately after cache preparation. A registry entry
alone never enables a candidate or permits compilation in the game process.
"""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class GameProfile:
    label: str
    legacy: tuple = ()
    numeric: tuple = ()
    hits: tuple = ()
    fused_replay: bool = True
    pending_validation: bool = False

    def mode_options(self):
        result = {"c512_qkv_library_720": True, "native_k8_720": True,
                  **dict(self.legacy)}
        if self.numeric:
            result["numeric_cleanup_720"] = dict(self.numeric)
        return result


PROFILES = MappingProxyType({
    "c512_k8_decoder": GameProfile("基准＋解码激活去舍入", (("decoder_gather_unround_720", True),),
                                   hits=("decoder_gather_unround_720_active",)),
    "c512_k8_c32_native": GameProfile("基准＋C32 原生三次激活", (("c32_hidden_native_720", True),),
                                      hits=("c32_hidden_native_720_active",)),
    "c512_k8_probability": GameProfile("基准＋C512 注意力概率去舍入",
                                       (("c512_probability_unround_720", True),),
                                       hits=("c512_probability_unround_720_active",)),
    "c512_k8_merge": GameProfile("基准＋解码合并输出去舍入", (("decoder_merge_unround_720", True),),
                                 hits=("decoder_merge_native_720_active",)),
    "c512_k8_merge_fma": GameProfile("基准＋解码合并原生乘加", (("decoder_merge_native_fma_720", True),),
                                     hits=("decoder_merge_native_720_active",)),
    "c512_k8_post_fma16": GameProfile("基准＋输出模块 FP16 融合乘加",
                                      (("post_native_fma_720", "fp16_fma"),),
                                      hits=("post_native_fma_720_active",)),
    "c512_k8_post_fma32": GameProfile("基准＋输出模块 FP32 融合乘加",
                                      (("post_native_fma_720", "fp32_fma"),),
                                      hits=("post_native_fma_720_active",)),
    "num_decoder_full_k": GameProfile("基准＋解码输入完整 K 累加",
                                       numeric=(("decoder_input_full_k", True),)),
    "num_branch_c64": GameProfile("基准＋C64 分支原生累加", numeric=(("branch_accum_families", ("c64",)),)),
    "num_branch_c128": GameProfile("基准＋C128 分支原生累加", numeric=(("branch_accum_families", ("c128",)),)),
    "num_branch_c256": GameProfile("基准＋C256 分支原生累加", numeric=(("branch_accum_families", ("c256",)),)),
    "num_vit_denominator_ordered": GameProfile("基准＋ViT 分母顺序融合",
                                              numeric=(("vit_denominator", "ordered_fused"),)),
    "num_vit_denominator_fp32": GameProfile("基准＋ViT 分母 FP32 归约",
                                           numeric=(("vit_denominator", "fp32_reduction"),)),
    "num_vit_norm_fma": GameProfile("基准＋ViT 归一化融合乘加", numeric=(("vit_norm_fma", True),)),
    "num_vit_exp_fma": GameProfile("基准＋ViT 指数融合乘加", numeric=(("vit_exp_fma", True),)),
    "num_vit_norm_exp_fma": GameProfile("基准＋ViT 归一化和指数融合乘加",
                                       numeric=(("vit_norm_fma", True), ("vit_exp_fma", True))),
    "num_history_fractional": GameProfile("基准＋历史分数采样原生 FP32",
                                          numeric=(("history_value", "fp32_fractional"),)),
    "num_history_dimension_rcp": GameProfile("基准＋历史尺寸倒数原生计算",
                                             numeric=(("history_dimension_rcp", "native"),)),
    "num_history_direct_pixel": GameProfile("基准＋历史像素坐标直接计算",
                                            numeric=(("history_coord", "direct_pixel"),)),
    "num_vit_qkv_full_k": GameProfile("基准＋ViT 输入投影完整累加",
                                     numeric=(("vit_qkv_full_k", True),), pending_validation=True),
    "num_vit_projection_full_k": GameProfile("基准＋ViT 输出投影完整累加",
                                            numeric=(("vit_projection_full_k", True),), pending_validation=True),
    "num_vit_exp_zero_constant": GameProfile("基准＋ViT 零指数常量",
                                           numeric=(("vit_exp_zero_constant", True),), pending_validation=True),
    "num_history_all_paths": GameProfile("基准＋历史全部采样原生 FP32",
                                        numeric=(("history_value", "fp32_all_paths"),), pending_validation=True),
    "num_history_reciprocal": GameProfile("基准＋历史权重倒数原生计算",
                                         numeric=(("history_reciprocal", "native"),), pending_validation=True),
    "num_post_sigmoid": GameProfile("基准＋输出亮度曲线原生计算",
                                    numeric=(("post_sigmoid", "native"),), pending_validation=True),
    "num_post_rne": GameProfile("基准＋输出使用原生就近舍入",
                                numeric=(("post_store", "rne"),), pending_validation=True),
    "num_post_rtz": GameProfile("基准＋输出使用原生截断舍入",
                                numeric=(("post_store", "native_rtz"),), pending_validation=True),
    "num_front_radius": GameProfile("基准＋输入噪声半径原生计算",
                                    numeric=(("front_noise", "native_radius"),), pending_validation=True),
    "num_front_trig": GameProfile("基准＋输入噪声角度原生计算",
                                  numeric=(("front_noise", "native_trig"),), pending_validation=True),
    "num_front_both": GameProfile("基准＋输入噪声半径和角度原生计算",
                                  numeric=(("front_noise", "native_both"),), pending_validation=True),
})

LEGACY_LABELS = MappingProxyType({
    "baseline": "旧基线（仅供回退）", "c512": "仅 C512 库矩阵", "k8": "仅原生 K8",
    "c512_k8": "对照基准：C512＋K8",
    "c512_k8_c32": "旧实验：C32 四站布局", "c512_k8_c32_post": "旧实验：C32 布局和输出尾段",
    "history_compact": "旧实验：历史采样合并", "post_fma_fp16": "旧实验：输出 FP16 乘加",
    "post_fma_fp32": "旧实验：输出 FP32 乘加", "vit_head": "旧实验：ViT 布局直通",
})
EXPERIMENT_LABELS = MappingProxyType({**LEGACY_LABELS, **{k: p.label for k, p in PROFILES.items()}})
EXPERIMENTS_720 = tuple(EXPERIMENT_LABELS)
FUSED_REPLAY_PROFILES = frozenset(k for k, p in PROFILES.items() if p.fused_replay)
HIT_FIELDS = MappingProxyType({
    "c512": ("c512_library_720_active",), "k8": ("native_k8_720_active",),
    "c512_k8": ("c512_library_720_active", "native_k8_720_active"),
    "history_compact": ("history_compact_720_active",),
    "post_fma_fp16": ("post_native_fma_720_active",), "post_fma_fp32": ("post_native_fma_720_active",),
    "vit_head": ("vit_head_720_active",),
    "c512_k8_c32": ("c512_library_720_active", "native_k8_720_active", "c32_window_chain_720_active"),
    "c512_k8_c32_post": ("c512_library_720_active", "native_k8_720_active",
                         "c32_window_chain_720_active", "post_rgb_tail_720_active"),
    **{k: ("c512_library_720_active", "native_k8_720_active") +
          (p.hits or ("numeric_cleanup_720_active",)) for k, p in PROFILES.items()},
})


def combined_mode_options(selected):
    """Combine independently selected mathematics, rejecting conflicting choices."""
    if not isinstance(selected, (tuple, list)) or any(type(key) is not str for key in selected):
        raise ValueError("Expected a list of installed 720p optimization names")
    if len(set(selected)) != len(selected) or set(selected) - set(PROFILES):
        raise ValueError("Unknown or duplicate 720p optimization")
    legacy, numeric = {}, {}
    for key in selected:
        profile = PROFILES[key]
        for name, value in profile.legacy:
            if name in legacy and legacy[name] != value:
                raise ValueError("同一计算位置不能同时选择两种实现")
            legacy[name] = value
        for name, value in profile.numeric:
            if name == "branch_accum_families":
                numeric[name] = tuple(sorted(set(numeric.get(name, ())) | set(value)))
            elif name == "front_noise":
                # Radius and angle are independent flags in the same kernel.
                bits = {"native_radius": 1, "native_trig": 2, "native_both": 3}
                combined = bits.get(numeric.get(name), 0) | bits[value]
                numeric[name] = {1: "native_radius", 2: "native_trig", 3: "native_both"}[combined]
            elif name == "history_value":
                # All paths already includes fractional sampling; run it once.
                values = {value, numeric.get(name, value)}
                if not values <= {"fp32_fractional", "fp32_all_paths"}:
                    raise ValueError("Unknown history sampling implementation")
                numeric[name] = "fp32_all_paths" if "fp32_all_paths" in values else "fp32_fractional"
            elif name in numeric and numeric[name] != value:
                raise ValueError("同一计算位置不能同时选择两种实现")
            else:
                numeric[name] = value
    if numeric.get("history_coord") == "direct_pixel" and numeric.get("history_dimension_rcp") == "native":
        raise ValueError("直接像素坐标与尺寸倒数替换是同一坐标路径的两种实现")
    result = {"c512_qkv_library_720": True, "native_k8_720": True, **legacy}
    if numeric:
        result["numeric_cleanup_720"] = numeric
    return result


def conflicting_profiles():
    """Use the same mathematical conflict rules in the webpage and HTTP API."""
    keys = tuple(PROFILES)
    result = []
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            try:
                combined_mode_options((left, right))
            except ValueError:
                result.append((left, right))
    return tuple(result)


def numeric_health(modes):
    """Read small existing CPU counters; never snapshot, verify files or touch GPU."""
    def all_sites(counter):
        calls = getattr(counter, "calls", None)
        return bool(calls and all(value > 0 for value in calls.values()))

    owner = getattr(modes, "numeric_cleanup_calls", None)
    children = getattr(owner, "children", {})
    child_hits = {}
    for label, child in children.items():
        # History near/fractional sites depend on motion and need not all occur.
        calls = (getattr(child, "owner_calls", None) or getattr(child, "calls", None) or {})
        child_hits[label] = bool(calls and any(value > 0 for value in calls.values()))
        if label == "history" and getattr(owner.options, "history_value", None) == "fp32_fractional":
            child_hits[label] = bool(calls.get("fractional.texture", 0) > 0)
    session = getattr(modes, "session", None)
    stack = getattr(session, "_stack", None)
    gather = getattr(stack, "decoder_gather", None)
    return {
        "c32_hidden_native_720_active": all_sites(getattr(modes, "c32_hidden_calls", None)),
        "c512_probability_unround_720_active": all_sites(getattr(modes, "c512_probability_calls", None)),
        "decoder_merge_native_720_active": all_sites(getattr(modes, "decoder_merge_calls", None)),
        "decoder_gather_unround_720_active": bool(
            getattr(modes, "height", None) == 720 and
            getattr(modes, "decoder_gather_unround_720", False) and
            getattr(gather, "unround_activations", False) and
            getattr(modes, "last_graph_used", False)),
        "numeric_cleanup_720_active": bool(
            owner and not owner.closed and children and all(child_hits.values()) and
            set(owner.preflights) == set(children) and getattr(modes, "last_graph_used", False)),
        "numeric_cleanup_sites": child_hits,
    }
