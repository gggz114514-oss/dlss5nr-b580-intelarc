"""Fuse window reshape/transpose/pixel gather into one exact half-data permutation."""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl
import nr_backend.attention as attention
import nr_backend.multihead_block as multihead


@triton.jit
def _pack(X, ORDER, OUT, TOTAL: tl.constexpr, W: tl.constexpr, HEADS: tl.constexpr,
          ROWS: tl.constexpr, COLS: tl.constexpr, SY: tl.constexpr, SX: tl.constexpr,
          SH: tl.constexpr, SC: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % 32
    pixel = (i // 32) % 64
    window = i // 2048
    col = window % COLS
    row = (window // COLS) % ROWS
    head = window // (COLS * ROWS)
    physical = tl.load(ORDER + pixel).to(tl.int32)
    y, x = row * 8 + physical // 8, col * 8 + physical % 8
    value = tl.load(X + y * SY + x * SX + head * SH + c * SC, i < TOTAL, other=0)
    tl.store(OUT + i, value, i < TOTAL)


@triton.jit
def _unpack(X, INVERSE, OUT, TOTAL: tl.constexpr, W: tl.constexpr, HEADS: tl.constexpr,
            SH: tl.constexpr, SR: tl.constexpr, SW: tl.constexpr,
            SP: tl.constexpr, SC: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % 32
    head = (i // 32) % HEADS
    pixel = i // (32 * HEADS)
    y, x = pixel // W, pixel % W
    local = (y % 8) * 8 + x % 8
    p = tl.load(INVERSE + local).to(tl.int32)
    value = tl.load(X + head * SH + (y // 8) * SR + (x // 8) * SW + p * SP + c * SC,
                    i < TOTAL, other=0)
    tl.store(OUT + i, value, i < TOTAL)


def pack(x, order):
    # Internal NR helper: order is the model's authenticated 64-pixel permutation.
    if x.ndim != 4 or x.shape[-1] != 32 or min(x.shape) <= 0 or x.shape[0] % 8 or x.shape[1] % 8:
        raise ValueError('Expected complete H,W,heads,32 windows')
    if x.device.type != 'xpu' or x.dtype != torch.float16 or order.device != x.device or order.shape != (64,) or order.dtype != torch.int64:
        raise ValueError('Expected half XPU features and owned int64 pixel order')
    h, w, heads, _ = x.shape
    out = torch.empty((heads, h // 8, w // 8, 64, 32), dtype=x.dtype, device=x.device)
    _pack[(triton.cdiv(out.numel(), 256),)](x, order, out, out.numel(), w, heads, h // 8, w // 8, *x.stride(), 256)
    return out


def unpack(x, inverse):
    if x.ndim != 5 or x.shape[-2:] != (64, 32) or min(x.shape) <= 0:
        raise ValueError('Expected head,row,col,64,32 window data')
    if x.device.type != 'xpu' or x.dtype != torch.float16 or inverse.device != x.device or inverse.shape != (64,) or inverse.dtype != torch.int64:
        raise ValueError('Expected half XPU windows and owned int64 inverse')
    heads, rows, cols = x.shape[:3]
    out = torch.empty((rows * 8, cols * 8, heads, 32), dtype=x.dtype, device=x.device)
    _unpack[(triton.cdiv(out.numel(), 256),)](x, inverse, out, out.numel(), cols * 8, heads, *x.stride(), 256)
    return out


class WindowLayout:
    """Scoped class overrides; same projections, quantization, attention and math."""
    def __init__(self):
        self.original_c32 = attention.C32AttentionCore.forward
        self.original_multi = multihead.MultiHeadAttention.forward
        self.calls = {}

    def c32(self, model, features):
        if attention.current_arithmetic_backend() != 'triton' or features.device.type != 'xpu':
            return self.original_c32(model, features)
        if features.ndim != 3 or features.shape[-1] != 32 or features.shape[0] % 8 or features.shape[1] % 8:
            raise ValueError('Expected padded HWC32 with dimensions divisible by eight')
        h, w = features.shape[:2]
        q, k, v = model.front(features)
        q = pack(q.reshape(h, w, 1, 32), model.pixel_order)[0]
        k = pack(k.reshape(h, w, 1, 32), model.pixel_order)[0]
        v = pack(attention.quantize_fp8(v).reshape(h, w, 1, 32), model.pixel_order)[0]
        result = attention.swin_attention_windows(q, k, v, model.bias)
        self.calls['c32'] = self.calls.get('c32', 0) + 1
        return unpack(result.unsqueeze(0), model.pixel_inverse).reshape(h, w, 32)

    def multi(self, model, features):
        if attention.current_arithmetic_backend() != 'triton' or features.device.type != 'xpu':
            return self.original_multi(model, features)
        height, width, channels = features.shape
        if channels != model.channels or height % 8 or width % 8:
            raise ValueError('Expected whole 8x8 attention windows')
        z = multihead.sm89_f16_dot(multihead.quantize_fp8(features), model.qkv, chunk_k=16).reshape(height, width, model.heads, 3, 32)
        q = multihead.quantize_fp8((multihead.normalize_c32(z[:, :, :, 0]) * model.scale[None, None, :, None]).half())
        k = multihead.quantize_fp8(multihead.normalize_c32(z[:, :, :, 1]))
        v = multihead.quantize_fp8(z[:, :, :, 2])
        q, k, v = [pack(t, model.pixel_order) for t in (q, k, v)]
        result = torch.stack([multihead.swin_attention_windows(q[h], k[h], v[h], model.bias[h]) for h in range(model.heads)])
        self.calls['multi'] = self.calls.get('multi', 0) + 1
        return unpack(result, model.pixel_inverse).reshape(height, width, channels)

    @contextmanager
    def installed(self):
        self.calls = {}
        assert attention.C32AttentionCore.forward is self.original_c32
        assert multihead.MultiHeadAttention.forward is self.original_multi
        c32 = lambda model, features: self.c32(model, features)
        multi = lambda model, features: self.multi(model, features)
        attention.C32AttentionCore.forward = c32
        multihead.MultiHeadAttention.forward = multi
        try:
            yield self
        finally:
            assert attention.C32AttentionCore.forward is c32 and multihead.MultiHeadAttention.forward is multi
            attention.C32AttentionCore.forward = self.original_c32
            multihead.MultiHeadAttention.forward = self.original_multi
