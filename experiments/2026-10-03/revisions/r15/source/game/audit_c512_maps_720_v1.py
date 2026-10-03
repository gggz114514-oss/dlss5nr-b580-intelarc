"""stdlib-only compact current720 window/query/pool maps.

Order is the owned model pixel_order: packed slot -> physical 8x8 pixel.
Every valid query occurs exactly once. Keys remain all 64 slots per window.
Map assets are uploaded once before capture by the owned loader.
"""
from __future__ import annotations

HEIGHT, WIDTH, CHANNELS, QUERY_TILE = 24, 40, 512, 16
SHIFTS = ((0, 0), (4, 4), (0, 4), (4, 0))


def build(order, shift):
    order = tuple(int(x) for x in order)
    if len(order) != 64 or sorted(order) != list(range(64)):
        raise ValueError("pixel_order must be the actual 64-slot permutation")
    shift = tuple(shift)
    if shift not in SHIFTS:
        raise ValueError("Only the current720 four window shifts are supported")
    sy, sx = shift
    hp, wp = ((HEIGHT + sy + 7) // 8 * 8, (WIDTH + sx + 7) // 8 * 8)
    source, bias_rows, destinations, windows = [], [], [], []
    for wy in range(hp // 8):
        for wx in range(wp // 8):
            valid = []
            for slot, physical in enumerate(order):
                y, x = wy * 8 + physical // 8, wx * 8 + physical % 8
                if sy <= y < sy + HEIGHT and sx <= x < sx + WIDTH:
                    valid.append((y * wp + x, slot, (y - sy) * WIDTH + x - sx))
            if len(valid) % QUERY_TILE:
                raise ValueError("Current720 per-window valid queries must tile exactly")
            for begin in range(0, len(valid), QUERY_TILE):
                windows.append(wy * (wp // 8) + wx)
                for src, slot, dst in valid[begin:begin + QUERY_TILE]:
                    source.append(src);bias_rows.append(slot);destinations.append(dst)
    if len(source) != HEIGHT * WIDTH or sorted(destinations) != list(range(HEIGHT * WIDTH)):
        raise AssertionError("valid query map lost or duplicated a current720 pixel")
    inverse = [0] * (HEIGHT * WIDTH)
    for compact, dst in enumerate(destinations):
        inverse[dst] = compact
    return dict(shift=shift, hp=hp, wp=wp, window_count=(hp // 8)*(wp // 8),
                query_groups=len(windows), source=source, bias_rows=bias_rows,
                destinations=destinations, windows=windows, inverse=inverse)


def pool_sources():
    """Current encoder7 ordered top/bottom pairs, original valid HWC pixels."""
    return [(2*y*WIDTH+2*x, 2*y*WIDTH+2*x+1,
             (2*y+1)*WIDTH+2*x, (2*y+1)*WIDTH+2*x+1)
            for y in range(HEIGHT // 2) for x in range(WIDTH // 2)]

