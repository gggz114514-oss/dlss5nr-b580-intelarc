"""派生一份采集记录，让改后的产品能被 `materials_entry_v1` / `gapfill_entry_v1` 接受。

为什么必须派生
--------------
`materials_entry_v1.py:141` 有 `assert paths.sha(path) == digest, path` —— 它把产品源文件的
sha256 钉在采集记录里，保证"跑的就是采集那一刻的产品"。本轮按批准把
`nr_backend/sampling.py` 的 `_sample_five_axes` 加了 motion-scope 短路后
（`fb13d07c…` → 见下方实测），上一份派生记录必然对不上。

**上一份记录是历史记录，本轮不动它。** 这里再派生一份到本轮目录：只把 `loaded_sources`
里 `sampling.py` 一项换成当前真实哈希，其余字段逐字照抄。

链条（每份都只动过 sampling.py 一项）
------------------------------------
    D:/Codex-NR-Experiments/nr-b580/fourway-face-review-v1/exact/validation.json   （锚点，冻结）
      → D:/warpsync-land-v1/exact-derived/validation.json                          （N4b 之后）
      → D:/allmerge-land-v1/exact-derived/validation.json                          （N4d 之后）
      → D:/motion-scope-land-v1/exact-derived/validation.json                      （本轮）

⚠️ 必须从**上一份派生记录**接着派生，不能从锚点直接派生：锚点是采集时刻的清单，
   `N2b`/`N4`/`N4b`/`N4d` 四轮落地都动过产品源，从锚点看会有**多个**文件对不上，
   那正是"只允许 sampling.py 变"这条断言要拦下的情况（2026-09-19 轮 A 三个臂就是这么全灭的）。

这份派生记录的边界（不许沉默）
------------------------------
1. **输入帧没有重采**：`frames[].file` 仍指向上游的 npz、`sha256` 保持原值，
   脚本逐项断言它们与磁盘现状一致 —— 对不上就报错退出，不硬凑。
2. `loaded_sources` 里**除 `sampling.py` 外每一项都必须仍然对得上**，
   否则说明还有别的东西被动过，同样报错退出。
3. 派生记录带 `derived_from` / `derived_from_sha256` / `derivation_note` 字段，
   任何下游一眼能看出它不是采集产物。
4. 因为路径 ≠ `paths.EXACT_VALIDATION`，`materials_entry_v1` 会把
   `frozen_reference` 记成 False。那是**路径判据**，不是"输入换了" ——
   输入帧的可比性由第 1 条保证，不靠这个标志。

用法：
    python motion_scope_land_derive_exact_v1.py --out D:/motion-scope-land-v1/exact-derived
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE = Path('D:/allmerge-land-v1/exact-derived/validation.json')
LANDED_SAMPLING = Path('E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/backend/nr_backend/sampling.py')
PRE_LANDING_SHA = 'fb13d07cdc0b0fe7636c210711f84d5f996102dd68762b10193602cc93b08b56'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()

    record = json.loads(args.base.read_text(encoding='utf-8-sig'))
    assert record.get('passed') is True, 'base record did not pass'

    landed_sha = sha(LANDED_SAMPLING)
    print('[derive] base             = %s' % args.base)
    print('[derive] base sha256      = %s' % sha(args.base))
    print('[derive] base derived_from= %s' % record.get('derived_from'))
    print('[derive] landed sampling  = %s  (%s)' % (landed_sha[:16], LANDED_SAMPLING))

    # ---- 第 1 条：输入帧必须逐项对得上磁盘现状（没有重采）----
    frames = record['frames']
    for row in frames:
        assert Path(row['file']).is_file(), 'missing input frame: %s' % row['file']
        assert sha(row['file']) == row['sha256'], (
            'input frame changed on disk: %s' % row['file'])
    print('[derive] %d input frames verified byte-for-byte against disk' % len(frames))

    # ---- 第 2 条：loaded_sources 里只允许 sampling.py 变 ----
    sources = record['loaded_sources']
    changed = []
    for path, digest in sources.items():
        now = sha(path)
        if now != digest:
            changed.append(path)
    assert len(changed) == 1, (
        'expected exactly one changed source (sampling.py), got %d: %r'
        % (len(changed), changed))
    target = Path(changed[0])
    assert target.resolve() == LANDED_SAMPLING.resolve(), (
        'the changed source is not the landed sampling.py: %s' % target)
    assert sources[changed[0]] == PRE_LANDING_SHA, (
        'base does not record the pre-landing sampling.py hash: %s' % sources[changed[0]])

    sources[changed[0]] = landed_sha
    record['derived_from'] = str(args.base)
    record['derived_from_sha256'] = sha(args.base)
    record['derivation_note'] = (
        'Derived record: input frames were NOT re-captured (frames[].file and sha256 '
        'are the upstream values and were re-verified against disk). Only the '
        'loaded_sources entry for nr-b580/backend/nr_backend/sampling.py was advanced '
        'from %s to %s, because that file was changed on purpose: _sample_five_axes now '
        'short-circuits the five bilinear taps to the centre tap when all six axis terms '
        'sit within the sampler own 1/256-pixel fixed-point grid, selectable at run time '
        'with NR_MOTION_SCOPE (default on, off = pre-landing five-tap form) and '
        'NR_MOTION_SCOPE_TOL (default 1/256). This is NOT bit-identical by construction: '
        'the neighbouring tap carries a ~1.7e-05 weight, so the output moves by O(1e-5). '
        'Every other loaded source still matches its upstream digest.'
        % (PRE_LANDING_SHA, landed_sha))
    record['derived_by'] = str(Path(__file__).resolve())

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / 'validation.json'
    out.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('[derive] wrote %s  sha256=%s' % (out, sha(out)))
    print('[derive] advanced source: %s' % changed[0])
    print('[derive]   %s -> %s' % (PRE_LANDING_SHA[:16], landed_sha[:16]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
