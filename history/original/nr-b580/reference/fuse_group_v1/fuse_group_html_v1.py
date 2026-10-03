"""`.md` → 本项目风格的 `.html`，附带 **mdsafe** 自检。

为什么需要 mdsafe
-----------------
本项目渲染管线的三条硬规矩（前几轮踩过，记在这里免得再踩）：

  1. 加粗标记必须**逐行**平衡 —— 跨行的加粗（开头在上一行、收尾落在下一行）会被判失败；
  2. 不能在加粗里面把 code span 包进去，会挂；
  3. 渲染后不许残留 `*` —— 这条抓的是「加粗标记个数为偶数、却配错对」，
     前两条都查不出来。反例就是本轮踩到的这一行：
     候选 1 的开头用了两对标记，一对包「候选 1（原推荐 」，一对包「 —— 见 §6.2）」，
     于是「已做完，结论为负」被拆成了独立粗体，而本意是整行加粗。

所以 `--check` 逐行验证这三条，`--html` 才渲染。渲染与检查走同一份「行」视图，
不会出现「检查过了但渲染的是另一份」。

用法
----
  python fuse_group_html_v1.py --md RESULT.md --check
  python fuse_group_html_v1.py --md RESULT.md --html RESULT.html
"""
import argparse
import html
import re
from pathlib import Path

CSS = """:root{
  --bg:#f7f8fa; --card:#ffffff; --ink:#141a22; --muted:#5c6675; --line:#e3e7ee;
  --accent:#1e5fd8; --accent-soft:#eef4ff; --warn:#b45309; --warn-soft:#fff7ed; --num:#0b3a8f;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:40px 22px 90px}
h1{font-size:26px;line-height:1.35;margin:0 0 12px;letter-spacing:-.2px}
h2{font-size:20px;margin:44px 0 6px;padding-bottom:8px;border-bottom:1px solid var(--line);letter-spacing:-.2px}
h3{font-size:16px;margin:26px 0 8px;color:#20304a}
h4{font-size:14px;margin:20px 0 6px;color:#33415c}
p{margin:8px 0}
hr{border:none;border-top:1px solid var(--line);margin:34px 0}
code{background:#eef1f6;border-radius:4px;padding:1px 5px;font-size:12.5px;
  font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace;color:#1c355e}
pre{background:#101827;color:#e8eefc;border-radius:10px;padding:16px 18px;overflow:auto;
  font-size:12.5px;line-height:1.65;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
pre code{background:none;color:inherit;padding:0;font-size:12.5px}
table{border-collapse:collapse;width:100%;margin:12px 0;background:var(--card);
  font-size:13px;box-shadow:0 1px 2px rgba(20,26,34,.06);border-radius:8px;overflow:hidden}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f1f4f9;font-weight:600;color:#2a3648;font-size:12.5px;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:#fafcff}
td.isnum,th.isnum{text-align:right;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--num)}
tr.total td{background:#f1f4f9;font-weight:700}
blockquote{background:var(--accent-soft);border-left:3px solid var(--accent);
  padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0;font-size:13.5px}
blockquote.warn{background:var(--warn-soft);border-left-color:var(--warn)}
blockquote p{margin:4px 0}
ul,ol{margin:10px 0;padding-left:24px}
li{margin:3px 0}
strong{font-weight:700}
a{color:var(--accent)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px}"""

NUM = re.compile(r'^[-+]?[\d][\d,]*\.?\d*%?x?$')
FENCE = re.compile(r'^```')
BULLET = re.compile(r'^(\d+\.|[-*])\s+')
HR = ('---', '***', '___')
CODE = re.compile(r'`([^`]+)`')


def mask_code(line):
    """把 code span 替换为等长占位（`\\x01`），得到「标记视图」。

    规矩 1 与 2 都必须在这上面判断 —— 文档里正当引用的 `` `**` `` 是**字面标记本身**，
    不该被当成加粗的开 / 闭标记：否则既会误报「不平衡」，也会把两个 code span
    之间的正文误认成「加粗里包了 code span」。占位符保留长度，便于定位。"""
    return CODE.sub(lambda m: '\x01' * len(m.group(0)), line)


def check(lines, name):
    """mdsafe：三条硬规矩，返回 (行号, 类型, 片段) 列表。

      1. 加粗标记逐行平衡；
      2. 加粗里不许包 code span；
      3. 渲染后不许残留 `*`（抓「标记数平衡但配错对」）。

    列表项与引用块的标记会被先剥掉，免得 `*`/`-` 项目符被当成残留。
    """
    bad = []
    in_fence = False
    for number, line in enumerate(lines, 1):
        if FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        masked = mask_code(line)
        if masked.count('**') % 2:
            bad.append((number, 'unbalanced **', line.strip()[:80]))
            continue
        for span in re.findall(r'\*\*(.+?)\*\*', masked):
            if '\x01' in span:
                bad.append((number, 'code span inside bold', span[:60]))
        body = BULLET.sub('', line.strip()).lstrip('>').strip()
        if not body or body in HR:
            continue
        if '*' in re.sub(r'<code>.*?</code>', '', inline(body)):
            bad.append((number, 'mis-paired ** (残留 *)', body[:80]))
    return bad


def inline(text):
    """行内标记。**先把 code span 摘出来（占位），最后再放回** —— 否则 `` `**` `` 这类
    含标记的 code span 会被后面的 bold/em 正则从内部撕开，既渲染错、又让 mdsafe 第 3 条误报。"""
    text = html.escape(text, quote=False)
    spans = []

    def stash(match):
        spans.append(match.group(1))
        return '\x00%d\x00' % (len(spans) - 1)

    text = CODE.sub(stash, text)
    text = re.sub(r'\*\*([^*]+)\*\*', lambda m: '<strong>' + m.group(1) + '</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', lambda m: '<em>' + m.group(1) + '</em>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: '<a href="' + m.group(2) + '">' + m.group(1) + '</a>', text)
    text = re.sub(r'\x00(\d+)\x00', lambda m: '<code>' + spans[int(m.group(1))] + '</code>', text)
    return text


def split_row(line):
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def render(lines):
    out, index = [], 0
    paragraph = []

    def flush():
        if paragraph:
            out.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()

    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()
        stripped = line.strip()
        if FENCE.match(stripped):
            flush()
            index += 1
            body = []
            while index < len(lines) and not FENCE.match(lines[index].strip()):
                body.append(html.escape(lines[index], quote=False))
                index += 1
            index += 1
            out.append('<pre><code>' + '\n'.join(body) + '</code></pre>')
            continue
        if not stripped:
            flush()
            index += 1
            continue
        if stripped.startswith('|') and index + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[index + 1].strip()):
            flush()
            head = split_row(stripped)
            rows, index = [], index + 2
            while index < len(lines) and lines[index].strip().startswith('|'):
                rows.append(split_row(lines[index].strip()))
                index += 1
            width = len(head)
            numeric = []
            for column in range(width):
                values = [row[column] for row in rows if column < len(row)]
                numeric.append(bool(values) and all(NUM.match(value.strip('`')) for value in values))
            head_html = ''.join(
                f'<th class="isnum">{inline(cell)}</th>' if numeric[column] else f'<th>{inline(cell)}</th>'
                for column, cell in enumerate(head))
            body = []
            for row in rows:
                cells = ''.join(
                    (f'<td class="isnum">{inline(row[column])}</td>' if column < len(row) and numeric[column]
                     else f'<td>{inline(row[column]) if column < len(row) else ""}</td>')
                    for column in range(width))
                body.append('<tr>' + cells + '</tr>')
            out.append('<table><thead><tr>' + head_html + '</tr></thead><tbody>'
                       + ''.join(body) + '</tbody></table>')
            continue
        if stripped in ('---', '***', '___'):
            flush()
            out.append('<hr>')
            index += 1
            continue
        heading = re.match(r'^(#{1,4})\s+(.*)$', stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            out.append(f'<h{level}>' + inline(heading.group(2)) + f'</h{level}>')
            index += 1
            continue
        if stripped.startswith('>'):
            flush()
            body = []
            while index < len(lines) and lines[index].strip().startswith('>'):
                body.append(lines[index].strip().lstrip('>').strip())
                index += 1
            warn = any('⚠' in item for item in body)
            out.append('<blockquote' + (' class="warn"' if warn else '') + '>'
                       + ''.join('<p>' + inline(item) + '</p>' for item in body if item) + '</blockquote>')
            continue
        bullet = re.match(r'^([-*])\s+(.*)$', stripped)
        ordered = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if bullet or ordered:
            flush()
            tag = 'ul' if bullet else 'ol'
            items = []
            while index < len(lines):
                probe = lines[index].strip()
                match = (re.match(r'^([-*])\s+(.*)$', probe) if bullet else re.match(r'^(\d+)\.\s+(.*)$', probe))
                if not match:
                    break
                items.append('<li>' + inline(match.group(2)) + '</li>')
                index += 1
            out.append(f'<{tag}>' + ''.join(items) + f'</{tag}>')
            continue
        paragraph.append(stripped)
        index += 1
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', type=Path, required=True)
    ap.add_argument('--html', type=Path, default=None)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    text = args.md.read_text(encoding='utf-8')
    lines = text.splitlines()
    bad = check(lines, args.md.name)
    if bad:
        print(f'[mdsafe] {args.md.name}: {len(bad)} 行失分')
        for number, kind, sample in bad:
            print(f'  line {number}: {kind} :: {sample}')
        raise SystemExit(1)
    print(f'[mdsafe] {args.md.name}: 通过（逐行 ** 平衡、无 bold 内 code span、无残留 *）')
    if args.check and args.html is None:
        return

    title = args.title or next(
        (re.sub(r'[*`]', '', line.lstrip('# ').strip()) for line in lines if line.startswith('# ')),
        args.md.stem)
    body = render(lines)
    document = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                f'<title>{html.escape(title)}</title>\n<style>\n{CSS}\n</style>\n</head>\n<body>\n'
                '<div class="wrap">\n' + '\n'.join(body) + '\n</div>\n</body>\n</html>\n')
    target = args.html or args.md.with_suffix('.html')
    target.write_text(document, encoding='utf-8')
    # 注意：这是字符数，不是磁盘字节数（写盘时 \n 会被转成 \r\n，磁盘会更大）
    print(f'[html] {target}  ({len(document)} 字符, 字面 ** = {document.count("**")})')


if __name__ == '__main__':
    main()
