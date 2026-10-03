# -*- coding: utf-8 -*-
"""从 D: 回收站无损还原 D:/corr-strip-v1 下指定前缀的子树。

严格按用户级技能 windows-recyclebin-tree-restore §2 的算法：
  收集 → 分类 → 阶段1 目录快照（深度升序）→ 阶段2 文件（深度升序）→ 阶段3 空目录补建
  冲突一律跳过并计数，**绝不覆盖**。

安全约束（本卷 rmdir 会递归上溯删到根）：
  * 全程只用 os.rename（MoveFileEx，同卷元数据搬运，逐字节不变）。
  * **不调用任何** os.rmdir / os.removedirs / shutil.rmtree。
  * 目标已存在且非空 ⇒ 递归合并，逐子项 skip-if-exists（绝不删目标里的东西）。

默认 dry-run，必须显式 --apply 才动手。

用法：
  python restore_from_recycle_v1.py --prefix D:/corr-strip-v1/r            # dry-run
  python restore_from_recycle_v1.py --prefix D:/corr-strip-v1/r --apply
  python restore_from_recycle_v1.py --prefix A,B --apply --json out.json
"""
import argparse
import datetime
import json
import os
import struct
import sys
from collections import defaultdict

RB_ROOT = "D:" + os.sep + "$RECYCLE.BIN"


# --------------------------------------------------------------------------- #
# 回收站解析
# --------------------------------------------------------------------------- #
def parse_i(raw):
    if len(raw) < 28:
        return None
    magic_ok = raw[0:2] == b"$I"
    try:
        ver = struct.unpack_from("<Q", raw, 0)[0]
    except struct.error:
        return None
    if not (magic_ok or ver in (1, 2, 3)):
        return None
    size, ftime, plen = struct.unpack_from("<QQI", raw, 8)
    path = raw[28:].decode("utf-16-le", "surrogatepass").split("\x00")[0]
    try:
        ts = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=ftime / 10)
        ts = ts + (datetime.datetime.now() - datetime.datetime.utcnow())
    except Exception:
        ts = None
    return {"ver": ver, "size": size, "ts": ts, "plen": plen, "path": path}


def collect():
    """扫回收站 SID 目录，返回记录列表（只读）。"""
    if not os.path.isdir(RB_ROOT):
        raise SystemExit("回收站根不存在：%s" % RB_ROOT)
    recs = []
    for sid in os.listdir(RB_ROOT):
        sd = os.path.join(RB_ROOT, sid)
        if not os.path.isdir(sd):
            continue
        try:
            names = os.listdir(sd)
        except OSError as e:
            print("!! 无法列目录 %s: %s" % (sd, e), file=sys.stderr)
            continue
        for nm in names:
            if not nm.startswith("$I"):
                continue
            try:
                with open(os.path.join(sd, nm), "rb") as fh:
                    raw = fh.read()
            except OSError:
                continue
            d = parse_i(raw)
            if not d:
                continue
            tag = nm[2:]
            rp = os.path.join(sd, "$R" + tag)
            d["sid"] = sid
            d["rtag"] = "$R" + tag
            d["rpath"] = rp
            d["has_r"] = os.path.exists(rp)
            if d["has_r"] and os.path.isdir(rp):
                d["kind"] = "dir"
                n = 0
                for _dp, _dn, _fn in os.walk(rp):
                    n += len(_fn)
                d["nfiles"] = n
            elif d["has_r"]:
                d["kind"] = "file"
                try:
                    d["nfiles"] = 1
                    d["rsize"] = os.path.getsize(rp)
                except OSError:
                    d["rsize"] = -1
            else:
                d["kind"] = "MISSING"
                d["nfiles"] = 0
            d["norm"] = d["path"].replace("\\", "/").rstrip("/")
            recs.append(d)
    return recs


def depth_of(p):
    return p.count("/")


# --------------------------------------------------------------------------- #
# 还原原语
# --------------------------------------------------------------------------- #
class Stats:
    def __init__(self):
        self.moved_dir = 0
        self.moved_file = 0
        self.merged = 0
        self.made_dir = 0
        self.skipped = 0
        self.bad = 0
        self.detail = []

    def note(self, action, path, extra=""):
        self.detail.append({"action": action, "path": path, "extra": extra})


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def merge_tree(src, dst, st, apply, depth=0):
    """把 src 子树合并进已存在的 dst，逐子项 skip-if-exists。不删任何东西。

    用 os.listdir（不用 os.walk：walk 在 $RECYCLE.BIN 内会被 ACL 静默吞掉返回空）。
    """
    if depth > 64:
        st.note("merge-depth-guard", dst)
        return
    try:
        names = os.listdir(src)
    except OSError as e:
        st.note("merge-listdir-fail", src, str(e))
        st.bad += 1
        return
    for name in names:
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        try:
            is_dir = os.path.isdir(s)
        except OSError:
            is_dir = False
        if is_dir:
            if not os.path.exists(d):
                if apply:
                    os.rename(s, d)
                    st.moved_dir += 1
                    st.note("merged-dir", d)
                else:
                    st.note("DRY merged-dir", d)
            else:
                merge_tree(s, d, st, apply, depth + 1)
        else:
            if os.path.exists(d):
                st.skipped += 1
                st.note("skip-exists", d)
            else:
                try:
                    if apply:
                        os.rename(s, d)
                        st.moved_file += 1
                        st.note("merged-file", d)
                    else:
                        st.note("DRY merged-file", d)
                except OSError as e:
                    st.bad += 1
                    st.note("rename-fail", d, str(e))


def restore_one(rec, st, apply):
    """还原单条记录。返回 'done' / 'skip' / 'bad'。"""
    src = rec["rpath"]
    dst = rec["norm"]
    kind = rec["kind"]

    if kind == "MISSING":
        st.bad += 1
        st.note("bad-missing-entity", dst, rec["rtag"])
        return "bad"

    if kind == "dir":
        if not os.path.exists(dst):
            ensure_parent(dst)
            if apply:
                try:
                    os.rename(src, dst)
                    st.moved_dir += 1
                    st.note("dir", dst, "%d files" % rec["nfiles"])
                except OSError as e:
                    st.bad += 1
                    st.note("dir-rename-fail", dst, str(e))
            else:
                st.note("DRY dir", dst, "%d files" % rec["nfiles"])
            return "done"
        if os.path.isdir(dst):
            # 目标已存在（多为被掏空的壳）⇒ 合并，绝不删目标
            before = st.merged
            merge_tree(src, dst, st, apply)
            st.merged += 1
            if st.merged - before == 1:
                st.note("merge-into-existing", dst)
            return "done"
        st.skipped += 1
        st.note("skip-exists-nondir", dst)
        return "skip"

    # file
    if os.path.exists(dst):
        st.skipped += 1
        try:
            same = os.path.getsize(dst) == rec.get("rsize", -1)
        except OSError:
            same = False
        st.note("skip-exists", dst, "same_size=%s" % same)
        return "skip"
    ensure_parent(dst)
    if apply:
        try:
            os.rename(src, dst)
            st.moved_file += 1
            st.note("file", dst, "%d B" % rec.get("rsize", -1))
        except OSError as e:
            st.bad += 1
            st.note("file-rename-fail", dst, str(e))
    else:
        st.note("DRY file", dst, "%d B" % rec.get("rsize", -1))
    return "done"


def restore_missing_dirs(recs, st, apply):
    """阶段3：空目录记录（无实体）按深度升序补建。"""
    made = 0
    for r in sorted(recs, key=lambda x: depth_of(x["norm"])):
        if r["kind"] == "MISSING" and not r["has_r"]:
            dst = r["norm"]
            if not os.path.exists(dst):
                if apply:
                    os.makedirs(dst, exist_ok=True)
                    made += 1
                st.note("make-empty-dir", dst)
    st.made_dir += made
    return made


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="逗号分隔的原始路径前缀（大小写不敏感，正斜杠）")
    ap.add_argument("--apply", action="store_true", help="真正执行（默认 dry-run）")
    ap.add_argument("--json", default=None, help="把统计写到该 JSON")
    ap.add_argument("--dump", action="store_true", help="只用打印记录，不还原")
    args = ap.parse_args()

    prefixes = tuple(p.strip().replace("\\", "/").rstrip("/").lower()
                     for p in args.prefix.split(",") if p.strip())

    recs = collect()
    sel = [r for r in recs if r["norm"].lower().startswith(prefixes)]
    print("[scan] 回收站总记录 %d，命中前缀 %s 的记录 %d"
          % (len(recs), list(prefixes), len(sel)))

    k = defaultdict(int)
    for r in sel:
        k[r["kind"]] += 1
    print("[scan] 命中记录构成：%s" % dict(k))

    if args.dump:
        for r in sorted(sel, key=lambda x: x["norm"]):
            print("  [%-8s] n=%-5d %10d B  %s"
                  % (r["kind"], r["nfiles"], r.get("size", 0), r["norm"]))
        return

    # 分类。
    # ⚠️ 不能按「实体里有多少文件」区分「快照」与「空壳」：os.walk 进不去 $RECYCLE.BIN
    #    （ACL 会让它静默吞异常返回 0，技能 §1 坑 2）⇒ nfiles 恒为 0 不可信。
    #    好在两种情形动作完全一致（目标缺则 rename；目标在则合并），无需区分。
    all_dirs = [r for r in sel if r["kind"] == "dir" and r["has_r"]]
    files = [r for r in sel if r["kind"] == "file" and r["has_r"]]

    # 阶段顺序不可颠倒：目录快照先落地
    plan = (sorted(all_dirs, key=lambda x: depth_of(x["norm"]))
            + sorted(files, key=lambda x: depth_of(x["norm"])))
    # 按原始路径去重：同路径多记录取删除时间新者
    by_path = {}
    for r in plan:
        prev = by_path.get(r["norm"])
        if prev is None:
            by_path[r["norm"]] = r
        else:
            ts_new = r["ts"] or datetime.datetime.min
            ts_old = prev["ts"] or datetime.datetime.min
            if ts_new > ts_old:
                by_path[r["norm"]] = r
    plan = list(by_path.values())
    plan.sort(key=lambda x: (0 if x["kind"] == "dir" else 1, depth_of(x["norm"])))

    print("[plan] 目录记录 %d｜文件记录 %d｜按深度升序还原" % (len(all_dirs), len(files)))
    print("[plan] 去重后实际动作 %d 条" % len(plan))
    if not args.apply:
        print("[plan] **dry-run**（未改动任何文件）。加 --apply 才执行。")

    st = Stats()
    for i, r in enumerate(plan, 1):
        restore_one(r, st, args.apply)
        if i % 2000 == 0:
            print("  ... %d/%d" % (i, len(plan)))
    made = restore_missing_dirs(plan, st, args.apply)

    print()
    print("=" * 70)
    print("动作统计：目录 %d｜文件 %d｜合并进已存在目录 %d｜补建空目录 %d｜跳过 %d｜坏记录 %d"
          % (st.moved_dir, st.moved_file, st.merged, made, st.skipped, st.bad))
    print("=" * 70)

    # 完整性：逐个 stat 与 $I 记录的原尺寸对表
    if args.apply:
        bad_size = []
        checked = 0
        for r in plan:
            if r["kind"] != "file":
                continue
            p = r["norm"]
            if not os.path.isfile(p):
                continue
            checked += 1
            if os.path.getsize(p) != r["size"]:
                bad_size.append((p, os.path.getsize(p), r["size"]))
        print("[verify] 抽查文件 %d 个，尺寸与 $I 记录不符 %d 个" % (checked, len(bad_size)))
        for p, a, b in bad_size[:20]:
            print("   !! %s 实际 %d 记录 %d" % (p, a, b))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"prefixes": list(prefixes), "selected": len(sel),
                       "kinds": dict(k), "actions": len(plan),
                       "moved_dir": st.moved_dir, "moved_file": st.moved_file,
                       "merged": st.merged, "made_dir": st.made_dir,
                       "skipped": st.skipped, "bad": st.bad,
                       "detail": st.detail if args.apply else []},
                      fh, ensure_ascii=False, indent=1)
        print("[json] %s" % args.json)


if __name__ == "__main__":
    main()
