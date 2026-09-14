"""Ordered diagnostic pipeline; every failed stage stops, no retries/fallbacks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent


def render(full,out):
    # Reuse the proven encoder/remux/independent decode checks, with four inputs
    # changed to source / native-verified exact / existing NR256 / fullsize fast.
    source=HERE/'review_fourway_render_v2.py'
    text=source.read_text(encoding='utf-8')
    replacements={
        "out=ROOT/'review-v2';out.mkdir(exist_ok=False)":"out=NEW_OUT;out.mkdir(exist_ok=False)",
        "fast=read(ROOT/'fast/validation.json')":"fast=read(FULL/'validation.json')",
        "labels=['原视频（未经过 NR）','4060 原版 NR · 864×480','B580 精确版 · 864×480','B580 快速版 · NR256＋色调修正']":"labels=['原视频（未经过 NR）','精确版 · 已验证4060一致','当前快速版 · NR256','快速算术 · 原尺寸864×480']",
        "pictures=[Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('u1')) for a in (source,nv,ex,quick)]":"oldrow=OLD_FAST['frames'][i]\n            assert sha(oldrow['file'])==oldrow['sha256'] and oldrow['input_sha256']==e['sha256']\n            with np.load(oldrow['file'],allow_pickle=False) as d:oldquick=d['fast'].astype('f4')\n            pictures=[Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('u1')) for a in (source,ex,oldquick,quick)]"
    }
    for old,new in replacements.items():
        if text.count(old)!=1:raise RuntimeError(f'Render adapter anchor changed: {old}')
        text=text.replace(old,new)
    root=Path('D:/Codex-NR-Experiments/nr-b580/fourway-face-review-v1')
    ns=dict(__file__=str(source),__name__='fullsize_render_adapter',FULL=full,NEW_OUT=out,
        OLD_FAST=json.loads((root/'fast/validation.json').read_text(encoding='utf-8')))
    exec(compile(text,str(source),'exec'),ns)
    ns['main']()


def main(root):
    root.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,stages=[],user_visual_review='pending')
    def run(name,args,receipt):
        with (root/f'{name}.stdout.log').open('wb') as stdout,(root/f'{name}.stderr.log').open('wb') as stderr:
            p=subprocess.Popen([sys.executable,'-X','utf8','-u',*map(str,args)],stdout=stdout,stderr=stderr,creationflags=0x08000000)
            row=dict(stage=name,pid=p.pid);report['stages'].append(row)
            (root/'pipeline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
            rc=p.wait();row['returncode']=rc
        if rc:raise RuntimeError(f'{name} failed, rc={rc}; see {root/name}.stderr.log')
        validation=json.loads(receipt.read_text(encoding='utf-8'))
        if not validation['passed']:raise RuntimeError(f'{name} validation failed')
    try:
        runner=HERE/'review_fast_fullsize_v1.py'
        run('collect',[runner,'--phase','collect','--out',root/'collect'],root/'collect/validation.json')
        run('compile',[HERE/'collect_fast_fullsize_v1.py','--catalog',root/'collect/catalog.json','--out',root/'compile.json'],root/'compile.json')
        run('gate',[runner,'--phase','gate','--out',root/'gate'],root/'gate/validation.json')
        run('full',[runner,'--phase','full','--gate',root/'gate','--out',root/'full'],root/'full/validation.json')
        run('render',[__file__,'--render-full',root/'full','--out',root/'review'],root/'review/validation.json')
        report['passed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        (root/'pipeline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--render-full',type=Path)
    a=p.parse_args()
    if a.render_full:render(a.render_full,a.out)
    else:main(a.out)
