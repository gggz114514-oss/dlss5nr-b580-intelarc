"""Complete list-shaped evidence excerpts and seal the reviewable public snapshot."""
import ast,hashlib,json,re,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'nr-b580-public';R=BASE/'nr-b580/reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');PRIVATE=D/'publication-v0.1.0-pre'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
index=js(OUT/'evidence/index.json')
for i in (10,11):
    row=index['reports'][i];original=R/row['report_id']
    if not original.exists():original=D/row['report_id']
    assert sha(original)==row['original_report_sha256']
    values=js(original);assert isinstance(values,list)
    # These two reports are numeric lists (texture rules/coordinate candidates),
    # not maps; preserve all list entries and reject any accidental local paths.
    encoded=json.dumps(values,ensure_ascii=False)
    assert not re.search(r'[A-Za-z]:[/\\]|192\.168\.|C:[/\\]+Users|BEGIN .*PRIVATE KEY',encoded)
    path=OUT/row['excerpt_path'];record=js(path)
    record['excerpt']={'measurements':values}
    record['provenance']['excerpt_fields']=['measurements']
    write(path,record);row['excerpt_fields']=['measurements'];row['excerpt_sha256']=sha(path)
write(OUT/'evidence/index.json',index)

source_manifest=js(OUT/'evidence/source-manifest.json')
for item in source_manifest['files']:
    assert sha(OUT/item['path'])==item['sha256']
    origin=(BASE/'nr-b580' if item['source'].startswith('exact/') else BASE/'nr-b580-int8')/item['source'].split('/',1)[1]
    assert sha(origin)==item['sha256']
current=js(OUT/'evidence/excerpts/16.json')['excerpt']
assert current['all_byte_equal'] and current['persisted_rgb_and_private_reread_equal'] and current['full_frame_comparisons']==55
main=js(OUT/'evidence/excerpts/17.json')['excerpt']
assert len(main['frames'])==13 and all(f['all_byte_equal'] for f in main['frames'])
checks=[]
for number,expected in ((4,496),(5,960),(6,1408)):
    record=js(OUT/f'evidence/excerpts/{number:02d}.json')['excerpt']['cases']
    count=0
    def walk(obj):
        nonlocal_placeholder=None
        if isinstance(obj,dict):
            if 'byte_equal' in obj:
                assert obj['byte_equal']
                if 'mismatches' in obj:assert obj['mismatches']==0
                return 1
            return sum(walk(v) for v in obj.values())
        if isinstance(obj,list):return sum(walk(v) for v in obj)
        return 0
    count=walk(record);assert count==expected,(number,count,expected)
    checks.append(dict(excerpt=f'{number:02d}',complete_buffer_comparisons=count))

for row in index['reports']:
    assert js(OUT/row['excerpt_path'])['excerpt']
    assert sha(OUT/row['excerpt_path'])==row['excerpt_sha256']
for p in OUT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:continue
    text=p.read_text(encoding='utf-8-sig')
    assert not re.search(r'192\.168\.|C:[/\\]+Users|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|github_pat_|gh[pousr]_[A-Za-z0-9]{20,}|HBHGGGZ|ASUS|\.ssh[/\\]',text),p
    if p.suffix=='.py':ast.parse(text,filename=str(p))
    assert p.suffix.lower() not in ('.dll','.exe','.cubin','.bin','.npy','.npyz','.mp4','.png','.zip'),p

write(OUT/'evidence/release-checks.json',dict(scope='Publication preparation checks, not a new full NR run',
    source_files_byte_identical=len(source_manifest['files']),exact_backend_files=35,
    historical_buffer_comparison_records_checked=checks,current_exact_stage_frames=55,current_exact_main_frames=13,
    asset_free_fma_witness='passed: 37.53125 versus FP32-then-half 37.5',
    current_published_half_fma_cpu_cases=24007,current_published_half_fma_cpu_mismatches=0,
    full_model_via_new_public_cli_tested=False,private_assets_included=False,passed=True))
files=[]
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or '.git' in p.parts or p==OUT/'evidence/file-manifest.json':continue
    files.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
write(OUT/'evidence/file-manifest.json',dict(schema=1,scope='All publication files except this manifest and Git metadata',files=files))
write(PRIVATE/'publication-finalization-v1.json',dict(passed=True,files=len(files)+1,
    bytes=sum(f['bytes'] for f in files),file_manifest_sha256=sha(OUT/'evidence/file-manifest.json'),
    finalizer_sha256=sha(__file__),local_original_files_unchanged=True))
print(json.dumps(dict(passed=True,files=len(files)+1,bytes=sum(f['bytes'] for f in files)),indent=2))
