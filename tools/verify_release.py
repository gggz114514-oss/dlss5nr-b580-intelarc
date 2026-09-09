"""Check file identity, source correspondence, syntax and local document links.

This verifies the public snapshot, not the authenticity of unpublished raw
experiments and not full NR model equivalence.
"""
import ast,hashlib,json,re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'evidence/file-manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=(ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(),item['path']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
    sources=json.loads((ROOT/'evidence/source-manifest.json').read_text(encoding='utf-8'))
    assert sources['backend_files']==35
    for item in sources['files']:
        assert item['byte_identical'] and sha(ROOT/item['path'])==item['sha256'],item['path']
    index=json.loads((ROOT/'evidence/index.json').read_text(encoding='utf-8'))
    assert index['current_exact_stage_comparisons']==55 and not index['complete_migration']
    for row in index['reports']:
        p=ROOT/row['excerpt_path'];assert sha(p)==row['excerpt_sha256']
        excerpt=json.loads(p.read_text(encoding='utf-8'))
        assert excerpt['provenance']['original_report_sha256']==row['original_report_sha256']
        assert excerpt['excerpt'],row['report_id']
    py_count=0;links=0
    for item in manifest['files']:
        p=ROOT/item['path']
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=item['path']);py_count+=1
        if p.suffix=='.md':
            for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
                target=target.strip('<>').split('#',1)[0]
                if not target or '://' in target or target.startswith('mailto:'):continue
                assert (p.parent/unquote(target)).exists(),f'{item["path"]}: {target}'
                links+=1
    print(json.dumps(dict(passed=True,files_verified=len(manifest['files']),
        original_sources=len(sources['files']),python_syntax_checked=py_count,local_links_checked=links,
        reports=len(index['reports']),scope='Snapshot identity and structure only; no model execution'),indent=2))

if __name__=='__main__':main()
