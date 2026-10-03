"""Extract an official PyTorch nightly wheel into an isolated D-drive test path."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
root=Path('D:/Codex-NR-Experiments/nr-b580/reference/toolchains/triton-xpu-3.8.0-git1e2d42a0')
root.mkdir(parents=True,exist_ok=True)
assert not list(root.iterdir()),'Refuse to overwrite an existing toolchain'
# The index's download-r2 mirror returned HTTP403; the canonical PyTorch host
# supplies the same filename and its SHA256 in an S3 metadata header.
url='https://download.pytorch.org/whl/nightly/triton_xpu-3.8.0%2Bgit1e2d42a0-cp313-cp313-win_amd64.whl'
expected='c363a2c6e5b0450015a18237fe58d830ab1d80f995947a470917f87ae26b37d8'
wheel=root/'triton_xpu-3.8.0+git1e2d42a0-cp313-cp313-win_amd64.whl'
with urllib.request.urlopen(url,timeout=40) as response,wheel.open('xb') as output:
    assert response.headers['x-amz-meta-checksum-sha256']==expected
    count=0
    while chunk:=response.read(2**20):
        count+=len(chunk);assert count<400*2**20;output.write(chunk)
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==expected
site=root/'site';site.mkdir()
with zipfile.ZipFile(wheel) as z:
    total=0
    for entry in z.infolist():
        target=(site/entry.filename).resolve();assert target.is_relative_to(site.resolve())
        total+=entry.file_size;assert total<1536*2**20
    z.extractall(site)
manifest={str(p.relative_to(site)):hashlib.sha256(p.read_bytes()).hexdigest() for p in site.rglob('*') if p.is_file()}
record=dict(source_url=url,wheel_sha256=expected,wheel_bytes=wheel.stat().st_size,extracted_bytes=total,
    files=manifest,site=str(site),installed_environment_modified=False,
    provisioner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'provision-v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2))
