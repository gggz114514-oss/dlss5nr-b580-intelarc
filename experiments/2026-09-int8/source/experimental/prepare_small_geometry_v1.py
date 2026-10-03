"""Stage bounded native small-geometry sequences; do not change B580 contracts."""
import hashlib
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent / 'nr-b580/reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/small-geometry-v1')
assert not D.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = R / 'inputs/visual-qa-01/apex.png'
old = json.loads((R / 'inputs/dimension-512/manifest.json').read_text())
assert sha(source) == old['source_sha256']
archive = R / 'video2dlssnr-aux-dimensions-v1.zip'
assert sha(archive) == '22b65afebe009744e7c12dec5f4e19ee8ee20526691aa25b89cb250be72e21a4'
D.mkdir()
stage = D / 'stage'
stage.mkdir()
scripts = []


def change(s, a, b):
    assert s.count(a) == 1, (a, s.count(a))
    return s.replace(a, b)


def write(name, value):
    path = HERE / name
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        f.write(value)
    scripts.append(path)
    (stage / name).write_bytes(path.read_bytes())


old_dims = "'512x512','864x480','1920x1080','2559x1439'"
new_dims = "'128x128','192x192','256x144'"
for leaf, new_leaf in [('Run-NrUiDimensionProbe.ps1', 'Run-NrSmallGeometryV1.ps1'),
                       ('Invoke-InteractiveUiDimension.ps1', 'Invoke-NrSmallGeometryV1.ps1')]:
    s = (R / leaf).read_text(encoding='utf-8-sig')
    s = change(s, old_dims, new_dims)
    s = change(s, "[string]$Dimension='512x512'", "[string]$Dimension='128x128'")
    s = change(s, "[string]$Mode='zero'", "[string]$Mode='fine'")
    s = change(s, "[ValidateSet('i05','i2','t05s15','i2t05s15','style1i05t05','style2i2t05')][string]$UiControl='i05'",
               "[ValidateSet('balanced')][string]$UiControl='balanced'")
    s = s.replace("$ErrorActionPreference = 'Stop'", "$ErrorActionPreference = 'Stop'\n$ProgressPreference='SilentlyContinue'")
    s = s.replace("$ErrorActionPreference='Stop'", "$ErrorActionPreference='Stop'\n$ProgressPreference='SilentlyContinue'")
    if leaf.startswith('Run-'):
        s = change(s, "$nrUiControls=@{", "$nrUiControls=@{\n    'balanced'=@{style='0';intensity='1';tone='1';structure='1'}")
        s = change(s, "$Case=if ($Dimension -eq '512x512') {'temporal-512-v1'} else {'temporal-full-'+$Dimension+'-v1'}",
                   "$Case='small-geometry-v1-'+$Dimension")
        s = change(s, "$prefix = 'ui-dimension-'", "$prefix = 'small-geometry-v1-'")
        s = change(s, '$inputPath = if ($Batch) { "$root\\inputs\\$Case" } else { "$root\\inputs\\$Case\\gradient.png" }',
                   '$inputPath = "$root\\experiments\\small-geometry-v1\\inputs\\$Dimension"\n'
                   '$fixtureManifest=Join-Path $inputPath "manifest.json"\n'
                   '$fixture=Get-Content -LiteralPath $fixtureManifest -Raw | ConvertFrom-Json\n'
                   'if ($fixture.dimension -ne $Dimension -or $fixture.frames.Count -ne 4) {throw "Bad fixture"}\n'
                   'foreach ($frame in $fixture.frames) {\n'
                   '    if ($frame.file -notmatch "^frame0[0-3][.]png$") {throw "Unexpected frame name"}\n'
                   '    if ((Get-FileHash -LiteralPath (Join-Path $inputPath $frame.file)).Hash -ne $frame.sha256) {throw "Changed fixture frame"}\n'
                   '}')
        s = change(s, "$runtime = Join-Path $bin 'nvngx_dlssnr.dll'",
                   "$runtime = Join-Path $bin 'nvngx_dlssnr.dll'\n"
                   "if ((Get-FileHash -LiteralPath (Join-Path $bin 'video2dlssnr.exe')).Hash -ne 'AD69B6C840CCAA3ADB977D10CCDDA6DF12BB68CCE11C03904A313BC6EDD89B53') {throw 'Unexpected reference executable'}\n"
                   "if ((Get-FileHash -LiteralPath (Join-Path $bin 'nr_nvapi_trace.dll')).Hash -ne '1D1A002C01195D20DD40E950C4C5333827A619CEFF1919751EB52223D6DF7620') {throw 'Unexpected trace DLL'}")
        s = change(s, '    dimension=$Dimension',
                   '    runnerSha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash\n'
                   '    fixtureManifestSha256=(Get-FileHash -LiteralPath $fixtureManifest).Hash\n'
                   '    dimension=$Dimension')
    else:
        s = change(s, "'Codex-NR-AuxTexture-'", "'Codex-NR-SmallGeometry-'" )
        s = change(s, "$script=Join-Path $root 'Run-NrUiDimensionProbe.ps1'",
                   "$script=Join-Path $root 'experiments\\small-geometry-v1\\Run-NrSmallGeometryV1.ps1'")
        s = change(s, "Get-ChildItem -LiteralPath (Join-Path $root 'runs') -Directory |",
                   "Get-ChildItem -LiteralPath (Join-Path $root 'runs') -Directory -Filter ('small-geometry-v1-'+$Dimension+'-*') |")
    write(new_leaf, s)

with Image.open(source) as image:
    image = image.convert('RGB')
    for w, h in ((128, 128), (192, 192), (256, 144)):
        dimension = f'{w}x{h}'
        dest = stage / 'inputs' / dimension
        dest.mkdir(parents=True)
        frames = []
        for i, shift in enumerate((0, -1, -2, 0)):
            box = (1000 + shift, 500, 1000 + shift + w, 500 + h)
            name = f'frame{i:02d}.png'
            image.crop(box).save(dest / name)
            frames.append(dict(file=name, sha256=sha(dest / name), crop_xyxy=box, reset=i in (0, 3)))
        fixture = dict(source=str(source), source_sha256=sha(source), dimension=dimension,
                       frames=frames, motion='fine synthetic spatial field, same existing native uploader',
                       depth='zero', controls='default, no auxiliary textures', resized=False)
        (dest / 'manifest.json').write_text(json.dumps(fixture, indent=2) + '\n', encoding='utf-8')

files = {str(p.relative_to(stage)).replace('\\', '/'): dict(sha256=sha(p), bytes=p.stat().st_size)
         for p in sorted(stage.rglob('*')) if p.is_file()}
manifest = dict(scope=__doc__, files=files, sources={str(p): sha(p) for p in
                [Path(__file__), R / 'Run-NrUiDimensionProbe.ps1', R / 'Invoke-InteractiveUiDimension.ps1', archive, source, *scripts]},
                total_stage_bytes=sum(v['bytes'] for v in files.values()), native_experiments_run=False)
(D / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
(stage / 'stage-manifest.json').write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(stage=str(stage), bytes=manifest['total_stage_bytes'], files=len(files)), indent=2))
