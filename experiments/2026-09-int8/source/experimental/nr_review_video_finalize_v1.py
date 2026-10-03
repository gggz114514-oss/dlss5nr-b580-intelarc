"""Independent FFmpeg9 finalization; verify container, SPS, timestamps and pixels."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

HIDDEN = 0x08000000


def read_exact(pipe, size):
    parts = []; left = size
    while left:
        block = pipe.read(left)
        if not block:
            raise EOFError('Incomplete video frame')
        parts.append(block); left -= len(block)
    return b''.join(parts)


def finalize(ffmpeg, encoded, output, width, height, count, fps, duration):
    ffmpeg, encoded, output = map(Path, (ffmpeg, encoded, output))
    assert not output.exists()
    probe_exe = ffmpeg.with_name('ffprobe.exe')
    command = [str(ffmpeg), '-v', 'error', '-i', str(encoded), '-map', '0:v:0',
               '-c:v', 'copy', '-bsf:v',
               'h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0',
               '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
               '-color_range', 'tv', '-movflags', '+faststart+write_colr', '-n', str(output)]
    done = subprocess.run(command, capture_output=True, creationflags=HIDDEN, timeout=60)
    assert done.returncode == 0, done.stderr.decode(errors='replace')

    def probe(path):
        return json.loads(subprocess.check_output([
            str(probe_exe), '-v', 'error', '-select_streams', 'v:0', '-show_packets',
            '-show_entries', 'packet=pts,dts,duration:stream=width,height,pix_fmt,avg_frame_rate,nb_frames,duration,time_base,color_space,color_range,color_transfer,color_primaries',
            '-of', 'json', str(path)], creationflags=HIDDEN, timeout=30))

    before, after = probe(encoded), probe(output)
    s = after['streams'][0]
    assert (s['width'], s['height'], s['pix_fmt'], s['avg_frame_rate'], int(s['nb_frames'])) == (width, height, 'yuv420p', fps, count)
    assert abs(float(s['duration']) - duration) < .001
    assert all(s.get(k) == 'bt709' for k in ('color_space', 'color_transfer', 'color_primaries')) and s['color_range'] == 'tv'
    assert before['packets'] == after['packets'] and len(after['packets']) == count
    assert before['streams'][0]['time_base'] == s['time_base']
    data = output.read_bytes(); pos = data.find(b'colrnclx')
    assert pos >= 4 and data.find(b'colrnclx', pos + 1) == -1
    assert int.from_bytes(data[pos-4:pos], 'big') == 19
    atom = [int.from_bytes(data[pos+i:pos+i+2], 'big') for i in (8, 10, 12)]
    assert atom == [1, 1, 1] and not data[pos+14] & 128
    del data
    trace = subprocess.run([str(ffmpeg), '-v', 'info', '-i', str(output), '-map', '0:v:0',
        '-c:v', 'copy', '-bsf:v', 'trace_headers', '-frames:v', '1', '-f', 'null', '-'],
        capture_output=True, creationflags=HIDDEN, timeout=30)
    assert trace.returncode == 0
    text = trace.stderr.decode(errors='replace'); vui = {}
    for key, expected in [('colour_primaries', 1), ('transfer_characteristics', 1),
                          ('matrix_coefficients', 1), ('video_full_range_flag', 0)]:
        values = {int(v) for v in re.findall(r'\b'+key+r'\s+[01]+\s+=\s+(\d+)', text)}
        assert values == {expected}, (key, values)
        vui[key] = expected
    processes = []; files = []; rows = []
    try:
        for label, path in [('encoded', encoded), ('final', output)]:
            err = (output.parent / (label+'-pixel-check.stderr.txt')).open('wb'); files.append(err)
            processes.append(subprocess.Popen([str(ffmpeg), '-v', 'error', '-threads', '2', '-i', str(path),
                '-map', '0:v:0', '-an', '-sn', '-dn', '-fps_mode', 'passthrough', '-pix_fmt', 'yuv420p',
                '-c:v', 'rawvideo', '-threads', '1', '-f', 'rawvideo', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=err, creationflags=HIDDEN))
        size = width * height * 3 // 2
        for i in range(count):
            a, b = [read_exact(p.stdout, size) for p in processes]
            assert a == b, ('Finalization changed decoded pixels', i)
            rows.append(dict(frame=i, bytes=size, byte_equal=True, sha256=hashlib.sha256(a).hexdigest()))
        for p in processes:
            assert p.stdout.read(1) == b'' and p.wait(timeout=30) == 0
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill(); p.wait()
        for f in files:
            f.close()
    return dict(passed=True, probe=s, nclx=atom, vui=vui,
                packet_timestamps_equal=True, all_decoded_frames_byte_equal=True, decoded_frames=rows,
                video=dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), bytes=output.stat().st_size))
