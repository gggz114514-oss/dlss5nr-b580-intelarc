"""Thin local ComfyUI adapter; inference runs in an isolated GPU worker."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import comfy.model_management as model_management
from comfy_api.latest import InputImpl

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
JOBS = Path('D:/Codex-NR-Experiments/nr-b580/comfy-jobs')
MODES = {'精确版': 'exact', '快速版（类 DLSS5）': 'fast'}


def execute_video(video, mode, combo, scale=1.0):
    if mode not in MODES:
        raise ValueError('请选择精确版或快速版（类 DLSS5）。')
    source = video.get_stream_source()
    if not isinstance(source, (str, os.PathLike)) or not Path(source).is_file():
        raise ValueError('请连接 Load Video 加载的本地文件型 VIDEO。')
    if video.get_active_trim_window() != (0.0, 0.0):
        raise ValueError('当前节点读取完整视频，请将 Load Video 的裁剪起点和时长设为 0。')
    job = JOBS / uuid.uuid4().hex
    job.mkdir(parents=True)
    request = job / 'request.json'
    request.write_text(json.dumps(dict(input=str(Path(source).resolve()), mode=MODES[mode],
                                      combo=combo, scale=scale), ensure_ascii=False), encoding='utf-8')
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    command = [str(BASE / 'ComfyUI-aki-v3-IntelArc/python/python.exe'), '-B', '-X', 'utf8',
               str(BASE / 'nr-b580/reference/gpu_lease_nr.py'), '--owner', 'comfy-nr-' + job.name,
               '--cwd', str(HERE), '--log', str(job / 'worker.log'), '--timeout-seconds', '1800',
               '--wait-seconds', '0', '--', str(BASE / 'ComfyUI-aki-v3-IntelArc/python/python.exe'),
               '-B', '-X', 'utf8', str(HERE / 'runner.py'), '--request', str(request)]
    env = os.environ.copy()
    env.update(TEMP=str(job), TMP=str(job), TORCHINDUCTOR_CACHE_DIR=str(job / 'inductor'))
    with (job / 'launcher.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while process.poll() is None:
                model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.25)
        except BaseException:
            (job / 'cancel.flag').touch()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                process.wait(timeout=10)
            raise
    result_path = job / 'result.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
    if process.returncode or not result.get('passed'):
        raise RuntimeError(f"NR 处理失败。日志：{job}\n{result.get('error', '请查看 launcher.log 和 worker.log。')}")
    return InputImpl.VideoFromFile(result['output']), result['output']


class NRB580Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'video': ('VIDEO',), 'mode': (list(MODES),)}}

    RETURN_TYPES = ('VIDEO', 'STRING')
    RETURN_NAMES = ('video', '文件路径')
    FUNCTION = 'execute'
    CATEGORY = 'NR B580'
    DESCRIPTION = '单 NR，同尺寸输出。快速版为类 DLSS5，画面与精确版有明显差异。'

    def execute(self, video, mode):
        return execute_video(video, mode, False)


class NRB580XeSSVideo(NRB580Video):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs['required']['sr_scale'] = ('FLOAT', {'default': 2.0, 'min': 1.0, 'max': 2.0,
                                                   'step': 0.01, 'tooltip': '1× 为 XeSS AA；1×、2×已测试，其他倍率由 SDK 校验。'})
        return inputs

    DESCRIPTION = 'NR → XeSS SR → FG，全 GPU Block 链路。快速版为类 DLSS5。FG 输出 2N−1 帧。'

    def execute(self, video, mode, sr_scale):
        return execute_video(video, mode, True, sr_scale)


NODE_CLASS_MAPPINGS = {'NRB580Video': NRB580Video, 'NRB580XeSSVideo': NRB580XeSSVideo}
NODE_DISPLAY_NAME_MAPPINGS = {'NRB580Video': 'NR 视频（B580）', 'NRB580XeSSVideo': 'NR + XeSS SR + FG（B580）'}
