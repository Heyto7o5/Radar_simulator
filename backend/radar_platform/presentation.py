"""Encode browser-rendered presentation frames; never modifies simulation data."""
import asyncio
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

router = APIRouter(prefix='/api/v1/presentations', tags=['presentations'])


class ExportSpec(BaseModel):
    title: str = Field(max_length=160)
    width: int = Field(default=1920)
    height: int = Field(default=1080)
    frames: int = Field(ge=1, le=600)
    seconds_per_frame: float = Field(ge=.04, le=10)


def root():
    return Path(os.environ.get('RADAR_RUNTIME_ROOT', '.runtime')).resolve() / 'presentations'


def folder(key):
    if not re.fullmatch(r'[a-f0-9]{32}', key):
        raise HTTPException(404, '导出任务不存在')
    path = root() / key
    if path.is_symlink() or not (path / 'spec.json').is_file():
        raise HTTPException(404, '导出任务不存在')
    return path


@router.post('')
def create(spec: ExportSpec):
    if (spec.width, spec.height) not in [(1280, 720), (1920, 1080)]:
        raise HTTPException(422, '仅支持 720p 或 1080p')
    if spec.frames * spec.seconds_per_frame > 600:
        raise HTTPException(422, '单段视频最长 10 分钟')
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise HTTPException(503, '请先安装展示依赖：pip install -e ".[presentation]"')
    key = uuid.uuid4().hex
    path = root() / key
    path.mkdir(parents=True)
    (path / 'spec.json').write_text(spec.model_dump_json())
    return {'id': key}


@router.put('/{key}/frames/{index}')
async def upload(key: str, index: int, request: Request):
    path = folder(key)
    spec = ExportSpec.model_validate_json((path / 'spec.json').read_text())
    if not 0 <= index < spec.frames or (path / 'encoding').exists():
        raise HTTPException(409, '帧号无效或视频正在编码')
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 12_000_000:
            raise HTTPException(413, '单帧图片过大')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != 'PNG' or im.size != (spec.width, spec.height):
                raise ValueError()
            im.verify()
    except Exception:
        raise HTTPException(422, '需要尺寸匹配的 PNG 图片')
    target = path / f'{index:04d}.png'
    temporary = target.with_suffix('.tmp')
    temporary.write_bytes(data)
    temporary.replace(target)
    return {'frame': index}


@router.post('/{key}/finish')
async def finish(key: str):
    path = folder(key)
    spec = ExportSpec.model_validate_json((path / 'spec.json').read_text())
    if any(not (path / f'{i:04d}.png').is_file() for i in range(spec.frames)):
        raise HTTPException(409, '仍有视频帧未上传')
    try:
        (path / 'encoding').touch(exist_ok=False)
    except FileExistsError:
        raise HTTPException(409, '正在编码，请勿重复提交')
    import imageio_ffmpeg
    try:
        result = await asyncio.to_thread(subprocess.run, [imageio_ffmpeg.get_ffmpeg_exe(),
            '-y', '-framerate', str(1 / spec.seconds_per_frame), '-i', str(path / '%04d.png'),
            '-vf', 'fps=30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(path / 'video.mp4')], capture_output=True, timeout=180)
        if result.returncode:
            raise HTTPException(500, '视频编码失败，请重新导出')
        shutil.copyfile(path / '0000.png', path / 'cover.png')
        return {'video': f'/api/v1/presentations/{key}/video', 'cover': f'/api/v1/presentations/{key}/cover'}
    finally:
        (path / 'encoding').unlink(missing_ok=True)


@router.get('/{key}/{product}')
def download(key: str, product: str):
    if product not in ('video', 'cover'):
        raise HTTPException(404)
    path = folder(key) / ('video.mp4' if product == 'video' else 'cover.png')
    if not path.is_file():
        raise HTTPException(404, '导出尚未完成')
    return FileResponse(path, filename=path.name, media_type='video/mp4' if product == 'video' else 'image/png')


@router.delete('/{key}')
def cancel(key: str):
    path = folder(key)
    if (path / 'encoding').exists():
        raise HTTPException(409, '编码正在结束，请稍后重试')
    shutil.rmtree(path)
    return {'deleted': True}
