"""Presentation exports preserve uploaded frames and reject incomplete output."""
import io
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from radar_platform.presentation import router


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip('imageio_ffmpeg')
    monkeypatch.setenv('RADAR_RUNTIME_ROOT', str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def create(client, **overrides):
    return client.post('/api/v1/presentations', json={
        'title': '展示测试', 'width': 1280, 'height': 720,
        'frames': 2, 'seconds_per_frame': .1, **overrides})


def png(color, size=(1280, 720)):
    out = io.BytesIO()
    Image.new('RGB', size, color).save(out, format='PNG')
    return out.getvalue()


def test_incomplete_and_malformed_frames(client):
    key = create(client).json()['id']
    base = f'/api/v1/presentations/{key}'
    assert client.post(base+'/finish').status_code == 409
    assert client.put(base+'/frames/2', content=png('red')).status_code == 409
    assert client.put(base+'/frames/0', content=png('red',(10,10))).status_code == 422
    assert client.put(base+'/frames/0', content=b'bad png').status_code == 422
    assert client.get(base+'/video').status_code == 404
    assert client.delete(base).status_code == 200
    assert client.get(base+'/cover').status_code == 404


def test_encode_and_download(client):
    key = create(client).json()['id']
    base = f'/api/v1/presentations/{key}'
    red = png('red')
    for i, data in enumerate([red,png('blue')]):
        assert client.put(base+f'/frames/{i}',content=data).status_code == 200
    response = client.post(base+'/finish')
    assert response.status_code == 200, response.text
    cover = client.get(response.json()['cover'])
    assert cover.content == red
    video = client.get(response.json()['video'])
    assert video.headers['content-type'] == 'video/mp4'
    assert b'ftyp' in video.content[:32]


def test_resource_limits(client):
    assert create(client,width=20000).status_code == 422
    assert create(client,frames=601).status_code == 422
    assert create(client,frames=600,seconds_per_frame=10).status_code == 422
