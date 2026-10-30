import io
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from scipy.io import savemat

from radar_platform.api import app
from radar_platform.models import C0
from radar_platform.unified.schema import Simulation
from radar_platform.unified.engine import simulate_frame, SimulationCancelled
from radar_platform.workbench import PreviewRequest, preview, presets, resolve_request


def tiny_request():
    value=presets()['points_pulsed_lfm']
    value['sequence']['frame_count']=1
    return preview(PreviewRequest(request=value))['request']


def wait_job(client,job):
    end=time.monotonic()+20
    while time.monotonic()<end:
        state=client.get(f'/api/v1/scene-simulations/{job}').json()
        if state['status'] in ('complete','failed','cancelled'):return state
        time.sleep(.02)
    raise AssertionError('job did not terminate')


@pytest.mark.parametrize('height',[3000,5000,10000])
@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_automatic_gate_tracks_height_and_keeps_real_bandwidth(height,kind):
    draft=presets()['mixed_'+kind]
    draft['platform']['position_ned_m'][2]=-height
    result=preview(PreviewRequest(request=draft))
    request=Simulation.model_validate(result['request'])
    assert request.waveform.bandwidth_hz==pytest.approx(10e6)
    assert len(result['geometry']['objects'])==4
    assert request.processing.range_limits_m[0]<height<request.processing.range_limits_m[1]
    if kind=='pulsed_lfm':
        assert request.waveform.adc_start_s<2*height/C0
    else:assert request.waveform.adc_start_s>2*height/C0


def test_auto_gate_covers_interior_closest_approach_not_only_endpoints():
    draft=presets()['points_pulsed_lfm']
    draft['platform']['position_ned_m']=[-1000,0,-5000]
    draft['platform']['velocity_ned_mps']=[1000,0,0]
    draft['scene']['objects']=[{'kind':'point','id':'p','position_ned_m':[0,0,0]}]
    draft['sequence'].update(frame_count=3,frame_interval_s=1)
    request,_=resolve_request(draft,True)
    assert request.processing.range_limits_m[0]<5000
    assert request.waveform.adc_start_s<2*5000/C0


def test_auto_mode_does_not_silently_change_unworkable_prf():
    draft=presets()['points_pulsed_lfm'];draft['waveform']['slot_period_s']=10e-6
    with pytest.raises(ValueError,match='PRF'):resolve_request(draft,True)


def test_cooperative_cancellation_does_not_change_uncancelled_numerics():
    request=Simulation.model_validate(tiny_request())
    before=simulate_frame(request)
    after=simulate_frame(request,cancelled=lambda:False)
    np.testing.assert_array_equal(before.adc,after.adc)
    with pytest.raises(SimulationCancelled):simulate_frame(request,cancelled=lambda:True)


def test_preview_and_matrix_views_have_axes_reference_and_saved_parameters(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        raw=presets()['points_pulsed_lfm'];raw['sequence']['frame_count']=1
        checked=client.post('/api/v1/workbench/preview',json={'request':raw,'auto_acquisition':True})
        assert checked.status_code==200
        request=checked.json()['request']
        created=client.post('/api/v1/scene-simulations',json=request);assert created.status_code==202
        job=created.json()['job_id'];state=wait_job(client,job);assert state['status']=='complete'
        assert client.get(f'/api/v1/workbench/jobs/{job}/request').json()==request
        references=[]
        for channel in ['combined','tx0_rx0','tx1_rx3']:
            data=client.get(f'/api/v1/workbench/jobs/{job}/frames/0/view',params={'channel':channel})
            assert data.status_code==200
            values=np.frombuffer(data.content,dtype='<f4')
            assert len(values)==int(data.headers['X-Rows'])*int(data.headers['X-Cols'])
            assert np.isfinite(values).all() and values.max()>-60
            references.append(data.headers['X-Reference-Power'])
        assert len(set(references))==1
        ra=client.get(f'/api/v1/workbench/jobs/{job}/frames/0/view?view=ra')
        assert int(ra.headers['X-Rows'])==len(state['frames'][0]['axes']['azimuth_deg'])
        assert client.get(f'/api/v1/workbench/jobs/{job}/frames/0/view?channel=tx9_rx0').status_code==422
        assert client.get(f'/api/v1/workbench/jobs/{job}/frames/-1/view').status_code==422


@pytest.fixture
def tiny_dataset(monkeypatch,tmp_path):
    from radar_platform.comparison_ui import real_configuration
    base=tmp_path/'datasets';root=base/'ramp_adc_77ghz'
    scene=root/'extracted/Automotive/test_sequence';radar=scene/'radar_raw_frame';labels=scene/'text_labels'
    radar.mkdir(parents=True);labels.mkdir()
    cfg=real_configuration()
    request=Simulation.model_validate({**cfg.model_dump(),'scene':{'objects':[{'kind':'point','id':'p','position_ned_m':[10,1,0]}],'amplitude_reference_range_m':10.}})
    adc=simulate_frame(request).adc
    savemat(radar/'000001.mat',{'adc':adc})
    for index in range(3):(labels/f'{index:010d}.csv').write_text(f'1,2,1,{10-index*.03},2,4\n')
    image=Image.new('RGB',(32,24),(40,80,120));buffer=io.BytesIO();image.save(buffer,format='JPEG')
    (root/'raw').mkdir()
    with zipfile.ZipFile(root/'raw/Automotive.zip','w') as archive:archive.writestr('Automotive/test_sequence/images_0/0000000001.jpg',buffer.getvalue())
    monkeypatch.setenv('RADAR_DATA_ROOT',str(base));monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path/'runtime'))
    return root


def test_comparison_real_camera_labels_editable_input_and_both_channel_views(tiny_dataset):
    with TestClient(app) as client:
        catalog=client.get('/api/v1/comparisons/ramp/catalog').json()
        assert catalog['available'] and catalog['frames']==1
        frame=catalog['sequences'][0]['frames'][0];assert frame['camera_available']
        prepared=client.post('/api/v1/comparisons/ramp/prepare',json={'frame_id':frame['id']})
        assert prepared.status_code==200
        record=prepared.json();assert record['is_real'] and record['template']
        assert record['labels'][0]['class_name']=='汽车'
        image=client.get(record['camera_url'])
        assert image.status_code==200 and image.headers['content-type']=='image/jpeg'
        assert Image.open(io.BytesIO(image.content)).size==(32,24)
        request=record['template'];request['scene']['objects'][0]['position_ned_m'][0]=12
        created=client.post(f"/api/v1/comparisons/ramp/records/{record['record_id']}/simulate",json={'request':request})
        assert created.status_code==202
        job=created.json()['job_id'];assert wait_job(client,job)['status']=='complete'
        for channel in ['combined','tx0_rx0','tx1_rx3']:
            measured=client.get(f"/api/v1/comparisons/ramp/records/{record['record_id']}/rd",params={'channel':channel})
            simulated=client.get(f"/api/v1/comparisons/ramp/records/{record['record_id']}/simulations/{job}/rd",params={'channel':channel})
            assert measured.status_code==simulated.status_code==200
            assert measured.headers['X-Reference-Power']==simulated.headers['X-Reference-Power']
            assert len(measured.content)==len(simulated.content)
            assert not np.array_equal(np.frombuffer(measured.content,'<f4'),np.frombuffer(simulated.content,'<f4'))
        request['waveform']['start_hz']=10e9
        assert client.post(f"/api/v1/comparisons/ramp/records/{record['record_id']}/simulate",json={'request':request}).status_code==422
        assert client.post('/api/v1/comparisons/ramp/prepare',json={'frame_id':'../../private'}).status_code==404


def test_comparison_missing_data_is_not_replaced_by_synthetic(monkeypatch):
    monkeypatch.delenv('RADAR_DATA_ROOT',raising=False)
    with TestClient(app) as client:
        status=client.get('/api/v1/comparisons/ramp/catalog').json()
        assert not status['configured'] and not status['available']
        assert client.post('/api/v1/comparisons/ramp/prepare',json={'frame_id':'missing'}).status_code==404
