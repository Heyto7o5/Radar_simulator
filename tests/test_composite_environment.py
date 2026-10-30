import json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from radar_platform.statistical_background import BackgroundRecipe,SpectrumComponent
from radar_platform.unified.schema import Simulation
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.processing import range_axis,selected_range_indices,range_compress,process_adc,process_range_iq,window
from radar_platform.unified.environment import prepare_environment,receiver_noise
from radar_platform.unified.engine import simulate_frame


def request(kind='pulsed_lfm',tx=1,rx=1):
    d=json.loads((Path(__file__).resolve().parents[1]/'configs/lab_moving_plate_po.json').read_text())
    d['scene']['objects'][0]['quadrature_order']=4
    d['array']={'layout':'siso' if tx*rx==1 else 'ula','num_tx':tx,'num_rx':rx}
    if kind=='fmcw':
        d['waveform']={'kind':kind,'start_hz':9e9,'slope_hz_s':1e12,'ramp_s':60e-6,'slot_period_s':200e-6,
                       'adc_start_s':2e-6,'samples':128,'sample_rate_hz':4e6,'loops':64}
    d['sequence']={'frame_count':2,'frame_interval_s':.04}
    d['processing'].update(azimuth_deg=list(np.arange(-40,41)),elevation_deg=[0.])
    return Simulation.model_validate(d)


def with_environment(r,mode='steered'):
    d=r.model_dump();layout=build_layout(r.array,r.waveform,r.platform)
    ranges=range_axis(r)[selected_range_indices(r)];nr=len(ranges);prf=1/layout.loop_period_s
    sign=1 if r.waveform.kind=='pulsed_lfm' else -1
    recipe=BackgroundRecipe(prf,tuple(ranges),tuple(np.full(nr,1e-5)),
        (SpectrumComponent(1.,sign*prf/8,prf/512),),.02,3.,.05,
        tuple(map(tuple,np.eye(nr))),tuple(map(tuple,np.zeros((nr,nr)))),'correlated_gaussian')
    d['receiver_noise']={'variance':1e-3,'seed':77}
    d['range_background']={'model':asdict(recipe),'power_scale':1.,'provenance':'synthetic test; no measured calibration',
        'channel_mode':mode,'azimuth_deg':20. if mode=='steered' else 0.,'seed':42}
    return Simulation.model_validate(d)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_split_processing_is_exactly_compatible(kind):
    r=request(kind);f=simulate_frame(r);layout=build_layout(r.array,r.waveform,r.platform)
    direct=process_adc(f.adc,r,layout);split=process_range_iq(range_compress(f.adc,r),r,layout)
    np.testing.assert_array_equal(direct.rd_channels,split.rd_channels)
    np.testing.assert_array_equal(direct.rad,split.rad)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
@pytest.mark.parametrize('tx,rx',[(1,1),(1,4),(2,4)])
def test_components_sum_complex_fields_and_raw_adc_does_not_include_background(kind,tx,rx):
    r=with_environment(request(kind,tx,rx));env=prepare_environment(r);f=simulate_frame(r,environment=env)
    c=f.components
    total=c['clean_rd_channels']+c['receiver_noise_rd_channels']+c['background_rd_channels']
    np.testing.assert_allclose(total,f.products.rd_channels,atol=3e-9,rtol=2e-5)
    layout=build_layout(r.array,r.waveform,r.platform)
    replay=process_range_iq(range_compress(f.adc,r)+c['background_range_iq'],r,layout)
    np.testing.assert_array_equal(replay.rd_channels,f.products.rd_channels)
    without=process_adc(f.adc,r,layout)
    assert not np.allclose(without.rd_channels,f.products.rd_channels,atol=1e-8)
    assert f.metadata['data_origin']=='synthetic_composite_simulation' and not f.metadata['is_real']
    assert f.metadata['range_background']['no_invented_raw_ADC_for_range_background']


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_known_direction_steered_background_has_correct_sign_and_TDM_angle(kind):
    r=with_environment(request(kind,2,4));env=prepare_environment(r)
    layout=build_layout(r.array,r.waveform,r.platform)
    p=process_range_iq(env.frame(r,0),r,layout)
    i,j,e,a=np.unravel_index(np.argmax(abs(p.rad)),p.rad.shape)
    assert abs(p.azimuth_deg[a]-20)<=1
    assert p.velocity_mps[j]>0 # recipe flips raw slow-frequency convention for FMCW.


def test_same_sequence_reuses_background_timeline_and_noise_is_deterministic():
    r=with_environment(request());env=prepare_environment(r)
    f0=env.frame(r,0);f1=env.frame(r,1)
    assert not np.array_equal(f0,f1)
    np.testing.assert_array_equal(f1,prepare_environment(r).frame(r,1))
    start=env.samples_per_frame_interval
    np.testing.assert_array_equal(f1[env.range_indices],env.background[start:start+r.waveform.loops].transpose(1,0,2,3))
    np.testing.assert_array_equal(receiver_noise(r,0),receiver_noise(r,0))
    assert not np.array_equal(receiver_noise(r,0),receiver_noise(r,1))
    changed=r.model_copy(deep=True);changed.range_background.seed+=1
    with pytest.raises(ValueError,match='different acquisition'):env.frame(changed,0)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_receiver_variance_follows_range_and_Doppler_window_gains(kind):
    r=request(kind);d=r.model_dump();d['receiver_noise']={'variance':2.,'seed':42};r=Simulation.model_validate(d)
    w=r.waveform;layout=build_layout(r.array,w,r.platform)
    rw=window(r.processing.range_window,w.samples if kind=='fmcw' else int(np.ceil(w.pulse_width_s*w.sample_rate_hz)))
    dw=window(r.processing.doppler_window,w.loops)
    expected=2*np.sum(rw**2)/np.sum(rw)**2*np.sum(dw**2)/np.sum(dw)**2
    actual=np.mean([np.mean(abs(process_adc(receiver_noise(r,i),r,layout).rd_channels)**2) for i in range(20)])
    assert actual==pytest.approx(expected,rel=.08)


@pytest.mark.parametrize('fault',['range','PRF','interval','angle','variance','domain','memory'])
def test_rejects_silent_grid_time_or_unit_mismatches(fault):
    d=with_environment(request()).model_dump(mode='json')
    if fault=='range':d['range_background']['model']['range_m'][0]+=.1
    elif fault=='PRF':d['range_background']['model']['prf_hz']*=2
    elif fault=='interval':d['sequence']['frame_interval_s']+=1e-5
    elif fault=='angle':d['range_background']['channel_mode']='independent'
    elif fault=='variance':d['receiver_noise']['variance_per_rx_tx']=[[1.,2.]]
    elif fault=='domain':d['range_background']['model']['domain']='raw_ADC'
    else:d['sequence']['frame_interval_s']=1000.
    with pytest.raises((ValidationError,ValueError)):Simulation.model_validate(d)


def test_composite_api_stores_replay_components(monkeypatch,tmp_path):
    from radar_platform.api import app
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    r=with_environment(request('fmcw'));r.sequence.frame_count=1
    with TestClient(app) as c:
        result=c.post('/api/v1/scene-simulations',json=r.model_dump(mode='json'))
        assert result.status_code==202,result.text
        job=result.json()['job_id']
        with c.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                event=ws.receive_json()
                if event['type'] in ['job_completed','job_failed']:
                    assert event['type']=='job_completed',event;break
        state=c.get(f'/api/v1/scene-simulations/{job}').json()
        assert state['data_origin']=='synthetic_composite_simulation'
        products=state['frames'][0]['products']
        for name in ['adc','background_range_iq','clean_rd_channels','receiver_noise_rd_channels','background_rd_channels']:
            response=c.get(f'/api/v1/scene-simulations/{job}/frames/0/data/{name}')
            assert response.status_code==200 and len(response.content)==products[name]['bytes']


def test_receiver_floor_is_not_counted_twice_and_clutter_power_is_preserved():
    r=with_environment(request());d=r.model_dump(mode='json')
    d['range_background']['floor_policy']='retain_model_floor'
    with pytest.raises(ValueError,match='counted twice'):Simulation.model_validate(d)
    r.range_background.model=replace(r.range_background.model,white_fraction=1.)
    e=prepare_environment(r)
    assert not np.any(e.background) and e.metadata['source_model_floor_fraction_removed']==1


def test_removed_floor_does_not_get_redistributed_into_clutter():
    from dataclasses import replace
    r=with_environment(request());r.receiver_noise=None
    r.range_background.model=replace(r.range_background.model,white_fraction=.5)
    removed=prepare_environment(r).background
    r.range_background.model=replace(r.range_background.model,white_fraction=0.,mean_power=tuple(.5*np.array(r.range_background.model.mean_power)))
    explicitly_scaled=prepare_environment(r).background
    np.testing.assert_array_equal(removed,explicitly_scaled)


def test_per_channel_adc_variance_and_signal_independent_noise():
    r=request('fmcw',2,4);d=r.model_dump(mode='json')
    variance=np.arange(1.,9.).reshape(4,2)
    d['receiver_noise']={'variance_per_rx_tx':variance.tolist(),'seed':9}
    r=Simulation.model_validate(d);noise=receiver_noise(r,0)
    np.testing.assert_allclose(np.mean(abs(noise)**2,axis=(0,1)),variance,rtol=.06)
    r.scene.objects[0].relative_reflectivity*=1000
    np.testing.assert_array_equal(noise,receiver_noise(r,0))


def test_white_background_baseline_requires_explicit_floor_semantics():
    d=with_environment(request()).model_dump(mode='json');d['receiver_noise']=None
    d['range_background']['model']['family']='white_gaussian'
    with pytest.raises(ValueError,match='noise-only baseline'):Simulation.model_validate(d)
    d['range_background']['floor_policy']='retain_model_floor'
    r=Simulation.model_validate(d);assert np.any(prepare_environment(r).background)


def test_disabling_ADC_persistence_does_not_leak_raw_noise_component(tmp_path):
    from radar_platform.unified.storage import save_frame
    r=with_environment(request());frame=simulate_frame(r)
    meta=save_frame(tmp_path,frame,False)
    assert 'adc' not in meta['products'] and 'receiver_noise_adc' not in meta['products']
    assert 'background_range_iq' in meta['products']
