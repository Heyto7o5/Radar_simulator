"""Finite-area dynamic surfaces: independent static/analytic checks, no real-data fitting."""
import numpy as np
import pytest
from pydantic import ValidationError
from radar_platform.models import C0
from radar_platform.physical_optics import simulate_po_surface
from radar_platform.unified.schema import Simulation, Scene
from radar_platform.unified.scene import compile_scene, object_mesh
from radar_platform.unified.engine import simulate_frame, synthesize_adc
from radar_platform.unified.geometry import build_layout


def plate_request(kind='pulsed_lfm', closing=0., order=12):
    waveform = ({'kind':kind,'carrier_hz':9e9,'bandwidth_hz':20e6,'sample_rate_hz':40e6,
                 'pulse_width_s':2e-6,'slot_period_s':200e-6,'samples':128,'loops':64,'adc_start_s':0.}
                if kind=='pulsed_lfm' else
                {'kind':kind,'start_hz':9e9,'slope_hz_s':1e12,'ramp_s':60e-6,'slot_period_s':200e-6,
                 'adc_start_s':2e-6,'samples':128,'sample_rate_hz':4e6,'loops':64})
    return Simulation.model_validate({'waveform':waveform,
        'platform':{'position_ned_m':[0,0,0],'boresight_ned':[1,0,0],'right_hint_ned':[0,1,0],'beam_half_angle_deg':30.},
        'array':{'layout':'siso','num_tx':1,'num_rx':1},
        'scene':{'objects':[{'kind':'plate','id':'plate','position_ned_m':[30,0,0],
                    'velocity_ned_mps':[-closing,0,0],'size_m':[.85,.65],'normal_ned':[-1,0,0],
                    'divisions':1,'scattering':'coherent','scattering_model':'scalar_po',
                    'quadrature_order':order,'relative_reflectivity':1.}]},
        'processing':{'range_limits_m':[5,70],'azimuth_deg':[0.],'elevation_deg':[0.],
                      'range_window':'hann','doppler_window':'hann'},
        'sequence':{'frame_count':2,'frame_interval_s':.02}})


def test_surface_quadrature_area_and_model_are_explicit():
    r=plate_request();s=compile_scene(r.scene)
    assert len(s.amplitudes)==2*12**2 and s.scalar_po.all()
    assert s.amplitudes.sum()==pytest.approx(.85*.65)
    np.testing.assert_allclose(s.phases_rad,np.pi/2)
    assert s.metadata()['scalar_po_nodes']==288
    assert s.objects[0]['scattering_model']=='scalar_po'


def test_po_rejects_random_diffuse_and_excessive_budget_and_broadband():
    d=plate_request().model_dump();d['scene']['objects'][0]['scattering']='diffuse'
    with pytest.raises(ValidationError,match='coherent'):Simulation.model_validate(d)
    d=plate_request().model_dump();d['waveform']['bandwidth_hz']=600e6;d['waveform']['sample_rate_hz']=600e6
    with pytest.raises(ValidationError,match='5%'):Simulation.model_validate(d)
    r=plate_request();r.scene.max_scatterers=100
    with pytest.raises(ValueError,match='quadrature exceeds'):compile_scene(r.scene)


def test_dynamic_po_matches_independent_static_frequency_response():
    r=plate_request();r.sequence.frame_count=1
    # A virtually unmodulated pulse isolates the surface field at fc. This
    # compares the ADC synthesis loop with the separate SFCW integral, not RD peaks.
    d=r.model_dump();d['waveform'].update(bandwidth_hz=1.,sample_rate_hz=4e6,samples=8,adc_start_s=.3e-6,loops=4)
    d['array']={'layout':'custom','num_tx':1,'num_rx':1,'tx_local_m':[[0,-.2,0]],'rx_local_m':[[0,.2,0]]}
    r=Simulation.model_validate(d);mesh=object_mesh(r.scene.objects[0])
    expected=simulate_po_surface(mesh,np.array([r.waveform.carrier_hz]),order=24)[0]
    s=compile_scene(r.scene);layout=build_layout(r.array,r.waveform,r.platform)
    adc,_=synthesize_adc(r,s,layout,0.)
    np.testing.assert_allclose(adc[0,0,0,0],expected,rtol=2e-6,atol=1e-9)
    # Scalar field units must not change when point-target reference range changes.
    r.scene.amplitude_reference_range_m=37.
    other,_=synthesize_adc(r,compile_scene(r.scene),layout,0.)
    np.testing.assert_allclose(adc,other,rtol=1e-6,atol=1e-10)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
@pytest.mark.parametrize('closing',[-3.,0.,3.])
def test_moving_plate_rd_global_peak_and_quadrature_convergence(kind,closing):
    r=plate_request(kind,closing,8);low=simulate_frame(r)
    r.scene.objects[0].quadrature_order=16;high=simulate_frame(r)
    # Stronger than normalized image similarity: compare complex ADC directly.
    relative=np.linalg.norm(low.adc-high.adc)/np.linalg.norm(high.adc)
    assert relative<.002
    p=high.products;i,j=np.unravel_index(np.argmax(p.rd_power),p.rd_power.shape)
    epoch=high.metadata['truth_epoch_s'];true_range=30-closing*epoch
    assert abs(p.range_m[i]-true_range)<np.diff(p.range_m)[0]
    assert abs(p.velocity_mps[j]-closing)<p.metadata['native_velocity_cell_mps']


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_plate_frame_phase_continuity_common_motion_and_reciprocity(kind):
    r=plate_request(kind,3,8);reference=simulate_frame(r).adc
    # Split the same continuous acquisition, not independently phase-reset frames.
    short=r.model_copy(deep=True);short.waveform.loops=32;short.sequence.frame_interval_s=32*r.waveform.slot_period_s
    combined=np.concatenate([simulate_frame(short,0).adc,simulate_frame(short,1).adc],axis=1)
    np.testing.assert_allclose(reference,combined,rtol=3e-6,atol=1e-9)
    r.platform.velocity_ned_mps=(1,2,0);r.scene.objects[0].velocity_ned_mps=(-2,2,0)
    np.testing.assert_allclose(reference,simulate_frame(r).adc,rtol=3e-6,atol=1e-9)


def test_front_back_and_default_centroid_compatibility():
    r=plate_request(order=8);front=simulate_frame(r)
    r.scene.objects[0].normal_ned=(1,0,0);back=simulate_frame(r)
    assert np.linalg.norm(front.adc)>0 and not np.any(back.adc)
    r.scene.objects[0].two_sided=True;two=simulate_frame(r)
    np.testing.assert_allclose(two.adc,front.adc,rtol=2e-5,atol=1e-8)
    d=plate_request().model_dump();d['scene']['objects'][0].pop('scattering_model')
    default=Simulation.model_validate(d);assert default.scene.objects[0].scattering_model=='centroid'
    assert not compile_scene(default.scene).scalar_po.any()


def test_plate_far_field_angular_envelope_matches_rectangular_aperture_sinc():
    values=[];angles=np.array([0.,.25,.5,1.,1.5])
    for angle in angles:
        d=plate_request(order=16).model_dump()
        d['waveform'].update(bandwidth_hz=1.,sample_rate_hz=4e6,samples=8,adc_start_s=6.8e-6,loops=4)
        d['scene']['objects'][0].update(position_ned_m=[1000,0,0],yaw_deg=float(angle))
        r=Simulation.model_validate(d)
        adc,_=synthesize_adc(r,compile_scene(r.scene),build_layout(r.array,r.waveform,r.platform),0.)
        values.append(abs(adc[0,0,0,0]))
    actual=np.array(values)/values[0]
    lam=C0/9e9
    # Independent far-field rectangular-area Fourier transform. Not the mesh solver.
    expected=abs(np.cos(np.radians(angles))*np.sinc(2*.85*np.sin(np.radians(angles))/lam))
    np.testing.assert_allclose(actual,expected,atol=.003,rtol=.015)


def test_tilted_translating_pulse_edge_refines_without_relaxing_threshold():
    r=plate_request(closing=3.,order=32);r.scene.objects[0].yaw_deg=1.
    r.scene.max_scatterers=8192
    a=simulate_frame(r)
    r.scene.objects[0].quadrature_order=48;b=simulate_frame(r)
    assert np.linalg.norm(a.adc-b.adc)/np.linalg.norm(b.adc)<.002
    assert not b.metadata['scene']['scalar_po_quadrature_convergence_verified']
    assert any('pulse/gate edges' in s for s in b.metadata['warnings'])


def test_surface_po_runs_through_existing_job_api_and_saves_complex_channels(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from radar_platform.api import app
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    r=plate_request('fmcw',closing=3.,order=4)
    r.sequence.frame_count=1;r.array.num_tx=2;r.array.num_rx=4;r.array.layout='ula'
    with TestClient(app) as client:
        response=client.post('/api/v1/scene-simulations',json=r.model_dump(mode='json'))
        assert response.status_code==202
        job=response.json()['job_id']
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                event=ws.receive_json()
                if event['type'] in ['job_completed','job_failed']:
                    assert event['type']=='job_completed',event
                    break
        state=client.get(f'/api/v1/scene-simulations/{job}').json()
        frame=state['frames'][0]
        assert frame['scene']['scalar_po_nodes']==32
        assert frame['processing']['rd_channels_axes']==['range','closing_doppler','rx','tx']
        data=client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/rd_channels')
        assert data.status_code==200 and len(data.content)>0
