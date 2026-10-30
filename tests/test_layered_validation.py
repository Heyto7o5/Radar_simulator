import numpy as np
import pytest
from fastapi.testclient import TestClient
from radar_platform.comparison import compare_power,average_rd_power,representative_indices
from radar_platform.kinematics import position_at,relative_geometry
from radar_platform.models import default_request,TargetConfig
from radar_platform.physics import platform_boresight_ned,request_warnings
from radar_platform.simulator import _clutter_scatterers,simulate_frame
from radar_platform.fmcw import FMCWConfig
from radar_platform.array_fmcw import TDMArray,simulate_motion_adc,process_array_adc
from radar_platform.detection import detect_rd_peaks


def config(chirps=64):return FMCWConfig(76.5e9,77.5e9,50e-6,60e-6,128,chirps,4e6,2e-6)


def target(r=10.,a=15.,v=1.):
    direction=np.array([np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a)),0])
    return TargetConfig(id='T',position_ned_m=tuple(r*direction),velocity_ned_mps=tuple(-v*direction))


def test_common_reference_does_not_hide_gain_error():
    m=np.ones((4,5));c=compare_power(m,4*m,reference_power=1.)
    assert c.metrics['mae_db']==pytest.approx(10*np.log10(4))
    assert np.all(c.difference_db>0)
    assert compare_power(m,m,reference_power=1.).metrics['mae_db']==0


def test_missing_predictions_kept_and_roi_does_not_hide_background():
    m=np.zeros((4,5));m[1,1]=1;mask=np.zeros(m.shape,dtype=bool);mask[1,1]=True
    c=compare_power(m,np.zeros_like(m),reference_power=1.,target_mask=mask)
    assert c.metrics['target_mae_db']==60
    assert c.metrics['mae_db']==3
    assert c.metrics['simulation_zero_power_fraction']==1


def test_power_mean_before_log_and_fixed_representatives():
    rd=np.array([[[1+0j]],[[3+0j]]]);assert average_rd_power(rd)[0,0]==5
    assert representative_indices([9.,2.,3.,1.,4.])==[1,2,3,0]
    with pytest.raises(ValueError):compare_power(np.ones(2),np.ones(3),reference_power=1)


@pytest.mark.parametrize('rv,tv,expected',[(3,0,3),(0,-3,3),(4,1,3),(3,3,0),(0,3,-3)])
def test_relative_closing_velocity(rv,tv,expected):
    r,u,v=relative_geometry([0,0,0],[rv,0,0],[10,0,0],[tv,0,0])
    assert v==expected
    assert np.dot([0,3,0],u)==0
    np.testing.assert_allclose(position_at([10,0,0],[tv,0,0],2),[10+2*tv,0,0])


def test_stationary_platform_and_world_fixed_clutter():
    request=default_request();request.platform.speed_mps=0;request.platform.boresight_ned=(0,0,1)
    request.scene.simplified_clutter_enabled=True
    a=_clutter_scatterers(request,np.array([0,0,-5000]))
    b=_clutter_scatterers(request,np.array([200,0,-4800]))
    np.testing.assert_array_equal([x.position_ned_m for x in a],[x.position_ned_m for x in b])
    assert [x.phase_rad for x in a]==[x.phase_rad for x in b]
    assert np.isfinite(platform_boresight_ned(request.platform)).all()
    assert isinstance(request_warnings(request),list)


def test_pulsed_moving_target_and_truth_free_detection():
    request=default_request();request.platform.speed_mps=0;request.platform.boresight_ned=(0,0,1)
    request.scene.targets=[TargetConfig(id='moving',position_ned_m=(0,0,0),velocity_ned_mps=(0,0,-30))]
    first=simulate_frame(request,0);second=simulate_frame(request,1)
    assert first.metadata.validations[0].true_radial_velocity_mps==30
    assert first.metadata.validations[0].validation_mode=='truth_assisted_analytic_check'
    assert second.metadata.target_positions_ned_m['moving'][2]<first.metadata.target_positions_ned_m['moving'][2]
    detected=first.metadata.blind_detections[0]
    assert abs(detected.apparent_velocity_mps-30)<4
    assert abs(detected.range_using_apparent_doppler_m-first.metadata.validations[0].true_range_m)<1.1


def test_detector_has_no_truth_inputs_and_wraps_doppler():
    p=np.zeros((8,9));p[0,4]=1;p[7,4]=.9;p[4,7]=.7
    d=detect_rd_peaks(p,np.arange(9),np.arange(-4,4),.03,1e12,doppler_guard=1,range_guard=1)
    assert len(d)==2
    assert detect_rd_peaks(p*0,np.arange(9),np.arange(-4,4),.03,1e12)==[]


@pytest.mark.parametrize('num_tx',[1,2])
@pytest.mark.parametrize('angle',[-30.,0.,25.])
def test_array_range_velocity_and_azimuth(num_tx,angle):
    c=config();a=TDMArray.ideal_ula(c,num_tx,4);t=target(a=angle)
    adc=simulate_motion_adc(c,[t],a);result=process_array_adc(adc,c,a)
    r,d,k=np.unravel_index(np.argmax(abs(result.rad)),result.rad.shape)
    assert abs(result.range_m[r]-10)<c.native_range_resolution_m
    assert abs(result.velocity_mps[d]-1)<c.wavelength_m/(2*c.chirps*a.loop_period_s)
    assert abs(result.azimuth_deg[k]-angle)<=1.1


def test_common_motion_is_zero_doppler_and_exchange_radars_targets():
    c=config();a=TDMArray.ideal_ula(c,1,1)
    base=TargetConfig(id='t',position_ned_m=(10,0,0))
    radar_moving=simulate_motion_adc(c,[base],a,platform_velocity_ned_mps=(1,0,0))
    target_moving=simulate_motion_adc(c,[base.model_copy(update={'velocity_ned_mps':(-1,0,0)})],a)
    np.testing.assert_allclose(radar_moving,target_moving,rtol=1e-6,atol=1e-6)
    same=simulate_motion_adc(c,[base.model_copy(update={'velocity_ned_mps':(1,0,0)})],a,platform_velocity_ned_mps=(1,0,0))
    result=process_array_adc(same,c,a)
    assert result.velocity_mps[np.unravel_index(np.argmax(abs(result.rd_channels)),result.rd_channels.shape)[1]]==0


def test_frame_phase_continuity():
    c=config();a=TDMArray.ideal_ula(c,2,4);t=target()
    full=simulate_motion_adc(c,[t],a)
    short=config(32)
    first=simulate_motion_adc(short,[t],a)
    second=simulate_motion_adc(short,[t],a,frame_start_s=32*a.loop_period_s)
    np.testing.assert_array_equal(full,np.concatenate([first,second],axis=1))


def test_tdm_compensation_uses_measured_bin_not_target_velocity():
    c=config(128);a=TDMArray.ideal_ula(c,2,4)
    # An exact slow FFT bin makes the TDM phase error test independent of interpolation.
    v=20*c.wavelength_m/(2*c.chirps*a.loop_period_s)
    adc=simulate_motion_adc(c,[target(a=15,v=v)],a)
    corrected=process_array_adc(adc,c,a)
    uncorrected=process_array_adc(adc,c,a,tdm_compensation=False)
    angle=lambda r:r.azimuth_deg[np.unravel_index(np.argmax(abs(r.rad)),r.rad.shape)[2]]
    assert abs(angle(corrected)-15)<=1
    assert abs(angle(uncorrected)-15)>abs(angle(corrected)-15)


def test_array_api_and_validation(monkeypatch,tmp_path):
    from radar_platform.api import app
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        response=client.post('/api/v1/fmcw-array-simulations',json={'targets':[target().model_dump()]})
        assert response.status_code==200
        meta=response.json();assert meta['is_real'] is False
        raw=client.get(meta['urls']['rad'])
        assert len(raw.content)==np.prod(meta['shapes']['rad'])*8
        assert client.get('/api/v1/fmcw-array-simulations/'+meta['job_id']).status_code==200
        assert client.post('/api/v1/fmcw-array-simulations',json={'targets':[target().model_dump()],'boresight_ned':[0,0,0]}).status_code==422


def test_partial_adc_window_and_geometry_failures():
    c=config();assert c.adc_duration_s<c.ramp_s<c.chirp_period_s
    assert c.native_range_resolution_m==pytest.approx(299792458/(2*c.slope_hz_s*c.adc_duration_s))
    with pytest.raises(ValueError):TDMArray([[0,0,0],[0,0,0]],[[0,0,0]],[0,0],1.)
    with pytest.raises(ValueError):FMCWConfig(77e9,78e9,30e-6,60e-6,128,64,4e6,2e-6)


def test_full_maroon_axis_order_and_restricted_reader(tmp_path):
    from radar_platform.maroon_full import load_full_frame
    data=np.zeros((96,96,128),dtype=np.complex64);data[2,3]=1+2j
    p=tmp_path/'frame.npy'
    np.save(p,{'xr':data,'fvec':np.linspace(72e9,82e9,128),'timestamp_us':np.uint64(123),'number_of_measurements':10})
    signal,f,meta=load_full_frame(p)
    np.testing.assert_array_equal(signal[2,1],np.full(128,1+2j))
    assert not np.any(signal[1,2])
    assert meta['source_axes']==['rx','tx','frequency']


def test_external_npy_cannot_execute_arbitrary_pickle_globals(tmp_path):
    from radar_platform.maroon_full import load_full_frame
    import os
    class Unsafe:
        def __reduce__(self):return os.system,('false',)
    p=tmp_path/'bad.npy';np.save(p,{'payload':Unsafe()})
    with pytest.raises(ValueError,match='forbidden pickle global'):load_full_frame(p)


def test_comparison_does_not_mutate_roi_mask():
    roi=np.ones((2,2),dtype=bool);valid=np.array([[True,False],[True,False]])
    compare_power(np.ones((2,2)),np.ones((2,2)),reference_power=1.,target_mask=roi,valid_mask=valid)
    assert roi.all()


def test_land_scattering_input_refuses_unsupported_near_nadir(tmp_path):
    from radar_platform.ground_scattering import load_maqu_sigma0
    path=tmp_path/'sigma0_vv_alpha0_azip00_9000-9875.dat'
    path.write_text('descriptor alpha0,+,- sigma0,+,-\n53 6 -6 -14.3 1.8 -3.1\n')
    obs=load_maqu_sigma0(path)[0]
    obs.require_coverage(9.5e9,53,'vv','alpine_meadow')
    assert obs.effective_patch_rcs_m2(100)==pytest.approx(100*10**(-1.43))
    with pytest.raises(ValueError,match='incidence'):obs.require_coverage(9.5e9,5,'vv','alpine_meadow')
    with pytest.raises(ValueError,match='frequency'):obs.require_coverage(77e9,53,'vv','alpine_meadow')
    with pytest.raises(ValueError,match='land-cover'):obs.require_coverage(9.5e9,53,'vv','sea')
