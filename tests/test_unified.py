import numpy as np
import pytest
from pydantic import ValidationError

from radar_platform.models import C0
from radar_platform.unified.schema import Simulation, Scene, Platform, Array
from radar_platform.unified.scene import compile_scene, terrain_mesh, object_mesh
from radar_platform.unified.geometry import sensor_basis, scan_directions, local_angles, build_layout
from radar_platform.unified.engine import simulate_frame, synthesize_adc
from radar_platform.unified.processing import process_adc, array_factor


def request(kind='pulsed_lfm', num_tx=2, num_rx=4, layout='ula', az=0., el=0., closing=0.):
    waveform = ({'kind':kind, 'carrier_hz':10e9, 'bandwidth_hz':5e6, 'sample_rate_hz':10e6,
                 'pulse_width_s':8e-6, 'slot_period_s':100e-6, 'adc_start_s':29e-6, 'samples':256, 'loops':32}
                if kind == 'pulsed_lfm' else
                {'kind':kind, 'start_hz':9.995e9, 'slope_hz_s':1e11, 'ramp_s':120e-6,
                 'slot_period_s':150e-6, 'adc_start_s':40e-6, 'samples':256, 'sample_rate_hz':4e6, 'loops':32})
    platform = Platform(beam_half_angle_deg=60.)
    unit = scan_directions([az], [el])[0, 0]@sensor_basis(platform).T
    point = np.array(platform.position_ned_m)+5000*unit
    return Simulation(waveform=waveform, platform=platform, array={'layout':layout,'num_tx':num_tx,'num_rx':num_rx},
        scene={'objects':[{'kind':'point','id':'p','position_ned_m':point.tolist(),'velocity_ned_mps':(-closing*unit).tolist()}]},
        processing={'range_limits_m':[4600,5400],'azimuth_deg':list(np.arange(-35.,36.)),
                    'elevation_deg':list(np.arange(-20.,21.)) if layout=='ura' else [0.]})


def peak(frame):
    p = frame.products
    r,d,e,a = np.unravel_index(np.argmax(abs(p.rad)),p.rad.shape)
    return p.range_m[r], p.velocity_mps[d], p.azimuth_deg[a], p.elevation_deg[e]


def test_sensor_coordinates_have_no_nadir_singularity():
    for forward in [(1,0,0),(0,0,1),(1,1,3),(0,0,-1)]:
        platform=Platform(boresight_ned=forward)
        basis=sensor_basis(platform)
        np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-14)
        assert np.linalg.det(basis)==pytest.approx(1)
        d=scan_directions([23.],[-14.])[0,0]@basis.T
        az,el=local_angles(d,basis)
        assert az==pytest.approx(23) and el==pytest.approx(-14)
    with pytest.raises(ValidationError):Platform(boresight_ned=(0,1,0))


def test_compiled_mixed_scene_is_fixed_and_seed_does_not_change_geometry():
    s=Scene(terrain={'kind':'hills','hills':[{}]},objects=[
        {'kind':'box','id':'b','position_ned_m':[100,0,-10]},
        {'kind':'plate','id':'plate','position_ned_m':[0,0,-10]},
        {'kind':'point','id':'moving','velocity_ned_mps':[3,4,0]}])
    a,b=compile_scene(s),compile_scene(s.model_copy(update={'seed':91}))
    np.testing.assert_array_equal(a.positions_ned_m,b.positions_ned_m)
    assert not np.array_equal(a.phases_rad,b.phases_rad)
    terrain=a.object_index==0
    np.testing.assert_array_equal(a.at(3)[terrain],a.positions_ned_m[terrain])
    assert np.all(a.normals_ned[terrain,2]<0)
    box=next(x for x in a.objects if x['id']=='b')
    assert box['surface_area_m2']==pytest.approx(2*(40*30+40*20+30*20))


def test_object_phase_is_stable_when_another_object_is_added():
    s=Scene(objects=[{'kind':'box','id':'b'}]);a=compile_scene(s)
    other=Scene(objects=[{'kind':'point','id':'a'},*s.objects]);b=compile_scene(other)
    np.testing.assert_array_equal(a.phases_rad,b.phases_rad[b.object_index==1])


def test_dem_input_is_geometry_not_just_visualization():
    flat=Scene(terrain={'kind':'dem','north_m':[-10,10],'east_m':[-10,10],'elevation_up_m':[[0,0],[0,0]]})
    raised=Scene(terrain={'kind':'dem','north_m':[-10,10],'east_m':[-10,10],'elevation_up_m':[[100,100],[100,100]]})
    a,b=compile_scene(flat),compile_scene(raised)
    np.testing.assert_allclose(b.positions_ned_m-a.positions_ned_m,[[0,0,-100]]*2)
    assert np.all(a.normals_ned[:,2]==-1)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
@pytest.mark.parametrize('az',[-25.,0.,25.])
@pytest.mark.parametrize('tx',[1,2])
def test_two_waveforms_share_distance_velocity_and_mimo_angle(kind,az,tx):
    r=request(kind,num_tx=tx,az=az,closing=6.)
    f=simulate_frame(r);distance,velocity,angle,_=peak(f)
    assert abs(distance-5000)<max(np.diff(f.products.range_m))
    assert abs(velocity-6.)<f.products.metadata['native_velocity_cell_mps']
    assert abs(angle-az)<=1
    assert f.adc.shape==(256,32,4,tx)
    assert f.products.rd_channels.shape[2:]==(4,tx)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_planar_mimo_observes_both_sensor_angles_at_nadir(kind):
    r=request(kind,num_tx=4,layout='ura',az=17,el=-11,closing=0.)
    f=simulate_frame(r)
    assert abs(peak(f)[2]-17)<=1
    assert abs(peak(f)[3]+11)<=1
    assert f.metadata['array']['two_dimensional_angle_geometry']


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_signal_linearity_uses_one_fixed_amplitude_reference(kind):
    r=request(kind,num_tx=1,num_rx=1)
    second=r.scene.objects[0].model_copy(update={'id':'q','position_ned_m':(200.,0.,-100.),'amplitude':.7,'phase_rad':.7})
    two=r.model_copy(deep=True);two.scene.objects.append(second)
    only=r.model_copy(deep=True);only.scene.objects=[second]
    np.testing.assert_allclose(simulate_frame(two).adc,simulate_frame(r).adc+simulate_frame(only).adc,rtol=3e-6,atol=2e-8)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_common_motion_zero_doppler_and_frame_phase_continuity(kind):
    r=request(kind)
    r.platform.velocity_ned_mps=(17,2,0)
    r.scene.objects[0].velocity_ned_mps=(17,2,0)
    first=simulate_frame(r)
    assert peak(first)[1]==0
    r=request(kind,closing=10.)
    whole=simulate_frame(r).adc
    short=r.model_copy(deep=True);short.waveform.loops=16
    layout=build_layout(short.array,short.waveform,short.platform)
    short.sequence.frame_count=2;short.sequence.frame_interval_s=16*layout.loop_period_s
    split=np.concatenate([simulate_frame(short,0).adc,simulate_frame(short,1).adc],axis=1)
    np.testing.assert_allclose(whole,split,rtol=1e-6,atol=1e-8)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_tdm_correction_uses_bin_not_truth(kind):
    r=request(kind,az=20)
    layout=build_layout(r.array,r.waveform,r.platform)
    speed=8*r.waveform.wavelength_m/(2*r.waveform.loops*layout.loop_period_s)
    direction=scan_directions([20.],[0.])[0,0]@layout.basis_ned.T
    r.scene.objects[0].velocity_ned_mps=tuple(-speed*direction)
    corrected=simulate_frame(r)
    r.processing.tdm_compensation=False
    uncorrected=process_adc(corrected.adc,r,layout)
    index=np.unravel_index(np.argmax(abs(uncorrected.rad)),uncorrected.rad.shape)
    wrong=uncorrected.azimuth_deg[index[-1]]
    assert abs(peak(corrected)[2]-20)<=1
    assert abs(wrong-20)>2
    assert not corrected.products.metadata['compensation_uses_truth']


def test_permuted_channel_storage_preserves_geometry_based_beamformer():
    r=request('fmcw',az=18)
    original=simulate_frame(r)
    old=build_layout(r.array,r.waveform,r.platform)
    # Physical positions AND time offsets move with the channel, not with list index.
    tx_order=[1,0];rx_order=[3,1,0,2]
    r.array=Array(layout='custom',num_tx=2,num_rx=4,tx_local_m=old.tx_local_m[tx_order].tolist(),
                  rx_local_m=old.rx_local_m[rx_order].tolist(),tx_offsets_s=old.tx_offsets_s[tx_order].tolist())
    r.processing.spatial_window='hamming'
    permuted=process_adc(original.adc[:,:,rx_order,:][:,:,:,tx_order],r,build_layout(r.array,r.waveform,r.platform))
    base=r.model_copy(deep=True);base.array=Array()
    reference=process_adc(original.adc,base,old)
    np.testing.assert_allclose(permuted.rad,reference.rad,rtol=1e-5,atol=1e-8)


def test_tdm_arbitrary_emission_order_has_explicit_offsets():
    r=request();r.array=Array(num_tx=3,num_rx=2,tx_order=[2,0,1])
    a=build_layout(r.array,r.waveform,r.platform)
    np.testing.assert_allclose(a.tx_offsets_s,[100e-6,200e-6,0])
    assert a.metadata()['chronological_tx_order']==[2,0,1]


def test_ula_virtual_array_matches_closed_form_factor_and_aperture_improves_resolution():
    r=request();a=build_layout(r.array,r.waveform,r.platform)
    scan=np.linspace(-50,50,2001)
    response=abs(array_factor(a,r.waveform.wavelength_m,scan,[0]))[0]
    u=np.pi*np.sin(np.deg2rad(scan));n=8
    expected=abs(np.sinc(n*u/(2*np.pi))/np.sinc(u/(2*np.pi)))
    np.testing.assert_allclose(response,expected,atol=2e-10)
    small=r.model_copy(deep=True);small.array.num_tx=1
    b=build_layout(small.array,small.waveform,small.platform)
    short=abs(array_factor(b,r.waveform.wavelength_m,scan,[0]))[0]
    width=lambda y:np.ptp(scan[y>=np.sqrt(.5)])
    assert width(response)<width(short)*.6


def test_siso_does_not_claim_an_observed_angle():
    r=request(num_tx=1,num_rx=1,layout='siso')
    f=simulate_frame(r)
    assert f.metadata['peak_candidates'][0]['azimuth_deg'] is None
    assert any('not observable' in x for x in f.metadata['warnings'])


def test_invalid_timing_geometry_and_memory_are_explicit():
    r=request().model_dump()
    r['array']['tx_offsets_s']=[0,1e-6]
    with pytest.raises(ValidationError,match='overlap'):Simulation.model_validate(r)
    r=request().model_dump();r['waveform']['adc_start_s']=.1
    with pytest.raises(ValidationError,match='gate'):Simulation.model_validate(r)
    r=request().model_dump();r['array']['tx_order']=[0,0]
    with pytest.raises(ValidationError,match='permutation'):Simulation.model_validate(r)
    r=request();r.processing.range_limits_m=(100,200)
    with pytest.raises(ValueError,match='range limits'):simulate_frame(r)
    with pytest.raises(ValidationError):Scene(objects=[{'kind':'point','id':'x'},{'kind':'point','id':'x'}])


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_mixed_terrain_building_and_moving_object_really_contribute_to_adc(kind):
    r=request(kind,num_tx=1,num_rx=1)
    r.scene.terrain=Scene(terrain={'north_bounds_m':[-30,30],'east_bounds_m':[-30,30],
        'spacing_m':60,'relative_reflectivity':.02}).terrain
    b=Scene(objects=[{'kind':'box','id':'building','position_ned_m':[100,0,-20],
                     'size_m':[30,30,40],'divisions':1}]).objects[0]
    r.scene.objects.append(b)
    combined=simulate_frame(r)
    sum_adc=np.zeros_like(combined.adc)
    for component in ['terrain','p','building']:
        one=r.model_copy(deep=True)
        one.scene.objects=[x for x in r.scene.objects if x.id==component]
        one.scene.terrain=r.scene.terrain if component=='terrain' else None
        frame=simulate_frame(one)
        assert np.max(abs(frame.adc))>0
        sum_adc+=frame.adc
    np.testing.assert_allclose(combined.adc,sum_adc,atol=1e-7,rtol=1e-5)
    assert {x['kind'] for x in combined.metadata['scene']['objects']}=={'flat','box','point'}


def test_real_adc_processing_contract_does_not_require_dummy_scene():
    from radar_platform.unified.schema import Acquisition
    r=request('fmcw')
    a=Acquisition.model_validate({key:r.model_dump()[key] for key in ['waveform','platform','array','processing']})
    assert not hasattr(a,'scene')
    actual=simulate_frame(r)
    independently=process_adc(actual.adc,a,build_layout(a.array,a.waveform,a.platform))
    np.testing.assert_array_equal(actual.products.rad,independently.rad)


def test_old_public_configs_have_explicit_input_bridge():
    from radar_platform.models import default_request
    from radar_platform.array_api import ArraySimulationRequest
    from radar_platform.unified.adapters import from_pulsed_request,from_array_request
    old=default_request()
    new=from_pulsed_request(old)
    assert len(new.scene.objects)==len(old.scene.targets)
    assert new.waveform.carrier_hz==old.radar.carrier_hz
    assert new.array.num_tx==1
    array=ArraySimulationRequest(targets=[{'id':'x','position_ned_m':[10,0,0]}])
    converted=from_array_request(array)
    assert converted.array.num_tx==2 and converted.scene.objects[0].id=='x'
    assert converted.waveform.slope_hz_s==pytest.approx(array.radar.config().slope_hz_s)


@pytest.mark.parametrize('kind,sign',[('pulsed_lfm',1),('fmcw',-1)])
def test_reflection_phase_is_conjugated_only_by_fmcw_dechirp(kind,sign):
    r=request(kind,num_tx=1,num_rx=1)
    a=simulate_frame(r).adc
    r.scene.objects[0].phase_rad=.7
    b=simulate_frame(r).adc
    np.testing.assert_allclose(b,a*np.exp(sign*.7j),rtol=3e-6,atol=1e-8)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_reversed_tdm_transmit_order_still_localizes_motion(kind):
    r=request(kind,az=23,closing=10)
    r.array.tx_order=[1,0]
    result=simulate_frame(r)
    assert abs(peak(result)[2]-23)<=1
    assert result.metadata['array']['chronological_tx_order']==[1,0]


def test_grating_lobes_and_duplicate_apertures_are_not_hidden():
    r=request('fmcw',az=20,closing=0)
    r.array.spacing_m=r.waveform.wavelength_m
    f=simulate_frame(r)
    assert any('grating lobes' in s for s in f.metadata['warnings'])
    a=build_layout(r.array,r.waveform,r.platform)
    aliases=np.array([20.,np.rad2deg(np.arcsin(np.sin(np.deg2rad(20))-1))])
    values=abs(array_factor(a,r.waveform.wavelength_m,sorted(aliases),[0],20))[0]
    np.testing.assert_allclose(values,1.,atol=1e-10)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_two_targets_at_same_range_velocity_are_resolved_by_larger_virtual_aperture(kind):
    from scipy.signal import find_peaks
    r=request(kind,az=-12)
    basis=sensor_basis(r.platform)
    second=np.array(r.platform.position_ned_m)+5000*(scan_directions([12.],[0.])[0,0]@basis.T)
    r.scene.objects.append(r.scene.objects[0].model_copy(update={'id':'q','position_ned_m':tuple(second)}))
    f=simulate_frame(r)
    i,j=np.unravel_index(np.argmax(f.products.rd_power),f.products.rd_power.shape)
    pattern=abs(f.products.rad[i,j,0])**2
    maxima,_=find_peaks(pattern,prominence=.1*pattern.max())
    angles=f.products.azimuth_deg[maxima]
    assert any(abs(angles+12)<4) and any(abs(angles-12)<4)
    left=pattern[(f.products.azimuth_deg<0)&(f.products.azimuth_deg>-20)].max()
    right=pattern[(f.products.azimuth_deg>0)&(f.products.azimuth_deg<20)].max()
    valley=pattern[np.argmin(abs(f.products.azimuth_deg))]
    assert 10*np.log10(min(left,right)/valley)>=3
    r.array.num_tx=1
    smaller=simulate_frame(r)
    i,j=np.unravel_index(np.argmax(smaller.products.rd_power),smaller.products.rd_power.shape)
    pattern=abs(smaller.products.rad[i,j,0])**2
    maxima,_=find_peaks(pattern,prominence=.1*pattern.max())
    assert len(maxima)==1


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_high_speed_alias_is_reported_not_claimed_as_unwrapped_velocity(kind):
    r=request(kind,closing=100.)
    f=simulate_frame(r);p=f.products
    i,j=np.unravel_index(np.argmax(p.rd_power),p.rd_power.shape)
    maximum=p.metadata['max_unambiguous_velocity_mps']
    folded=(100+maximum)%(2*maximum)-maximum
    assert abs(p.velocity_mps[j]-folded)<p.metadata['native_velocity_cell_mps']
    assert any('Doppler ambiguity' in s for s in f.metadata['warnings'])
    assert not p.metadata['tdm_ambiguity_resolved']


def test_outside_beam_remains_empty_without_fabricated_detections():
    r=request(az=25)
    r.platform.beam_half_angle_deg=5
    f=simulate_frame(r)
    assert not np.any(f.adc)
    assert f.metadata['peak_candidates']==[]
    assert f.metadata['visibility']['illuminated_scatterer_slots']==0
