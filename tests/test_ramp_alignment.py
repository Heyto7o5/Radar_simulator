import numpy as np
import pytest
from radar_platform.ramp_comparison import SimulationTuning,label_scatterers,simulate_label_conditioned_adc,simulate_label_conditioned_adc_legacy


def labels():return [{'uid':'22','class_id':80,'x_m':1.,'y_m':10.,'width_m':1.7,'length_m':.6,'range_m':np.hypot(1,10),'radial_velocity_mps':.5}]


def test_single_point_is_exact_annotation_center_and_noise_seed_cannot_move_it():
    cfg=SimulationTuning('point',noise_std=0)
    point=label_scatterers(labels(),cfg)[0]
    assert point['x_m']==1 and point['y_m']==10
    np.testing.assert_array_equal(simulate_label_conditioned_adc(labels(),cfg,1),simulate_label_conditioned_adc(labels(),cfg,99))
    assert not np.array_equal(simulate_label_conditioned_adc_legacy(labels(),cfg,1),simulate_label_conditioned_adc_legacy(labels(),cfg,99))


def test_multiple_points_are_uid_stable_and_translation_equivariant():
    cfg=SimulationTuning('extended',scatterers_per_target=5,noise_std=0)
    first=label_scatterers(labels(),cfg)
    second=label_scatterers([dict(labels()[0],x_m=3.,y_m=11.)],cfg)
    for a,b in zip(first,second):
        assert a['normalized_offset']==b['normalized_offset']
        assert b['x_m']-a['x_m']==pytest.approx(2)
        assert b['y_m']-a['y_m']==pytest.approx(1)
        assert a['phase_rad']==b['phase_rad']
    with_other=label_scatterers([dict(labels()[0],uid='99'),*labels()],cfg)
    assert first==[x for x in with_other if x['uid']=='22']


def test_missing_speed_is_explicit_and_duplicate_ids_rejected():
    cfg=SimulationTuning('point')
    assert not label_scatterers([dict(labels()[0],radial_velocity_mps=None)],cfg)[0]['velocity_available']
    with pytest.raises(ValueError,match='duplicate UID'):label_scatterers(labels()*2,cfg)


def test_noise_remains_seed_controlled_without_changing_geometry():
    cfg=SimulationTuning('noise',noise_std=.1)
    assert not np.array_equal(simulate_label_conditioned_adc(labels(),cfg,1),simulate_label_conditioned_adc(labels(),cfg,2))


def test_target_gain_excludes_background_and_never_reads_validation():
    from radar_platform.alignment_metrics import fit_target_power
    s=np.zeros((6,8));s[2:4,3:5]=3
    roi=s>0;background=~roi
    a=fit_target_power([s*4+10],[s],[roi],[background])
    b=fit_target_power([s*4+1000],[s],[roi],[background])
    assert a.power_gain==pytest.approx(4)
    assert b.power_gain==pytest.approx(4)
    assert b.background_power==1000
    with pytest.raises(ValueError,match='disjoint'):fit_target_power([s],[s],[roi],[roi])
    # Symmetric fluctuations around an equal background have no target energy.
    m=np.full((6,8),10.);m[2:4,3:5]=[[8,12],[12,8]]
    with pytest.raises(ValueError,match='no identifiable'):fit_target_power([m],[s],[roi],[background])


def test_global_delay_fit_recovers_calibration_only_and_keeps_missing_channels():
    from radar_platform.alignment_metrics import fit_global_delay_gain,apply_delay_gain
    from radar_platform.models import C0
    f=np.linspace(72e9,82e9,128);rng=np.random.default_rng(1)
    s=rng.normal(size=(4,128))+1j*rng.normal(size=(4,128));s[3]=0
    real=2*np.exp(.3j)*s*np.exp(-4j*np.pi*f*.004/C0)
    mask=np.array([True,True,False,False])
    cal=fit_global_delay_gain(s,real,f,mask)
    assert cal['equivalent_range_bias_m']==pytest.approx(.004,abs=1e-7)
    corrupted=real.copy();corrupted[2:]=999
    assert fit_global_delay_gain(s,corrupted,f,mask)==cal
    np.testing.assert_allclose(apply_delay_gain(s,f,cal),real,atol=1e-5)
    assert not np.any(apply_delay_gain(s,f,cal)[3])


def test_annotation_velocity_smoothing_uses_only_annotation_files(tmp_path):
    from radar_platform.ramp_alignment import AnnotationTracks,PUBLIC_FRAME_PERIOD_S
    for i in range(21):
        t=i*PUBLIC_FRAME_PERIOD_S
        (tmp_path/f'{i:010d}.csv').write_text(f'1,2,{1+2*t},{10-t},2,4\n')
    raw=AnnotationTracks(tmp_path).labels(10,smooth=False)[0]
    smooth=AnnotationTracks(tmp_path).labels(10,smooth=True)[0]
    np.testing.assert_allclose(raw['velocity_ned_mps'],[-1,2,0],atol=1e-12)
    np.testing.assert_allclose(smooth['velocity_ned_mps'],[-1,2,0],atol=1e-12)
    assert len(smooth['trajectory_support_frames'])==11
    assert smooth['velocity_source']=='annotation_local_quadratic'


def test_shared_real_sim_rad_grid_and_full_doppler_roi():
    from radar_platform.ramp_alignment import aligned_config,aligned_array,process_aligned_adc,simulate_track_adc,annotation_masks
    source=dict(labels()[0],velocity_ned_mps=[-1,0,0])
    adc=simulate_track_adc([source])
    result=process_aligned_adc(adc)
    assert result.rad.shape==(128,255,81)
    assert result.metadata['windows']=={'range':'hamming','doppler':'hamming','spatial':'hamming'}
    t,b,valid=annotation_masks([source],result.range_m,result.velocity_mps,result.azimuth_deg)
    assert not np.any(t&b)
    np.testing.assert_array_equal(t[:,0],t[:,-1])
    assert aligned_array().loop_period_s==pytest.approx(120e-6)
    assert aligned_config().observed_bandwidth_hz==pytest.approx(672e6)


def test_duplicate_annotation_ids_are_not_merged_into_fake_trajectory(tmp_path):
    from radar_platform.ramp_alignment import AnnotationTracks
    (tmp_path/'0000000000.csv').write_text('1.00,2,0,10,2,4\n')
    (tmp_path/'0000000001.csv').write_text('1,2,0,10,2,4\n1.00,2,8,18,2,4\n')
    (tmp_path/'0000000002.csv').write_text('1,2,9,18,2,4\n')
    tracks=AnnotationTracks(tmp_path)
    ambiguous=tracks.labels(1,smooth=True)
    assert len({x['uid'] for x in ambiguous})==2
    assert all(x['tracking_ambiguous'] and x['radial_velocity_mps'] is None for x in ambiguous)
    assert tracks.labels(2,smooth=True)[0]['trajectory_support_frames']==[2]


def test_signal_support_metric_retains_missed_echoes_and_exposes_empty_background():
    from radar_platform.comparison import compare_power
    m=np.zeros((10,10));m[4,4]=1
    result=compare_power(m,np.zeros_like(m),reference_power=1)
    assert result.metrics['mae_db']==pytest.approx(.6)
    assert result.metrics['signal_support_mae_db']==60
    assert result.metrics['joint_floor_fraction']==pytest.approx(.99)
    assert result.metrics['signal_support_cells']==1


def test_rigid_corner_uses_geometry_only_and_preserves_published_vertex():
    from radar_platform.surface_scene import TriangleMesh
    from radar_platform.geometry_alignment import rigid_corner_from_template,corner_descriptor
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    faces=np.array([[0,1,2],[0,1,3],[0,2,3]])
    template=TriangleMesh(vertices,faces)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
    observed=TriangleMesh((vertices*np.array([1.02,.99,.98]))@rotation+[3,4,5],faces)
    corrected,info=rigid_corner_from_template(template,observed)
    descriptor=corner_descriptor(corrected)[2]
    np.testing.assert_allclose(descriptor['edge_lengths_m'],[1,1,1])
    np.testing.assert_allclose(descriptor['edge_angles_deg'],[90,90,90])
    np.testing.assert_allclose(descriptor['corner_ned_m'],[3,4,5])
    assert info['edge_fit_rms_m']>0


def test_relative_channel_calibration_recovers_known_response_and_flags_inconsistent_data():
    from radar_platform.alignment_metrics import fit_relative_channel_response
    rng=np.random.default_rng(11);theta=np.linspace(-.6,.6,100)
    steering=np.exp(1j*np.pi*np.sin(theta[:,None])*np.arange(8)[None,:])
    truth=np.linspace(.9,1.2,8)*np.exp(1j*np.linspace(-.2,.4,8))
    common=np.exp(1j*rng.uniform(-np.pi,np.pi,100))
    fitted=fit_relative_channel_response(steering*truth*common[:,None],steering)
    response=np.array(fitted['relative_response_real'])+1j*np.array(fitted['relative_response_imag'])
    np.testing.assert_allclose(response,truth/truth[0],rtol=1e-10)
    assert fitted['stable_enough_for_empirical_trial']
    bad=fit_relative_channel_response(steering*np.exp(1j*rng.uniform(-np.pi,np.pi,(100,8))),steering)
    assert not bad['stable_enough_for_empirical_trial']


def test_phase_curvature_is_invariant_to_unknown_target_direction_and_common_phase():
    from radar_platform.alignment_metrics import fit_phase_curvature
    rng=np.random.default_rng(9);index=np.arange(8)-3.5
    phase=.05*(index**2-np.mean(index**2))
    source=rng.uniform(-2,2,100)[:,None]*index[None,:]+rng.uniform(-np.pi,np.pi,100)[:,None]
    result=fit_phase_curvature(np.exp(1j*(source+phase)))
    np.testing.assert_allclose(result['phase_rad'],phase,atol=1e-12)
    assert result['stable_enough_for_trial']
    assert result['unidentified_modes']==['common phase','linear phase / absolute azimuth zero']


def test_phase_response_preserves_power_and_correct_virtual_channel_order():
    from radar_platform.alignment_metrics import apply_channel_phase_response
    adc=np.ones((4,5,4,2),dtype=np.complex64);phase=np.arange(8)*.1
    out=apply_channel_phase_response(adc,phase)
    np.testing.assert_allclose(abs(out)**2,abs(adc)**2,atol=2e-7)
    np.testing.assert_allclose(out[0,0,2,1],np.exp(.6j),atol=1e-7)
    np.testing.assert_allclose(apply_channel_phase_response(out,-phase),adc,atol=1e-7)


def test_noise_variance_estimator_accounts_for_fft_windows_and_channel_order():
    from radar_platform.alignment_metrics import add_calibrated_noise,estimate_adc_noise_variance
    variance=np.arange(1,9,dtype=float)
    x=add_calibrated_noise(np.zeros((128,255,4,2),dtype=np.complex64),variance,44)
    rw,dw=np.hamming(128),np.hamming(255)
    rd=np.fft.fft(np.fft.fft(x*rw[:,None,None,None],axis=0)*dw[None,:,None,None],axis=1)/(rw.sum()*dw.sum())
    fitted=estimate_adc_noise_variance(rd,np.ones((128,255),dtype=bool),rw,dw)
    np.testing.assert_allclose(fitted,variance,rtol=.05)
    np.testing.assert_array_equal(x,add_calibrated_noise(np.zeros_like(x),variance,44))


def test_public_ramp_display_has_physical_closing_sign_and_no_phantom_empty_targets():
    from radar_platform.ramp_dataset import synthesize_adc,process_adc,normalize_adc_axes
    source=[dict(labels()[0],x_m=0.,y_m=10.,range_m=10.,radial_velocity_mps=2.)]
    adc,_=synthesize_adc(source);rd,_,r,v=process_adc(adc)
    d,k=np.unravel_index(np.argmax(rd),rd.shape)
    assert abs(v[d]-2)<abs(v[1]-v[0])
    assert abs(r[k]-10)<r[1]-r[0]
    empty,returned=synthesize_adc([])
    assert returned==[] and not np.any(empty)
    with pytest.raises(ValueError,match='complex'):normalize_adc_axes(np.zeros((128,255,4,2)))
    broken=adc.copy();broken[0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='non-finite'):normalize_adc_axes(broken)


def test_public_label_adapter_does_not_join_ambiguous_ids(tmp_path):
    from radar_platform.ramp_dataset import load_labels
    radar=tmp_path/'radar_raw_frame';annotations=tmp_path/'text_labels'
    radar.mkdir();annotations.mkdir()
    for frame,text in enumerate(['1,2,0,10,2,4\n','1.00,2,0,9,2,4\n1,2,8,18,2,4\n','1,2,0,8,2,4\n']):
        (radar/f'{frame:010d}.mat').touch()
        (annotations/f'{frame:010d}.csv').write_text(text)
    middle=load_labels(radar/'0000000001.mat')
    assert len(middle)==2 and all(x['tracking_ambiguous'] for x in middle)
    assert all(x['radial_velocity_mps'] is None for x in middle)
    assert load_labels(radar/'0000000002.mat')[0]['radial_velocity_mps'] is None
    (annotations/'0000000001.csv').write_text('1.00,2,0,9,2,4\n')
    assert load_labels(radar/'0000000001.mat')[0]['radial_velocity_mps']>0
    assert not load_labels(radar/'0000000001.mat')[0]['velocity_is_surveyed_truth']
