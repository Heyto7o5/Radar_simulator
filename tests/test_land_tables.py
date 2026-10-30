import copy
import numpy as np
import pytest
from radar_platform.land_tables import BackscatterTable,TabulatedBackscatterSet
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.land_response import land_amplitude_factor


def table(label='A',db=-20,pol='vv'):
    return dict(land_cover=label,polarization=pol,frequencies_hz=[9e9,11e9],incidence_deg=[0,80],
                sigma0_db=[[db,db],[db,db]],evidence='assumed',source='Explicit analytic test constants, not real terrain measurements')


def request():
    return Simulation.model_validate(dict(waveform=dict(kind='pulsed_lfm',carrier_hz=10e9,bandwidth_hz=2e6,
        sample_rate_hz=4e6,pulse_width_s=2e-6,slot_period_s=100e-6,adc_start_s=0,samples=64,loops=8),
        platform=dict(position_ned_m=[0,0,-10],boresight_ned=[0,0,1],beam_half_angle_deg=40),
        array=dict(layout='siso',num_tx=1,num_rx=1),
        scene=dict(seed=42,amplitude_reference_range_m=10,terrain=dict(kind='flat',
            north_bounds_m=[0,2],east_bounds_m=[0,2],spacing_m=1,relative_reflectivity=1,
            land_scattering=dict(land_cover='A',polarization='vv',tables=[table(),table('B',-10)]),
            land_cover_map=[['A','B'],['B','A']])),
        processing=dict(range_limits_m=[0,100],azimuth_deg=[0.],elevation_deg=[0.])))


def test_frequency_incidence_interpolation_and_polarization_selection():
    raw=table();raw['sigma0_db']=[[-30,-10],[-10,10]]
    model=TabulatedBackscatterSet([BackscatterTable(**raw),BackscatterTable(**table(pol='hh',db=-40))])
    assert model.predict_db(10e9,np.array([40]),'vv','A')[0]==pytest.approx(-10)
    assert model.predict_db(10e9,np.array([40]),'hh','A')[0]==pytest.approx(-40)
    assert model.metadata()['contains_assumed_parameters']
    for frequency,angle,pol,land in [(8e9,40,'vv','A'),(10e9,81,'vv','A'),(10e9,40,'hv','A'),(10e9,40,'vv','forest')]:
        with pytest.raises(ValueError):model.predict_db(frequency,np.array([angle]),pol,land)


def test_cell_classes_map_to_two_triangles_and_strength_not_double_cosine():
    r=request();scene=compile_scene(r.scene)
    assert scene.land_cover_labels.tolist()==['A','A','B','B','B','B','A','A']
    delta=scene.positions_ned_m-np.array(r.platform.position_ned_m)
    distance=np.linalg.norm(delta,axis=1);cos=-np.sum(delta*scene.normals_ned,axis=1)/distance
    f=land_amplitude_factor(scene,slice(None),cos[:,None],cos[:,None],delta[:,None,:],delta[:,None,:],
        distance[:,None],distance[:,None],np.ones((8,1),bool),10e9,cos[:,None])[:,0]
    np.testing.assert_allclose(f,np.where(scene.land_cover_labels=='A',.1,np.sqrt(.1)))
    assert np.sum((scene.amplitudes*f)**2)==pytest.approx(2*.01+2*.1)


def test_class_change_scales_echo_and_does_not_change_geometry_or_seed_phase():
    raw=request().model_dump();raw['scene']['terrain']['land_cover_map']=[['A','A'],['A','A']]
    a=simulate_frame(Simulation.model_validate(raw))
    raw['scene']['terrain']['land_cover_map']=[['B','B'],['B','B']]
    b=simulate_frame(Simulation.model_validate(raw))
    np.testing.assert_allclose(b.adc,a.adc*np.sqrt(10),rtol=1e-6,atol=1e-7)
    assert any('not measured' in x for x in b.metadata['warnings'])
    assert b.metadata['scene']['objects'][0]['land_cover_counts']=={'B':8}


@pytest.mark.parametrize('fault',['shape','unknown','sweep','evidence','both','cosine'])
def test_invalid_or_uncovered_tables_are_rejected(fault):
    raw=request().model_dump();ground=raw['scene']['terrain'];land=ground['land_scattering']
    if fault=='shape':ground['land_cover_map']=[['A']]
    elif fault=='unknown':ground['land_cover_map'][0][0]='unknown'
    elif fault=='sweep':raw['waveform']['carrier_hz']=11e9
    elif fault=='evidence':del land['tables'][0]['evidence']
    elif fault=='cosine':land['mode']='cosine_power'
    else:
        land['observations']=[dict(source='test',frequency_min_hz=9e9,frequency_max_hz=11e9,polarization='vv',
            incidence_center_deg=40,incidence_min_deg=39,incidence_max_deg=41,sigma0_db=-10,
            upper_uncertainty_db=1,lower_uncertainty_db=-1)]
    with pytest.raises(ValueError):Simulation.model_validate(raw)


def test_scene_evidence_does_not_get_upgraded_from_assumed_to_measured():
    r=request();scene=compile_scene(r.scene)
    metadata=scene.metadata()['empirical_ground_models']['0']
    assert all(t['evidence']=='assumed' for t in metadata['tables'])
    assert metadata['absolute_radar_power_calibrated'] is False
