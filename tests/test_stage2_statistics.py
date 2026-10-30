import importlib.util
from pathlib import Path
import numpy as np
import pytest
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.radiation import TransmitAperture,aperture_voltage,directions,Pattern,element_voltage


def test_mixed_land_and_element_pattern_ensemble_power_matches_area_sum():
    tables=[dict(land_cover=name,polarization='vv',frequencies_hz=[9e9,11e9],incidence_deg=[0,80],
        sigma0_db=[[db,db],[db,db]],evidence='assumed',source='statistical unit check, not material measurements') for name,db in [('A',-20),('B',-10)]]
    r=Simulation.model_validate(dict(waveform=dict(kind='pulsed_lfm',carrier_hz=10e9,bandwidth_hz=1,
        sample_rate_hz=100e6,pulse_width_s=1e-6,slot_period_s=1e-3,adc_start_s=.12e-6,samples=8,loops=4),
        platform=dict(position_ned_m=[0,0,-10],boresight_ned=[0,0,1]),array=dict(layout='siso',num_tx=1,num_rx=1),
        radiation=dict(tx_element=dict(kind='cosine_power',power_exponent=2),rx_element=dict(kind='cosine_power',power_exponent=2)),
        scene=dict(amplitude_reference_range_m=10,terrain=dict(kind='flat',north_bounds_m=[0,2],east_bounds_m=[0,2],spacing_m=1,
            relative_reflectivity=1,land_scattering=dict(tables=tables,land_cover='A',polarization='vv'),land_cover_map=[['A','B'],['B','A']])),
        processing=dict(range_limits_m=[0,30],azimuth_deg=[0.],elevation_deg=[0.])))
    scene=compile_scene(r.scene);ranges=np.linalg.norm(scene.positions_ned_m-np.array(r.platform.position_ned_m),axis=1)
    sigma=np.where(scene.land_cover_labels=='A',.01,.1)
    expected=np.sum(scene.amplitudes**2*sigma*10**4/ranges**4*(10/ranges)**4)
    power=[]
    for seed in range(512):
        r.scene.seed=seed;power.append(abs(simulate_frame(r).adc[0,0,0,0])**2)
    assert np.mean(power)==pytest.approx(expected,rel=.18)


def test_sidelobe_reduction_is_not_hidden_by_separate_peak_normalization():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('input_report',root/'scripts/report_stage2_inputs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    angles=np.linspace(-80,80,16001);u=directions(angles,np.zeros_like(angles))
    positions=np.zeros((16,3));positions[:,1]=(np.arange(16)-7.5)*299792458/10e9/2
    metrics=[]
    for weights in [np.ones(16),np.hamming(16)]:
        ap=TransmitAperture(positions_local_m=positions.tolist(),amplitudes=weights.tolist())
        gain=aperture_voltage(u,np.zeros_like(angles),ap,10e9)*element_voltage(u[:,0],Pattern(kind='cosine_power'))**2
        metrics.append(module.beam_metrics(angles,abs(gain)**2))
    uniform,hamming=metrics
    assert -14<uniform['peak_sidelobe_relative_to_own_main_db']< -12
    assert hamming['peak_sidelobe_relative_to_own_main_db']< -35
    assert hamming['half_power_width_deg']>uniform['half_power_width_deg']
    assert hamming['peak_relative_drive_power_db']<uniform['peak_relative_drive_power_db']


def test_planar_transmit_array_steers_both_azimuth_and_elevation():
    coordinates=np.arange(4)-1.5;y,z=np.meshgrid(coordinates,coordinates)
    positions=np.c_[np.zeros(y.size),y.ravel(),z.ravel()]*299792458/10e9/2
    aperture=TransmitAperture(positions_local_m=positions.tolist(),steering_azimuth_deg=20,steering_elevation_deg=15)
    az,el=np.meshgrid(np.arange(-30,31),np.arange(-30,31))
    gain=aperture_voltage(directions(az,el),np.zeros_like(az,dtype=float),aperture,10e9)
    peak=np.unravel_index(np.argmax(abs(gain)),gain.shape)
    assert az[peak]==20 and el[peak]==15
    assert abs(gain[peak])**2==pytest.approx(16)


def test_large_element_spacing_produces_grating_lobe_instead_of_hiding_it():
    positions=np.zeros((8,3));positions[:,1]=(np.arange(8)-3.5)*299792458/10e9
    aperture=TransmitAperture(positions_local_m=positions.tolist(),steering_azimuth_deg=20)
    grating=np.rad2deg(np.arcsin(np.sin(np.deg2rad(20))-1))
    az=np.array([20,grating]);gain=aperture_voltage(directions(az,np.zeros(2)),np.zeros(2),aperture,10e9)
    np.testing.assert_allclose(abs(gain)**2,[8,8],rtol=1e-12)
