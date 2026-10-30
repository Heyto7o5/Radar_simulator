import json
import numpy as np
import pytest
from radar_platform.array_fmcw import TDMArray, simulate_motion_adc
from radar_platform.fmcw import FMCWConfig
from radar_platform.models import TargetConfig
from radar_platform.unified.array_adapter import motion_array_configuration
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.session import SimulationSession


def inputs():
    cfg=FMCWConfig(76.5e9,77.5e9,50e-6,60e-6,128,16,4e6,2e-6)
    return cfg,TDMArray.ideal_ula(cfg),[TargetConfig(id='t',position_ned_m=(10,2,0),velocity_ned_mps=(-2,0,0))]


@pytest.mark.parametrize('direction',[(1,0,0),(0,1,0),(0,0,1)])
def test_ned_array_offsets_are_not_rotated_with_boresight(direction):
    cfg,array,targets=inputs()
    request=motion_array_configuration(cfg,targets,array,boresight_ned=direction,beam_half_angle_deg=180)
    layout=build_layout(request.array,request.waveform,request.platform)
    np.testing.assert_allclose(layout.tx_ned_m,array.tx_ned_m,atol=1e-15)
    np.testing.assert_allclose(layout.rx_ned_m,array.rx_ned_m,atol=1e-15)
    np.testing.assert_array_equal(layout.tx_offsets_s,array.tx_offsets_s)
    expected=SimulationSession(request).frame(0).adc
    actual=simulate_motion_adc(cfg,targets,array,boresight_ned=direction,beam_half_angle_deg=180)
    np.testing.assert_array_equal(actual,expected)
    assert np.max(abs(actual))>0


def test_array_api_saves_reusable_config(tmp_path,monkeypatch):
    from radar_platform.array_api import ArraySimulationRequest,run_array_request
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    request=ArraySimulationRequest(targets=[TargetConfig(id='t',position_ned_m=(10,0,0))])
    result=run_array_request(request)
    root=tmp_path/'array_jobs'/result['job_id']
    from radar_platform.unified.schema import Simulation
    common=Simulation.model_validate_json((root/'simulation_config.json').read_text())
    frame=SimulationSession(common).frame(0)
    for name,expected in [('adc',frame.adc),('rd_channels',frame.products.rd_channels),('rad',frame.products.rad[:,:,0,:])]:
        stored=np.fromfile(root/(name+'.c64'),dtype='<c8').reshape(expected.shape)
        np.testing.assert_array_equal(expected,stored)
    assert result['azimuth_coordinate_frame']=='sensor_local_right_positive'
    assert common.processing.azimuth_deg==request.azimuth_deg
    np.testing.assert_array_equal(result['velocity_mps'],frame.products.velocity_mps)
    assert json.loads((root/'metadata.json').read_text())['simulation_engine']=='unified'


def test_configuration_rejects_tdm_overlap_before_running():
    cfg,array,targets=inputs()
    bad=TDMArray(array.tx_ned_m,array.rx_ned_m,[0,1e-6],array.loop_period_s)
    with pytest.raises(ValueError,match='overlap'):
        motion_array_configuration(cfg,targets,bad)


def test_smallest_public_array_request_remains_supported():
    from radar_platform.array_api import ArraySimulationRequest
    request=ArraySimulationRequest(radar={'samples':4,'chirps':4},
        targets=[TargetConfig(id='t',position_ned_m=(2,0,0))])
    adc=simulate_motion_adc(request.radar.config(),request.targets,request.array())
    assert adc.shape==(4,4,4,2)
    assert np.isfinite(adc).all() and np.max(abs(adc))>0


def test_legacy_endfire_scan_endpoints_are_not_silently_dropped():
    cfg,array,targets=inputs()
    config=motion_array_configuration(cfg,targets,array,azimuth_deg=[-90,0,90])
    result=SimulationSession(config).frame(0).products
    np.testing.assert_array_equal(result.azimuth_deg,[-90,0,90])
    assert np.isfinite(result.rad).all()
