import copy
import json
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from radar_platform.api import app
from radar_platform.unified.applicability import inspect_configuration
from radar_platform.unified.session import session_from_config

ROOT=Path(__file__).resolve().parents[1]


def config():
    r=json.loads((ROOT/'configs/unified/lab_fmcw_mimo.json').read_text())
    r['scene']['objects']=r['scene']['objects'][:1]
    r['scene']['objects'][0]['velocity_ned_mps']=[0,0,0]
    r['array']={'layout':'siso','num_tx':1,'num_rx':1}
    return r


def indexed(report):return {x['code']:x for x in report['checks']}


def test_static_well_sampled_scene_usable_and_check_does_not_modify_input():
    r=config();original=copy.deepcopy(r)
    report=inspect_configuration(r)
    assert report['status']=='usable'
    assert r==original
    assert report['coverage']['all_scatterers_checked']
    assert not report['coverage']['all_chirps_checked']


def test_1200_mps_is_not_silently_reduced_and_range_walk_is_reported():
    r=json.loads((ROOT/'configs/unified/high_altitude_1ghz.json').read_text())
    report=inspect_configuration(r);checks=indexed(report)
    assert checks['doppler_ambiguity']['status']=='usable'
    assert checks['doppler_ambiguity']['metrics']['closing_speed_max_mps']>1199
    assert checks['range_walk']['status']=='risk'
    assert checks['range_walk']['metrics']['sampled_max_cpi_range_change_bins']>4
    r['waveform']['carrier_hz']=10e9
    assert indexed(inspect_configuration(r))['doppler_ambiguity']['status']=='risk'


def test_tdm_uses_same_transmitter_period_not_slot_prf():
    r=config();a=inspect_configuration(r)['derived']['maximum_unambiguous_speed_mps']
    r['array']={'layout':'ula','num_tx':2,'num_rx':4}
    b=inspect_configuration(r)['derived']['maximum_unambiguous_speed_mps']
    assert np.isclose(b,a/2)


def test_fmcw_startup_and_out_of_band_are_separate_risks():
    r=config();r['waveform']['adc_start_s']=0
    assert indexed(inspect_configuration(r))['fmcw_receive_gate']['status']=='risk'
    r['scene']['objects'][0]['position_ned_m']=[1000,0,0]
    assert indexed(inspect_configuration(r))['fmcw_beat_band']['status']=='risk'


def test_pulse_truncation_and_outside_pri_not_silently_called_range_ambiguity_model():
    r=json.loads((ROOT/'configs/unified/high_altitude_1ghz.json').read_text())
    r['scene']['objects']=[{'kind':'point','id':'far','position_ned_m':[0,0,20000]}]
    r['platform']['boresight_ned']=[0,0,1]
    checks=indexed(inspect_configuration(r))
    assert checks['range_ambiguity']['status']=='risk'
    assert checks['pulse_receive_gate']['metrics']['illuminated_fully_missing']>0


def test_bad_timing_and_large_cube_are_unsupported():
    r=config();r['waveform']['ramp_s']=1
    assert inspect_configuration(r)['status']=='unsupported'
    r=config();r['waveform'].update(samples=32768,sample_rate_hz=1e9,loops=512)
    r['array']={'layout':'ula','num_tx':8,'num_rx':8}
    assert inspect_configuration(r)['status']=='unsupported'


def test_check_api_does_not_create_job_and_saved_run_has_report(tmp_path):
    from radar_platform.unified.jobs import manager
    before=set(manager.jobs)
    with TestClient(app) as client:
        result=client.post('/api/v1/scene-check',json=config())
        assert result.status_code==200 and result.json()['status']=='usable'
        bad=config();bad['waveform']['samples']=-1
        assert client.post('/api/v1/scene-check',json=bad).json()['status']=='unsupported'
    assert set(manager.jobs)==before
    session=session_from_config(config());session.save_inputs(tmp_path)
    assert json.loads((tmp_path/'applicability.json').read_text())['status']=='usable'
