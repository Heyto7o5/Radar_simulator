"""Cross-process recovery, provenance and display contracts of the research UI."""
import io
import json
import time
import zipfile
import numpy as np
from fastapi.testclient import TestClient
from radar_platform.api import app
from radar_platform.unified.jobs import manager
from radar_platform.workbench import preview,PreviewRequest,presets
from radar_platform.workbench_archive import recover,owned_root


def create_finished(client):
    raw=presets()['points_pulsed_lfm'];raw['sequence']['frame_count']=2
    raw['visibility']={'report_coverage':True}
    raw['radiation']={'tx_element':{'kind':'cosine_power','power_exponent':2},'replace_legacy_cone':True}
    request=preview(PreviewRequest(request=raw))['request']
    response=client.post('/api/v1/scene-simulations',json=request);assert response.status_code==202
    job_id=response.json()['job_id']
    for _ in range(1000):
        state=client.get(f'/api/v1/scene-simulations/{job_id}').json()
        if state['status'] not in ('queued','running'):break
        time.sleep(.01)
    assert state['status']=='complete',state.get('error')
    return job_id,request


def test_disk_recovery_keeps_arrays_and_does_not_relaunch(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as c:
        job_id,request=create_finished(c)
        original=c.get(f'/api/v1/workbench/jobs/{job_id}/frames/0/view').content
        manager.jobs.pop(job_id)
        restored=c.get(f'/api/v1/scene-simulations/{job_id}').json()
        assert restored['status']=='complete' and job_id not in manager.tasks
        assert c.get(f'/api/v1/workbench/jobs/{job_id}/request').json()==request
        assert c.get(f'/api/v1/workbench/jobs/{job_id}/frames/0/view').content==original
        saved=c.post(f'/api/v1/workbench/experiments/{job_id}/save',json={'title':'A frozen','display':{'frame':1}})
        assert saved.status_code==200
        exp=c.get(f'/api/v1/workbench/experiments/{job_id}').json()
        assert exp['saved']['display']['frame']==1 and len(exp['config_hash'])==64
        record=next(x for x in c.get('/api/v1/workbench/experiments').json() if x['id']==job_id)
        assert record['saved'] and record['name']=='A frozen'


def test_partial_recovery_and_owned_paths(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as c:
        job_id,_=create_finished(c);root=manager.get(job_id).root
        state=json.loads((root/'job.json').read_text());state['status']='running';(root/'job.json').write_text(json.dumps(state))
        restored=recover(job_id)
        assert restored.status=='cancelled' and len(restored.frames)==2 and '中断' in restored.error
        assert c.get('/api/v1/workbench/experiments/not-a-job').status_code==404
        for value in ['../request.json','/etc/passwd','a'*32+'/../../x']:
            try:owned_root(value)
            except KeyError:pass
            else:raise AssertionError('accepted path')
        target=tmp_path/'scene_jobs'/('b'*32);target.symlink_to(root,target_is_directory=True)
        try:owned_root('b'*32)
        except KeyError:pass
        else:raise AssertionError('accepted symlink')


def test_auxiliary_epoch_and_coverage_match_solver(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as c:
        job_id,request=create_finished(c);job=manager.get(job_id)
        for i in range(2):
            a=c.get(f'/api/v1/workbench/jobs/{job_id}/frames/{i}/auxiliary').json()
            assert a['time_s']==job.frames[i]['frame_start_s']
            np.testing.assert_allclose(a['platform_position_ned_m'],job.frames[i]['platform_position_ned_m'])
            expected=job.frames[i]['visibility']['coverage']
            assert a['coverage']['sampled_echo_channel_slots']==expected['sampled_echo_channel_slots']
            assert a['coverage']['channel_slot_denominator']==expected['channel_slot_denominator']
            assert max(a['pattern']['normalized_power'])==1
            assert a['pattern']['time_s']==a['time_s']
        assert c.get(f'/api/v1/workbench/jobs/{job_id}/frames/9/auxiliary').status_code==404
        geometry=c.get(f'/api/v1/workbench/jobs/{job_id}/geometry').json()
        assert len(geometry['positions_ned_m'])==3


def test_linear_data_and_reproduction_package(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as c:
        job_id,request=create_finished(c)
        data=c.get(f'/api/v1/workbench/jobs/{job_id}/frames/0/display-data').json()
        from radar_platform.workbench import read_product
        np.testing.assert_array_equal(data['power'],read_product(manager.get(job_id),0,'rd_power').T)
        assert len(data['power'])==len(data['axes']['velocity_mps'])
        package=c.get(f'/api/v1/workbench/experiments/{job_id}/export');assert package.status_code==200
        with zipfile.ZipFile(io.BytesIO(package.content)) as z:
            assert json.loads(z.read('request.json'))==request
            manifest=json.loads(z.read('experiment.json'));assert not manifest['partial']
            assert 'frame_0001/rd_channels.bin' in z.namelist()
        assert not list((tmp_path/'exports').glob('*.zip'))


def test_draft_and_capabilities_preserve_advanced_input(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as c:
        raw=presets()['points_pulsed_lfm'];raw['input_provenance']=['fixture source'];raw['visibility']={'report_coverage':True}
        body={'request':raw,'features':{},'preset':'custom','auto':False}
        assert c.put('/api/v1/workbench/draft',json=body).status_code==200
        assert c.get('/api/v1/workbench/draft').json()=={**body,'result_job_id':None}
        caps=c.get('/api/v1/workbench/capabilities').json()
        assert 'radiation' in caps['schema']['properties']
        assert len(c.get('/api/v1/workbench/cases').json())==6


def test_case_inputs_are_normalized_and_invalid_import_rejected():
    with TestClient(app) as c:
        for case_id in ['array','scan','occlusion']:
            r=c.get(f'/api/v1/workbench/cases/{case_id}/request').json()
            assert r['schema_version']=='1.0' and 'start_time_s' in r['sequence']
            assert all('velocity_ned_mps' in o for o in r['scene']['objects'])
        assert c.post('/api/v1/workbench/normalize',json={'scene':{'objects':[{'kind':'nope'}]}}).status_code==422
        raw=presets()['points_pulsed_lfm'];raw['platform']['position_ned_m'][2]=-7000
        r=c.post('/api/v1/workbench/normalize',json=raw).json()
        assert r['waveform']==raw['waveform']
        assert r['platform']['position_ned_m'][2]==-7000


def test_sampled_trajectory_preview_uses_evaluator_epoch():
    raw=presets()['points_pulsed_lfm'];raw['sequence'].update(frame_count=1,start_time_s=.5)
    raw['platform']['trajectory']={'time_s':[0,1],'position_ned_m':[[0,0,-5000],[20,0,-5000]],'velocity_ned_mps':[[20,0,0],[20,0,0]],'quaternion_xyzw':[[0,0,0,1],[0,0,0,1]],'provenance':'test trajectory'}
    result=preview(PreviewRequest(request=raw,auto_acquisition=False))
    assert result['state']['time_s']==.5
    np.testing.assert_allclose(result['state']['platform_position_ned_m'],[10,0,-5000])
    np.testing.assert_allclose(result['state']['basis_ned'],np.eye(3))
