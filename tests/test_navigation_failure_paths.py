import json
import numpy as np
import pytest

from radar_platform.navigation.controller import Controller,Estimate
from radar_platform.navigation.dynamics import State,Command,Limits,propagate
from radar_platform.navigation.episode import Evaluator,default_episode,model_fingerprint,execution_fingerprint,run_episode
from radar_platform.unified.schema import Simulation


def test_lost_observation_emits_bounded_braking_command():
    controller=Controller(Limits())
    controller.step(Estimate(0.,np.array([30.,0.,-30.]),np.array([12.,5.,3.])))
    command,info=controller.step(Estimate(.5,np.zeros(3),np.zeros(3),valid=False))
    assert np.linalg.norm(command.acceleration_body_mps2[:2])<=2.+1e-12
    assert abs(command.acceleration_body_mps2[2])<=1
    assert np.dot(command.acceleration_body_mps2,controller.last_velocity)<0
    assert info['mode']=='reacquire' and not info['estimated_arrival']


def test_failure_is_scored_at_first_observed_crossing_not_future_endpoint():
    cfg=Simulation(scene=dict(terrain=dict(kind='dem',north_m=[-100,100],east_m=[-100,100],elevation_up_m=[[0,0],[0,0]])))
    evaluator=Evaluator(cfg)
    start=State(0.,np.array([59.99,0.,30.]),np.array([3.,0.,0.]),0.)
    evaluator.observe([start]);future,samples=propagate(start,Command.coast(),.25)
    evaluator.observe(samples[1:]);summary=evaluator.summary(future,evaluator.failure)
    assert summary['termination']=='left_agricultural_experiment_bounds'
    assert summary['elapsed_simulation_s']<future.time_s
    assert summary['elapsed_simulation_s']==evaluator.samples[-1]['time_s']
    assert summary['final_position_neu_m']==evaluator.samples[-1]['position_neu_m']
    assert not summary['success']


def test_cached_learned_flight_rejects_changed_weights(project_tmp_path):
    tmp_path = project_tmp_path
    model=tmp_path/'model';model.mkdir()
    (model/'status.json').write_text(json.dumps({'status':'complete'}))
    (model/'training_protocol.json').write_text(json.dumps({'seeds':[11]}))
    for name in ['acquisition_axes.npz','uncertainty.json','seed_11.pt']:(model/name).write_bytes(b'original')
    spec=default_episode();output=tmp_path/'flight';output.mkdir()
    (output/'protocol.json').write_text(json.dumps(dict(spec=spec,mode='learned',model_sha256=model_fingerprint(model),execution_sha256=execution_fingerprint())))
    (output/'complete.json').write_text(json.dumps(dict(success=True)))
    assert run_episode(spec,output,model_dir=model)=={'success':True}
    (model/'seed_11.pt').write_bytes(b'retrained weights')
    with pytest.raises(ValueError,match='cached flight'):run_episode(spec,output,model_dir=model)
