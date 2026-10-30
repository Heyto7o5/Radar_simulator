"""Causal radar/plant/control loop, with truth isolated in the evaluator."""
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np

from ..dem_los_reference import height_at
from .io import local_path,write_json,read_json,digest
from .scene import navigation_scene
from .dynamics import State,Command,Limits,propagate,body_to_neu
from .acquisition import RadarWorld,pair
from .controller import Controller,Estimate
from .timeline import covered_cpi_end


def model_fingerprint(model_dir):
    """Bind cached learned flights to exact weights and inference settings."""
    if model_dir is None:raise ValueError('learned run requires a frozen model')
    root=local_path(model_dir)
    if read_json(root/'status.json')['status']!='complete':raise ValueError('model training is incomplete')
    protocol=read_json(root/'training_protocol.json')
    names=['training_protocol.json','acquisition_axes.npz','uncertainty.json']
    if (root/'metrics.json').exists():names.append('metrics.json')
    if (root/'acquisition_timing.json').exists():names.append('acquisition_timing.json')
    names.extend(f'seed_{seed}.pt' for seed in protocol['seeds'])
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def execution_fingerprint():
    files=list(local_path('backend/radar_platform/unified').glob('*.py'))
    files.extend(local_path(f'backend/radar_platform/navigation/{name}.py') for name in
                 ['episode','controller','state_filter','timeline','dynamics','acquisition','inference','model','model_v2','model_v3','features_v2','observation','scene','io'])
    files.extend(local_path(f'backend/radar_platform/{name}.py') for name in
                 ['models','surface_scene','physical_optics','ground_scattering','land_tables','dem_buildings','dem_los_reference'])
    return {str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}


def default_episode():
    yaw=np.arctan2(18,24)+np.deg2rad(60)
    return dict(id='default',position_neu_m=[-24.,-18.,30.],
        velocity_neu_mps=(3*np.array([np.cos(yaw),np.sin(yaw),0.])).tolist(),
        yaw_rad=float(yaw),phase_seed=300001,noise_seed=400001,radar_interval_s=.25,
        correction_interval_s=.5,max_time_s=60.,noise_multiplier=1.,terrain_only=False,
        limits=asdict(Limits()))


class Evaluator:
    """May inspect truth for scoring/physical termination, never for guidance."""
    def __init__(self,config):
        self.terrain=config.scene.terrain
        self.buildings=config.scene.objects
        self.samples=[];self.held_since=None;self.failure=None;self.minimum_clearance=float('inf')
        self.last_state=None

    def observe(self,states):
        for state in states:
            self.last_state=state
            p=state.position_neu_m;v=state.velocity_neu_mps
            self.samples.append(dict(time_s=state.time_s,position_neu_m=p.tolist(),
                                     velocity_neu_mps=v.tolist(),yaw_rad=state.yaw_rad))
            if np.any(abs(p[:2])>60):
                self.failure='left_agricultural_experiment_bounds';return
            t=self.terrain
            ground=float(height_at(t.north_m,t.east_m,t.elevation_up_m,*p[:2]))
            self.minimum_clearance=min(self.minimum_clearance,float(p[2]-ground))
            if p[2]<=ground:
                self.failure='terrain_contact';return
            for obj in self.buildings:
                if obj.kind!='box':continue
                center=np.array(obj.position_ned_m)*[1,1,-1];half=np.array(obj.size_m)/2
                if np.all(abs(p-center)<=half):
                    self.failure='building_contact';return
            error=p-np.array([0.,0.,30.])
            held=np.linalg.norm(error[:2])<=2 and abs(error[2])<=1.5 and np.linalg.norm(v)<=.3
            if not held:self.held_since=None
            elif self.held_since is None:self.held_since=state.time_s

    def summary(self,state,termination):
        # A collision/boundary can occur inside an integrated interval. Score
        # the first observed terminal sample, not its unobserved future endpoint.
        if self.last_state is None:raise ValueError('evaluation requires observed plant samples')
        state=self.last_state
        p=state.position_neu_m;error=p-[0,0,30]
        held=0. if self.held_since is None else state.time_s-self.held_since
        points=np.array([s['position_neu_m'] for s in self.samples])
        return dict(termination=termination,success=bool(not self.failure and termination=='estimated_arrival' and held>=3.-1e-8),
            elapsed_simulation_s=state.time_s,final_position_neu_m=p.tolist(),
            horizontal_error_m=float(np.linalg.norm(error[:2])),height_error_m=float(abs(error[2])),
            hover_target_error_3d_m=float(np.linalg.norm(error)),ground_target_distance_3d_m=float(np.linalg.norm(p)),
            final_speed_mps=float(np.linalg.norm(state.velocity_neu_mps)),truth_hold_duration_s=held,
            trajectory_length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            minimum_ground_clearance_m=float(self.minimum_clearance) if np.isfinite(self.minimum_clearance) else None,
            truth_failure=self.failure)


def run_episode(spec,output,*,mode='learned',model_dir=None,threads=4,maximum_frames=None,state_filter=False,gentle_terminal=False):
    if mode not in {'learned','oracle','no_correction'}:raise ValueError('unknown experiment mode')
    root=local_path(output)
    if state_filter and mode!='learned':raise ValueError('radar state filtering is for learned runs only')
    fingerprint=model_fingerprint(model_dir) if mode=='learned' else None
    code_fingerprint=execution_fingerprint()
    if (root/'complete.json').exists():
        old=read_json(root/'protocol.json')
        if old['spec']!=spec or old['mode']!=mode:raise ValueError('episode output belongs to a different run')
        if bool(old.get('state_filter'))!=state_filter:raise ValueError('cached flight uses different state filtering')
        if old.get('gentle_terminal',False)!=gentle_terminal:raise ValueError('cached flight uses different terminal guidance')
        if mode=='learned' and old.get('model_sha256')!=fingerprint:
            raise ValueError('cached flight belongs to different or unverified model weights; use a new output')
        if old.get('execution_sha256')!=code_fingerprint:
            raise ValueError('cached flight belongs to different or unverified controller/simulation source; use a new output')
        return read_json(root/'complete.json')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'controls.jsonl').exists():raise ValueError('partial episode preserved; replay in a new directory')
    cfg,provenance=navigation_scene(terrain_only=spec['terrain_only'],seed=spec['phase_seed'])
    raw=cfg.model_dump(mode='json');raw['receiver_noise']['seed']=spec['noise_seed']
    raw['receiver_noise']['variance']*=spec['noise_multiplier']
    from ..unified.schema import Simulation
    cfg=Simulation.model_validate(raw);world=RadarWorld(cfg)
    limits=Limits(**spec['limits']);controller=Controller(limits,gentle_terminal=gentle_terminal)
    predictor=None
    temporal_filter=None;filter_protocol=None
    if mode=='learned':
        if model_dir is None:raise ValueError('learned run requires a frozen model')
        from .inference import Predictor
        predictor=Predictor(model_dir,threads)
        if state_filter:
            from .state_filter import RadarStateFilter,FILTER_VERSION
            measurement_std=read_json(local_path(model_dir)/'metrics.json')['val']['axis_rmse']
            temporal_filter=RadarStateFilter(measurement_std,limits)
            filter_protocol=dict(version=FILTER_VERSION,measurement_std=temporal_filter.std.tolist(),
                calibration='Frozen validation residual RMSE; no per-episode truth tuning',
                prediction='Previous RDA estimate and issued body acceleration/yaw-rate, no world pose input')
    protocol=dict(spec=spec,mode=mode,model_dir=None if model_dir is None else str(local_path(model_dir)),model_sha256=fingerprint,
        execution_sha256=code_fingerprint,
        state_filter=filter_protocol,
        gentle_terminal=gentle_terminal,
        guidance=dict(position_gain=controller.position_gain,horizontal_deadband_m=.75 if gentle_terminal else 0.,
            vertical_deadband_m=.3 if gentle_terminal else 0.,estimated_arrival_radius_m=controller.arrival_radius,
            estimated_arrival_height_m=controller.arrival_height,estimated_arrival_speed_mps=controller.arrival_speed,
            estimated_arrival_dwell_s=controller.arrival_dwell_s,
            actual_evaluation_limits_unchanged=[2.,1.5,.3,3.]),
        observation_boundary='RDA numeric volumes, acquisition axes and timestamps only',
        controller_boundary='estimated target-relative body position and velocity only',
        plant='bounded continuous kinematics with yaw; level pitch/roll',
        command_timing='after complete second CPI; no future observations',
        success='estimated stop plus actual horizontal <=2m, height <=1.5m, speed <=0.3m/s for 3s',
        oracle_scope='explicit truth-control evaluation reference only; never mixed into learned guidance')
    write_json(root/'protocol.json',protocol);write_json(root/'scene_provenance.json',provenance)
    write_json(root/'scene.json',cfg.model_dump(mode='json'))
    state=State(0.,np.array(spec['position_neu_m']),np.array(spec['velocity_neu_mps']),spec['yaw_rad'])
    evaluator=Evaluator(cfg);evaluator.observe([state]);command=Command.coast()
    previous=None;next_correction=0.;invalid_since=None;corrections=0;start=time.time();termination='timeout'
    write_json(root/'status.json',dict(status='running',frame=0,simulation_time_s=0))
    frame_limit=int(np.floor((spec['max_time_s']-world.duration_s)/spec['radar_interval_s']))+1
    if maximum_frames is not None:frame_limit=min(frame_limit,maximum_frames)
    with (root/'controls.jsonl').open('x') as control_log,(root/'acquisition.jsonl').open('x') as acquisition_log:
        for index in range(frame_limit):
            frame_start=index*spec['radar_interval_s']
            if state.time_s<frame_start:
                state,samples=propagate(state,command,frame_start,limits);evaluator.observe(samples[1:])
            if evaluator.failure:termination=evaluator.failure;break
            state,samples=propagate(state,command,covered_cpi_end(cfg,frame_start),limits)
            evaluator.observe(samples[1:])
            if evaluator.failure:termination=evaluator.failure;break
            full_directory=root/'full_adc'/f'frame_{index:04d}' if index in [0,20] else None
            current,diagnostic,request=world.acquire(samples,index,
                save_directory=full_directory,
                assessment=True)
            if full_directory is not None:
                # Geometry/work estimates are reusable, applicability is pose-dependent.
                write_json(full_directory/'applicability.json',diagnostic['applicability'])
            frame_dir=root/'frames';frame_dir.mkdir(exist_ok=True)
            np.savez_compressed(frame_dir/f'{index:04d}.npz',magnitude=current.magnitude,
                range_m=current.range_m,velocity_mps=current.velocity_mps,elevation_deg=current.elevation_deg,
                azimuth_deg=current.azimuth_deg,frame_start_s=current.start_s,frame_end_s=current.end_s)
            # Global truth and visibility are not part of the online observation files.
            truth_dir=root/'truth';truth_dir.mkdir(exist_ok=True)
            write_json(truth_dir/f'{index:04d}.json',dict(state_body=state.label().tolist(),
                position_neu_m=state.position_neu_m.tolist(),velocity_neu_mps=state.velocity_neu_mps.tolist(),
                yaw_rad=state.yaw_rad,time_s=state.time_s,diagnostic=diagnostic,
                request=request.model_dump(mode='json')))
            acquisition_log.write(json.dumps(dict(frame=index,start_s=frame_start,end_s=state.time_s,
                                                  elapsed_s=diagnostic['elapsed_s']))+'\n');acquisition_log.flush()
            if previous is not None and state.time_s+1e-9>=next_correction:
                observation=pair(previous,current)
                obs_dir=root/'observations';obs_dir.mkdir(exist_ok=True)
                np.savez_compressed(obs_dir/f'{index:04d}.npz',**observation.arrays())
                estimate=None;prediction={}
                raw_prediction=None
                if mode=='learned':estimate,prediction=predictor.predict(observation)
                elif mode=='oracle':
                    label=state.label();estimate=Estimate(state.time_s,label[:3],label[3:])
                    prediction=dict(reference='explicit simulator-truth controller; excluded from learned results')
                if estimate is not None:
                    if temporal_filter is not None:
                        raw_prediction=dict(relative_position_body_m=estimate.relative_position_body_m.tolist(),
                            velocity_body_mps=estimate.velocity_body_mps.tolist(),valid=estimate.valid,
                            error_scale_m=estimate.error_scale_m,time_s=estimate.time_s)
                        estimate,filter_diagnostic=temporal_filter.update(estimate,command)
                        prediction['state_filter']=filter_diagnostic
                    command,control=controller.step(estimate)
                    if not estimate.valid:
                        if invalid_since is None:invalid_since=state.time_s
                    else:invalid_since=None
                    prediction.update(relative_position_body_m=estimate.relative_position_body_m.tolist(),
                        velocity_body_mps=estimate.velocity_body_mps.tolist(),valid=estimate.valid,
                        error_scale_m=estimate.error_scale_m,time_s=estimate.time_s)
                else:control=dict(mode='no_correction',estimated_arrival=False)
                row=dict(frame=index,time_s=state.time_s,prediction=prediction,control=control,
                    command=dict(acceleration_body_mps2=command.acceleration_body_mps2.tolist(),
                                 yaw_rate_rad_s=command.yaw_rate_rad_s))
                if raw_prediction is not None:row['raw_prediction']=raw_prediction
                control_log.write(json.dumps(row)+'\n');control_log.flush();corrections+=int(mode!='no_correction')
                next_correction=state.time_s+spec['correction_interval_s']
                if control['estimated_arrival']:termination='estimated_arrival';break
                if invalid_since is not None and state.time_s-invalid_since>=2.-1e-9:
                    termination='radar_localization_lost';break
            previous=current
            write_json(root/'status.json',dict(status='running',frame=index,simulation_time_s=state.time_s,
                       wall_time_s=time.time()-start,corrections=corrections))
            print(json.dumps(dict(episode=spec['id'],mode=mode,frame=index,simulation_time_s=state.time_s,
                                  radar_seconds=diagnostic['elapsed_s'])),flush=True)
    if maximum_frames is not None and termination=='timeout':termination='development_frame_limit'
    write_json(root/'truth_trajectory.json',evaluator.samples)
    summary=evaluator.summary(state,termination)
    summary.update(mode=mode,corrections=corrections,wall_time_s=time.time()-start,episode=spec['id'],
                   complete_experiment=maximum_frames is None)
    write_json(root/'complete.json',summary);write_json(root/'status.json',dict(status='complete',**summary))
    return summary
