#!/usr/bin/env python3
"""Exercise new HTTP contracts in-process; do not expose any network ports."""
import argparse,json,os,time
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from fastapi.testclient import TestClient
from radar_platform.api import app
from radar_platform.array_api import ArraySimulationRequest
from radar_platform.unified.array_adapter import motion_array_configuration
from radar_platform.unified.session import SimulationSession
from radar_platform.models import default_request,TargetConfig
from radar_platform.jobs import job_manager


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={}
    with TemporaryDirectory(prefix='radar_layered_api_') as temporary:
        previous=os.environ.get('RADAR_RUNTIME_ROOT');os.environ['RADAR_RUNTIME_ROOT']=temporary
        try:
            with TestClient(app) as client:
                request=ArraySimulationRequest(targets=[TargetConfig(id='moving',position_ned_m=(12,2,0),velocity_ned_mps=(-2,.5,0))],frame_start_s=.5)
                response=client.post('/api/v1/fmcw-array-simulations',json=request.model_dump(mode='json'))
                response.raise_for_status();meta=response.json()
                config=motion_array_configuration(request.radar.config(),request.targets,request.array(),frame_start_s=.5,azimuth_deg=request.azimuth_deg)
                frame=SimulationSession(config).frame(0)
                for name,value in [('adc',frame.adc),('rd_channels',frame.products.rd_channels),('rad',frame.products.rad[:,:,0,:])]:
                    response=client.get(meta['urls'][name]);response.raise_for_status()
                    array=np.frombuffer(response.content,dtype='<c8').reshape(meta['shapes'][name])
                    np.testing.assert_array_equal(array,value)
                    result[name]={'shape':list(array.shape),'bytes':len(response.content),'bitwise_equal_to_direct_call':True}
                assert meta['is_real'] is False
                pulse=default_request();pulse.platform.speed_mps=0;pulse.platform.boresight_ned=(0,0,1)
                pulse.scene.targets=[TargetConfig(id='moving',position_ned_m=(0,0,0),velocity_ned_mps=(0,0,-30))]
                pulse.sequence.frame_count=1
                created=client.post('/api/v1/simulations',json=pulse.model_dump(mode='json'));created.raise_for_status()
                job=created.json()['job_id'];deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    state=client.get('/api/v1/simulations/'+job).json()
                    if state['status'] in ('complete','failed'):break
                    time.sleep(.02)
                assert state['status']=='complete',state
                frame=state['frames'][0]
                assert frame['validations'][0]['validation_mode']=='truth_assisted_analytic_check'
                assert frame['blind_detections']
                assert (Path(temporary)/'jobs'/job/'request.json').exists()
                result['pulsed_api']={'stationary_radar_moving_target':True,'truth_assisted_separated':True,
                                      'blind_candidates':len(frame['blind_detections']),'input_manifest_saved':True}
                job_manager.clear()
        finally:
            if previous is None:os.environ.pop('RADAR_RUNTIME_ROOT',None)
            else:os.environ['RADAR_RUNTIME_ROOT']=previous
    result['network_ports_opened']=False
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
