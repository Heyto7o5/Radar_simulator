#!/usr/bin/env python3
"""Check broad-pattern full-context visibility using an independent height-grid algorithm."""
import argparse,json
from pathlib import Path
import numpy as np
from radar_platform.dem_los_reference import reference_los


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('coverage audit already exists')
    request=json.loads((a.run/'request.json').read_text())
    if request['array']['num_tx']!=1 or request['array']['num_rx']!=1 or request['platform']['trajectory'] is not None:
        raise ValueError('this independent campaign audit is for the declared constant-motion co-located SISO cases')
    meta=json.loads((a.run/'frame_0000/metadata.json').read_text());coverage=meta['visibility']['coverage']
    points=np.asarray(coverage['positions_at_frame_start_ned_m'])
    indices=np.unique(np.linspace(0,len(points)-1,min(1024,len(points)),dtype=int))
    expected=np.zeros(len(indices),int);ambiguous=np.zeros(len(indices),bool)
    terrain=request['scene']['terrain'];w=request['waveform'];platform=request['platform']
    start=request['sequence']['start_time_s'];poses=[]
    for slot in range(w['loops']):
        time=start+slot*w['slot_period_s']
        origin=np.array(platform['position_ned_m'])+np.array(platform['velocity_ned_mps'])*time
        result=reference_los(terrain['north_m'],terrain['east_m'],terrain['elevation_up_m'],origin,points[indices],
                             endpoint_epsilon_m=request['visibility']['endpoint_epsilon_m'])
        if not all(r['covered'] for r in result):raise ValueError('ray leaves modeled context')
        expected+=np.array([not r['blocked'] for r in result]);ambiguous|=np.array([r['near_tangent'] for r in result])
        poses.append(origin.tolist())
    actual=np.array(coverage['unblocked_channel_slots'])[indices]
    illumination=np.array(coverage['illuminated_tx_slots'])[indices]
    if not np.all(illumination==w['loops']):raise ValueError('audit must separately account for angular support; cannot equate it with occlusion')
    mismatch=(expected!=actual)&~ambiguous
    report={'source_run':str(a.run),'selection':'1024 evenly spaced scatterer indices fixed without using visibility values',
        'checked_scatterers':len(indices),'checked_slots':w['loops'],'ray_evaluations':len(indices)*w['loops'],
        'non_tangent_count_mismatches':int(mismatch.sum()),'near_tangent_scatterers':int(ambiguous.sum()),
        'reference_blocked_ray_evaluations':int((w['loops']-expected).sum()),
        'indices':indices.tolist(),'reference_clear_slots':expected.tolist(),'engine_clear_slots':actual.tolist(),
        'observer_positions_ned_m':poses,
        'scope':'full 5.76 km context, including sidelobe illumination; independent algorithm on the same real DSM, not field observations'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['indices','reference_clear_slots','engine_clear_slots','observer_positions_ned_m']}))
    if mismatch.any():raise SystemExit(1)


if __name__=='__main__':main()
