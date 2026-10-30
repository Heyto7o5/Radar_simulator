import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from radar_platform.navigation.io import ROOT


def test_last_window_samples_do_not_fabricate_a_three_second_hold(project_tmp_path):
    tmp_path = project_tmp_path
    root=tmp_path/'run';root.mkdir();out=tmp_path/'audit'
    times=np.r_[np.arange(0,3.001,.02),3.005]
    velocity=np.zeros((len(times),3));velocity[:,0]=np.maximum(0,.31-times)
    position=np.zeros((len(times),3));position[:,2]=30.
    position[1:,0]=np.cumsum((velocity[:-1,0]+velocity[1:,0])/2*np.diff(times))
    trace=[dict(time_s=float(t),position_neu_m=p.tolist(),velocity_neu_mps=v.tolist(),yaw_rad=0.)
           for t,p,v in zip(times,position,velocity)]
    (root/'truth_trajectory.json').write_text(json.dumps(trace))
    (root/'controls.jsonl').write_text('')
    (root/'complete.json').write_text(json.dumps(dict(success=False,termination='estimated_arrival')))
    # The last-three-second slice starts at 0.005s, dropping the invalid t=0
    # sample. All remaining samples pass, but their actual held span is 2.985s.
    result=subprocess.run([sys.executable,str(ROOT/'scripts/audit_navigation.py'),
        '--episode',str(root),'--output',str(out)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    audit=json.loads((out/'audit.json').read_text())['episode']
    assert audit['recomputed_hold_duration_s']<3
    assert not audit['success_recomputed_from_last_3_seconds']
