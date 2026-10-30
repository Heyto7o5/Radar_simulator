import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from radar_platform.unified.trajectory import RigidTrajectory

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('geometry_support',scripts/'audit_dart_geometry_rd_support.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_projection_sign_visibility_and_identical_dilation():
    trajectory=RigidTrajectory(time_s=[0,1],position_ned_m=[(0,0,0),(1,0,0)],velocity_ned_mps=[(1,0,0)]*2,
                               quaternion_xyzw=[(0,0,0,1)]*2,provenance='analytic test')
    request=SimpleNamespace(waveform=SimpleNamespace(loops=4),array=SimpleNamespace(loop_period_s=.001),
                            platform=SimpleNamespace(trajectory=trajectory,beam_half_angle_deg=45))
    ranges=np.arange(0,20,.1);velocity=np.arange(-2,2.01,.1)
    mask,limit=m.support(np.array([[10.,0,0]]),request,ranges,velocity)
    assert mask[100,30] and not mask[100,10]
    assert mask.sum()==5*7
    assert limit==1.05
    hidden,_=m.support(np.array([[-10.,0,0]]),request,ranges,velocity)
    assert not hidden.any()
