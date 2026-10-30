import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('dart_scale',Path(__file__).resolve().parents[1]/'scripts/audit_dart_surface_scale.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_covariance_diagnostic_plane_isotropic_and_known_thickness():
    x,y,z=np.meshgrid([-1.,1.],[-1.,1.],[-1.,1.])
    p=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    ratio,rms,_=module.covariance_metrics(p)
    assert ratio==pytest.approx(1/3) and rms==pytest.approx(1)
    p[:,2]=0
    ratio,rms,_=module.covariance_metrics(p)
    assert ratio==pytest.approx(0) and rms==pytest.approx(0)
    p[:,2]=z.ravel()*.04
    ratio,rms,_=module.covariance_metrics(p)
    assert rms==pytest.approx(.04)
    assert ratio==pytest.approx(.04**2/(2+.04**2))
