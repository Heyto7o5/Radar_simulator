import numpy as np
import pytest
from radar_platform.plane_extraction import extract_planes, nearest_plane_distance, FittedPlane, supported_plane_mesh


def test_two_planes_seed_reproducibility_and_holdout():
    rng=np.random.default_rng(5)
    xy=rng.uniform(-1,1,(1000,2))
    points=np.concatenate([np.column_stack([xy,np.zeros(1000)]),
                           np.column_stack([np.full(1000,2.),xy])])
    kwargs=dict(tolerance_m=.01,max_planes=2,trials=80,min_inliers=100,seed=9)
    a=extract_planes(points[::2],**kwargs);b=extract_planes(points[::2],**kwargs)
    assert len(a)==len(b)==2
    for p,q in zip(a,b):assert np.allclose(p.normal,q.normal)
    assert np.quantile(nearest_plane_distance(points[1::2],a),.9)<.01
    assert nearest_plane_distance(np.array([[20.,20.,0.]]),a)[0]>10


def test_degenerate_and_invalid():
    line=np.column_stack([np.arange(20),np.zeros((20,2))])
    assert extract_planes(line,tolerance_m=.1,min_inliers=3)==[]
    assert np.isinf(nearest_plane_distance(line,[])).all()
    with pytest.raises(ValueError):extract_planes(line,tolerance_m=-1)


def test_supported_mesh_preserves_unobserved_holes():
    plane=FittedPlane(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),
                      np.array([0.,1.,0.]),np.array([[0.,0.],[10.,1.]]),2)
    points=np.array([[.2,.2,0],[9.2,.2,0],[5.,.2,1.]])
    vertices,faces=supported_plane_mesh(plane,points,cell_m=1.,tolerance_m=.01)
    assert vertices.shape==(8,3) and faces.shape==(4,3)
    assert not np.any((vertices[:,0]>1)&(vertices[:,0]<9))
    triangles=vertices[faces]
    assert np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2==pytest.approx(2.)
