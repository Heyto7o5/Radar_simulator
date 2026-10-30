import numpy as np
import pytest
from radar_platform.lidar_surface import planar_patches


def test_plane_normals_mesh_area_and_no_radar_input():
    x,y=np.meshgrid(np.linspace(-1,1,31),np.linspace(-1,1,31))
    p=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    surface=planar_patches(p,spacing_m=.2,neighborhood_m=.3,observer_m=[0,0,2])
    assert len(surface.centers_m)>0
    assert np.allclose(surface.normals,[0,0,1])
    vertices,faces=surface.triangle_mesh()
    tri=vertices[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
    assert area==pytest.approx(len(surface.centers_m)*.2**2)
    assert np.all(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])[:,2]>0)


def test_line_is_not_a_verified_surface():
    p=np.column_stack([np.linspace(-1,1,100),np.zeros((100,2))])
    surface=planar_patches(p,spacing_m=.2,neighborhood_m=.3,observer_m=[0,0,2])
    assert len(surface.centers_m)==0
    assert surface.rejected_count==surface.candidate_count
    with pytest.raises(ValueError):planar_patches(p,spacing_m=-1,neighborhood_m=.3,observer_m=[0,0,2])
