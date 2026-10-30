import numpy as np
import pytest
from fastapi.testclient import TestClient

from radar_platform.surface_scene import TriangleMesh,mesh_from_dem,mesh_from_obj,mesh_from_legacy_dem_mat
from radar_platform.surface_echo import SurfaceRadar,specular_paths,simulate_surface_spectrum,range_response
from radar_platform.models import C0


def plane():
    return TriangleMesh(np.array([[1.,-2.,-2.],[1.,2.,-2.],[1.,0.,2.]]),np.array([[0,1,2]]))


def test_dem_ned_heights_area_and_obj_scale(tmp_path):
    mesh=mesh_from_dem([0,2],[0,3],[[5,5],[5,5]])
    assert len(mesh.faces)==2
    assert np.all(mesh.vertices_m[:,2]==-5)
    assert mesh.areas_m2.sum()==pytest.approx(6)
    obj=tmp_path/"plate.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
    transform=np.diag([2.,2.,2.,1.]);transform[:3,3]=[5,6,7]
    imported=mesh_from_obj(obj,transform)
    assert imported.areas_m2[0]==pytest.approx(2.)
    np.testing.assert_allclose(imported.vertices_m[0],[5,6,7])


def test_plane_specular_distance_and_finite_aperture():
    tx=np.array([[0.,0.,0.],[0.,10.,0.]])
    lengths,valid=specular_paths(plane(),tx,tx,(0,))
    assert valid.tolist()==[True,False]
    assert lengths[0]==pytest.approx(2.)


def test_plane_occlusion_blocks_specular_path():
    mesh=plane()
    blocker=mesh.vertices_m.copy(); blocker[:,0]=.5
    combined=TriangleMesh(np.r_[mesh.vertices_m,blocker],np.array([[0,1,2],[3,4,5]]))
    _,valid=specular_paths(combined,np.zeros((1,3)),np.zeros((1,3)),(0,))
    assert not valid[0]


def test_two_orthogonal_reflections_and_reciprocity():
    # Two large vertical mirrors x=0 and y=0. Unfolded Tx is (-1,-2,1).
    mesh=TriangleMesh(np.array([[0,0,-10],[0,20,-10],[0,0,20],
                                [0,0,-10],[20,0,-10],[0,0,20]],float),np.array([[0,1,2],[3,4,5]]))
    tx=np.array([[1.,2.,1.]]);rx=np.array([[2.,1.,1.]])
    length,valid=specular_paths(mesh,tx,rx,(0,1))
    reverse,reverse_valid=specular_paths(mesh,rx,tx,(1,0))
    assert valid[0] and reverse_valid[0]
    assert length[0]==pytest.approx(np.sqrt(18))
    np.testing.assert_allclose(length,reverse)


def test_three_orthogonal_reflections():
    v=np.array([[0,0,0],[10,0,0],[0,10,0],[0,0,10]],float)
    mesh=TriangleMesh(v,np.array([[0,2,3],[0,1,3],[0,1,2]]))
    tx=np.array([[1.,2.,3.]]);rx=np.array([[3.,2.,1.]])
    # Image method gives (-1,-2,-3) and total length sqrt(48).
    L,valid=specular_paths(mesh,tx,rx,(0,1,2))
    assert valid[0]
    assert L[0]==pytest.approx(np.sqrt(48))


def test_spectrum_phase_matches_analytic_plane():
    f=np.linspace(8e9,10e9,32)
    radar=SurfaceRadar(f,[[0,0,0]],[[0,0,0]],1)
    result=simulate_surface_spectrum(plane(),radar)
    expected=-np.exp(-2j*np.pi*f*2/C0)/2
    np.testing.assert_allclose(result.signal[0,0],expected,rtol=1e-6,atol=1e-7)
    assert result.path_counts[1][0,0]==1
    np.testing.assert_array_equal(result.signal,simulate_surface_spectrum(plane(),radar).signal)


def test_range_transform_uses_frequency_spacing_not_zero_padding_resolution():
    f=np.linspace(72e9,82e9,128); r0=.4
    signal=np.exp(-4j*np.pi*f*r0/C0)
    response,ranges=range_response(signal,f,512)
    assert abs(ranges[np.argmax(abs(response))]-r0)<=ranges[1]
    assert C0/(2*(f[-1]-f[0]))==pytest.approx(.0149896229)
    with pytest.raises(ValueError,match="uniform"):
        range_response(signal,f**1.1,512)


def test_invalid_inputs():
    with pytest.raises(ValueError,match="degenerate"):
        TriangleMesh(np.zeros((3,3)),np.array([[0,1,2]]))
    with pytest.raises(ValueError,match="increasing"):
        SurfaceRadar([2e9,1e9],[[0,0,0]],[[0,0,0]])
    with pytest.raises(ValueError,match="shape"):
        mesh_from_dem([0,1],[0,1],[[0]])


def test_dem_shared_edge_does_not_duplicate_specular_path():
    mesh=mesh_from_dem([0,1],[0,1],[[0,0],[0,0]])
    radar=SurfaceRadar([1e9,2e9],[[.2,.2,-1]],[[.2,.2,-1]],1)
    result=simulate_surface_spectrum(mesh,radar)
    assert result.path_counts[1][0,0]==1
    np.testing.assert_allclose(abs(result.signal),.5,rtol=1e-6)


def test_legacy_dem_mat_roundtrip(tmp_path):
    from scipy.io import savemat
    path=tmp_path/'terrain.mat'
    savemat(path,{'z':np.full((2,2),7.),'x0':10.,'y0':20.,'resolution':2.,'normal_vector':np.zeros((2,2,3))})
    mesh=mesh_from_legacy_dem_mat(path)
    np.testing.assert_allclose(mesh.vertices_m[0],[10,20,-7])
    assert mesh.areas_m2.sum()==pytest.approx(4.)


def test_gain_calibration_does_not_see_evaluation_values():
    from radar_platform.validation_metrics import fit_complex_gain
    s=np.array([[1,2],[3,4]],dtype=complex)
    m=(2+3j)*s;mask=np.array([True,False])
    first=fit_complex_gain(s,m,mask)
    m[1]=10000-40000j
    second=fit_complex_gain(s,m,mask)
    assert first==pytest.approx(2+3j)
    assert second==first


def test_gain_metrics_are_sensitive_to_phase_not_only_peak():
    from radar_platform.validation_metrics import complex_metrics
    m=np.ones((2,4),dtype=complex);s=m.copy();s[1]=-1
    score=complex_metrics(s,m,np.array([True,True]))
    assert score['complex_coherence']==0
    assert score['complex_nmse']==pytest.approx(2.)


def test_surface_api_roundtrip_and_dem(monkeypatch,tmp_path):
    from radar_platform.api import app
    monkeypatch.setenv("RADAR_RUNTIME_ROOT",str(tmp_path))
    client=TestClient(app)
    f=np.linspace(72e9,82e9,16).tolist()
    payload={"frequencies_hz":f,"tx_ned_m":[[0,0,0]],"rx_ned_m":[[0,0,0]],
             "vertices_ned_m":plane().vertices_m.tolist(),"faces":[[0,1,2]],"max_bounces":1}
    response=client.post('/api/v1/surface-simulations',json=payload)
    assert response.status_code==200
    meta=response.json(); data=client.get(meta['spectrum_url'])
    assert data.status_code==200 and data.headers['X-Shape']=='1,1,16'
    observed=np.frombuffer(data.content,dtype='<c8')
    np.testing.assert_allclose(observed,-np.exp(-4j*np.pi*np.array(f)/C0)/2,rtol=1e-6)
    assert not meta['has_doppler_axis']
    assert client.get('/api/v1/surface-simulations/not-a-uuid/spectrum').status_code==404
    payload['frequencies_hz']=[5,4]
    assert client.post('/api/v1/surface-simulations',json=payload).status_code==422
    dem={"frequencies_hz":f,"tx_ned_m":[[.2,.2,-1]],"rx_ned_m":[[.2,.2,-1]],
         "dem_north_m":[0,1],"dem_east_m":[0,1],"dem_elevation_up_m":[[0,0],[0,0]],"max_bounces":1}
    assert client.post('/api/v1/surface-simulations',json=dem).status_code==200
