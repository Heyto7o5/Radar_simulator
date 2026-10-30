import numpy as np
import pytest
from radar_platform.dem_los_reference import height_at,reference_los
from radar_platform.surface_scene import mesh_from_dem
from radar_platform.unified.visibility import blocked_segments
from radar_platform.unified.visibility_bvh import TriangleBVH


def test_independent_grid_reference_matches_exact_random_grid_visibility():
    rng=np.random.default_rng(42)
    north=np.linspace(-100,100,17);east=np.linspace(-100,100,15)
    h=rng.uniform(0,50,(15,17));mesh=mesh_from_dem(north,east,h)
    xy=rng.uniform(-90,90,(200,2));z=height_at(north,east,h,xy[:,0],xy[:,1])
    targets=np.c_[xy,-z];origin=np.array([-95.,-95.,-60.])
    reference=reference_los(north,east,h,origin,targets)
    assert all(r['covered'] for r in reference)
    expected=blocked_segments(origin,targets,mesh.triangles)
    mask=np.array([not r['near_tangent'] for r in reference])
    np.testing.assert_array_equal(expected[mask],np.array([r['blocked'] for r in reference])[mask])
    assert expected.any() and (~expected).any()


def test_bvh_matches_brute_force_without_changing_geometry():
    rng=np.random.default_rng(17)
    vertices=rng.normal(size=(100,3,3))*10
    origins=rng.normal(size=(100,3))*20;targets=rng.normal(size=(100,3))*20
    expected=blocked_segments(origins,targets,vertices)
    work={'aabb_tests':0,'triangle_tests':0}
    result=TriangleBVH(vertices).blocked(origins,targets,work=work)
    np.testing.assert_array_equal(result,expected)
    assert work['triangle_tests']>0
    with pytest.raises(ValueError,match='budget'):TriangleBVH(vertices).blocked(origins,targets,work_limit=1)


def test_reference_does_not_assume_outside_map_is_clear():
    r=reference_los([0,10],[0,10],[[0,0],[0,0]],[-1,5,-10],[[5,5,0]])
    assert r[0]['covered'] is False and r[0]['blocked'] is None


def test_geotiff_pixel_centers_units_and_nodata(tmp_path):
    rasterio=pytest.importorskip('rasterio');pytest.importorskip('pyproj')
    from rasterio.transform import from_origin
    from radar_platform.dem_import import import_copernicus_patch
    path=tmp_path/'source.tif';transform=from_origin(7.4,46.6,.001,.001)
    rr,cc=np.mgrid[:200,:200]
    lon=7.4+(cc+.5)*.001;lat=46.6-(rr+.5)*.001
    data=(1000+100*(lat-46.5)+200*(lon-7.5)).astype('float64')
    with rasterio.open(path,'w',driver='GTiff',width=200,height=200,count=1,dtype='float64',crs='EPSG:4326',transform=transform,nodata=-9999) as ds:ds.write(data,1)
    patch=import_copernicus_patch(path,46.5,7.5,width_m=600,spacing_m=30)
    assert patch.elevation_up_m.shape==(21,21)
    assert patch.metadata['origin_orthometric_height_m']==pytest.approx(1000,abs=1e-8)
    assert abs(patch.elevation_up_m[10,10])<1e-9
    assert patch.elevation_up_m[11,10]>0 and patch.elevation_up_m[10,11]>0
    data[99:102,99:102]=-9999
    with rasterio.open(path,'r+') as ds:ds.write(data,1)
    with pytest.raises(ValueError,match='nodata'):import_copernicus_patch(path,46.5,7.5,width_m=600,spacing_m=30)


def test_larger_real_grid_requires_explicit_opt_in():
    from radar_platform.unified.schema import Terrain,Scene
    from radar_platform.unified.scene import compile_scene
    fields=dict(kind='dem',north_m=np.arange(65).tolist(),east_m=np.arange(65).tolist(),elevation_up_m=np.zeros((65,65)).tolist())
    with pytest.raises(ValueError,match='max_grid_vertices'):Terrain(**fields)
    terrain=Terrain(**fields,max_grid_vertices=65536)
    with pytest.raises(ValueError,match='max_scatterers'):compile_scene(Scene(terrain=terrain))
    scene=compile_scene(Scene(terrain=terrain,max_scatterers=131072))
    assert len(scene.amplitudes)==8192
