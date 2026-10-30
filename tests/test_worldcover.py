import numpy as np
import pytest
from radar_platform.worldcover import classes_on_dem_cells


def test_reprojection_preserves_categories_axis_order_and_mixtures(tmp_path):
    rasterio=pytest.importorskip('rasterio');pytest.importorskip('pyproj')
    from rasterio.transform import from_origin
    path=tmp_path/'classes.tif'
    data=np.zeros((100,100),np.uint8)
    data[:50,:50]=10;data[:50,50:]=80;data[50:,:50]=30;data[50:,50:]=50
    with rasterio.open(path,'w',driver='GTiff',width=100,height=100,count=1,dtype='uint8',crs='EPSG:4326',
                       transform=from_origin(7.495,46.505,.0001,.0001)) as f:f.write(data,1)
    meta={'local_horizontal_projection':'+proj=aeqd +lat_0=46.5 +lon_0=7.5 +datum=WGS84 +units=m'}
    dominant,purity,codes,fractions,m=classes_on_dem_cells(path,[-30,0,30],[-30,0,30],meta)
    np.testing.assert_array_equal(dominant,[[30,10],[50,80]])
    assert purity.min()>.95
    np.testing.assert_allclose(fractions.sum(axis=0),1,atol=1e-5)
    dominant,purity,*_=classes_on_dem_cells(path,[-30,30],[-30,30],meta)
    assert dominant[0,0] in [10,30,50,80]
    assert purity[0,0]==pytest.approx(.25,abs=.02)
