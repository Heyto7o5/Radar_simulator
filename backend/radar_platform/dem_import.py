"""Real GeoTIFF DSM -> local metric height grid, with explicit datum limitations."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class LocalDEM:
    north_m: np.ndarray
    east_m: np.ndarray
    elevation_up_m: np.ndarray
    metadata: dict

    def save(self,path):
        import json
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,north_m=self.north_m,east_m=self.east_m,elevation_up_m=self.elevation_up_m,
                            metadata_json=np.array(json.dumps(self.metadata)))


def import_copernicus_patch(path,latitude_deg,longitude_deg,*,width_m=1920.,spacing_m=30.):
    import rasterio
    from rasterio.windows import Window
    from pyproj import CRS,Transformer
    path=Path(path)
    if not np.isfinite([latitude_deg,longitude_deg,width_m,spacing_m]).all() or not -90<latitude_deg<90 or not -180<=longitude_deg<=180:
        raise ValueError('finite coordinates and grid scales required')
    if not 0<spacing_m<=width_m or width_m>20000:raise ValueError('patch must have positive spacing and width <=20 km')
    intervals=round(width_m/spacing_m)
    if not np.isclose(intervals*spacing_m,width_m):raise ValueError('width must be an integer number of grid intervals')
    if (intervals+1)**2>1_000_000:raise ValueError('DEM import exceeds one million grid nodes')
    axis=np.linspace(-width_m/2,width_m/2,intervals+1)
    north,east=np.meshgrid(axis,axis)
    local=CRS.from_proj4(f'+proj=aeqd +lat_0={latitude_deg} +lon_0={longitude_deg} +datum=WGS84 +units=m')
    to_geo=Transformer.from_crs(local,'EPSG:4326',always_xy=True)
    lon,lat=to_geo.transform(east,north)
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg()!=4326:raise ValueError('Copernicus input must declare EPSG:4326')
        if src.count!=1:raise ValueError('single-band elevation raster required')
        col,row=(~src.transform)*(lon,lat)
        # GDAL transform addresses pixel edges; center coordinates are offset by 0.5.
        col=np.asarray(col)-.5;row=np.asarray(row)-.5
        if row.min()<0 or col.min()<0 or row.max()>src.height-1 or col.max()>src.width-1:
            raise ValueError('patch exceeds available source tile; provide a real mosaic instead of padding missing terrain')
        r0=max(0,int(np.floor(row.min()))-1);r1=min(src.height,int(np.ceil(row.max()))+2)
        c0=max(0,int(np.floor(col.min()))-1);c1=min(src.width,int(np.ceil(col.max()))+2)
        data=src.read(1,window=Window(c0,r0,c1-c0,r1-r0),masked=True)
        if np.ma.getmaskarray(data).any() or not np.isfinite(data).all():
            raise ValueError('source window contains nodata; no silent hole filling')
        height=map_coordinates(np.asarray(data,dtype=float),[row-r0,col-c0],order=1,mode='nearest',prefilter=False)
        ccenter,rcenter=(~src.transform)*(longitude_deg,latitude_deg)
        origin_height=float(map_coordinates(np.asarray(data,dtype=float),[[rcenter-.5-r0],[ccenter-.5-c0]],order=1,prefilter=False)[0])
        tags=src.tags()
        metadata={'source_file':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_crs':str(src.crs),'source_transform':list(src.transform),'source_width':src.width,'source_height':src.height,
            'source_pixel_registration_tag':tags.get('AREA_OR_POINT'),
            'source_vertical_datum':'EGM2008 orthometric meters (Copernicus product specification)',
            'origin_latitude_deg':latitude_deg,'origin_longitude_deg':longitude_deg,'origin_orthometric_height_m':origin_height,
            'local_horizontal_projection':local.to_string(),'local_axis_order':'elevation[east,north]; NED z=-relative elevation',
            'height_conversion':'source orthometric height minus interpolated origin height; no vertical scaling',
            'spacing_m':spacing_m,'width_m':width_m,'resampling':'bilinear from native raster centers; not raster overviews',
            'orthometric_height_min_m':float(height.min()),'orthometric_height_max_m':float(height.max()),
            'local_earth_curvature_sag_estimate_m':(width_m/np.sqrt(2))**2/(2*6371000),
            'limits':['DSM contains vegetation/buildings, not bare-earth DTM',
                      'Local projected flat coordinates plus relative orthometric relief, not geoid-corrected ECEF/ENU',
                      'Resampled grid spacing is not source accuracy or independent measured resolution',
                      'No land-cover or radar scattering parameters inferred from elevation']}
    return LocalDEM(axis.copy(),axis.copy(),height-origin_height,metadata)
