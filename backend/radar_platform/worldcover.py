"""WorldCover classes to local DEM cells; never interpolate numeric class IDs."""
from pathlib import Path
import hashlib
import numpy as np

CLASSES={0:'unknown',10:'tree_cover',20:'shrubland',30:'grassland',40:'cropland',50:'built_up',
         60:'bare_sparse',70:'snow_ice',80:'permanent_water',90:'herbaceous_wetland',95:'mangroves',100:'moss_lichen'}


def classes_on_dem_cells(path,north,east,dem_metadata):
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_origin
    from rasterio.warp import reproject,Resampling
    from pyproj import CRS,Transformer
    north,east=map(lambda x:np.asarray(x,float),(north,east))
    dn,de=np.diff(north),np.diff(east)
    if min(len(north),len(east))<2 or np.any(dn<=0) or np.any(de<=0) or not np.allclose(dn,dn[0]) or not np.allclose(de,de[0]):
        raise ValueError('uniform increasing metric DEM grid required')
    local=CRS.from_user_input(dem_metadata['local_horizontal_projection'])
    transform=Transformer.from_crs(local,'EPSG:4326',always_xy=True)
    n,e=np.meshgrid(north,east);lon,lat=transform.transform(e,n)
    path=Path(path)
    with rasterio.open(path) as source:
        if source.crs is None or source.crs.to_epsg()!=4326 or source.count!=1:raise ValueError('single-band EPSG:4326 WorldCover required')
        c,r=(~source.transform)*(lon,lat)
        r0=int(np.floor(r.min()))-2;r1=int(np.ceil(r.max()))+2
        c0=int(np.floor(c.min()))-2;c1=int(np.ceil(c.max()))+2
        if min(r0,c0)<0 or r1>source.height or c1>source.width:raise ValueError('classification tile does not cover DEM cells; no padding')
        window=Window(c0,r0,c1-c0,r1-r0);values=source.read(1,window=window)
        if not set(np.unique(values)).issubset(CLASSES):raise ValueError('unrecognized WorldCover code; not silently substituted')
        src_transform=source.window_transform(window)
        dst_transform=from_origin(east[0],north[-1],de[0],dn[0])
        codes=np.array(sorted(CLASSES),dtype=np.uint8)
        fractions=np.zeros((len(codes),len(north)-1,len(east)-1),np.float32)
        for i,code in enumerate(codes):
            reproject((values==code).astype(np.float32),fractions[i],src_transform=src_transform,
                      src_crs=source.crs,dst_transform=dst_transform,dst_crs=local.to_wkt(),
                      resampling=Resampling.average,num_threads=1)
    if not np.allclose(fractions.sum(axis=0),1,atol=2e-5):raise ValueError('class fractions do not cover output cells')
    # GDAL raster [north descending,east ascending] -> platform [east,north ascending].
    fractions=fractions.transpose(0,2,1)[:,:,::-1].copy()
    dominant=codes[np.argmax(fractions,axis=0)];purity=fractions.max(axis=0)
    sorted_f=np.sort(fractions,axis=0)
    metadata={'source_file':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'dataset':'ESA WorldCover 2021 v200','source':'https://esa-worldcover.org/en/data-access',
        'class_names':CLASSES,'output_axes':['east_cell','north_cell'],
        'method':'average individual one-hot class masks into metric DEM cells; choose largest area fraction',
        'tie_rule':'lowest numeric code when fractions are exactly equal',
        'near_tie_cells':int((sorted_f[-1]-sorted_f[-2]<=1e-6).sum()),
        'mixed_cells_below_80_percent':int((purity<.8).sum()),'total_cells':int(purity.size),
        'dominant_class_counts':{str(int(k)):int((dominant==k).sum()) for k in np.unique(dominant)},
        'license':'CC-BY-4.0','attribution':'© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium',
        'limits':['Remote-sensing classification, not local field verification',
                  'Classification year 2021 differs from the DEM acquisition epoch',
                  'Mixed cells retain fractions, but current echo model uses the dominant class',
                  'No dielectric, roughness, moisture or radar NRCS values supplied by this map']}
    return dominant,purity,codes,fractions,metadata
