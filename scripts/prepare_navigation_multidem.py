#!/usr/bin/env python3
"""Acquire small public source windows inside the project and freeze geography."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os

import httpx
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window

from radar_platform.dem_import import import_copernicus_patch
from radar_platform.worldcover import classes_on_dem_cells, CLASSES
from radar_platform.navigation.io import local_path, read_json, write_json, digest
from radar_platform.navigation_multidem.protocol import tile_name, validate_regions


def download_window(url, target, region, width_m):
    target = local_path(target)
    metadata_path = target.with_suffix('.json')
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata['url'] != url or metadata['region'] != region or metadata['width_m'] != width_m:
            raise ValueError('cached source window request differs')
        if hashlib.sha256(target.read_bytes()).hexdigest() != metadata['crop_sha256']:
            raise ValueError('cached source window hash mismatch')
        return metadata
    target.parent.mkdir(parents=True, exist_ok=True)
    lat, lon = region['latitude_deg'], region['longitude_deg']
    local = CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    to_geo = Transformer.from_crs(local, 'EPSG:4326', always_xy=True)
    extent = width_m/2 + 150
    xx, yy = np.meshgrid(np.linspace(-extent, extent, 9), np.linspace(-extent, extent, 9))
    longitude, latitude = to_geo.transform(xx, yy)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.head(url)
        response.raise_for_status()
        etag = response.headers.get('etag')
        source_size = response.headers.get('content-length')
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT='30', GDAL_HTTP_MAX_RETRY='1'):
        with rasterio.open(url) as source:
            cols, rows = (~source.transform)*(longitude, latitude)
            c0, r0 = int(np.floor(cols.min()))-3, int(np.floor(rows.min()))-3
            c1, r1 = int(np.ceil(cols.max()))+3, int(np.ceil(rows.max()))+3
            if c0 < 0 or r0 < 0 or c1 > source.width or r1 > source.height:
                raise ValueError('requested source window crosses tile boundary')
            window = Window(c0, r0, c1-c0, r1-r0)
            values = source.read(1, window=window, masked=True)
            if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
                raise ValueError('source window has nodata; no silent filling')
            profile = dict(driver='GTiff', height=r1-r0, width=c1-c0, count=1,
                           dtype=source.dtypes[0], crs=source.crs, transform=source.window_transform(window),
                           nodata=source.nodata, compress='deflate')
            temporary = target.with_suffix('.tmp.tif')
            with rasterio.open(temporary, 'w', **profile) as output:
                output.write(np.asarray(values), 1)
                output.update_tags(**source.tags())
            temporary.replace(target)
            metadata = dict(url=url, region=region, width_m=width_m, source_etag=etag,
                            source_content_length=source_size, source_window=[c0, r0, c1-c0, r1-r0],
                            source_shape=[source.height, source.width], native_pixel_size_deg=list(source.res),
                            crop_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            accessed_utc=datetime.now(timezone.utc).isoformat(),
                            operation='Lossless pixel window from native COG; no overview/resampling; hash covers saved crop, not full remote tile')
    write_json(metadata_path, metadata)
    return metadata


def prepare_region(plan, region, root):
    output = root/'regions'/region['id']
    signature = digest(dict(plan=plan, region=region))
    if (output/'complete.json').exists():
        result = read_json(output/'complete.json')
        if result['request_sha256'] != signature:
            raise ValueError('existing region differs; use a new output')
        for name, expected in result['file_sha256'].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'region artifact changed: {name}')
        return result
    dem_tile = tile_name(region['latitude_deg'], region['longitude_deg'])
    cover_tile = tile_name(region['latitude_deg'], region['longitude_deg'], worldcover=True)
    dem_url = f'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{dem_tile}/{dem_tile}.tif'
    cover_url = f'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{cover_tile}_Map.tif'
    dem_path, cover_path = output/'source_dem.tif', output/'source_worldcover.tif'
    width = region.get('selection_width_m', plan['selection_width_m'])
    sources = [download_window(url, path, region, width) for url, path in [(dem_url, dem_path), (cover_url, cover_path)]]
    dem = import_copernicus_patch(dem_path, region['latitude_deg'], region['longitude_deg'],
                                 width_m=width, spacing_m=plan['measured_grid_m'])
    dominant, purity, codes, fractions, cover_meta = classes_on_dem_cells(cover_path, dem.north_m, dem.east_m, dem.metadata)
    cover = np.vectorize(CLASSES.__getitem__)(dominant)
    candidates = []
    half = plan['context_width_m']/2
    for j in range(cover.shape[0]-3):
        for i in range(cover.shape[1]-3):
            n, e = dem.north_m[i+2], dem.east_m[j+2]
            if min(n-dem.north_m[0], dem.north_m[-1]-n, e-dem.east_m[0], dem.east_m[-1]-e) < half:
                continue
            if np.all(cover[j:j+4, i:i+4] == 'cropland'):
                candidates.append((float(n*n+e*e), j, i))
    if not candidates:
        raise ValueError(f'{region["id"]}: no eligible real cropland window; region not silently substituted')
    _, j, i = min(candidates)
    parent = {'scene': {'terrain': {'kind': 'dem', 'north_m': dem.north_m.tolist(),
                'east_m': dem.east_m.tolist(), 'elevation_up_m': dem.elevation_up_m.tolist(),
                'land_cover_map': cover.tolist()}},
              'input_provenance': [f'Copernicus GLO-30 DSM source: {dem_url}',
                f'Native source crop SHA256={sources[0]["crop_sha256"]}',
                'DSM, not bare-earth DTM; no relief scaling; local 30m sampling does not add measured detail.',
                f'ESA WorldCover 2021 v200: {cover_url}; classifications are not field verification.']}
    write_json(output/'parent.json', parent)
    write_json(output/'provenance.json', dict(region=region, sources=sources, dem=dem.metadata, land_cover=cover_meta,
        selected_cell_row=j, selected_cell_col=i, selected_origin_ne_m=[float(dem.north_m[i+2]), float(dem.east_m[j+2])],
        selection=plan['selection_rule'], candidate_windows=len(candidates),
        attribution=['produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved',
                     '© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium']))
    np.savez_compressed(output/'land_cover.npz', dominant=dominant, purity=purity, codes=codes, fractions=fractions)
    names = ['source_dem.tif', 'source_worldcover.tif', 'parent.json', 'provenance.json', 'land_cover.npz']
    result = dict(region_id=region['id'], split=region['split'], request_sha256=signature, dem_tile=dem_tile,
                  selected_cell_row=j, selected_cell_col=i, candidate_windows=len(candidates),
                  file_sha256={name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in names})
    write_json(output/'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/navigation_multidem_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    plan, root = read_json(args.plan), local_path(args.output)
    groups = validate_regions(plan)
    if (root/'plan.json').exists() and read_json(root/'plan.json') != plan:
        raise ValueError('frozen geographic plan changed')
    write_json(root/'plan.json', plan)
    write_json(root/'status.json', dict(status='preparing', pid=os.getpid(), groups=groups))
    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prepare_region, plan, region, root): region['id'] for region in plan['regions']}
        for future in as_completed(futures):
            try:
                result = future.result(); completed.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                errors.append(dict(region_id=futures[future], error=str(exc)))
                print(json.dumps(errors[-1]), flush=True)
            write_json(root/'status.json', dict(status='preparing', pid=os.getpid(), completed=completed, errors=errors))
    write_json(root/'status.json', dict(status='complete' if not errors else 'failed', groups=groups, completed=completed, errors=errors))
    if errors:
        raise SystemExit('One or more declared regions failed; inspect status.json. No substitutions made.')


if __name__ == '__main__':
    main()
