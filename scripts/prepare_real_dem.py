#!/usr/bin/env python3
"""Prepare every predeclared real DSM patch, retaining native source provenance."""
import argparse,json,os
from pathlib import Path
from radar_platform.dem_import import import_copernicus_patch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/real_dem_validation_plan.json')
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    data=Path(os.environ.get('RADAR_DATA_ROOT','/nas/datasets/hyt/datasets/radar_open'))/'copernicus_glo30'
    source=data/'raw'/plan['tile']/(plan['tile']+'.tif')
    destination=data/'processed'/'fixed_patches_20261022';destination.mkdir(parents=True,exist_ok=True)
    for location in plan['locations']:
        for spacing in plan['comparison_grid_spacing_m']:
            path=destination/f'{location["id"]}_{spacing}m.npz'
            if path.exists():raise FileExistsError('Refusing to replace previously prepared patch: '+str(path))
            patch=import_copernicus_patch(source,location['latitude_deg'],location['longitude_deg'],width_m=plan['patch_width_m'],spacing_m=spacing)
            patch.metadata['selection_plan']=plan;patch.save(path)
            print(json.dumps({'patch':str(path),'shape':patch.elevation_up_m.shape,
                              'height_range_m':[patch.metadata['orthometric_height_min_m'],patch.metadata['orthometric_height_max_m']],
                              'no_height_scaling':True}),flush=True)


if __name__=='__main__':main()
