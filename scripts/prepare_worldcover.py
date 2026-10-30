#!/usr/bin/env python3
"""Co-register real public classes to the already fixed real DEM grids."""
import json,os
from pathlib import Path
import numpy as np
from radar_platform.worldcover import classes_on_dem_cells


def main():
    data=Path(os.environ.get('RADAR_DATA_ROOT','/nas/datasets/hyt/datasets/radar_open'))
    source=data/'worldcover_2021/raw/ESA_WorldCover_10m_2021_v200_N45E006_Map.tif'
    out=data/'worldcover_2021/processed/fixed_dem_cells_20261022';out.mkdir(parents=True,exist_ok=True)
    for site in ['alpine','valley','foothill']:
        path=out/f'{site}_30m_cells.npz'
        if path.exists():raise FileExistsError('existing class grid preserved: '+str(path))
        dem=data/f'copernicus_glo30/processed/fixed_context_20261022/{site}_30m_context.npz'
        with np.load(dem,allow_pickle=False) as f:
            result=classes_on_dem_cells(source,f['north_m'],f['east_m'],json.loads(str(f['metadata_json'])))
        dominant,purity,codes,fractions,meta=result
        np.savez_compressed(path,dominant_code=dominant,dominant_fraction=purity,class_codes=codes,
                            class_fractions=fractions,metadata_json=np.array(json.dumps(meta)))
        print(json.dumps({'site':site,'class_counts':meta['dominant_class_counts'],
                          'mixed_cells':meta['mixed_cells_below_80_percent'],'total':meta['total_cells']}),flush=True)


if __name__=='__main__':main()
