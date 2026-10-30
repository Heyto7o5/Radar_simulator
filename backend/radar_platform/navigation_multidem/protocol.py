"""Geographic separation checks; this module never enters model inference."""
import math
from itertools import combinations


def tile_name(latitude, longitude, *, worldcover=False):
    step = 3 if worldcover else 1
    lat, lon = (math.floor(float(x) / step) * step for x in (latitude, longitude))
    ns = f'{"N" if lat >= 0 else "S"}{abs(lat):02d}'
    ew = f'{"E" if lon >= 0 else "W"}{abs(lon):03d}'
    return ns + ew if worldcover else f'Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM'


def distance_km(a, b):
    lat1, lat2 = map(math.radians, (a['latitude_deg'], b['latitude_deg']))
    dlat = lat2 - lat1
    dlon = math.radians(b['longitude_deg'] - a['longitude_deg'])
    q = math.sin(dlat / 2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon / 2)**2
    return 12742 * math.asin(min(1., math.sqrt(q)))


def validate_regions(plan):
    regions = plan['regions']
    if len({r['id'] for r in regions}) != len(regions):
        raise ValueError('duplicate region id')
    splits = {'train', 'val', 'test'}
    if {r['split'] for r in regions} != splits:
        raise ValueError('train, val and test geographic regions required')
    if any(sum(r['split'] == s for r in regions) < 2 for s in splits):
        raise ValueError('at least two independent regions per split required')
    for r in regions:
        if not r['id'].replace('_', '').isalnum():
            raise ValueError('region ids must be plain alphanumeric names')
        lat, lon = r['latitude_deg'], r['longitude_deg']
        if not math.isfinite(lat + lon) or not -90 < lat < 90 or not -180 <= lon < 180:
            raise ValueError('invalid geographic coordinates')
    for a, b in combinations(regions, 2):
        if distance_km(a, b) < plan['minimum_region_distance_km']:
            raise ValueError(f'regions too close: {a["id"]}, {b["id"]}')
        if a['split'] != b['split'] and tile_name(a['latitude_deg'], a['longitude_deg']) == tile_name(b['latitude_deg'], b['longitude_deg']):
            raise ValueError('source DEM tile shared across splits')
    return {s: [r['id'] for r in regions if r['split'] == s] for s in sorted(splits)}


def validate_records(records, plan):
    validate_regions(plan)
    regions = {r['id']: r['split'] for r in plan['regions']}
    ids = set()
    for record in records:
        if record['id'] in ids:
            raise ValueError('duplicate episode id')
        ids.add(record['id'])
        if record.get('region_id') not in regions or record['split'] != regions[record['region_id']]:
            raise ValueError('episode split differs from whole-region split')
