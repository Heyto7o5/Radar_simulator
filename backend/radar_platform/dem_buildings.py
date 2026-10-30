"""Place axis-aligned synthetic landmarks on the existing triangulated DEM.

Heights are relative to the highest terrain point under the footprint. A solid
foundation extends to its lowest point; no terrain heights are edited. This is
a geometry preparation tool, not a calibrated building scattering model.
"""
import numpy as np
from pydantic import Field

from .unified.schema import StrictModel, Simulation, Box


class BuildingSpec(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    center_ne_m: tuple[float, float]
    footprint_ne_m: tuple[float, float]
    height_above_highest_ground_m: float = Field(gt=0)
    max_foundation_relief_m: float = Field(ge=0)
    relative_reflectivity: float = Field(gt=0)
    divisions: int = Field(2, ge=1, le=12)


def _clip(poly, axis, bound, sign):
    """Clip an N/E/elevation polygon, interpolating elevation on each edge."""
    result = []
    for previous, current in zip(poly[-1:] + poly[:-1], poly):
        p_in = sign * (previous[axis] - bound) >= 0
        c_in = sign * (current[axis] - bound) >= 0
        if p_in != c_in:
            t = (bound - previous[axis]) / (current[axis] - previous[axis])
            result.append(previous + t * (current - previous))
        if c_in:
            result.append(current)
    return result


def footprint_heights(terrain, center, size):
    """Exact extrema of piecewise-linear DEM over a rectangular footprint."""
    n, e, h = map(np.asarray, (terrain.north_m, terrain.east_m, terrain.elevation_up_m))
    center, size = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    if not np.isfinite(np.r_[center, size]).all() or np.any(size <= 0):
        raise ValueError('finite center and positive footprint dimensions required')
    lo, hi = center - size / 2, center + size / 2
    if np.any(lo < [n[0], e[0]]) or np.any(hi > [n[-1], e[-1]]):
        raise ValueError('building footprint extends outside DEM')
    values = []
    for j in np.flatnonzero((e[:-1] < hi[1]) & (e[1:] > lo[1])):
        for i in np.flatnonzero((n[:-1] < hi[0]) & (n[1:] > lo[0])):
            a = np.array([n[i], e[j], h[j, i]])
            b = np.array([n[i+1], e[j], h[j, i+1]])
            c = np.array([n[i], e[j+1], h[j+1, i]])
            d = np.array([n[i+1], e[j+1], h[j+1, i+1]])
            for triangle in ([a, b, d], [a, d, c]):
                poly = triangle
                for axis in (0, 1):
                    poly = _clip(poly, axis, lo[axis], 1)
                    poly = _clip(poly, axis, hi[axis], -1)
                values.extend(p[2] for p in poly)
    return float(min(values)), float(max(values))


def add_dem_buildings(config: Simulation, specs: list[BuildingSpec]):
    terrain = config.scene.terrain
    if terrain is None or terrain.kind != 'dem':
        raise ValueError('an explicit DEM scene is required')
    if not specs:
        raise ValueError('at least one building is required')
    boxes, report, footprints = [], [], []
    for spec in specs:
        low, high = footprint_heights(terrain, spec.center_ne_m, spec.footprint_ne_m)
        if high - low > spec.max_foundation_relief_m + 1e-9:
            raise ValueError(f'{spec.id}: foundation relief exceeds declared limit')
        center, size = np.array(spec.center_ne_m), np.array(spec.footprint_ne_m)
        lo, hi = center - size/2, center + size/2
        if any(np.all(hi > a) and np.all(b > lo) for a, b in footprints):
            raise ValueError('new building footprints overlap')
        footprints.append((lo, hi))
        roof = high + spec.height_above_highest_ground_m
        box = Box(id=spec.id, position_ned_m=(*spec.center_ne_m, -(low+roof)/2),
                  size_m=(*spec.footprint_ne_m, roof-low), divisions=spec.divisions,
                  relative_reflectivity=spec.relative_reflectivity)
        boxes.append(box.model_dump(mode='json'))
        report.append(dict(id=spec.id, ground_min_up_m=low, ground_max_up_m=high,
                           foundation_relief_m=high-low, roof_up_m=roof,
                           scattering='assumed relative diffuse centroid surface'))
    data = config.model_dump(mode='json')
    data['scene']['objects'].extend(boxes)
    data['visibility'].update(occlusion_enabled=True, report_coverage=True,
                              occluders='all_surfaces', accelerator='bvh')
    data['input_provenance'].append(
        'Synthetic axis-aligned buildings: roof above highest footprint terrain; '
        'foundation extends to lowest terrain. DEM unchanged. Relative scattering '
        'is assumed, not measured. All-surface occlusion enabled. Existing object '
        'intersections and geographic land use require separate inspection.')
    return Simulation.model_validate(data), report
