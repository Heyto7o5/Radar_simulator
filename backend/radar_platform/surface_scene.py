"""Geometry inputs shared by DEM and imported finite triangular reflectors.

Coordinates and lengths are SI, NED.  Unlike a height field, TriangleMesh can
represent vertical walls and the three faces of a trihedral reflector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TriangleMesh:
    vertices_m: np.ndarray
    faces: np.ndarray
    name: str = "surface"

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        source_faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("vertices must be finite [vertex,3] coordinates in metres")
        if source_faces.ndim != 2 or source_faces.shape[1] != 3 or not len(source_faces):
            raise ValueError("faces must be a non-empty [face,3] array")
        if not np.issubdtype(source_faces.dtype, np.integer):
            raise ValueError("face indices must be integers")
        faces = source_faces.astype(np.int64)
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("face index outside vertex array")
        tri = vertices[faces]
        if np.any(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) < 1e-14):
            raise ValueError("degenerate triangle")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices_m[self.faces]

    @property
    def normals(self) -> np.ndarray:
        t = self.triangles
        normals = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
        return normals / np.linalg.norm(normals, axis=1)[:, None]

    @property
    def areas_m2(self) -> np.ndarray:
        t = self.triangles
        return np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1) / 2


def mesh_from_obj(path: str | Path, transform: np.ndarray | None = None) -> TriangleMesh:
    """Read positions and polygon faces; triangulate polygons without eval/pickle."""
    vertices, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "v":
            if len(parts) < 4:
                raise ValueError("OBJ vertex needs three coordinates")
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            if len(parts) < 4:
                raise ValueError("OBJ face needs at least three vertices")
            indices = [int(x.split("/")[0]) for x in parts[1:]]
            if 0 in indices:
                raise ValueError("OBJ indices are one-based; zero is invalid")
            indices = [i-1 if i > 0 else len(vertices)+i for i in indices]
            faces.extend((indices[0], indices[i], indices[i+1]) for i in range(1, len(indices)-1))
    v = np.asarray(vertices, dtype=np.float64)
    if transform is not None:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("transform must be a finite affine 4x4 matrix")
        # Permit similarity scale: the source optical mesh is not in metres.
        v = (np.c_[v, np.ones(len(v))] @ matrix.T)[:, :3]
    return TriangleMesh(v, np.asarray(faces, dtype=np.int64), Path(path).stem)


def mesh_from_dem(north_m: np.ndarray, east_m: np.ndarray, elevation_up_m: np.ndarray) -> TriangleMesh:
    """Convert a DEM [east,north] to two triangles per cell in NED."""
    north = np.asarray(north_m, dtype=float)
    east = np.asarray(east_m, dtype=float)
    z = np.asarray(elevation_up_m, dtype=float)
    if north.ndim != 1 or east.ndim != 1 or min(len(north),len(east)) < 2:
        raise ValueError("DEM needs at least two samples on each horizontal axis")
    if not np.all(np.diff(north) > 0) or not np.all(np.diff(east) > 0):
        raise ValueError("DEM horizontal axes must be strictly increasing")
    if z.shape != (len(east), len(north)):
        raise ValueError("DEM elevation must have shape [east,north]")
    n, e = np.meshgrid(north, east)
    vertices = np.column_stack((n.ravel(), e.ravel(), -z.ravel()))
    faces = []
    for j in range(len(east)-1):
        for i in range(len(north)-1):
            a=j*len(north)+i; b=a+1; c=a+len(north); d=c+1
            faces.extend(((a,b,d),(a,d,c)))
    return TriangleMesh(vertices, np.asarray(faces), "DEM")


def mesh_from_legacy_dem_mat(path: str | Path) -> TriangleMesh:
    """Read old TerrainGenerator MAT: x0,y0,resolution,z (z positive up).

    Explicit convention: legacy x -> North, y -> East. A georeferenced DEM
    in other axes must be projected/reoriented before calling this adapter.
    Stored old normals are ignored; face normals follow the actual geometry.
    """
    from scipy.io import loadmat
    raw=loadmat(path)
    for key in ("x0","y0","resolution","z"):
        if key not in raw: raise ValueError(f"missing legacy DEM field: {key}")
    z=np.asarray(raw['z'],dtype=float)
    if z.ndim!=2:raise ValueError("legacy DEM z must be 2-D")
    x0=float(np.asarray(raw['x0']).item());y0=float(np.asarray(raw['y0']).item())
    step=float(np.asarray(raw['resolution']).item())
    if not np.isfinite(step) or step<=0:raise ValueError('DEM resolution must be positive')
    return mesh_from_dem(x0+np.arange(z.shape[1])*step,y0+np.arange(z.shape[0])*step,z)


def points_in_triangle(points: np.ndarray, triangle: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    a, b, c = triangle
    u, v, w = b-a, c-a, points-a
    uu,uv,vv = u@u,u@v,v@v
    wu,wv = w@u,w@v
    denominator = uu*vv-uv*uv
    if denominator < 1e-28:
        raise ValueError("degenerate triangle")
    s=(vv*wu-uv*wv)/denominator; t=(uu*wv-uv*wu)/denominator
    return (s >= -tol) & (t >= -tol) & (s+t <= 1+tol)
