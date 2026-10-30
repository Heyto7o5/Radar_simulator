"""Seeded multi-plane geometry fitting; independent of radar intensities."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FittedPlane:
    center: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    bounds_uv: np.ndarray
    inlier_count: int

    def finite_distance(self, points):
        """Distance to enclosing rectangle, not an infinite-plane extrapolation.

        Rectangle can fill geometry holes: this is a coverage diagnostic only,
        not an assertion that every point of the rectangle is a real surface.
        """
        delta=np.asarray(points)-self.center
        uv=np.column_stack([delta@self.tangent_u,delta@self.tangent_v])
        outside=np.maximum(self.bounds_uv[0]-uv,0)+np.maximum(uv-self.bounds_uv[1],0)
        return np.sqrt((delta@self.normal)**2+np.sum(outside**2,axis=1))


def extract_planes(points, *, tolerance_m, max_planes=8, trials=256, min_inliers=100, seed=42):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('finite Nx3 points required')
    if not np.isfinite(tolerance_m) or tolerance_m<=0 or max_planes<1 or trials<1 or min_inliers<3:
        raise ValueError('invalid extraction parameters')
    rng=np.random.default_rng(seed);remaining=points.copy();planes=[]
    for _ in range(max_planes):
        if len(remaining)<min_inliers:break
        probe=remaining[rng.choice(len(remaining),min(2048,len(remaining)),replace=False)]
        best=None;best_count=0
        for _ in range(trials):
            a,b,c=remaining[rng.choice(len(remaining),3,replace=False)]
            normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
            if length<1e-10:continue
            normal/=length
            count=int(np.sum(abs((probe-a)@normal)<=tolerance_m))
            if count>best_count:best=(a,normal);best_count=count
        if best is None:break
        center,normal=best
        keep=abs((remaining-center)@normal)<=tolerance_m
        if keep.sum()<min_inliers:break
        for _ in range(3):
            subset=remaining[keep];center=subset.mean(axis=0)
            _,_,vh=np.linalg.svd(subset-center,full_matrices=False)
            normal=vh[-1];keep=abs((remaining-center)@normal)<=tolerance_m
            if keep.sum()<min_inliers:break
        if keep.sum()<min_inliers:break
        # Do not pretend collinear data determine a unique surface.
        subset=remaining[keep];center=subset.mean(axis=0)
        _,s,vh=np.linalg.svd(subset-center,full_matrices=False)
        if s[1]<1e-8:break
        normal=vh[-1];u=vh[0];v=np.cross(normal,u)
        uv=np.column_stack([(subset-center)@u,(subset-center)@v])
        planes.append(FittedPlane(center,normal,u,v,np.stack([uv.min(axis=0),uv.max(axis=0)]),len(subset)))
        remaining=remaining[~keep]
    return planes


def nearest_plane_distance(points,planes):
    if not planes:return np.full(len(points),np.inf)
    return np.min(np.stack([plane.finite_distance(points) for plane in planes]),axis=0)


def supported_plane_mesh(plane, support_points, *, cell_m, tolerance_m):
    """Only tile projected cells touched by observed support; do not fill bounds.

    Each touched cell is still a finite-resolution area assumption. Empty cells
    remain holes, and disconnected components remain disconnected.
    """
    points=np.asarray(support_points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('finite Nx3 support required')
    if not np.isfinite([cell_m,tolerance_m]).all() or cell_m<=0 or tolerance_m<=0:raise ValueError('positive finite support scales required')
    delta=points-plane.center
    delta=delta[abs(delta@plane.normal)<=tolerance_m]
    uv=np.column_stack([delta@plane.tangent_u,delta@plane.tangent_v])
    cells=np.unique(np.floor(uv/cell_m).astype(np.int64),axis=0)
    corners=(cells[:,None,:]+np.array([[0,0],[1,0],[1,1],[0,1]]))*cell_m
    vertices=(plane.center+corners[:,:,0,None]*plane.tangent_u+corners[:,:,1,None]*plane.tangent_v).reshape(-1,3)
    faces=(np.arange(len(cells))[:,None,None]*4+np.array([[0,1,2],[0,2,3]])).reshape(-1,3)
    return vertices,faces
