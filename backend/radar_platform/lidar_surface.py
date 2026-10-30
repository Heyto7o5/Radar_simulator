"""Geometry-only local planar patches; not material or scattering ground truth."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SurfacePatches:
    centers_m: np.ndarray
    normals: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    width_m: float
    candidate_count: int
    rejected_count: int

    def triangle_mesh(self):
        """Disjoint square patches; no claim of watertight reconstruction."""
        offsets = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) * self.width_m/2
        vertices = (self.centers_m[:,None,:] + offsets[None,:,0,None]*self.tangent_u[:,None,:]
                    + offsets[None,:,1,None]*self.tangent_v[:,None,:]).reshape(-1,3)
        faces = (np.arange(len(self.centers_m))[:,None,None]*4 + np.array([[0,1,2],[0,2,3]])).reshape(-1,3)
        return vertices,faces


def planar_patches(points, *, spacing_m, neighborhood_m, observer_m,
                   max_normal_variance_ratio=.03, min_neighbors=8):
    """Fit normals from local PCA; orient toward observer, not global solid interior.

    Each occupied coarse cell contributes at most one patch. Patch area is a
    modeling choice, not measured illuminated area. Ambiguous/nonplanar cells
    are counted rather than silently relabeled as physical planes.
    """
    points=np.asarray(points,dtype=float);observer=np.asarray(observer_m,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<min_neighbors or not np.isfinite(points).all():
        raise ValueError('finite point cloud with sufficient points required')
    if observer.shape!=(3,) or not np.isfinite(observer).all():raise ValueError('invalid observer')
    if not 0<spacing_m<neighborhood_m or min_neighbors<3 or not 0<max_normal_variance_ratio<1/3:
        raise ValueError('invalid patch scale or planarity threshold')
    keys=np.floor(points/spacing_m).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    count=np.bincount(inverse)
    centers=np.column_stack([np.bincount(inverse,weights=points[:,i])/count for i in range(3)])
    tree=cKDTree(points)
    accepted=[];normals=[];us=[];vs=[]
    for center,indices in zip(centers,tree.query_ball_point(centers,neighborhood_m)):
        if len(indices)<min_neighbors:continue
        local=points[indices];origin=local.mean(axis=0)
        cov=(local-origin).T@(local-origin)/len(local)
        eigenvalues,eigenvectors=np.linalg.eigh(cov)
        if eigenvalues.sum()<=0 or eigenvalues[1]<=1e-10 or eigenvalues[0]/eigenvalues.sum()>max_normal_variance_ratio:continue
        normal=eigenvectors[:,0]
        if np.dot(normal,observer-center)<0:normal=-normal
        center=center-normal*np.dot(center-origin,normal)
        u=eigenvectors[:,2];v=np.cross(normal,u)
        accepted.append(center);normals.append(normal);us.append(u);vs.append(v)
    def array(values):return np.asarray(values,dtype=float).reshape(-1,3)
    return SurfacePatches(array(accepted),array(normals),array(us),array(vs),spacing_m,
                          len(centers),len(centers)-len(accepted))
