"""Opaque triangle line-of-sight blocking; no reflection, diffraction or penetration.

Rays are evaluated at each Tx slot. Endpoint tolerance is in meters, not a
fraction of slant range; a surface does not block its own endpoint reflection.
"""
import numpy as np


def blocked_segments(origins, endpoints, triangles, *, endpoint_epsilon_m=1e-4):
    origins,endpoints,triangles=map(lambda x:np.asarray(x,dtype=float),(origins,endpoints,triangles))
    if endpoints.ndim!=2 or endpoints.shape[1]!=3 or triangles.ndim!=3 or triangles.shape[1:]!=(3,3):
        raise ValueError('endpoints [N,3] and triangles [M,3,3] required')
    origins=np.broadcast_to(origins,endpoints.shape)
    if not all(np.isfinite(x).all() for x in (origins,endpoints,triangles)) or endpoint_epsilon_m<=0:
        raise ValueError('finite geometry and positive endpoint tolerance required')
    result=np.zeros(len(endpoints),dtype=bool)
    for start in range(0,len(endpoints),128):
        o=origins[start:start+128];direction=endpoints[start:start+128]-o
        lengths=np.linalg.norm(direction,axis=1)
        if np.any(lengths<=2*endpoint_epsilon_m):raise ValueError('ray too short for endpoint tolerance')
        epsilon=endpoint_epsilon_m/lengths
        hit=np.zeros(len(o),dtype=bool)
        for first in range(0,len(triangles),256):
            tri=triangles[first:first+256];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
            cross=np.cross(direction[:,None,:],e2[None,:,:])
            det=np.sum(e1[None,:,:]*cross,axis=-1)
            scale=lengths[:,None]*np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)
            nonparallel=abs(det)>1e-12*scale
            inv=np.divide(1.,det,out=np.zeros_like(det),where=nonparallel)
            rel=o[:,None,:]-tri[None,:,0,:]
            u=np.sum(rel*cross,axis=-1)*inv
            q=np.cross(rel,e1[None,:,:])
            v=np.sum(direction[:,None,:]*q,axis=-1)*inv
            t=np.sum(e2[None,:,:]*q,axis=-1)*inv
            valid=nonparallel&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)
            valid&=(t>epsilon[:,None])&(t<1-epsilon[:,None])
            hit|=valid.any(axis=1)
            if hit.all():break
        result[start:start+len(o)]=hit
    return result


class VisibilityEvaluator:
    def __init__(self,scene,settings):
        self.settings=settings
        triangles=scene.occluder_triangles_m
        if triangles is None:triangles=np.empty((0,3,3))
        keep=np.ones(len(triangles),bool)
        if settings.occluders=='terrain':
            keep=scene.occluder_is_terrain if scene.occluder_is_terrain is not None else np.zeros(len(triangles),bool)
        self.triangles=triangles[keep]
        self.velocities=scene.occluder_velocities_mps[keep] if len(triangles) else np.empty((0,3))
        self.moving=bool(np.any(self.velocities))
        self.work={'aabb_tests':0,'triangle_tests':0}
        self.bvh=None
        if settings.occlusion_enabled and settings.accelerator=='bvh' and not self.moving:
            from .visibility_bvh import TriangleBVH
            self.bvh=TriangleBVH(self.triangles)
        self._compiled_tree=None

    def clear(self,origins,endpoints,time_s):
        if not self.settings.occlusion_enabled or not len(self.triangles):return np.ones(len(endpoints),bool)
        triangles=self.triangles+self.velocities[:,None,:]*time_s if self.moving else self.triangles
        if self.settings.accelerator=='bvh':
            from .visibility_bvh import TriangleBVH
            tree=self.bvh if self.bvh is not None else TriangleBVH(triangles)
            if self.settings.bvh_kernel=='numba':
                from .visibility_bvh_numba import CompiledBVH
                if self._compiled_tree is None or self.moving:
                    self._compiled_tree=CompiledBVH(tree)
                return ~self._compiled_tree.blocked(origins,endpoints,
                    endpoint_epsilon_m=self.settings.endpoint_epsilon_m,
                    work=self.work,work_limit=self.settings.max_ray_triangle_tests)
            return ~tree.blocked(origins,endpoints,endpoint_epsilon_m=self.settings.endpoint_epsilon_m,
                                 work=self.work,work_limit=self.settings.max_ray_triangle_tests)
        return ~blocked_segments(origins,endpoints,triangles,endpoint_epsilon_m=self.settings.endpoint_epsilon_m)


def visibility_budget(request,scene):
    count=0 if scene.occluder_triangles_m is None else len(scene.occluder_triangles_m)
    if request.visibility.occluders=='terrain':count=0 if scene.occluder_is_terrain is None else int(scene.occluder_is_terrain.sum())
    pairs=len(scene.amplitudes)*count*request.waveform.loops*request.array.num_tx*(1+request.array.num_rx)
    if request.visibility.occlusion_enabled and request.visibility.accelerator=='brute_force' and pairs>request.visibility.max_ray_triangle_tests:
        raise ValueError('occlusion exceeds explicit ray-triangle budget; coarsen geometry, shorten CPI or increase the bounded budget')
    return {'enabled':request.visibility.occlusion_enabled,'occluder_triangles':count,'accelerator':request.visibility.accelerator,
            'estimated_ray_triangle_tests_per_frame':pairs if request.visibility.occlusion_enabled else 0,
            'estimate_scope':'brute-force upper bound; BVH enforces actual AABB + triangle work during execution',
            'limit':request.visibility.max_ray_triangle_tests,'sampling':'each Tx slot; frozen within slot'}
