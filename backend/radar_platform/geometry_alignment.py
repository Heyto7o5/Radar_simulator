"""Geometry-only rigid consistency checks for repeated captures of one trihedral."""
from itertools import permutations
import numpy as np
from .surface_scene import TriangleMesh


def corner_descriptor(mesh: TriangleMesh):
    if len(mesh.faces)!=3 or len(mesh.vertices_m)!=4:raise ValueError('expected three-face four-vertex trihedral')
    common=set(mesh.faces[0]).intersection(*map(set,mesh.faces[1:]))
    if len(common)!=1:raise ValueError('one common reflector vertex required')
    corner=common.pop();outer=[i for i in range(4) if i!=corner]
    edges=mesh.vertices_m[outer]-mesh.vertices_m[corner]
    lengths=np.linalg.norm(edges,axis=1);unit=edges/lengths[:,None]
    angles=np.rad2deg(np.arccos(np.clip((unit@unit.T)[np.triu_indices(3,1)],-1,1)))
    return corner,outer,{'edge_lengths_m':lengths.tolist(),'edge_angles_deg':angles.tolist(),'corner_ned_m':mesh.vertices_m[corner].tolist()}


def rigid_corner_from_template(template: TriangleMesh,observed: TriangleMesh):
    """Preserve published corner position; fit only rotation to published edges.

    No radar samples enter. Assumes the same physical target should have the
    same dimensions and right-angle structure in different captures. This is
    a declared input-consistency hypothesis, not a corrected ground truth mesh.
    """
    tc,to,td=corner_descriptor(template);oc,oo,od=corner_descriptor(observed)
    source=template.vertices_m[to]-template.vertices_m[tc]
    best=None
    for order in permutations(oo):
        target=observed.vertices_m[list(order)]-observed.vertices_m[oc]
        u,_,vt=np.linalg.svd(source.T@target)
        if np.linalg.det(u@vt)<0:u[:,-1]*=-1
        rotation=u@vt
        error=float(np.sqrt(np.mean(np.sum((source@rotation-target)**2,axis=1))))
        if best is None or error<best[0]:best=(error,rotation,list(order))
    error,rotation,order=best
    vertices=(template.vertices_m-template.vertices_m[tc])@rotation+observed.vertices_m[oc]
    result=TriangleMesh(vertices,template.faces.copy(),'rigid_reference_corner')
    return result,{'hypothesis':'same physical object; fixed template dimensions; no radar-derived geometry fit',
                   'template':td,'published_capture':od,'aligned':corner_descriptor(result)[2],
                   'edge_fit_rms_m':error,'rotation_row_vector':rotation.tolist(),'matched_outer_indices':order,
                   'published_corner_position_preserved':True}
