"""Compile dataset-independent geometry to one persistent scatterer collection.

Default cells are phenomenological point-scatterer approximations. Explicit
scalar_po plates/meshes instead use coherent area quadrature. Neither model
includes vector diffraction, mutual coupling or calibrated material RCS.
Surface triangles are retained for the optional shared visibility evaluator.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from ..surface_scene import TriangleMesh, mesh_from_dem
from ..physical_optics import triangle_quadrature
from ..ground_scattering import EmpiricalBackscatterCurve
from ..land_tables import compile_land_model
from .schema import Scene, Box, Plate, Mesh, Terrain


@dataclass(frozen=True)
class CompiledScene:
    positions_ned_m: np.ndarray
    velocities_ned_mps: np.ndarray
    amplitudes: np.ndarray
    phases_rad: np.ndarray
    normals_ned: np.ndarray  # zero vector for point/cloud
    two_sided: np.ndarray
    object_index: np.ndarray
    scalar_po: np.ndarray
    objects: list[dict]
    terrain: dict | None
    reference_range_m: float
    ground_models: dict | None = None
    occluder_triangles_m: np.ndarray | None = None
    occluder_velocities_mps: np.ndarray | None = None
    occluder_is_terrain: np.ndarray | None = None
    land_cover_labels: np.ndarray | None = None

    def at(self, time_s):
        return self.positions_ned_m+self.velocities_ned_mps*np.asarray(time_s)[..., None, None]

    def metadata(self):
        return {'scatterer_count': len(self.amplitudes), 'objects': self.objects,
                'reference_range_m': self.reference_range_m,
                'amplitude_model': 'point amplitude; diffuse surface amplitude = coefficient * sqrt(area_m2); coherent surface amplitude = coefficient * area_m2',
                'surface_directionality': 'centroid: sqrt(cos_Tx*cos_Rx); scalar_po: arithmetic mean on visible sides; two-sided optional',
                'propagation_amplitude': 'fixed reference_range^2 / (Tx_range * Rx_range)',
                'surface_model': 'per-object centroid cells or scalar PEC single-bounce area quadrature; not full-wave or calibrated RCS',
                'scalar_po_nodes': int(self.scalar_po.sum()),
                'scalar_po_quadrature_convergence_verified': False,
                'empirical_ground_objects': len(self.ground_models or {}),
                'empirical_ground_scope': 'measured footprint averages or evidence-labelled NRCS tables applied to local surface area; synthetic diffuse phases, no extra Lambert cosine, no absolute receiver calibration',
                'empirical_ground_models': {str(k):{**v[0].metadata(),'max_bistatic_angle_assumption_deg':v[1].max_bistatic_angle_deg} for k,v in (self.ground_models or {}).items()},
                'scalar_po_response': 'i*fc/c * integral mean(positive cos_Tx,cos_Rx)/(Rt*Rr) exp(-ikL) dA; carrier-amplitude approximation; no reference-range scaling',
                'world_fixed_terrain': True, 'environment_occlusion': False,
                'seed_changes_geometry': False, 'target_materials_calibrated': False}


def yaw_matrix(angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])


def rectangular_mesh(center, u, v, width, height, divisions, outward):
    x, y = np.meshgrid(np.linspace(-width/2, width/2, divisions+1),
                       np.linspace(-height/2, height/2, divisions+1))
    vertices = np.asarray(center)+x.ravel()[:, None]*u+y.ravel()[:, None]*v
    faces = []
    for j in range(divisions):
        for i in range(divisions):
            a = j*(divisions+1)+i
            faces.extend([(a, a+1, a+divisions+2), (a, a+divisions+2, a+divisions+1)])
    faces = np.array(faces)
    if np.dot(np.cross(u, v), outward) < 0:
        faces = faces[:, ::-1]
    return TriangleMesh(vertices, faces)


def object_mesh(obj: Box | Plate | Mesh) -> TriangleMesh:
    if isinstance(obj, Box):
        vertices, faces, count = [], [], 0
        axes = np.eye(3)
        for normal_axis in range(3):
            other = [i for i in range(3) if i != normal_axis]
            for sign in [-1, 1]:
                normal = sign*axes[normal_axis]
                mesh = rectangular_mesh(normal*obj.size_m[normal_axis]/2,
                    axes[other[0]], axes[other[1]], obj.size_m[other[0]], obj.size_m[other[1]], obj.divisions, normal)
                vertices.append(mesh.vertices_m)
                faces.append(mesh.faces+count)
                count += len(mesh.vertices_m)
        mesh = TriangleMesh(np.concatenate(vertices), np.concatenate(faces))
    elif isinstance(obj, Plate):
        normal = np.array(obj.normal_ned, dtype=float)
        normal /= np.linalg.norm(normal)
        hint = np.eye(3)[np.argmin(abs(normal))]
        u = hint-normal*np.dot(normal, hint)
        u /= np.linalg.norm(u)
        mesh = rectangular_mesh(np.zeros(3), u, np.cross(normal, u), *obj.size_m, obj.divisions, normal)
    else:
        mesh = TriangleMesh(np.array(obj.vertices_local_m), np.array(obj.faces))
    rotation = yaw_matrix(obj.yaw_deg)
    return TriangleMesh(mesh.vertices_m@rotation.T+obj.position_ned_m, mesh.faces, obj.id)


def terrain_mesh(spec: Terrain):
    if spec.kind == 'dem':
        north, east, elevation = map(np.asarray, [spec.north_m, spec.east_m, spec.elevation_up_m])
    else:
        north = np.linspace(*spec.north_bounds_m, int(np.ceil(np.diff(spec.north_bounds_m)[0]/spec.spacing_m))+1)
        east = np.linspace(*spec.east_bounds_m, int(np.ceil(np.diff(spec.east_bounds_m)[0]/spec.spacing_m))+1)
        n, e = np.meshgrid(north, east)
        elevation = np.full(n.shape, spec.base_elevation_m, dtype=float)
        for hill in spec.hills:
            elevation += hill.height_m*np.exp(-.5*(((n-hill.center_ne_m[0])/hill.sigma_ne_m[0])**2+
                                                  ((e-hill.center_ne_m[1])/hill.sigma_ne_m[1])**2))
    raw = mesh_from_dem(north, east, elevation)
    # Legacy triangulator has down-facing normals. Ground outward means UP.
    mesh = TriangleMesh(raw.vertices_m, raw.faces[:, ::-1], spec.id)
    return mesh, {'north_m': north.tolist(), 'east_m': east.tolist(), 'elevation_up_m': elevation.tolist(),
                  'axes': ['east', 'north'], 'kind': spec.kind, 'id': spec.id}


def stable_phases(seed, object_id, count):
    digest = hashlib.sha256(f'{seed}:{object_id}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little')).uniform(-np.pi, np.pi, count)


def compile_scene(scene: Scene) -> CompiledScene:
    pieces, descriptors, terrain = [], [], None
    ground_models={}
    occluders=[];occluder_velocities=[];terrain_flags=[];land_labels=[]
    sources = sorted(scene.objects, key=lambda x: x.id)
    if scene.terrain:
        sources = [scene.terrain, *sources]
    for index, source in enumerate(sources):
        use_po = getattr(source, 'scattering_model', None) == 'scalar_po'
        land=getattr(source,'land_scattering',None)
        if land is not None:ground_models[index]=(compile_land_model(land),land)
        if source.kind == 'point':
            positions = np.array([source.position_ned_m])
            amplitude, phase = np.array([source.amplitude]), np.array([source.phase_rad])
            normals, sides = np.zeros((1, 3)), np.ones(1, dtype=bool)
            velocity = np.array([source.velocity_ned_mps])
            area = None
        elif source.kind == 'cloud':
            positions = np.array(source.points_local_m)+source.position_ned_m
            amplitude = np.array(source.amplitudes)
            phase = np.array(source.phases_rad if source.phases_rad is not None else np.zeros(len(amplitude)))
            normals, sides = np.zeros_like(positions), np.ones(len(positions), dtype=bool)
            velocity = np.broadcast_to(source.velocity_ned_mps, positions.shape)
            area = None
        else:
            if isinstance(source, Terrain):
                mesh, terrain = terrain_mesh(source)
                v, two_sided = (0., 0., 0.), False
            else:
                mesh = object_mesh(source)
                v, two_sided = source.velocity_ned_mps, source.two_sided
            occluders.append(mesh.triangles)
            occluder_velocities.append(np.broadcast_to(v,(len(mesh.faces),3)))
            terrain_flags.append(np.full(len(mesh.faces),isinstance(source,Terrain),dtype=bool))
            if use_po:
                nodes = len(mesh.faces)*source.quadrature_order**2
                if nodes + sum(len(p[0]) for p in pieces) > scene.max_scatterers:
                    raise ValueError('scalar_po quadrature exceeds max_scatterers; reduce mesh subdivisions or quadrature order')
                positions, area, normals = triangle_quadrature(mesh, source.quadrature_order)
            else:
                positions = mesh.triangles.mean(axis=1)
                area = mesh.areas_m2
                normals = mesh.normals
            amplitude = source.relative_reflectivity*(np.sqrt(area) if source.scattering == 'diffuse' else area)
            phase = stable_phases(scene.seed, source.id, len(area)) if source.scattering == 'diffuse' else np.zeros(len(area))
            if use_po:
                phase += np.pi/2  # physical PO field prefactor; conjugated by FMCW dechirping.
            sides = np.full(len(area), two_sided)
            velocity = np.broadcast_to(v, positions.shape)
        labels=np.full(len(positions),'' if land is None else land.land_cover,dtype=object)
        if land is not None and isinstance(source,Terrain) and source.land_cover_map is not None:
            labels=np.repeat(np.asarray(source.land_cover_map,dtype=object).ravel(),2)
            if len(labels)!=len(positions):raise ValueError('terrain class-to-triangle mapping mismatch')
        land_labels.append(labels)
        descriptors.append({'id': source.id, 'kind': source.kind, 'scatterers': len(positions),
                            'land_cover_counts':{str(k):int((labels==k).sum()) for k in np.unique(labels) if k},
                            'initial_center_ned_m': positions.mean(axis=0).tolist(),
                            'velocity_ned_mps': velocity[0].tolist(),
                            'bounds_ned_m': [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
                            'surface_area_m2': None if area is None else float(area.sum()),
                            'relative_reflectivity_multiplier':getattr(source,'relative_reflectivity',None),
                            'scattering': getattr(source, 'scattering', 'explicit_points'),
                            'scattering_model': ('tabulated_sigma0_diffuse' if land.tables else 'empirical_sigma0_diffuse') if land is not None else getattr(source, 'scattering_model', 'centroid' if area is not None else 'explicit_points'),
                            'quadrature_order': source.quadrature_order if use_po else None})
        pieces.append((positions, velocity, amplitude, phase, normals, sides, np.full(len(positions), index),np.full(len(positions),use_po,dtype=bool)))
        if sum(len(p[0]) for p in pieces) > scene.max_scatterers:
            raise ValueError('compiled scene exceeds max_scatterers; coarsen geometry explicitly')
    if not pieces:
        columns=[np.empty((0,3)),np.empty((0,3)),np.empty(0),np.empty(0),
                 np.empty((0,3)),np.empty(0,dtype=bool),np.empty(0,dtype=int),np.empty(0,dtype=bool)]
    else:
        columns = [np.concatenate([p[i] for p in pieces]) for i in range(8)]
    triangles=np.concatenate(occluders) if occluders else np.empty((0,3,3))
    speeds=np.concatenate(occluder_velocities) if occluders else np.empty((0,3))
    flags=np.concatenate(terrain_flags) if occluders else np.empty(0,dtype=bool)
    return CompiledScene(*columns, descriptors, terrain, scene.amplitude_reference_range_m,ground_models,
                         triangles,speeds,flags,np.concatenate(land_labels) if land_labels else np.empty(0,dtype=object))
