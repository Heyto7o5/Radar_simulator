"""Independent physical scenes and separate offline supervision."""
import numpy as np

from ..navigation.dynamics import body_to_neu
from ..navigation.dataset import generate_case
from ..navigation.io import local_path, read_json, write_json, digest
from .model import encode_rda
from .protocol import validate_records
from .scene import make_scene


def records_for_plan(plan, *, train_per_region=512, val_per_region=128, test_per_region=128):
    counts = {'train': train_per_region, 'val': val_per_region, 'test': test_per_region}
    result = []
    for index, region in enumerate(plan['regions']):
        rng = np.random.default_rng(np.random.SeedSequence([plan['seed'], index]))
        for k in range(counts[region['split']]):
            position = rng.uniform([-45, -45, 22], [45, 45, 60])
            if k % 4 == 0:
                position = rng.uniform([-8, -8, 26], [8, 8, 38])
            yaw = float(rng.uniform(-np.pi, np.pi))
            velocity = rng.uniform([-1., -1., -1.5], [8., 1., 1.5])
            if k % 5 == 0:
                velocity *= .1
            elif k % 5 == 4:
                velocity[0] = rng.uniform(8, 12)
            spec = plan['building']
            result.append(dict(id=f'{region["id"]}_{k:05d}', region_id=region['id'], split=region['split'],
                position_neu_m=position.tolist(), velocity_neu_mps=(body_to_neu(yaw)@velocity).tolist(), yaw_rad=yaw,
                acceleration_body_mps2=rng.uniform([-1.2, -1.2, -.5], [1.2, 1.2, .5]).tolist(),
                yaw_rate_rad_s=float(np.deg2rad(rng.uniform(-45, 45))),
                phase_seed=int(rng.integers(1, 2**32-1)), noise_seed=int(rng.integers(1, 2**32-1)),
                terrain_reflectivity=float(np.exp(rng.uniform(np.log(.001), np.log(.004)))),
                noise_variance=float(np.exp(rng.uniform(np.log(3e-6), np.log(3e-5)))),
                building=dict(footprint_m=[float(rng.uniform(*spec['width_m'])), float(rng.uniform(*spec['length_m']))],
                    height_m=float(rng.uniform(*spec['height_above_ground_m'])), yaw_deg=float(rng.uniform(*spec['yaw_deg'])),
                    relative_reflectivity=float(rng.uniform(*spec['relative_reflectivity'])))))
    validate_records(result, plan)
    return result


def generate_multidem_case(job):
    geography, record, output = job
    output = local_path(output)
    config, metadata = make_scene(geography, record)
    complete = generate_case((config.model_dump(mode='json'), record, str(output), False))
    write_json(output/'scene_truth.json', metadata)
    # The old generator is reused only for unchanged echo/trajectory physics.
    # Its old preprocessing cache is deliberately not a new model's input.
    with np.load(output/'rda.npz', allow_pickle=False) as observation:
        features = encode_rda(observation['magnitude'])
        sampling = {name: observation[name].tolist() for name in
                    ['range_m', 'velocity_mps', 'elevation_deg', 'azimuth_deg']}
        sampling['frame_interval_s'] = float(np.diff(observation['frame_start_s'])[0])
        sampling['cpi_duration_s'] = float(observation['frame_end_s'][0]-observation['frame_start_s'][0])
        sampling['shape'] = list(observation['magnitude'].shape)
    np.save(output/'rda_features.tmp.npy', features.astype(np.float16))
    (output/'rda_features.tmp.npy').replace(output/'rda_features.npy')
    write_json(output/'sampling.json', sampling)
    return dict(**complete, region_id=record['region_id'], split=record['split'], sampling_sha256=digest(sampling))
