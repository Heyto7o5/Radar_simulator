"""Physical radar acquisition adapter; only this side may access world truth."""
from dataclasses import dataclass
import time
import numpy as np

from ..unified.schema import Simulation
from ..unified.session import SimulationSession
from ..unified.preflight import preflight
from ..unified.applicability import assess_scene
from .dynamics import sampled_trajectory
from .observation import Observation


def cpi_duration(config):
    w, a = config.waveform, config.array
    period = a.loop_period_s or a.num_tx*w.slot_period_s
    offsets = a.tx_offsets_s or list(np.arange(a.num_tx)*w.slot_period_s)
    return (w.loops-1)*period+max(offsets)+w.adc_start_s+w.samples/w.sample_rate_hz


@dataclass(frozen=True)
class AcquiredFrame:
    magnitude: np.ndarray
    range_m: np.ndarray
    velocity_mps: np.ndarray
    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    start_s: float
    end_s: float


def pair(previous, current):
    for name in ['range_m','velocity_mps','elevation_deg','azimuth_deg']:
        if not np.array_equal(getattr(previous,name),getattr(current,name)):
            raise ValueError('acquisition changed within radar pair')
    observation = Observation(np.stack([previous.magnitude,current.magnitude]),
        current.range_m,current.velocity_mps,current.elevation_deg,current.azimuth_deg,
        np.array([previous.start_s,current.start_s]),np.array([previous.end_s,current.end_s]))
    observation.validate()
    return observation


class RadarWorld:
    def __init__(self, config):
        self.config = Simulation.model_validate(config.model_dump(mode='json'))
        self.scene, self.estimate = preflight(self.config)
        self.duration_s = cpi_duration(config)
        self.phase_snapshot = self.scene.phases_rad.copy()

    def acquire(self, states, frame_number, *, save_directory=None, assessment=False):
        start_s = states[0].time_s
        end_s = start_s+self.duration_s
        if states[-1].time_s < end_s:
            raise ValueError('plant must cover the complete CPI before observation is available')
        raw = self.config.model_dump(mode='json')
        raw['platform']['trajectory'] = sampled_trajectory(states).model_dump(mode='json')
        raw['sequence'].update(start_time_s=start_s, frame_count=1)
        if raw['receiver_noise'] is not None:
            raw['receiver_noise']['seed'] = int(np.random.SeedSequence(
                [self.config.receiver_noise.seed, int(frame_number)]).generate_state(1)[0])
        request = Simulation.model_validate(raw)
        session = SimulationSession(request, compiled=self.scene, estimate=self.estimate)
        started = time.perf_counter()
        result = session.frame(0)
        elapsed = time.perf_counter()-started
        if not np.array_equal(self.scene.phases_rad,self.phase_snapshot):
            raise RuntimeError('physical surface phases changed during episode')
        p = result.products
        frame = AcquiredFrame(np.abs(p.rad).astype(np.float32),p.range_m,p.velocity_mps,
                              p.elevation_deg,p.azimuth_deg,start_s,end_s)
        diagnostic = dict(elapsed_s=elapsed, frame_number=int(frame_number),
                          start_s=start_s,end_s=end_s,visibility=result.metadata['visibility'])
        if assessment:
            diagnostic['applicability'] = assess_scene(request,self.scene)
        if save_directory is not None:
            session.save_inputs(save_directory)
            session.save_frame(save_directory,result)
        return frame, diagnostic, request
