"""One configured acquisition session for offline, API and dataset callers.

No dataset names or measured-result fitting belong here. A session owns one
compiled world and one background realization for the entire sequence.
"""
from . import engine
from .schema import Simulation
from .environment import prepare_environment
from .preflight import preflight


class SimulationSession:
    def __init__(self, request: Simulation, *, compiled=None, estimate=None):
        # Revalidate and detach from mutable caller-owned configuration.
        self.request = Simulation.model_validate(request.model_dump(mode='json'))
        if compiled is None:
            compiled, estimate = preflight(self.request)
        self.scene = compiled
        self.estimate = estimate
        self._environment = None
        self._prepared = False

    def frame(self, index: int, *, cancelled=None):
        if not 0 <= index < self.request.sequence.frame_count:
            raise ValueError('frame index outside requested sequence')
        if not self._prepared:
            self._environment = prepare_environment(self.request)
            self._prepared = True
        return engine.simulate_frame(self.request, index, compiled=self.scene,
                                     environment=self._environment, cancelled=cancelled)

    @property
    def frame_count(self):
        return self.request.sequence.frame_count

    def save_inputs(self, root):
        from .storage import save_scene, write_json
        from pathlib import Path
        save_scene(root, self.request, self.scene)
        if self.estimate is not None:
            write_json(Path(root)/'applicability.json',self.estimate['applicability'])

    def save_frame(self, root, frame):
        from .storage import save_frame
        return save_frame(root, frame, self.request.processing.save_adc)

    @property
    def data_origin(self):
        if self.request.receiver_noise is not None or self.request.range_background is not None:
            return 'synthetic_composite_simulation'
        return 'ideal_simulation'


def session_from_config(raw):
    """Select a physical acquisition model by config, never by dataset name."""
    if raw.get('kind') == 'legacy_separable_if':
        from .separable_if import SeparableIFRequest, SeparableIFSession
        return SeparableIFSession(SeparableIFRequest.model_validate(raw))
    if raw.get('kind') in ('statistical_background','block_texture_background'):
        from .background import BackgroundRequest, ClutterSimulationRequest, BackgroundSession
        cls=BackgroundRequest if raw['kind']=='statistical_background' else ClutterSimulationRequest
        return BackgroundSession(cls.model_validate(raw))
    if raw.get('waveform') == 'sfcw':
        from .sfcw import SurfaceSimulationRequest, SFCWSession
        return SFCWSession(SurfaceSimulationRequest.model_validate(raw))
    return SimulationSession(Simulation.model_validate(raw))
