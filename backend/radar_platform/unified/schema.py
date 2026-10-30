"""Versioned SI/NED input contract shared by pulsed LFM and FMCW."""
from __future__ import annotations

from typing import Annotated, Literal
import math
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import C0
from ..statistical_background import BackgroundRecipe
from ..ground_scattering import Sigma0Observation, EmpiricalBackscatterCurve
from .trajectory import RigidTrajectory
from ..land_tables import BackscatterTable,compile_land_model
from .radiation import Radiation

Vec3 = tuple[float, float, float]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Motion(StrictModel):
    """Position at global simulation t=0; no hidden frame-relative rebasing."""
    position_ned_m: Vec3 = (0., 0., 0.)
    velocity_ned_mps: Vec3 = (0., 0., 0.)


class Platform(Motion):
    # When provided, sampled motion overrides the constant position/velocity/orientation fields.
    trajectory: RigidTrajectory | None = None
    position_ned_m: Vec3 = (0., 0., -5000.)
    boresight_ned: Vec3 = (0., 0., 1.)
    right_hint_ned: Vec3 = (0., 1., 0.)
    beam_half_angle_deg: float = Field(15., gt=0, le=180.)
    beam_model: Literal['hard_cone','cosine_edge'] = 'hard_cone'
    beam_taper_deg: float = Field(5.,gt=0,le=89.)

    @model_validator(mode='after')
    def orientation(self):
        if self.beam_model=='cosine_edge' and self.beam_taper_deg>self.beam_half_angle_deg:
            raise ValueError('beam taper width must not exceed beam half angle')
        f, r = np.array(self.boresight_ned), np.array(self.right_hint_ned)
        if np.linalg.norm(f) < 1e-9 or np.linalg.norm(np.cross(f, r)) < 1e-9:
            raise ValueError('boresight and right_hint must be nonzero and nonparallel')
        return self


class PulseLFM(StrictModel):
    kind: Literal['pulsed_lfm'] = 'pulsed_lfm'
    carrier_hz: float = Field(10e9, gt=0)
    bandwidth_hz: float = Field(20e6, gt=0)
    sample_rate_hz: float = Field(40e6, gt=0)
    pulse_width_s: float = Field(5e-6, gt=0)
    slot_period_s: float = Field(100e-6, gt=0)
    adc_start_s: float = Field(28e-6, ge=0)
    samples: int = Field(1024, ge=8, le=32768)
    loops: int = Field(64, ge=4, le=512)

    @model_validator(mode='after')
    def timing(self):
        if self.sample_rate_hz < self.bandwidth_hz:
            raise ValueError('complex pulse ADC rate must cover the bandwidth')
        if self.pulse_width_s >= self.slot_period_s:
            raise ValueError('pulse must end before the next Tx slot')
        if self.adc_start_s + self.samples / self.sample_rate_hz > self.slot_period_s * (1 + 1e-10):
            raise ValueError('receive gate must fit inside a Tx slot')
        if self.bandwidth_hz >= self.carrier_hz or self.pulse_width_s*self.sample_rate_hz < 4:
            raise ValueError('invalid baseband pulse bandwidth or fewer than four transmit samples')
        return self

    @property
    def wavelength_m(self):
        return C0 / self.carrier_hz

    @property
    def slope_hz_s(self):
        return self.bandwidth_hz / self.pulse_width_s


class FMCW(StrictModel):
    kind: Literal['fmcw'] = 'fmcw'
    start_hz: float = Field(9.98e9, gt=0)
    slope_hz_s: float = Field(2e11, gt=0)
    ramp_s: float = Field(250e-6, gt=0)
    slot_period_s: float = Field(300e-6, gt=0)
    adc_start_s: float = Field(45e-6, ge=0)
    sample_rate_hz: float = Field(20e6, gt=0)
    samples: int = Field(2048, ge=4, le=32768)
    loops: int = Field(64, ge=4, le=512)

    @model_validator(mode='after')
    def timing(self):
        if self.ramp_s > self.slot_period_s:
            raise ValueError('ramp exceeds Tx slot')
        if self.adc_start_s + self.samples/self.sample_rate_hz > self.ramp_s*(1+1e-10):
            raise ValueError('ADC window must fit inside ramp')
        return self

    @property
    def carrier_hz(self):
        return self.start_hz+self.slope_hz_s*(self.adc_start_s+self.samples/(2*self.sample_rate_hz))

    @property
    def wavelength_m(self):
        return C0/self.carrier_hz

    @property
    def bandwidth_hz(self):
        return self.slope_hz_s*self.samples/self.sample_rate_hz


Waveform = Annotated[PulseLFM | FMCW, Field(discriminator='kind')]


class Array(StrictModel):
    layout: Literal['siso', 'ula', 'ura', 'custom'] = 'ula'
    num_tx: int = Field(2, ge=1, le=8)
    num_rx: int = Field(4, ge=1, le=32)
    spacing_m: float | None = Field(None, gt=0)
    tx_local_m: list[Vec3] | None = None
    rx_local_m: list[Vec3] | None = None
    tx_order: list[int] | None = None
    tx_offsets_s: list[float] | None = None
    loop_period_s: float | None = Field(None, gt=0)

    @model_validator(mode='after')
    def geometry(self):
        if self.num_tx*self.num_rx > 64:
            raise ValueError('maximum 64 virtual channels')
        if self.layout == 'siso' and (self.num_tx != 1 or self.num_rx != 1):
            raise ValueError('SISO requires exactly one Tx and one Rx')
        if self.layout == 'custom':
            if self.tx_local_m is None or self.rx_local_m is None:
                raise ValueError('custom array requires both element lists')
            if len(self.tx_local_m) != self.num_tx or len(self.rx_local_m) != self.num_rx:
                raise ValueError('element counts do not match coordinates')
        elif self.tx_local_m is not None or self.rx_local_m is not None:
            raise ValueError('coordinates require custom array layout')
        if self.tx_order is not None and sorted(self.tx_order) != list(range(self.num_tx)):
            raise ValueError('tx_order must be a permutation of every Tx index')
        if self.tx_order is not None and self.tx_offsets_s is not None:
            raise ValueError('specify tx_order OR explicit per-Tx offsets')
        if self.tx_offsets_s is not None:
            if len(self.tx_offsets_s) != self.num_tx or min(self.tx_offsets_s) < 0:
                raise ValueError('one nonnegative offset per physical Tx required')
        return self


class Point(Motion):
    kind: Literal['point'] = 'point'
    id: str = Field(min_length=1, max_length=100)
    amplitude: float = Field(1., gt=0)
    phase_rad: float = 0.


class Cloud(Motion):
    kind: Literal['cloud'] = 'cloud'
    id: str = Field(min_length=1, max_length=100)
    points_local_m: list[Vec3] = Field(min_length=1, max_length=4096)
    amplitudes: list[float]
    phases_rad: list[float] | None = None

    @model_validator(mode='after')
    def lengths(self):
        if len(self.amplitudes) != len(self.points_local_m) or min(self.amplitudes) <= 0:
            raise ValueError('one positive amplitude per cloud point')
        if self.phases_rad is not None and len(self.phases_rad) != len(self.points_local_m):
            raise ValueError('one phase per cloud point')
        return self


class LandScattering(StrictModel):
    """Measured footprint coefficients used as an explicit local-patch approximation."""
    observations: list[Sigma0Observation] = Field(default_factory=list,max_length=128)
    tables: list[BackscatterTable] = Field(default_factory=list,max_length=64)
    mode: Literal['interpolate','cosine_power'] = 'interpolate'
    polarization: Literal['vv','hh','hv','vh']
    land_cover: str = 'alpine_meadow'
    max_bistatic_angle_deg: float = Field(5.,gt=0,le=5.)

    @model_validator(mode='after')
    def coverage(self):
        if bool(self.observations)==bool(self.tables):
            raise ValueError('supply observations OR frequency/incidence tables, not both or neither')
        if self.tables:
            if self.mode!='interpolate':raise ValueError('table mode uses bounded interpolation, not a fitted cosine law')
            model=compile_land_model(self)
            if (self.land_cover,self.polarization) not in model.tables:
                raise ValueError('default land-cover/polarization not present in tables')
            return self
        EmpiricalBackscatterCurve(tuple(self.observations),self.mode)
        for o in self.observations:
            if not 0<o.frequency_min_hz<o.frequency_max_hz or not 0<=o.incidence_min_deg<=o.incidence_center_deg<=o.incidence_max_deg<=90:
                raise ValueError('invalid measured frequency or incidence bounds')
        if any(o.polarization!=self.polarization or o.land_cover!=self.land_cover for o in self.observations):
            raise ValueError('land scattering observations do not match declared polarization/land cover')
        return self


class Surface(Motion):
    id: str = Field(min_length=1, max_length=100)
    yaw_deg: float = 0.
    relative_reflectivity: float = Field(.1, gt=0)
    scattering: Literal['diffuse', 'coherent'] = 'diffuse'
    two_sided: bool = False
    scattering_model: Literal['centroid', 'scalar_po'] = 'centroid'
    quadrature_order: int = Field(12, ge=2, le=48)
    land_scattering: LandScattering | None = None

    @model_validator(mode='after')
    def surface_integration(self):
        if self.land_scattering is not None:
            if self.scattering!='diffuse' or self.scattering_model!='centroid' or self.two_sided:
                raise ValueError('empirical land scattering requires one-sided diffuse area cells, not scalar PEC or coherent phases')
            if getattr(self,'kind',None)=='box':raise ValueError('empirical land scattering is for ground patches/meshes, not box objects')
        if self.scattering_model == 'scalar_po':
            if self.scattering != 'coherent':
                raise ValueError('scalar_po requires coherent scattering; random diffuse phases are not a PEC model')
            if getattr(self, 'kind', None) == 'box':
                raise ValueError('scalar_po currently supports plates/explicit meshes; box occlusion is not implemented')
        return self


class Box(Surface):
    kind: Literal['box'] = 'box'
    size_m: Vec3 = (40., 30., 20.)
    divisions: int = Field(2, ge=1, le=12)

    @model_validator(mode='after')
    def sizes(self):
        if min(self.size_m) <= 0:
            raise ValueError('box dimensions must be positive')
        return self


class Plate(Surface):
    kind: Literal['plate'] = 'plate'
    size_m: tuple[float, float] = (20., 20.)
    divisions: int = Field(4, ge=1, le=32)
    normal_ned: Vec3 = (0., 0., -1.)

    @model_validator(mode='after')
    def sizes(self):
        if min(self.size_m) <= 0 or np.linalg.norm(self.normal_ned) < 1e-9:
            raise ValueError('plate dimensions and normal must be nonzero')
        return self


class Mesh(Surface):
    kind: Literal['mesh'] = 'mesh'
    vertices_local_m: list[Vec3] = Field(min_length=3, max_length=8192)
    faces: list[tuple[int, int, int]] = Field(min_length=1, max_length=4096)

    @model_validator(mode='after')
    def valid_mesh(self):
        from ..surface_scene import TriangleMesh
        TriangleMesh(np.array(self.vertices_local_m), np.array(self.faces))
        return self


Object = Annotated[Point | Cloud | Box | Plate | Mesh, Field(discriminator='kind')]


class Hill(StrictModel):
    center_ne_m: tuple[float, float] = (0., 0.)
    height_m: float = Field(100., ge=0)
    sigma_ne_m: tuple[float, float] = (200., 200.)

    @model_validator(mode='after')
    def scale(self):
        if min(self.sigma_ne_m) <= 0:
            raise ValueError('positive hill horizontal scales required')
        return self


class Terrain(StrictModel):
    id: str = 'terrain'
    kind: Literal['flat', 'hills', 'dem'] = 'flat'
    max_grid_vertices: int = Field(4096,ge=4,le=65536)
    north_bounds_m: tuple[float, float] = (-400., 400.)
    east_bounds_m: tuple[float, float] = (-400., 400.)
    spacing_m: float = Field(100., gt=0)
    base_elevation_m: float = 0.
    hills: list[Hill] = Field(default_factory=list, max_length=16)
    north_m: list[float] | None = None
    east_m: list[float] | None = None
    elevation_up_m: list[list[float]] | None = None
    relative_reflectivity: float = Field(.002, gt=0)
    scattering: Literal['diffuse', 'coherent'] = 'diffuse'
    land_scattering: LandScattering | None = None
    land_cover_map: list[list[str]] | None = None

    @model_validator(mode='after')
    def grid(self):
        if self.land_scattering is not None and self.scattering!='diffuse':
            raise ValueError('empirical terrain coefficients require diffuse area cells')
        if self.kind == 'dem':
            if any(v is None for v in [self.north_m, self.east_m, self.elevation_up_m]):
                raise ValueError('DEM requires both axes and elevation array')
            from ..surface_scene import mesh_from_dem
            mesh_from_dem(self.north_m, self.east_m, self.elevation_up_m)
            if len(self.north_m)*len(self.east_m) > self.max_grid_vertices:
                raise ValueError('DEM exceeds declared max_grid_vertices')
        else:
            if any(v is not None for v in [self.north_m, self.east_m, self.elevation_up_m]):
                raise ValueError('explicit elevation arrays require DEM kind')
            lengths = [b[1]-b[0] for b in [self.north_bounds_m, self.east_bounds_m]]
            if min(lengths) <= 0:
                raise ValueError('terrain bounds must increase')
            if math.prod(math.ceil(x/self.spacing_m)+1 for x in lengths) > self.max_grid_vertices:
                raise ValueError('terrain exceeds declared max_grid_vertices')
        if self.kind == 'hills' and not self.hills:
            raise ValueError('hills terrain requires at least one hill')
        if self.kind != 'hills' and self.hills:
            raise ValueError('hill parameters only apply to hills terrain')
        if self.land_cover_map is not None:
            if self.land_scattering is None:raise ValueError('land_cover_map requires explicit land scattering parameters')
            ne,nn=(len(self.east_m)-1,len(self.north_m)-1) if self.kind=='dem' else (
                math.ceil((self.east_bounds_m[1]-self.east_bounds_m[0])/self.spacing_m),
                math.ceil((self.north_bounds_m[1]-self.north_bounds_m[0])/self.spacing_m))
            labels=np.asarray(self.land_cover_map)
            if labels.shape!=(ne,nn):raise ValueError('land_cover_map must be [east_cell,north_cell], not grid nodes')
            available={t.land_cover for t in self.land_scattering.tables if t.polarization==self.land_scattering.polarization} if self.land_scattering.tables else {self.land_scattering.land_cover}
            if not set(labels.ravel()).issubset(available):raise ValueError('land-cover map contains a class without a matching polarization table')
        return self


class Scene(StrictModel):
    allow_empty: bool = False
    objects: list[Object] = Field(default_factory=list, max_length=128)
    terrain: Terrain | None = None
    seed: int = Field(42, ge=0, le=2**32-1)
    amplitude_reference_range_m: float = Field(1000., gt=0)
    max_scatterers: int = Field(4096, ge=1, le=131072)

    @model_validator(mode='after')
    def names(self):
        if not self.objects and self.terrain is None and not self.allow_empty:
            raise ValueError('scene needs objects or terrain')
        ids = [o.id for o in self.objects] + ([self.terrain.id] if self.terrain else [])
        if len(set(ids)) != len(ids):
            raise ValueError('object and terrain IDs must be unique')
        return self


class Processing(StrictModel):
    range_limits_m: tuple[float, float] = (4000., 6000.)
    range_window: Literal['hann', 'hamming', 'rectangular'] = 'hann'
    doppler_window: Literal['hann', 'hamming', 'rectangular'] = 'hann'
    spatial_window: Literal['hann', 'hamming', 'rectangular'] = 'rectangular'
    azimuth_deg: list[float] = Field(default_factory=lambda: np.arange(-30., 31., 1.).tolist(), min_length=1, max_length=361)
    elevation_deg: list[float] = Field(default_factory=lambda: [0.], min_length=1, max_length=181)
    tdm_compensation: bool = True
    range_fft_size: int | None = Field(None, ge=8, le=32768)
    doppler_fft_size: int | None = Field(None, ge=4, le=1024)
    save_adc: bool = True
    max_peaks: int = Field(10, ge=1, le=64)

    @model_validator(mode='after')
    def axes(self):
        if not 0 <= self.range_limits_m[0] < self.range_limits_m[1]:
            raise ValueError('nonnegative increasing range limits required')
        for values in [self.azimuth_deg, self.elevation_deg]:
            if np.any(np.diff(values) <= 0) or max(abs(x) for x in values) > 90:
                raise ValueError('angle scan must increase within [-90,90] degrees')
        return self


class Sequence(StrictModel):
    frame_count: int = Field(1, ge=1, le=100)
    frame_interval_s: float = Field(.1, gt=0)
    start_time_s: float = Field(0., ge=0)


class ReceiverNoise(StrictModel):
    """E[|n|^2] per complex ADC sample; no signal-dependent SNR normalization."""
    variance: float = Field(0., ge=0, le=1e24)
    variance_per_rx_tx: list[list[float]] | None = None
    fast_time_spectral_power: list[float] | None = Field(None,min_length=8,max_length=32768)
    provenance: str | None = Field(None,min_length=1,max_length=1000)
    seed: int = Field(42, ge=0, lt=2**32)


class RangeBackground(StrictModel):
    """Explicitly scaled background AFTER range processing, BEFORE Doppler."""
    model: BackgroundRecipe
    power_scale: float = Field(ge=0, le=1e24)
    provenance: str = Field(min_length=1, max_length=1000)
    seed: int = Field(42, ge=0, lt=2**32)
    channel_mode: Literal['independent', 'steered'] = 'independent'
    azimuth_deg: float = Field(0., gt=-90, lt=90)
    elevation_deg: float = Field(0., gt=-90, lt=90)
    frequency_convention: Literal['raw_slow_time_hz'] = 'raw_slow_time_hz'
    floor_policy: Literal['remove_receiver_like_floor','retain_model_floor'] = 'remove_receiver_like_floor'


class Acquisition(StrictModel):
    """Processing an observed ADC never requires inventing a simulation target."""
    waveform: Waveform = Field(default_factory=PulseLFM)
    platform: Platform = Field(default_factory=Platform)
    array: Array = Field(default_factory=Array)
    processing: Processing = Field(default_factory=Processing)

    @model_validator(mode='after')
    def budget(self):
        w, a, p = self.waveform, self.array, self.processing
        if p.range_fft_size and (p.range_fft_size < w.samples or w.kind != 'fmcw'):
            raise ValueError('range FFT padding applies only to FMCW and cannot truncate ADC')
        if p.doppler_fft_size and p.doppler_fft_size < w.loops:
            raise ValueError('Doppler FFT must not truncate loops')
        if w.samples*w.loops*a.num_tx*a.num_rx > 8_000_000:
            raise ValueError('ADC exceeds 8 million complex samples per frame')
        offsets = np.array(a.tx_offsets_s if a.tx_offsets_s is not None else np.arange(a.num_tx)*w.slot_period_s)
        if len(offsets) > 1 and np.min(np.diff(np.sort(offsets))) < w.slot_period_s*(1-1e-10):
            raise ValueError('TDM Tx slots overlap')
        period = a.loop_period_s or a.num_tx*w.slot_period_s
        if max(offsets)+w.slot_period_s > period*(1+1e-10):
            raise ValueError('TDM slots do not fit the loop period')
        return self


class Visibility(StrictModel):
    occlusion_enabled: bool = False
    report_coverage: bool = False
    occluders: Literal['terrain', 'all_surfaces'] = 'all_surfaces'
    accelerator: Literal['brute_force','bvh'] = 'brute_force'
    bvh_kernel: Literal['numpy','numba'] = 'numpy'
    batch_bvh: bool = True
    endpoint_epsilon_m: float = Field(1e-4,gt=0,le=.1)
    max_ray_triangle_tests: int = Field(100_000_000,ge=1,le=500_000_000)


class Simulation(Acquisition):
    schema_version: Literal['1.0'] = '1.0'
    name: str = 'unified ideal scene'
    input_provenance: list[Annotated[str, Field(min_length=1,max_length=2000)]] = Field(default_factory=list,max_length=64)
    scene: Scene
    visibility: Visibility = Field(default_factory=Visibility)
    radiation: Radiation | None = None
    sequence: Sequence = Field(default_factory=Sequence)
    receiver_noise: ReceiverNoise | None = None
    range_background: RangeBackground | None = None
    save_components: bool = True

    @model_validator(mode='after')
    def sequence_timing(self):
        if self.radiation is not None:
            w=self.waveform
            lo,hi=(w.carrier_hz-w.bandwidth_hz/2,w.carrier_hz+w.bandwidth_hz/2) if w.kind=='pulsed_lfm' else (w.start_hz,w.start_hz+w.slope_hz_s*w.ramp_s)
            for pattern in [self.radiation.tx_element,self.radiation.rx_element]:
                if pattern.kind=='axisymmetric_table' and (lo<pattern.frequency_min_hz or hi>pattern.frequency_max_hz):
                    raise ValueError('complete RF sweep must fit antenna-pattern frequency band')
            aperture=self.radiation.transmit_aperture
            if aperture is not None:
                positions=np.asarray(aperture.positions_local_m)
                diameter=float(np.linalg.norm(positions[:,None]-positions[None,:],axis=-1).max())
                if np.pi*(hi-lo)*diameter/C0>aperture.max_wideband_phase_error_rad:
                    raise ValueError('transmit aperture exceeds declared narrowband phase-error budget; use smaller aperture or bandwidth')
        period = self.array.loop_period_s or self.array.num_tx*self.waveform.slot_period_s
        if self.platform.trajectory is not None:
            trajectory=self.platform.trajectory
            offsets=self.array.tx_offsets_s if self.array.tx_offsets_s is not None else list(np.arange(self.array.num_tx)*self.waveform.slot_period_s)
            first=self.sequence.start_time_s
            last=first+(self.sequence.frame_count-1)*self.sequence.frame_interval_s+(self.waveform.loops-1)*period+max(offsets)+self.waveform.adc_start_s+self.waveform.samples/self.waveform.sample_rate_hz
            if first<trajectory.time_s[0] or last>trajectory.time_s[-1]:
                raise ValueError('sampled trajectory must cover every Tx slot and ADC sample; extrapolation forbidden')
            if self.range_background is not None:
                raise ValueError('sampled platform trajectory with post-range statistical background is not yet supported; geometry-driven echoes and ADC noise are supported')
        if self.sequence.frame_count > 1 and self.sequence.frame_interval_s < self.waveform.loops*period*(1-1e-10):
            raise ValueError('sequence frame interval must cover the complete CPI')
        if any(getattr(o, 'scattering_model', None) == 'scalar_po' for o in self.scene.objects):
            if self.waveform.bandwidth_hz / self.waveform.carrier_hz > .05:
                raise ValueError('dynamic scalar_po uses a carrier-frequency amplitude approximation; bandwidth must not exceed 5% of carrier')
        for obj in [*self.scene.objects,self.scene.terrain]:
            land=getattr(obj,'land_scattering',None)
            if land is not None:
                w=self.waveform
                lo,hi=(w.carrier_hz-w.bandwidth_hz/2,w.carrier_hz+w.bandwidth_hz/2) if w.kind=='pulsed_lfm' else (w.start_hz,w.start_hz+w.slope_hz_s*w.ramp_s)
                if land.tables:
                    classes=set(np.asarray(obj.land_cover_map).ravel()) if getattr(obj,'land_cover_map',None) is not None else {land.land_cover}
                    sources=[t for t in land.tables if t.land_cover in classes and t.polarization==land.polarization]
                    if any(lo<t.frequencies_hz[0] or hi>t.frequencies_hz[-1] for t in sources):
                        raise ValueError('complete waveform RF sweep must fit every used land table; extrapolation forbidden')
                else:
                    source=land.observations[0]
                    if lo<source.frequency_min_hz or hi>source.frequency_max_hz:
                        raise ValueError('complete waveform RF sweep must fit measured land-coefficient band; extrapolation forbidden')
        if self.receiver_noise is not None:
            noise=self.receiver_noise
            if noise.fast_time_spectral_power is not None:
                q=np.asarray(noise.fast_time_spectral_power,float)
                if q.shape!=(self.waveform.samples,) or np.any(q<0) or np.any(q>1e24):
                    raise ValueError('colored ADC background requires nonnegative bounded PSD at every native fast-time FFT bin')
                if noise.variance!=0 or noise.variance_per_rx_tx is not None:
                    raise ValueError('choose colored PSD OR scalar/channel white variance')
                if noise.provenance is None:raise ValueError('colored ADC background requires provenance; it is not a calibrated thermal-noise model')
            if noise.variance_per_rx_tx is not None:
                a=np.asarray(noise.variance_per_rx_tx,float)
                if a.shape!=(self.array.num_rx,self.array.num_tx) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1e24):
                    raise ValueError('receiver variance matrix must be finite nonnegative [rx,tx]')
                if noise.variance!=0:raise ValueError('choose scalar variance OR variance_per_rx_tx, not both')
        if self.range_background is not None:
            b=self.range_background;w=self.waveform
            if b.model.family=='white_gaussian' and b.floor_policy=='remove_receiver_like_floor':
                raise ValueError('white_gaussian is a noise-only baseline; use receiver_noise or explicitly retain its floor without ADC noise')
            if self.receiver_noise is not None:
                noise=self.receiver_noise
                active=noise.variance>0 or (noise.variance_per_rx_tx is not None and np.any(np.asarray(noise.variance_per_rx_tx)>0)) or (noise.fast_time_spectral_power is not None and np.any(np.asarray(noise.fast_time_spectral_power)>0))
                if active and b.floor_policy=='retain_model_floor' and (b.model.white_fraction>0 or b.model.family=='white_gaussian'):
                    raise ValueError('receiver-like floor would be counted twice; remove the model floor before adding ADC noise')
            if b.channel_mode=='independent' and (b.azimuth_deg!=0 or b.elevation_deg!=0):
                raise ValueError('a background angle requires steered channel mode')
            if not np.isclose(b.model.prf_hz,1/period,rtol=1e-10,atol=1e-10):
                raise ValueError('background PRF must equal SAME-Tx loop rate, not Tx-slot rate')
            step=self.sequence.frame_interval_s/period
            if self.sequence.frame_count>1 and not np.isclose(step,round(step),atol=1e-8,rtol=0):
                raise ValueError('background frame starts must lie on the slow-time clock')
            n=max(512,(self.sequence.frame_count-1)*round(step)+w.loops+1)
            if n>131072 or n*len(b.model.range_m)*self.array.num_rx*self.array.num_tx>5_000_000:
                raise ValueError('background sequence exceeds timeline/channel memory budget')
            axis=((w.adc_start_s+np.arange(w.samples)/w.sample_rate_hz)*C0/2 if w.kind=='pulsed_lfm'
                  else np.arange(self.processing.range_fft_size or w.samples)*w.sample_rate_hz/(self.processing.range_fft_size or w.samples)*C0/(2*w.slope_hz_s))
            axis=axis[(axis>=self.processing.range_limits_m[0])&(axis<=self.processing.range_limits_m[1])]
            if not len(axis):raise ValueError('empty range-processing grid')
            distances=abs(np.array(b.model.range_m)[:,None]-axis[None,:]);indices=np.argmin(distances,axis=1)
            if np.any(distances[np.arange(len(indices)),indices]>1e-6) or len(set(indices))!=len(indices):
                raise ValueError('background ranges must match distinct processed range bins exactly; no implicit resampling')
        return self
