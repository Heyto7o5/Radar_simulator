"""One ideal propagation path for mixed geometry, motion, waveforms and arrays.

Bistatic ranges are recomputed at every Tx slot, then linearly evolved within
fast time. This retains intrachirp Doppler without adding it twice. Optional
sampled platform translation/attitude updates every Tx slot, including rotating
element velocity. Optional opaque-triangle occlusion is evaluated at Tx slots.
Retarded motion, within-slot acceleration and multipath are not modelled.
Raw ADC and unaveraged channels are retained.
"""
from dataclasses import dataclass
import time
import numpy as np

from ..models import C0
from .schema import Simulation
from .scene import CompiledScene, compile_scene
from .geometry import Layout, build_layout, local_angles
from .processing import Products, process_adc, range_compress, process_range_iq, doppler_channels, selected_range_indices, peak_candidates
from .environment import prepare_environment, receiver_noise as make_receiver_noise
from .land_response import land_amplitude_factor
from .trajectory import TrajectoryEvaluator
from .antenna import two_way_voltage_gain
from .visibility import VisibilityEvaluator, visibility_budget
from .radiation import transmit_response,receive_response
from .visibility_batch import clear_masks


@dataclass
class Frame:
    adc: np.ndarray
    products: Products
    metadata: dict
    components: dict | None = None


class SimulationCancelled(Exception):
    """Cooperative UI cancellation; no numerical changes when not requested."""


def synthesize_adc(request: Simulation, scene: CompiledScene, layout: Layout, time_s: float, cancelled=None):
    w, platform = request.waveform, request.platform
    fast = w.adc_start_s+np.arange(w.samples)/w.sample_rate_hz
    slow = time_s+np.arange(w.loops)*layout.loop_period_s
    p0, pv = np.array(platform.position_ned_m), np.array(platform.velocity_ned_mps)
    out = np.zeros((w.samples, w.loops, request.array.num_rx, request.array.num_tx), dtype=np.complex64)
    cosine = np.cos(np.deg2rad(platform.beam_half_angle_deg))
    forward = layout.basis_ned[:, 0]
    trajectory=TrajectoryEvaluator(platform.trajectory) if platform.trajectory is not None else None
    diagnostics = {'illuminated_scatterer_slots': 0, 'partly_sampled_scatterer_slots': 0,
                   'outside_gate_scatterer_slots': 0, 'out_of_band_scatterer_slots': 0}
    visibility_budget(request,scene)
    visibility=VisibilityEvaluator(scene,request.visibility)
    collect=request.visibility.report_coverage or request.visibility.occlusion_enabled
    beam_counts=np.zeros(len(scene.amplitudes),dtype=np.int64)
    clear_counts=np.zeros_like(beam_counts);echo_counts=np.zeros_like(beam_counts)
    diagnostics['occluded_scatterer_channel_slots']=0
    diagnostics['occlusion_evaluated']=request.visibility.occlusion_enabled
    # Bound the [scatterer,sample,loop] transient arrays independently of scene size.
    chunk = max(1, min(16, 1_000_000//(w.samples*w.loops)))
    radiation=request.radiation;aperture_diameter=None
    if radiation is not None and radiation.transmit_aperture is not None:
        elements=np.asarray(radiation.transmit_aperture.positions_local_m)
        aperture_diameter=float(np.linalg.norm(elements[:,None]-elements[None,:],axis=-1).max())
        chunk=min(chunk,max(1,262144//(w.loops*len(elements))))
    for tx, offset in enumerate(layout.tx_offsets_s):
        starts = slow+offset
        if trajectory is not None:
            platform_positions,platform_velocities,bases,angular_velocity=trajectory.at(starts)
            tx_offsets=np.einsum('nij,j->ni',bases,layout.tx_local_m[tx])
            tx_velocities=platform_velocities+np.cross(angular_velocity,tx_offsets)
        masks=clear_masks(request,scene,layout,visibility,starts,tx,
                          (platform_positions,bases) if trajectory is not None else None,cancelled,aperture_diameter)
        for begin in range(0, len(scene.amplitudes), chunk):
            if cancelled is not None and cancelled():raise SimulationCancelled()
            sl = slice(begin, begin+chunk)
            velocity = scene.velocities_ned_mps[sl]-pv
            delta = scene.positions_ned_m[sl, None, :]-p0+velocity[:, None, :]*starts[None, :, None]
            if trajectory is not None:
                delta=scene.positions_ned_m[sl,None,:]+scene.velocities_ned_mps[sl,None,:]*starts[None,:,None]-platform_positions[None,:,:]
            central = np.linalg.norm(delta, axis=-1)
            if np.any(central <= 1e-6):
                raise ValueError('scatterer intersects radar at a Tx slot')
            beam_cosine=np.sum(delta*forward,axis=-1)/central
            if trajectory is not None:
                beam_cosine=np.sum(delta*bases[None,:,:,0],axis=-1)/central
            illuminated=beam_cosine>=cosine
            beam_gain=two_way_voltage_gain(beam_cosine,platform.beam_half_angle_deg,platform.beam_model,platform.beam_taper_deg)
            tx_gain=None
            if radiation is not None:
                tx_delta=delta-(tx_offsets[None,:,:] if trajectory is not None else layout.tx_ned_m[tx])
                tx_gain=transmit_response(tx_delta,bases if trajectory is not None else layout.basis_ned,
                                          starts,radiation,w,aperture_diameter)
                if radiation.replace_legacy_cone:
                    illuminated=abs(tx_gain)>0;beam_gain=np.ones_like(beam_gain)
                else:illuminated&=abs(tx_gain)>0
            diagnostics['illuminated_scatterer_slots'] += int(illuminated.sum())
            if collect:beam_counts[sl]+=illuminated.sum(axis=1)
            if not illuminated.any():
                continue
            tx_clear=np.ones_like(illuminated) if masks is None else masks[0][sl]
            if request.visibility.occlusion_enabled and masks is None:
                for k,t in enumerate(starts):
                    selected=illuminated[:,k]
                    if not selected.any():continue
                    endpoints=scene.positions_ned_m[sl][selected]+scene.velocities_ned_mps[sl][selected]*t
                    origin=(platform_positions[k]+tx_offsets[k] if trajectory is not None
                            else p0+pv*t+layout.tx_ned_m[tx])
                    tx_clear[selected,k]=visibility.clear(origin,endpoints,t)
            dt = delta-layout.tx_ned_m[tx]
            if trajectory is not None:dt=delta-tx_offsets[None,:,:]
            rt = np.linalg.norm(dt, axis=-1)
            for rx, rx_position in enumerate(layout.rx_ned_m):
                if cancelled is not None and cancelled():raise SimulationCancelled()
                dr = delta-rx_position
                if trajectory is not None:
                    rx_offsets=np.einsum('nij,j->ni',bases,layout.rx_local_m[rx])
                    rx_velocities=platform_velocities+np.cross(angular_velocity,rx_offsets)
                    dr=delta-rx_offsets[None,:,:]
                rr = np.linalg.norm(dr, axis=-1)
                if np.any(rt <= 1e-6) or np.any(rr <= 1e-6):
                    raise ValueError('scatterer intersects antenna')
                antenna_gain=None
                if radiation is not None:
                    antenna_gain=tx_gain*receive_response(dr,bases if trajectory is not None else layout.basis_ned,radiation)
                    if w.kind=='fmcw':antenna_gain=antenna_gain.conj()
                visible=illuminated&tx_clear
                if request.visibility.occlusion_enabled:
                    visible=visible.copy()
                    if masks is not None:visible&=masks[1][sl,:,rx]
                    else:
                        for k,t in enumerate(starts):
                            selected=visible[:,k]
                            if not selected.any():continue
                            endpoints=scene.positions_ned_m[sl][selected]+scene.velocities_ned_mps[sl][selected]*t
                            origin=(platform_positions[k]+rx_offsets[k] if trajectory is not None
                                    else p0+pv*t+rx_position)
                            visible[selected,k]&=visibility.clear(origin,endpoints,t)
                    diagnostics['occluded_scatterer_channel_slots']+=int((illuminated&~visible).sum())
                if collect:clear_counts[sl]+=visible.sum(axis=1)
                rate = np.sum(dt*velocity[:, None, :], axis=-1)/rt+np.sum(dr*velocity[:, None, :], axis=-1)/rr
                if trajectory is not None:
                    rate=np.sum(dt*(scene.velocities_ned_mps[sl,None,:]-tx_velocities[None,:,:]),axis=-1)/rt
                    rate+=np.sum(dr*(scene.velocities_ned_mps[sl,None,:]-rx_velocities[None,:,:]),axis=-1)/rr
                path = (rt+rr)[:, None, :]+rate[:, None, :]*fast[None, :, None]
                if np.any(path <= 0):
                    raise ValueError('within-slot linear path crosses antenna; shorten waveform or reduce speed')
                delay = path/C0
                normal = scene.normals_ned[sl, None, :]
                ct = -np.sum(normal*dt, axis=-1)/rt
                cr = -np.sum(normal*dr, axis=-1)/rr
                sides = scene.two_sided[sl, None]
                ct, cr = np.where(sides, abs(ct), ct), np.where(sides, abs(cr), cr)
                directional = np.sqrt(np.maximum(ct, 0)*np.maximum(cr, 0))
                po = scene.scalar_po[sl, None]
                # Scalar single-bounce PO, same angular factor as the static
                # plate solver, but only illuminate the declared visible side.
                po_directional = np.where((ct>0)&(cr>0), .5*(ct+cr), 0.)
                directional = np.where(po, po_directional, directional)
                directional[np.linalg.norm(scene.normals_ned[sl], axis=1) == 0] = 1.
                directional=land_amplitude_factor(scene,sl,ct,cr,dt,dr,rt,rr,visible,w.carrier_hz,directional)
                amplitude = scene.amplitudes[sl, None]*scene.reference_range_m**2/(rt*rr)*directional*visible
                if platform.beam_model!='hard_cone':amplitude*=beam_gain
                # The i phase lives in scene.phases_rad. PO has its own field
                # normalization and must not acquire the point reference range.
                amplitude *= np.where(po, w.carrier_hz/C0/scene.reference_range_m**2, 1.)
                if not np.any(amplitude):
                    continue
                delayed = fast[None, :, None]-delay
                if w.kind == 'pulsed_lfm':
                    active = (delayed >= 0) & (delayed < w.pulse_width_s)
                    phase = np.pi*w.slope_hz_s*(delayed-w.pulse_width_s/2)**2-2*np.pi*w.carrier_hz*delay
                    full = (delay[:, 0, :] >= fast[0]) & (delay[:, -1, :]+w.pulse_width_s <= fast[-1]+1/w.sample_rate_hz)
                    in_band = np.ones_like(illuminated)
                else:
                    active = (delayed >= 0) & (delayed < w.ramp_s)
                    phase = 2*np.pi*(w.start_hz*delay+w.slope_hz_s*fast[None, :, None]*delay-.5*w.slope_hz_s*delay**2)
                    full = (delay[:, 0, :] <= fast[0]) & (fast[-1] <= w.ramp_s)
                    # Complex positive-IF sampling interval [0,fs), not a real-ADC half spectrum.
                    beat = w.slope_hz_s*(rt+rr)/C0+w.carrier_hz*rate/C0
                    in_band = (beat >= 0) & (beat < w.sample_rate_hz)
                    active &= in_band[:, None, :]
                # Scene phase describes the physical reflection coefficient.
                # FMCW tx*conj(rx) conjugates it, unlike the received pulse IQ.
                phase += (1 if w.kind == 'pulsed_lfm' else -1)*scene.phases_rad[sl, None, None]
                if collect:
                    contributing=(amplitude!=0)&active.any(axis=1)
                    if antenna_gain is not None:contributing&=abs(antenna_gain)>0
                    echo_counts[sl]+=contributing.sum(axis=1)
                if rx == 0:
                    any_sample = active.any(axis=1)
                    diagnostics['partly_sampled_scatterer_slots'] += int((illuminated & any_sample & ~full).sum())
                    diagnostics['outside_gate_scatterer_slots'] += int((illuminated & ~any_sample).sum())
                    diagnostics['out_of_band_scatterer_slots'] += int((illuminated & ~in_band).sum())
                if antenna_gain is None:
                    contribution = np.sum(amplitude[:, None, :]*active*np.exp(1j*phase), axis=0)
                else:
                    contribution = np.sum((amplitude*antenna_gain)[:,None,:]*active*np.exp(1j*phase),axis=0)
                out[:, :, rx, tx] += contribution.astype(np.complex64)
    if collect:
        diagnostics['coverage']={
            'positions_at_frame_start_ned_m':(scene.positions_ned_m+scene.velocities_ned_mps*time_s).tolist(),
            'object_index':scene.object_index.tolist(),'objects':[x['id'] for x in scene.objects],
            'illuminated_tx_slots':beam_counts.tolist(),'unblocked_channel_slots':clear_counts.tolist(),
            'sampled_echo_channel_slots':echo_counts.tolist(),
            'tx_slot_denominator':w.loops*request.array.num_tx,
            'channel_slot_denominator':w.loops*request.array.num_tx*request.array.num_rx,
            'occlusion_evaluated':request.visibility.occlusion_enabled,
            'coordinate_frame':'NED_m','positions_epoch_s':time_s,
            'scope':'scatterer samples, not a continuous area fraction; geometry and visibility frozen within each Tx slot'}
    if request.visibility.occlusion_enabled and request.visibility.accelerator=='bvh':
        diagnostics['bvh_work']=visibility.work
    return out, diagnostics


def scene_truth(request, scene, layout, time_s):
    position = np.array(request.platform.position_ned_m)+np.array(request.platform.velocity_ned_mps)*time_s
    platform_velocity=np.array(request.platform.velocity_ned_mps)
    if request.platform.trajectory is not None:
        position,platform_velocity,_,_=TrajectoryEvaluator(request.platform.trajectory).at(time_s)
    points = scene.positions_ned_m+scene.velocities_ned_mps*time_s
    delta = points-position
    radius = np.linalg.norm(delta, axis=1)
    if np.any(radius <= 1e-6):
        raise ValueError('scatterer intersects radar at truth epoch')
    unit = delta/radius[:, None]
    closing = np.sum((platform_velocity-scene.velocities_ned_mps)*unit, axis=1)
    azimuth, elevation = local_angles(unit, layout.basis_ned)
    objects = []
    for index, descriptor in enumerate(scene.objects):
        mask = scene.object_index == index
        center = points[mask].mean(axis=0)
        objects.append({**descriptor, 'center_ned_m': center.tolist(), 'range_extent_m': [float(radius[mask].min()), float(radius[mask].max())],
                        'closing_velocity_extent_mps': [float(closing[mask].min()), float(closing[mask].max())],
                        'azimuth_extent_deg': [float(azimuth[mask].min()), float(azimuth[mask].max())],
                        'elevation_extent_deg': [float(elevation[mask].min()), float(elevation[mask].max())]})
    return objects


def simulate_frame(request: Simulation, frame_index=0, *, compiled=None, cancelled=None, environment=None) -> Frame:
    if not 0 <= frame_index < request.sequence.frame_count:
        raise ValueError('frame index outside requested sequence')
    started = time.perf_counter()
    selected_range_indices(request)  # Reject oversized output before synthesizing ADC.
    scene = compile_scene(request.scene) if compiled is None else compiled
    layout = build_layout(request.array, request.waveform, request.platform)
    time_s = request.sequence.start_time_s+frame_index*request.sequence.frame_interval_s
    if request.platform.trajectory is not None:
        reference_time=time_s+(request.waveform.loops-1)*layout.loop_period_s/2+float(layout.tx_offsets_s.mean())
        reference_time+=request.waveform.adc_start_s+request.waveform.samples/(2*request.waveform.sample_rate_hz)
        _,_,basis,_=TrajectoryEvaluator(request.platform.trajectory).at(reference_time)
        reference_platform=request.platform.model_copy(update={'boresight_ned':tuple(basis[:,0]),'right_hint_ned':tuple(basis[:,1])})
        layout=build_layout(request.array,request.waveform,reference_platform)
    adc, diagnostics = synthesize_adc(request, scene, layout, time_s, cancelled)
    if cancelled is not None and cancelled():raise SimulationCancelled()
    components={};environment_meta=None
    if request.receiver_noise is None and request.range_background is None:
        products = process_adc(adc, request, layout)
    else:
        clean_adc=adc
        noise=make_receiver_noise(request,frame_index)
        adc=(clean_adc+noise).astype(np.complex64)
        ranges=range_compress(adc,request)
        background=np.zeros_like(ranges,dtype=np.complex64)
        if request.range_background is not None:
            environment=prepare_environment(request) if environment is None else environment
            background=environment.frame(request,frame_index)
            environment_meta=environment.metadata
            # Required for replay: saved ADC does not contain this post-range component.
            components['background_range_iq']=background
        products=process_range_iq(ranges+background,request,layout)
        if request.save_components:
            components['clean_rd_channels']=doppler_channels(range_compress(clean_adc,request),request,layout)[0]
            components['receiver_noise_rd_channels']=doppler_channels(range_compress(noise,request),request,layout)[0]
            components['background_rd_channels']=doppler_channels(background,request,layout)[0]
            if request.receiver_noise is not None:components['receiver_noise_adc']=noise
            summed=components['clean_rd_channels']+components['receiver_noise_rd_channels']+components['background_rd_channels']
            products.metadata['component_sum_relative_error']=float(np.max(abs(summed-products.rd_channels))/max(float(np.max(abs(products.rd_channels))),1e-30))
    midpoint = time_s+(request.waveform.loops-1)*layout.loop_period_s/2+float(layout.tx_offsets_s.mean())
    midpoint += request.waveform.adc_start_s+request.waveform.samples/(2*request.waveform.sample_rate_hz)
    warnings = []
    if request.radiation is not None:
        warnings.append('Scalar element patterns and optional coherent Tx subaperture use relative feed-power normalization, not calibrated absolute gain; no mutual coupling or vector polarization.')
        if request.radiation.replace_legacy_cone:
            warnings.append('Legacy cone illumination is replaced by radiation-pattern support; sidelobes are not clipped by the old beam half-angle. Finite scene extent still limits modeled clutter.')
        if request.radiation.transmit_aperture is not None:
            warnings.append('Each logical Tx feeds a narrowband far-field coherent subaperture; logical Tx channels remain TDM. Scan weights update at Tx slots, not within a chirp.')
    if request.visibility.occlusion_enabled:
        warnings.append('Opaque triangle occlusion evaluated on Tx and Rx paths at each Tx slot; no transmission, diffraction or multipath; within-slot shadow changes are not resolved.')
    if request.receiver_noise is not None and request.receiver_noise.fast_time_spectral_power is not None:
        warnings.append('Fast-time colored Gaussian ADC background is an empirical PSD surrogate; slow-time/channels are independent. It is not calibrated thermal noise or a geometry-driven scattering model.')
    if request.platform.beam_model!='hard_cone':
        warnings.append('Cosine beam-edge taper is an uncalibrated common-phase-center Tx/Rx approximation, not a measured antenna pattern; no sidelobes are modeled.')
    if request.platform.trajectory is not None:
        warnings.append('Sampled platform trajectory uses Hermite translation and SLERP orientation at each Tx slot, including rotating-element velocity. RD retains motion; angle processing uses CPI-midpoint orientation and has no motion compensation.')
    if scene.ground_models:
        if any(spec.tables for _,spec in scene.ground_models.values()):
            warnings.append('Ground NRCS tables are interpolated only inside declared frequency/incidence/class/polarization support; evidence labels are retained. Diffuse phases and absolute radar gain remain uncalibrated.')
            if any(t.evidence!='measured' for _,spec in scene.ground_models.values() for t in spec.tables):
                warnings.append('Ground scattering includes assumed or analytic table entries, not measured scene material truth.')
        else:
            warnings.append('Empirical ground coefficients are footprint averages applied to local cells; phases are synthetic and absolute receiver gain/material texture remain unvalidated.')
    if request.range_background is not None:
        warnings.append('Statistical background is injected AFTER range processing with an explicit relative power scale; it is not included in the raw ADC file.')
    if scene.scalar_po.any():
        warnings.append('Scalar PO quadrature is not automatically convergence-certified; compare increasing orders, especially at pulse/gate edges.')
    layout_meta = layout.metadata()
    if request.radiation is not None and request.radiation.transmit_aperture is not None:
        sub=np.asarray(request.radiation.transmit_aperture.positions_local_m)
        physical=layout.tx_local_m[:,None,:]+sub[None,:,:]
        layout_meta.update(logical_tx_channels=request.array.num_tx,
                           physical_tx_elements_per_logical_channel=len(sub),
                           physical_tx_element_count=int(request.array.num_tx*len(sub)),
                           physical_tx_local_m=physical.tolist(),
                           transmit_subaperture_model='far-field coherent elements per logical TDM channel',
                           transmit_weight_normalization='sum squared element drive amplitudes = 1 per logical Tx')
        if len(np.unique(np.round(physical.reshape(-1,3),10),axis=0))<request.array.num_tx*len(sub):
            warnings.append('Transmit subapertures overlap in physical coordinates; independent hardware realization is not validated.')
    if layout_meta['virtual_aperture_rank'] == 0:
        warnings.append('No spatial aperture: angle is not observable.')
    if layout_meta['virtual_aperture_rank'] < 2 and len(request.processing.elevation_deg) > 1:
        warnings.append('Array has fewer than two independent baselines: joint azimuth/elevation is ambiguous.')
    virtual = layout.virtual_local_m
    if layout_meta['unique_virtual_positions'] < len(virtual):
        warnings.append('Some virtual channels share the same position; channel count is not independent aperture size.')
    for axis in [1, 2]:
        positions = np.unique(np.round(virtual[:, axis], 10))
        if len(positions)>1 and np.max(np.diff(positions)) > request.waveform.wavelength_m/2*(1+1e-6):
            warnings.append('Virtual element gaps exceed half a wavelength in a transverse axis; angular grating lobes may occur.')
            break
    for key, text in [('partly_sampled_scatterer_slots', 'Some illuminated echoes only partly overlap the receive gate.'),
                      ('outside_gate_scatterer_slots', 'Some illuminated echoes fall outside the current waveform sampling gate.'),
                      ('out_of_band_scatterer_slots', 'Some FMCW beat frequencies exceed the configured positive-IF sampling band and were excluded.')]:
        if diagnostics[key]:
            warnings.append(text)
    truths = scene_truth(request, scene, layout, midpoint)
    aperture = float(np.linalg.norm(virtual[:, None, :]-virtual[None, :, :],axis=-1).max())
    layout_meta['far_field_reference_distance_m'] = 2*aperture**2/request.waveform.wavelength_m
    if truths and min(t['range_extent_m'][0] for t in truths) < layout_meta['far_field_reference_distance_m']:
        warnings.append('Scene enters the aperture near field; propagation is bistatic but the angle processor is far-field.')
    if any(max(abs(v) for v in t['closing_velocity_extent_mps']) >= products.metadata['max_unambiguous_velocity_mps'] for t in truths):
        warnings.append('Single-loop PRF Doppler ambiguity: reported velocities and TDM correction use aliased bins, not unwrapped truth.')
    return Frame(adc, products, {
        'schema_version': request.schema_version, 'data_origin': 'synthetic_composite_simulation' if request.receiver_noise is not None or request.range_background is not None else 'ideal_simulation', 'is_real': False,
        'frame_index': frame_index, 'frame_start_s': time_s, 'truth_epoch_s': midpoint,
        'elapsed_s': time.perf_counter()-started, 'coordinate_frame': 'NED_m',
        'velocity_convention': 'positive closing', 'platform_position_ned_m': ((np.array(request.platform.position_ned_m)+np.array(request.platform.velocity_ned_mps)*time_s) if request.platform.trajectory is None else TrajectoryEvaluator(request.platform.trajectory).at(time_s)[0]).tolist(),
        'platform_motion': 'constant_velocity_fixed_attitude' if request.platform.trajectory is None else 'sampled_Hermite_translation_SLERP_attitude',
        'beam_response': {'model':('radiation_pattern' if request.radiation is not None and request.radiation.replace_legacy_cone else request.platform.beam_model),'half_angle_deg':request.platform.beam_half_angle_deg,
                          'legacy_cone_applies':not (request.radiation is not None and request.radiation.replace_legacy_cone),
                          'taper_deg':request.platform.beam_taper_deg if request.platform.beam_model!='hard_cone' else None,
                          'measured_pattern':False,'cosine_edge_convention':'one-way power P; two-way voltage factor P; two-way power factor P squared'},
        'waveform': request.waveform.model_dump(), 'scene': {**scene.metadata(),'environment_occlusion':request.visibility.occlusion_enabled}, 'array': layout_meta,
        'visibility_model':request.visibility.model_dump(),
        'radiation_model':None if request.radiation is None else request.radiation.model_dump(mode='json'),
        'processing': products.metadata, 'visibility': diagnostics, 'object_truth': truths,
        'receiver_noise': None if request.receiver_noise is None else {
            **request.receiver_noise.model_dump(), 'stage':'complex_ADC_before_range_processing',
            'variance_definition':'E[abs(n)**2] per complex ADC sample','seed_rule':'SeedSequence([seed,frame_index])',
            'spectral_power_definition':'native fast-time FFT PSD; constant q gives ADC variance q; general per-sample variance is mean(q)' if request.receiver_noise.fast_time_spectral_power is not None else None,
            'absolute_noise_figure_calibrated':False},
        'range_background': environment_meta,
        'adc_contains': 'target_echo_and_optional_receiver_noise; range_background_is_a_separate_product',
        'peak_candidates': peak_candidates(products, request.processing.max_peaks, layout),
        'warnings': warnings,
        'limitations': ['relative amplitude only; no material or absolute RCS calibration',
                       'fixed attitude and constant Cartesian velocities' if request.platform.trajectory is None else 'sampled platform interpolation is not independent trajectory truth; target Cartesian velocities remain constant; no array motion compensation',
                       'linearized bistatic range within each Tx slot; propagation-retardation motion neglected',
                       'centroid or scalar PO surface model; optional opaque-triangle occlusion; no edge diffraction or environmental multipath',
                       'dynamic scalar PO uses carrier-frequency amplitude; finite quadrature requires convergence checks',
                       'far-field narrowband angle processing; no Doppler ambiguity resolution',
                       'optional white or fast-time colored Gaussian ADC background; no calibrated receiver noise figure or other receiver nonidealities',
                       'optional statistical range-domain background is not a geometry-driven environment']}, components or None)
