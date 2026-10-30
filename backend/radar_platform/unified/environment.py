"""Stage-aware stochastic components for generic mixed radar scenes.

Noise variance is in raw complex ADC units. Background power is explicitly
scaled into normalized range-processor output units, never added to ADC or dB.
Background is one finite stochastic timeline per requested sequence; no
independent-call phase-continuity or geometry-driven environment claim.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import numpy as np
from ..statistical_background import simulate_background
from .geometry import build_layout,scan_directions
from .processing import range_axis,selected_range_indices


def environment_signature(request):
    data=request.model_dump(mode='json',include={'range_background','waveform','array','platform','processing','sequence'})
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


@dataclass
class EnvironmentRealization:
    background: np.ndarray  # [timeline,provided range,rx,tx]
    range_indices: np.ndarray
    samples_per_frame_interval: int
    metadata: dict

    def frame(self,request,index):
        if environment_signature(request)!=self.metadata['configuration_sha256']:
            raise ValueError('cached background belongs to a different acquisition configuration')
        count=len(selected_range_indices(request));w=request.waveform
        result=np.zeros((count,w.loops,request.array.num_rx,request.array.num_tx),np.complex64)
        start=index*self.samples_per_frame_interval
        result[self.range_indices]=self.background[start:start+w.loops].transpose(1,0,2,3)
        return result


def receiver_noise(request,index):
    spec=request.receiver_noise
    w,a=request.waveform,request.array
    shape=(w.samples,w.loops,a.num_rx,a.num_tx)
    if spec is None:return np.zeros(shape,np.complex64)
    if spec.fast_time_spectral_power is not None:
        from ..colored_adc_background import colored_adc_noise
        return colored_adc_noise(shape,spec.fast_time_spectral_power,np.random.SeedSequence([spec.seed,index]))
    variance=np.asarray(spec.variance_per_rx_tx,float) if spec.variance_per_rx_tx is not None else np.full((a.num_rx,a.num_tx),spec.variance)
    rng=np.random.default_rng(np.random.SeedSequence([spec.seed,index]))
    x=(rng.standard_normal(shape)+1j*rng.standard_normal(shape))*np.sqrt(variance[None,None,:,:]/2)
    return x.astype(np.complex64)


def prepare_environment(request):
    spec=request.range_background
    if spec is None:return None
    layout=build_layout(request.array,request.waveform,request.platform)
    step=round(request.sequence.frame_interval_s/layout.loop_period_s)
    count=max(512,(request.sequence.frame_count-1)*step+request.waveform.loops+1)
    axes=range_axis(request)[selected_range_indices(request)]
    indices=np.argmin(abs(np.asarray(spec.model.range_m)[:,None]-axes[None,:]),axis=1)
    if np.any(abs(axes[indices]-spec.model.range_m)>1e-6):raise ValueError('background range grid changed after validation')
    nr=len(indices);a=request.array
    data=np.empty((count,nr,a.num_rx,a.num_tx),np.complex64)
    frequency=np.fft.fftfreq(count,layout.loop_period_s)
    model=spec.model
    remove=spec.floor_policy=='remove_receiver_like_floor'
    removed_fraction=model.white_fraction if remove else 0.
    if remove and removed_fraction<1:
        # Preserve clutter power instead of redistributing the removed floor.
        model=replace(model,white_fraction=0.,mean_power=tuple((1-removed_fraction)*np.asarray(model.mean_power)))
    if remove and removed_fraction==1:
        data.fill(0)
    elif spec.channel_mode=='steered':
        base=simulate_background(model,count,spec.seed)
        direction=scan_directions([spec.azimuth_deg],[spec.elevation_deg])[0,0]
        virtual=layout.virtual_local_m.reshape(a.num_tx,a.num_rx,3)
        sign=1 if request.waveform.kind=='pulsed_lfm' else -1
        phase=np.exp(sign*2j*np.pi/request.waveform.wavelength_m*(virtual@direction))
        spectrum=np.fft.fft(base,axis=0)
        for tx,dt in enumerate(layout.tx_offsets_s):
            # Band-limited time shift: different Tx slots observe the same field
            # at different times, before any TDM phase compensation downstream.
            shifted=np.fft.ifft(spectrum*np.exp(2j*np.pi*frequency[:,None]*dt),axis=0)
            for rx in range(a.num_rx):data[:,:,rx,tx]=shifted*phase[tx,rx]
    else:
        seeds=np.random.SeedSequence(spec.seed).generate_state(a.num_tx*a.num_rx)
        for tx in range(a.num_tx):
            for rx in range(a.num_rx):
                data[:,:,rx,tx]=simulate_background(model,count,int(seeds[tx*a.num_rx+rx]))
    data*=np.sqrt(spec.power_scale)
    return EnvironmentRealization(data,indices,step,{
        'configuration_sha256':environment_signature(request),
        'injection_stage':'after_range_processing_before_doppler_window',
        'source_domain':spec.model.domain,'output_power_units':'normalized range-processor units',
        'power_scale':spec.power_scale,'absolute_power_calibration_verified':False,'provenance':spec.provenance,
        'floor_policy':spec.floor_policy,'source_model_floor_fraction_removed':removed_fraction,
        'floor_physical_identification':'receiver-like statistical component; not independently identified thermal noise',
        'raw_slow_time_frequency_convention':True,'channel_mode':spec.channel_mode,
        'angle_assumption':{'azimuth_deg':spec.azimuth_deg,'elevation_deg':spec.elevation_deg} if spec.channel_mode=='steered' else 'independent channels; not a measured angular clutter covariance',
        'timeline_samples':count,'provided_range_bins':len(indices),'unprovided_bins':'zero background contribution',
        'timeline_origin_s':request.sequence.start_time_s,
        'same_sequence_uses_one_realization':True,'cross_request_phase_continuity_promised':False,
        'geometry_driven_environment':False,'TDMTx_time_shift_applied':spec.channel_mode=='steered',
        'no_invented_raw_ADC_for_range_background':True})
