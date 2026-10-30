"""Common physical axes and coherent array processing for both waveform families."""
from dataclasses import dataclass
import numpy as np
from scipy.fft import next_fast_len

from ..models import C0
from .geometry import Layout, scan_directions
from .schema import Acquisition


def window(name, count):
    # Two-element Hann is identically zero; a unity taper is explicit and useful.
    if count <= 2:
        return np.ones(count)
    return {'hann': np.hanning, 'hamming': np.hamming, 'rectangular': np.ones}[name](count)


def spatial_weights(layout, name):
    """Coordinate-based taper, invariant to channel storage permutation."""
    positions = layout.virtual_local_m
    result = np.ones(len(positions))
    for dimension in [1, 2]:
        unique, inverse = np.unique(np.round(positions[:, dimension], 10), return_inverse=True)
        result *= window(name, len(unique))[inverse]
    return result


def range_axis(request):
    w = request.waveform
    if w.kind == 'pulsed_lfm':
        return (w.adc_start_s+np.arange(w.samples)/w.sample_rate_hz)*C0/2
    count = request.processing.range_fft_size or w.samples
    return np.arange(count)*w.sample_rate_hz/count*C0/(2*w.slope_hz_s)


def selected_range_indices(request):
    r = range_axis(request)
    lower, upper = request.processing.range_limits_m
    keep = np.flatnonzero((r >= lower) & (r <= upper))
    if not len(keep):
        raise ValueError('processing range limits do not intersect sampled range bins')
    p, w = request.processing, request.waveform
    nd = p.doppler_fft_size or w.loops
    if len(keep)*nd*len(p.azimuth_deg)*len(p.elevation_deg) > 16_000_000:
        raise ValueError('RAD exceeds 16 million complex cells; restrict physical range/angle span or sampling')
    return keep


@dataclass
class Products:
    rd_channels: np.ndarray  # [range,closing_doppler,rx,tx], NOT spatially averaged
    rad: np.ndarray  # [range,closing_doppler,elevation,azimuth]
    rd_power: np.ndarray
    ra_power: np.ndarray
    range_m: np.ndarray
    closing_doppler_hz: np.ndarray
    velocity_mps: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    metadata: dict


def beamform(rd_channels, raw_slow_hz, request, layout):
    p, w = request.processing, request.waveform
    sign = 1 if w.kind == 'pulsed_lfm' else -1
    compensated = np.array(rd_channels, dtype=np.complex128, copy=True)
    if p.tdm_compensation:
        compensated *= np.exp(-2j*np.pi*raw_slow_hz[None, :, None, None]*layout.tx_offsets_s[None, None, None, :])
    virtual = compensated.transpose(0, 1, 3, 2).reshape(*compensated.shape[:2], -1)
    directions = scan_directions(p.azimuth_deg, p.elevation_deg).reshape(-1, 3)
    steering = np.exp(sign*2j*np.pi/w.wavelength_m*(directions@layout.virtual_local_m.T))
    weights = spatial_weights(layout, p.spatial_window)
    result = np.empty((*rd_channels.shape[:2], len(directions)), dtype=np.complex64)
    for start in range(0, len(result), 32):
        result[start:start+32] = np.einsum('rdc,ac->rda', virtual[start:start+32]*weights, steering.conj(), optimize=True)/weights.sum()
    return result.reshape(*rd_channels.shape[:2], len(p.elevation_deg), len(p.azimuth_deg))


def range_compress(adc, request: Acquisition):
    """[fast,slow,rx,tx] -> selected [range,slow,rx,tx], before Doppler window."""
    w, p = request.waveform, request.processing
    x = np.asarray(adc)
    expected = (w.samples, w.loops, request.array.num_rx, request.array.num_tx)
    if x.shape != expected or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError(f'finite complex ADC with shape {expected} required')
    keep = selected_range_indices(request)
    if w.kind == 'fmcw':
        rw = window(p.range_window, w.samples)
        ranges = np.fft.fft(x*rw[:, None, None, None], p.range_fft_size or w.samples, axis=0)/rw.sum()
    else:
        times = np.arange(int(np.ceil(w.pulse_width_s*w.sample_rate_hz)))/w.sample_rate_hz
        tx = np.exp(1j*np.pi*w.slope_hz_s*(times-w.pulse_width_s/2)**2)
        rw = window(p.range_window, len(times))
        kernel = (tx*rw).conj()[::-1]
        nfft = next_fast_len(w.samples+len(tx)-1)
        convolution = np.fft.ifft(np.fft.fft(x, nfft, axis=0)*np.fft.fft(kernel, nfft)[:, None, None, None], axis=0)/rw.sum()
        ranges = convolution[len(tx)-1:len(tx)-1+w.samples]
    return ranges[keep]


def doppler_channels(ranges, request: Acquisition, layout: Layout):
    """Linear Doppler transform retaining every spatial channel."""
    w,p=request.waveform,request.processing
    keep=selected_range_indices(request)
    expected=(len(keep),w.loops,request.array.num_rx,request.array.num_tx)
    ranges=np.asarray(ranges)
    if ranges.shape!=expected or not np.iscomplexobj(ranges) or not np.isfinite(ranges).all():
        raise ValueError(f'finite complex range IQ with shape {expected} required')
    sign=1 if w.kind=='pulsed_lfm' else -1
    dw = window(p.doppler_window, w.loops)
    nd = p.doppler_fft_size or w.loops
    channels = np.fft.fftshift(np.fft.fft(ranges*dw[None, :, None, None], nd, axis=1), axes=1)/dw.sum()
    raw_slow = np.fft.fftshift(np.fft.fftfreq(nd, layout.loop_period_s))
    closing = sign*raw_slow
    order = np.argsort(closing)
    channels, closing, raw_slow = channels[:, order], closing[order], raw_slow[order]
    channels = channels.astype(np.complex64)
    return channels,closing,raw_slow


def process_range_iq(ranges, request: Acquisition, layout: Layout) -> Products:
    w,p=request.waveform,request.processing
    keep=selected_range_indices(request)
    sign=1 if w.kind=='pulsed_lfm' else -1
    channels,closing,raw_slow=doppler_channels(ranges,request,layout)
    rad = beamform(channels, raw_slow, request, layout)
    rd_power = np.mean(abs(channels.astype(np.complex128))**2, axis=(2, 3)).astype(np.float32)
    ra_power = np.mean(abs(rad.astype(np.complex128))**2, axis=1).astype(np.float32)
    metadata = {'adc_axes': ['sample', 'loop', 'rx', 'tx'],
                'rd_channels_axes': ['range', 'closing_doppler', 'rx', 'tx'],
                'rad_axes': ['range', 'closing_doppler', 'elevation', 'azimuth'],
                'rd_power_axes': ['range', 'closing_doppler'], 'ra_power_axes': ['range', 'elevation', 'azimuth'],
                'range_resolution_m': C0/(2*w.bandwidth_hz),
                'native_velocity_cell_mps': w.wavelength_m/(2*w.loops*layout.loop_period_s),
                'max_unambiguous_velocity_mps': w.wavelength_m/(4*layout.loop_period_s),
                'slow_frequency_to_closing_sign': sign, 'spatial_phase_sign': sign,
                'tdm_compensation': p.tdm_compensation, 'compensation_uses_truth': False,
                'tdm_ambiguity_resolved': False, 'range_doppler_coupling_corrected': False,
                'windows': {'range': p.range_window, 'doppler': p.doppler_window, 'spatial': p.spatial_window},
                'angle_method': 'far-field coherent Bartlett in sensor-local coordinates',
                'rd_average': 'noncoherent mean power over physical Tx/Rx pairs',
                'ra_average': 'linear-power Doppler mean of coherent RAD',
                'angular_grid_is_resolution': False, 'absolute_power_calibrated': False}
    return Products(channels, rad, rd_power, ra_power, range_axis(request)[keep], closing,
                    closing*w.wavelength_m/2, np.array(p.azimuth_deg), np.array(p.elevation_deg), metadata)


def process_adc(adc, request: Acquisition, layout: Layout) -> Products:
    return process_range_iq(range_compress(adc,request),request,layout)


def peak_candidates(products: Products, count: int, layout: Layout):
    """Truth-free RD peak candidates; not CFAR, tracking or multi-target association."""
    power = products.rd_power.copy()
    maximum = float(power.max())
    if maximum <= 0:
        return []
    virtual = layout.virtual_local_m
    az_observable = np.ptp(virtual[:, 1]) > 1e-9 and len(products.azimuth_deg)>1
    el_observable = np.ptp(virtual[:, 2]) > 1e-9 and len(products.elevation_deg)>1
    joint_observable = az_observable and el_observable and layout.metadata()['virtual_aperture_rank']>=2
    results = []
    for _ in range(count):
        r, d = np.unravel_index(np.argmax(power), power.shape)
        peak = float(power[r, d])
        if peak < maximum*10**(-25/10):
            break
        el, az = np.unravel_index(np.argmax(abs(products.rad[r, d])), products.rad.shape[2:])
        results.append({'range_m': float(products.range_m[r]), 'apparent_closing_velocity_mps': float(products.velocity_mps[d]),
                        'azimuth_deg': float(products.azimuth_deg[az]) if az_observable else None,
                        'elevation_deg': float(products.elevation_deg[el]) if el_observable else None,
                        'power_relative_db': float(10*np.log10(peak/maximum)),
                        'range_bin': int(r), 'doppler_bin': int(d), 'elevation_bin': int(el), 'azimuth_bin': int(az),
                        'uses_target_truth': False, 'angle_conditional_on_scan_plane': not joint_observable})
        ds = np.arange(d-2, d+3) % power.shape[1]
        power[np.ix_(np.arange(max(0, r-3), min(len(power), r+4)), ds)] = 0
    return results


def array_factor(layout, wavelength_m, azimuth_deg, elevation_deg, look_azimuth_deg=0., look_elevation_deg=0., taper='rectangular'):
    scan = scan_directions(azimuth_deg, elevation_deg)
    look = scan_directions([look_azimuth_deg], [look_elevation_deg])[0, 0]
    phase = 2*np.pi/wavelength_m*((scan-look)@layout.virtual_local_m.T)
    weights = spatial_weights(layout, taper)
    return np.sum(np.exp(1j*phase)*weights, axis=-1)/weights.sum()
