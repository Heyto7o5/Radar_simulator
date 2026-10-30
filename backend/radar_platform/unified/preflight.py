"""Resource checks shared by every simulation entry point."""
from .scene import compile_scene
from .processing import selected_range_indices
from .applicability import assess_scene
from .visibility import visibility_budget


def preflight(request):
    keep = selected_range_indices(request)
    scene = compile_scene(request.scene)
    w, a, p = request.waveform, request.array, request.processing
    nd = p.doppler_fft_size or w.loops
    adc = w.samples*w.loops*a.num_tx*a.num_rx*8
    rd = len(keep)*nd*a.num_tx*a.num_rx*8
    rad = len(keep)*nd*len(p.azimuth_deg)*len(p.elevation_deg)*8
    total = request.sequence.frame_count*((adc if p.save_adc else 0)+rd+rad+len(keep)*nd*4+len(keep)*len(p.azimuth_deg)*len(p.elevation_deg)*4)
    if request.range_background is not None:
        total += request.sequence.frame_count*len(keep)*w.loops*a.num_rx*a.num_tx*8
    if request.save_components and (request.receiver_noise is not None or request.range_background is not None):
        total += request.sequence.frame_count*(3*rd+(adc if request.receiver_noise is not None and p.save_adc else 0))
    if total > 2_000_000_000:
        raise ValueError('sequence output exceeds 2 GB; reduce frames, processing span, grid or disable ADC persistence')
    work = w.samples*w.loops*a.num_rx*a.num_tx*len(scene.amplitudes)
    if request.radiation is not None and request.radiation.transmit_aperture is not None:
        work+=len(scene.amplitudes)*w.loops*a.num_tx*len(request.radiation.transmit_aperture.positions_local_m)
    if work > 4_000_000_000:
        raise ValueError('scene exceeds CPU work guard; coarsen geometry explicitly or use a smaller sampled gate')
    visibility=visibility_budget(request,scene)
    assessment=assess_scene(request,scene)
    return scene, {'estimated_output_bytes': total, 'sample_scatter_channel_work_per_frame': work,
                   'scatterers': len(scene.amplitudes),'applicability':assessment,'visibility':visibility}
