#!/usr/bin/env python3
"""Reconcile official acquisition template with published axes, not fit RD peaks."""
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
from radar_platform.models import C0
from radar_platform.unified.schema import FMCW, Array

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    urls={
        'manual_init.lua':'https://raw.githubusercontent.com/thetianshuhuang/rover/main/radar/templates/manual_init.lua',
        'radar_processing.py':'https://raw.githubusercontent.com/thetianshuhuang/rover/main/processing/process/radar.py',
        'frame_alignment.py':'https://raw.githubusercontent.com/WiseLabCMU/dart/main/tools/dataset.py'}
    sources=[];folder=ROOT/'raw/source_snapshots';folder.mkdir(parents=True,exist_ok=True)
    for name,url in urls.items():
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read(200001)
        if len(data)>200000:raise ValueError('unexpected source size')
        (folder/name).write_bytes(data)
        sources.append({'url':url,'file':str(folder/name),'sha256':hashlib.sha256(data).hexdigest()})
    # Explicit transcription of public template, not inferred from a measured peak.
    # Slot length differs from same-Tx repetition; two Tx chirps occupy only part of 1 ms.
    waveform=FMCW(start_hz=77e9,slope_hz_s=63.343e12,ramp_s=63.14e-6,
                  slot_period_s=73.14e-6,adc_start_s=6e-6,sample_rate_hz=9.121e6,
                  samples=512,loops=256)
    array=Array(layout='ula',num_tx=2,num_rx=4,tx_offsets_s=[0.,73.14e-6],loop_period_s=.001)
    sensor=json.loads((ROOT/'extracted/lab-1/sensor.json').read_text())
    ranges=np.linspace(*sensor['r']);vel=np.linspace(*sensor['d'])
    dr=C0*waveform.sample_rate_hz/(2*waveform.slope_hz_s*waveform.samples)
    vmax=waveform.wavelength_m/(4*.001)
    report={'status':'candidate_configuration_only','sources':sources,
            'acquisition_template_is_not_record_specific_log':True,
            'waveform':waveform.model_dump(),'array_timing_candidate':array.model_dump(),
            'array_element_positions_verified':False,'velocity_sign_verified':False,
            'derived':{'adc_duration_s':waveform.samples/waveform.sample_rate_hz,
                       'adc_bandwidth_hz':waveform.bandwidth_hz,'effective_carrier_hz':waveform.carrier_hz,
                       'native_range_bin_m_exact_c':dr,'published_range_bin_m':float(np.diff(ranges)[0]),
                       'relative_range_bin_difference':float(dr/np.diff(ranges)[0]-1),
                       'unambiguous_speed_mps_exact_c':vmax,'published_max_abs_speed_mps':float(abs(vel[0])),
                       'cpi_s':256*.001,'published_frame_stride_s':64*.001},
            'published_processing':{'range_fft':512,'range_bins_start':2,'range_bins_kept':128,
                                    'range_axis_offset_m':-.05671,'doppler_fft':256,'angle_fft':8,
                                    'range_and_doppler_window':'symmetric Hann',
                                    'zero_doppler_correction':'subtract per-range-angle median over frames from center 3 velocity bins; clip to zero',
                                    'output':'magnitude divided by 1e6, clipped to 65504 and cast float16'},
            'do_not_silently_replace_published_axes':True,
            'next_checks':['record-specific acquisition configuration unavailable',
                           'Tx/Rx physical coordinate and ordering',
                           'velocity sign and ADC conjugation convention',
                           'background subtraction fit scope; must not use held-out frames for calibration',
                           'range zero calibration reference and quantization'],
            'not_independent_scattering_validation':True}
    (OUT/'radar_configuration_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['derived']),flush=True)

if __name__=='__main__':main()
