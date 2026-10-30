#!/usr/bin/env python3
"""Transfer real multi-channel ADC into the shared processor; no label/target fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

from radar_platform.ramp_alignment import read_ramp_adc, process_aligned_adc, aligned_array
from radar_platform.unified.schema import Acquisition
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.processing import process_adc, selected_range_indices
from radar_platform.unified.storage import write_json


def acquisition():
    array=aligned_array(-1)
    return Acquisition.model_validate({
        'waveform':{'kind':'fmcw','start_hz':77e9,'slope_hz_s':21e12,'ramp_s':32e-6,'slot_period_s':60e-6,
                    'adc_start_s':0.,'samples':128,'loops':255,'sample_rate_hz':4e6},
        'platform':{'position_ned_m':[0,0,0],'boresight_ned':[1,0,0],'right_hint_ned':[0,1,0]},
        'array':{'layout':'custom','num_tx':2,'num_rx':4,'tx_local_m':array.tx_ned_m.tolist(),
                 'rx_local_m':array.rx_ned_m.tolist(),'tx_offsets_s':array.tx_offsets_s.tolist(),'loop_period_s':array.loop_period_s},
        'processing':{'range_limits_m':[1,24],'range_window':'hamming','doppler_window':'hamming','spatial_window':'hamming',
                      'azimuth_deg':list(np.linspace(-80,80,81)),'elevation_deg':[0.],'tdm_compensation':True},
    })


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    selected=json.loads((args.root/'selection.json').read_text())['frames']
    config=acquisition();layout=build_layout(config.array,config.waveform,config.platform)
    write_json(args.output/'protocol.json',{'acquisition':config.model_dump(mode='json'),'frames':selected,
        'uses_scene_or_target_labels':False,'native_range_bins':128,
        'expected_cropped_rad_shape':[len(selected_range_indices(config)),255,1,81],
        'range_crop_m':[1,24],'relative_complex_tolerance':2e-5,
        'scope':'real-ADC pipeline transfer/regression, not measured angular accuracy or a scene digital twin',
        'orientation':'fixed nominal geometry and previously established handedness; no new calibration',
        'frequency_origin':'published start-frequency approximation; physical ADC-start delay remains unknown'})
    project=Path(__file__).resolve().parents[1]
    files=[*project.glob('backend/radar_platform/unified/*.py'),project/'backend/radar_platform/ramp_alignment.py',
           project/'backend/radar_platform/array_fmcw.py',Path(__file__).resolve()]
    with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(f,f.relative_to(project))
    rows=[]
    for index,spec in enumerate(selected):
        path=args.root/spec['frame_id'];adc=read_ramp_adc(path)
        before=process_aligned_adc(adc);after=process_adc(adc,config,layout)
        keep=(before.range_m>=1)&(before.range_m<=24)
        assert np.array_equal(before.range_m[keep],after.range_m)
        np.testing.assert_allclose(before.velocity_mps,after.velocity_mps,rtol=1e-12,atol=1e-12)
        def error(a,b):return float(np.max(abs(a-b))/max(float(np.max(abs(b))),1e-30))
        rd_error=error(after.rd_channels,before.rd_channels[keep])
        rad_error=error(after.rad[:,:,0,:],before.rad[keep])
        rows.append({'frame_id':spec['frame_id'],'sequence':spec['sequence'],'adc_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'rd_relative_complex_error':rd_error,'rad_relative_complex_error':rad_error,
                     'channels_preserved':after.rd_channels.shape[2:]==(4,2),'passed':rd_error<2e-5 and rad_error<2e-5})
        if index%50==0:print('real ADC transfer',index,'/',len(selected),flush=True)
    with (args.output/'frames.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'frames':len(rows),'sequences':len({r['sequence'] for r in rows}),'passed':sum(r['passed'] for r in rows),
             'max_RD_relative_complex_error':max(r['rd_relative_complex_error'] for r in rows),
             'max_RAD_relative_complex_error':max(r['rad_relative_complex_error'] for r in rows),
             'scope':'processing compatibility, not physical fidelity certification','labels_used':False,'is_real_input':True}
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if summary['passed']!=len(rows):raise SystemExit(1)


if __name__=='__main__':main()
