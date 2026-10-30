#!/usr/bin/env python3
"""Check whether channel-phase residuals remain after eliminating angle uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np
from radar_platform.ramp_alignment import AnnotationTracks,read_ramp_adc,aligned_config,aligned_array
from radar_platform.array_fmcw import process_array_adc
from radar_platform.alignment_metrics import fit_phase_curvature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads(a.calibration.read_text());files=source['calibration_frame_ids'];cfg=aligned_config();array=aligned_array(-1)
    tracks={};vectors=[]
    for name in files:
        file=a.root/name;seq=file.parent.parent.name
        if seq not in tracks:tracks[seq]=AnnotationTracks(file.parent.parent/'text_labels')
        labels=tracks[seq].labels(int(file.stem),smooth=True)
        if len(labels)!=1:raise ValueError('calibration selection changed')
        label=labels[0]
        result=process_array_adc(read_ramp_adc(file),cfg,array,azimuth_deg=[0.],closing_doppler_sign=-1,spatial_phase_sign=-1,
                                 range_window='hamming',doppler_window='hamming',spatial_window='hamming')
        width=1.5+.5*(abs(label['x_m']/label['range_m'])*label['width_m']+abs(label['y_m']/label['range_m'])*label['length_m'])
        rr=np.flatnonzero((abs(result.range_m-label['range_m'])<=width)&(result.range_m>=1)&(result.range_m<=24))
        power=np.mean(abs(result.rd_channels)**2,axis=(2,3));r,d=np.unravel_index(np.argmax(power[rr]),(len(rr),cfg.chirps));r=int(rr[r])
        vector=result.rd_channels[r,d].T.reshape(8)
        f=-result.closing_doppler_hz[d]
        vectors.append(vector*np.exp(-2j*np.pi*f*np.repeat(array.tx_offsets_s,4)))
    result=fit_phase_curvature(vectors);result['calibration_frame_ids']=files
    (a.output/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.output/'calibration_vectors.npz',vectors=np.asarray(vectors))
    print(json.dumps({k:v for k,v in result.items() if k!='calibration_frame_ids'},indent=2))


if __name__=='__main__':main()
