#!/usr/bin/env python3
"""Separate zero predicted channels from genuinely missing target-region energy."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from radar_platform.maroon_full import load_full_frame,full_scene_geometry
from radar_platform.surface_echo import range_response
from radar_platform.repeat_metrics import distribution


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--published-run',type=Path,required=True)
    p.add_argument('--rigid-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path('/nas/datasets/hyt/datasets/radar_open/maroon_static_subset/raw/maroon')
    rows=[]
    for scene in ['24_corner_reflector/30','24_corner_reflector/50']:
        mesh,radar=full_scene_geometry(root/scene)
        model_masks={}
        for method,path in [('published',a.published_run),('rigid',a.rigid_run)]:
            with np.load(path/(scene.replace('/','_')+'_predictions.npz')) as z:model_masks[method]=np.any(abs(z['before'])>0,axis=-1)
        files=sorted((root/scene/'radar_72.0_82.0_128/calibrated_data').glob('*_emptyfiltered.npy'))
        for index,file in enumerate(files):
            if scene.endswith('/30') and index<5:continue
            spectrum,f,meta=load_full_frame(file);response,ranges=range_response(spectrum,f);power=abs(response).astype(float)**2
            valid=np.any(abs(spectrum)>0,axis=-1)
            radii=np.linalg.norm(mesh.vertices_m,axis=1)
            roi=(ranges>=radii.min()-.05)&(ranges<=radii.max()+.05)
            bg=(ranges>=radii.max()+.15)&(ranges<=radii.max()+.45)
            prominence=10*np.log10(np.maximum(power[:,:,roi].max(axis=2),1e-30)/np.maximum(np.median(power[:,:,bg],axis=2),1e-30))
            roi_power=power[:,:,roi].sum(axis=2)
            for method,predicted in model_masks.items():
                missing=valid & ~predicted
                rows.append({'scene':scene,'frame':file.name,'geometry':method,'valid_channels':int(valid.sum()),
                    'predicted_channel_fraction':float(predicted[valid].mean()),
                    'measured_roi_energy_in_zero_prediction_channels_fraction':float(roi_power[missing].sum()/roi_power[valid].sum()),
                    'zero_prediction_channels_with_20db_roi_peak':int(np.sum(missing&(prominence>=20))),
                    'zero_prediction_channels_total':int(missing.sum()),
                    'zero_prediction_20db_peak_fraction_of_valid':float(np.sum(missing&(prominence>=20))/valid.sum()),
                    'scope':'ROI energy includes residual background; 20dB peak-to-background is a diagnostic, not calibrated CFAR or exact target power'})
    with (a.output/'frame_audit.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for scene in ['24_corner_reflector/30','24_corner_reflector/50']:
        for method in ['published','rigid']:
            group=[r for r in rows if r['scene']==scene and r['geometry']==method]
            summaries.append({'scene':scene,'geometry':method,
                **{key:distribution([r[key] for r in group]) for key in ['predicted_channel_fraction',
                   'measured_roi_energy_in_zero_prediction_channels_fraction','zero_prediction_channels_with_20db_roi_peak','zero_prediction_20db_peak_fraction_of_valid']}})
    (a.output/'summary.json').write_text(json.dumps({'results':summaries,'threshold_db':20,'no_model_fitted':True},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
