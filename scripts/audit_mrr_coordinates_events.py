#!/usr/bin/env python3
"""Document instrument-native velocity and source event associations, not drop truth."""
from datetime import timezone
from pathlib import Path
from collections import Counter
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from radar_platform.mrr_dataset import load_mrr_spectra,load_precipitation_events,precipitation_label
from render_scene import configure_font

def main():
    root=Path('/nas/datasets/hyt/datasets/radar_open/mrr_ace')
    out=Path(__file__).resolve().parents[1]/'docs/land_extension/2026-10-15'
    for p in [out/'images',out/'metrics']:p.mkdir(parents=True,exist_ok=True)
    events,rejected=load_precipitation_events(root/'raw/PrecipitationEvents.txt',event_timezone=timezone.utc)
    files=sorted((root/'extracted').rglob('*.nc'));rows=[];configure_font()
    fig,axes=plt.subplots(1,len(files),figsize=(14,4),layout='constrained')
    for ax,path in zip(np.atleast_1d(axes),files):
        data=load_mrr_spectra(path);v=data.instrument_velocity_axis()
        labels=[precipitation_label(t,events) for t in data.utc]
        counts=Counter(x['label'] for x in labels)
        good=(data.quality_percent>=50)&(data.good_spectra_count>0)&np.any(np.isfinite(data.power),axis=(1,2))
        available=np.flatnonzero(good);index=int(available[len(available)//2])
        h=data.height_m>0;power=data.power[index,h];reference=float(np.nanmax(power))
        im=ax.pcolormesh(v,data.height_m[h],10*np.log10(np.maximum(power/reference,1e-6)),shading='auto',cmap='turbo',vmin=-60,vmax=0)
        fig.colorbar(im,ax=ax,label='本图峰值参考 / dB',shrink=.8)
        label=labels[index]['label']
        ax.set(xlabel='仪器原生表观下落速度 / (m/s)',ylabel='相对仪器高度 / m',title=f'{data.utc[index]}\n事件关联：{label}')
        rows.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(data.utc),
            'event_counts_all_records':dict(counts),'display_index':index,'display_label':labels[index],
            'native_velocity_mps':[float(v[0]),float(v[-1])],'native_bin_spacing_mps':data.velocity_bin_spacing_mps,
            'alias_period_mps':float(64*data.velocity_bin_spacing_mps),
            'uses_file_specific_spacing':True,'dealiasing_performed':False,'ship_motion_or_vertical_wind_corrected':False,
            'event_timezone_assumption':'UTC; event table does not itself declare a zone',
            'missing_label_means':'unlabelled, not confirmed dry/no-precipitation'})
    fig.suptitle('MRR 数据口径补齐：仪器原生坐标可解释，但不等于真实雨滴下落速度\n约 0–12 m/s 是有模糊的仪器区间；未做解模糊、船体运动或垂直风修正；功率仍是工程单位',fontsize=11)
    fig.savefig(out/'images/mrr_coordinates_events.png',dpi=155);plt.close(fig)
    result={'source':'https://zenodo.org/records/3929289','license':'CC BY 4.0',
        'coordinate_sources':['https://metek.de/faqs/','https://amt.copernicus.org/articles/5/2661/2012/',
                              'https://github.com/maahn/IMProToo/blob/master/IMProToo/core.py'],
        'coordinate_interpretation':'standard MRR-2 zero-origin fall-positive raw bins; file-specific spacing, not another instrument default',
        'valid_event_intervals':len(events),'malformed_source_rows_retained':rejected,'files':rows,
        'known_hardware_affected_bins_from_standard_MRR2_paper':[0,1,63],
        'hardware_mask_applied':False,'no_physical_precipitation_forward_validation_claimed':True}
    (out/'metrics/mrr_coordinate_event_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
