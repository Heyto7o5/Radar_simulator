#!/usr/bin/env python3
"""Exact-geometry full MAROON: frozen reference, all available repeated frames."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from radar_platform.maroon_full import load_full_frame,full_scene_geometry
from radar_platform.surface_echo import simulate_surface_spectrum,range_response
from radar_platform.validation_metrics import fit_complex_gain
from radar_platform.comparison import compare_power,representative_indices
from radar_platform.repeat_metrics import distribution
from validate_common_real import plot_curves
from matplotlib import font_manager
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((a.root/'selection.json').read_text())
    calibration_scene='maroon/24_corner_reflector/30'
    metadata=[];excluded=[];forward={};measurements={}
    for name in selection['scenes']:
        scene=a.root/name
        expected=sorted(n for values in selection['members'].values() for n in values if n.startswith(name+'/') and n.endswith('.npy'))
        files=[a.root/n for n in expected]
        missing=[str(f) for f in files if not f.exists()]
        if missing:
            excluded.append({'scene':name,'reason':'missing selected frames','files':missing});continue
        try:
            mesh,radar=full_scene_geometry(scene)
            if len(mesh.faces)>32:
                excluded.append({'scene':name,'reason':'mesh exceeds current exact-image solver size; no undocumented simplification',
                                 'triangles':len(mesh.faces),'downloaded_frames':len(files)});continue
            response=simulate_surface_spectrum(mesh,radar)
            np.savez_compressed(a.output/(name.replace('/','_')+'_forward.npz'),signal=response.signal,
                                 vertices_ned_m=mesh.vertices_m,faces=mesh.faces,frequencies_hz=radar.frequencies_hz)
            forward[name]=response.signal
            frames=[]
            for file in files:
                signal,f,meta=load_full_frame(file)
                if not np.array_equal(f,radar.frequencies_hz):
                    if not np.allclose(f,radar.frequencies_hz,rtol=0,atol=1e-2):raise ValueError('frequency mismatch')
                frames.append((file,signal,meta))
            measurements[name]=frames
            metadata.append({'scene':name,'frames':len(frames),'triangles':len(mesh.faces),
                             'vertices_ned_m':mesh.vertices_m.tolist(),'mesh_sha256':hashlib.sha256((scene/'photogrammetry/mesh_synthetic.obj').read_bytes()).hexdigest()})
        except Exception as exc:excluded.append({'scene':name,'reason':str(exc)})
    if calibration_scene not in measurements:raise ValueError('reference scene unavailable; no substitution allowed')
    reference_frames=measurements[calibration_scene][:5]
    reference=np.mean([x[1] for x in reference_frames],axis=0)
    valid=np.any(abs(reference)>0,axis=-1)
    gain=fit_complex_gain(forward[calibration_scene],reference,valid)
    f=np.linspace(72e9,82e9,128);ref_rd,axis=range_response(reference,f)
    pref=float(np.mean(abs(ref_rd[valid])**2,axis=0).max())
    rows=[];cases=[];frame_sources={}
    for name,frames in measurements.items():
        predicted=(gain*forward[name]).astype(np.complex64);pr,_=range_response(predicted,f);sp=abs(pr)**2
        model_has_path=np.any(abs(forward[name])>0,axis=-1)
        mesh,_=full_scene_geometry(a.root/name)
        vertex_range=np.linalg.norm(mesh.vertices_m,axis=1)
        roi=(axis>=vertex_range.min()-.05)&(axis<=vertex_range.max()+.05)
        for index,(file,signal,meta) in enumerate(frames):
            role='calibration' if name==calibration_scene and index<5 else 'development_temporal_or_distance_holdout'
            mr,_=range_response(signal,f);mp=abs(mr)**2;valid=np.any(abs(signal)>0,axis=-1)
            mask=np.broadcast_to(roi,(int(valid.sum()),len(axis)))
            comp=compare_power(mp[valid],sp[valid],reference_power=pref,target_mask=mask)
            profile=compare_power(mp[valid].mean(axis=0),sp[valid].mean(axis=0),reference_power=pref,target_mask=roi)
            row={'scene':name,'frame':file.name,'role':role,'timestamp_us':meta['timestamp_us'],
                 'source_number_of_measurements_field':meta['number_of_measurements'],
                 'source_valid_channels':int(valid.sum()),'predicted_nonzero_fraction':float(model_has_path[valid].mean()),
                 'channel_mae_db':comp.metrics['mae_db'],'channel_target_mae_db':comp.metrics['target_mae_db'],
                 'profile_mae_db':profile.metrics['mae_db'],'profile_target_mae_db':profile.metrics['target_mae_db']}
            rows.append(row)
            frame_sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            if role!='calibration':cases.append({'name':name.split('/')[1]+'/'+name.split('/')[-1]+'/'+file.stem[:6],
                                               'axis':axis,'plot_roi':roi,'comparison':profile})
        print(name,'all saved frames',len(frames),flush=True)
    with (a.output/'frame_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for name in measurements:
        selected=[r for r in rows if r['scene']==name and r['role']!='calibration']
        summaries.append({'scene':name,'evaluated_saved_frames':len(selected),
                          'channel_mae_db':distribution([r['channel_mae_db'] for r in selected]),
                          'channel_target_mae_db':distribution([r['channel_target_mae_db'] for r in selected]),
                          'profile_mae_db':distribution([r['profile_mae_db'] for r in selected]),
                          'nonzero_fraction':distribution([r['predicted_nonzero_fraction'] for r in selected])})
    indices=representative_indices([c['comparison'].metrics['mae_db'] for c in cases])
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9})
    plot_curves(a.output/'maroon_repeated_common_mae.png',[cases[i] for i in indices],
                'MAROON 官方逐次采集：平均距离剖面（不是 RD）\n图上 MAE 为平均剖面；全通道 MAE 与漏算比例另列，不按测试帧重新标定')
    summary={'selected_scenes':selection['scenes'],'eligible_scenes':metadata,'excluded_scenes':excluded,'scene_summaries':summaries,
        'saved_frames_processed':len(rows),'calibration_frames':len(reference_frames),'evaluation_frames':sum(r['role']!='calibration' for r in rows),
        'calibration':{'scene':calibration_scene,'saved_frame_indices':[0,1,2,3,4],'complex_gain':[gain.real,gain.imag],'reference_power':pref},
        'new_geometry_input':True,'source_axes_correction':'original xr[rx,tx,f] -> [tx,rx,f] before comparison',
        'scope':'static distance/temporal repeat validation; successive saved frames are not independent scenes; no coherent Doppler claim',
        'selected_figures':[cases[i]['name'] for i in indices],'frame_sha256':frame_sources}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'frames':len(rows),'evaluated':summary['evaluation_frames'],'excluded':excluded},ensure_ascii=False))


if __name__=='__main__':main()
