#!/usr/bin/env python3
"""Frozen before/after static MAROON alignment: one calibration delay + gain."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from radar_platform.maroon_full import load_full_frame,full_scene_geometry
from radar_platform.surface_echo import simulate_surface_spectrum,range_response
from radar_platform.validation_metrics import fit_complex_gain,complex_metrics
from radar_platform.alignment_metrics import fit_global_delay_gain,apply_delay_gain
from radar_platform.comparison import compare_power,representative_indices
from radar_platform.repeat_metrics import distribution
from radar_platform.geometry_alignment import rigid_corner_from_template,corner_descriptor
from validate_common_real import plot_curves
from matplotlib import font_manager
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry-mode',choices=['as_published','rigid_reference_corner'],default='as_published');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path('/nas/datasets/hyt/datasets/radar_open/maroon_static_subset/raw/maroon')
    scenes=['24_corner_reflector/30','24_corner_reflector/50']
    f=np.linspace(72e9,82e9,128);records={};models={};meshes={};geometry_checks={}
    template,_=full_scene_geometry(root/scenes[0])
    plan={'scenes':scenes,'calibration':'30cm scene saved frames 000000..000004 only',
          'heldout':'remaining 30cm frames and every 50cm frame','max_global_range_bias_m':.015,
          'geometry_mode':a.geometry_mode,'no_radar_geometry_fit':True,'no_channelwise_fit':True,
          'scope':'development alignment, not independent new acquisition study'}
    (a.output/'selection.json').write_text(json.dumps(plan,indent=2)+'\n')
    for name in scenes:
        mesh,radar=full_scene_geometry(root/name);meshes[name]=mesh
        geometry_checks[name]={'published':corner_descriptor(mesh)[2]}
        forward_mesh=mesh
        if a.geometry_mode=='rigid_reference_corner' and name!=scenes[0]:
            forward_mesh,check=rigid_corner_from_template(template,mesh);geometry_checks[name]['rigid_alignment']=check
        models[name]=simulate_surface_spectrum(forward_mesh,radar).signal
        files=sorted((root/name/'radar_72.0_82.0_128/calibrated_data').glob('*_emptyfiltered.npy'))
        records[name]=[(file,*load_full_frame(file)) for file in files]
        if len(files)!=10:raise ValueError('expected all ten selected saved frames')
    reference=np.mean([r[1] for r in records[scenes[0]][:5]],axis=0)
    calmask=np.any(abs(reference)>0,axis=-1)
    gain=fit_complex_gain(models[scenes[0]],reference,calmask)
    aligned=fit_global_delay_gain(models[scenes[0]],reference,f,calmask,max_equivalent_range_bias_m=.015)
    baseline={'delay_s':0.,'gain_real':gain.real,'gain_imag':gain.imag,'equivalent_range_bias_m':0.}
    (a.output/'calibration.json').write_text(json.dumps({'before':baseline,'after':aligned,'fit_record':plan['calibration']},indent=2)+'\n')
    reference_range,axis=range_response(reference,f);pref=float((abs(reference_range[calmask])**2).mean(axis=0).max())
    rows=[];channel_rows=[];cases=[]
    for name,frames in records.items():
        predictions={method:apply_delay_gain(models[name],f,cal) for method,cal in [('before',baseline),('after',aligned)]}
        projected={key:abs(range_response(value,f)[0])**2 for key,value in predictions.items()}
        radii=np.linalg.norm(meshes[name].vertices_m,axis=1);roi=(axis>radii.min()-.05)&(axis<radii.max()+.05)
        for index,(file,measured,freq,meta) in enumerate(frames):
            if not np.allclose(f,freq,rtol=0,atol=.01):raise ValueError('frequency mismatch')
            role='calibration' if name==scenes[0] and index<5 else 'evaluation'
            valid=np.any(abs(measured)>0,axis=-1);mp=abs(range_response(measured,f)[0])**2
            for method,sp in projected.items():
                comparison=compare_power(mp[valid],sp[valid],reference_power=pref,target_mask=np.broadcast_to(roi,(int(valid.sum()),len(axis))))
                profile=compare_power(mp[valid].mean(axis=0),sp[valid].mean(axis=0),reference_power=pref,target_mask=roi)
                predicted_nonzero=np.any(abs(predictions[method])>0,axis=-1)
                rr=axis[roi];real_peak=rr[np.argmax(mp[valid].mean(axis=0)[roi])];sim_peak=rr[np.argmax(sp[valid].mean(axis=0)[roi])]
                rows.append({'scene':name,'frame':file.name,'role':role,'method':method,
                    'channel_mae_db':comparison.metrics['mae_db'],'channel_target_mae_db':comparison.metrics['target_mae_db'],
                    'profile_mae_db':profile.metrics['mae_db'],'profile_target_mae_db':profile.metrics['target_mae_db'],
                    'profile_peak_error_m':float(abs(real_peak-sim_peak)),
                    'predicted_nonzero_fraction':float(predicted_nonzero[valid].mean()),**complex_metrics(predictions[method],measured,valid)})
                if role=='evaluation':
                    difference=comparison.difference_db
                    for j,(tx,rx) in enumerate(zip(*np.where(valid))):
                        channel_rows.append({'scene':name,'frame':file.name,'method':method,'tx':int(tx),'rx':int(rx),
                            'mae_db':float(abs(difference[j]).mean()),'target_mae_db':float(abs(difference[j,roi]).mean()),
                            'prediction_nonzero':bool(predicted_nonzero[tx,rx])})
                if method=='after' and role=='evaluation':cases.append({'name':name+'/'+file.stem[:6],'axis':axis,'plot_roi':roi,'comparison':profile})
        np.savez_compressed(a.output/(name.replace('/','_')+'_predictions.npz'),before=predictions['before'],after=predictions['after'],range_m=axis,frequencies_hz=f)
    for filename,values in [('frame_metrics.csv',rows),('channel_metrics.csv',channel_rows)]:
        with (a.output/filename).open('w',encoding='utf-8-sig',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    summaries=[]
    for scene in scenes:
        for method in ['before','after']:
            group=[r for r in rows if r['scene']==scene and r['method']==method and r['role']=='evaluation']
            summaries.append({'scene':scene,'method':method,'saved_frames':len(group),
                **{key:distribution([r[key] for r in group]) for key in ['channel_mae_db','channel_target_mae_db','profile_mae_db','profile_peak_error_m','complex_coherence','predicted_nonzero_fraction']}})
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9})
    selected=representative_indices([r['comparison'].metrics['mae_db'] for r in cases])
    plot_curves(a.output/'aligned_comparison.png',[cases[i] for i in selected],
        'MAROON：只用校准帧估计一次全局时延/复增益，再冻结\n静态距离响应；图为平均剖面 MAE，全通道 MAE 与漏算另列')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,title in zip(axes,['channel_target_mae_db','profile_peak_error_m'],['目标区全通道 MAE / dB','平均剖面主峰偏差 / m']):
        for j,name in enumerate(scenes):
            values=[next(s[key]['mean'] for s in summaries if s['scene']==name and s['method']==m) for m in ['before','after']]
            ax.plot([0,1],values,'o-',label=name)
        ax.set_xticks([0,1],['原单一复增益','增加全局时延对齐']);ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(a.output/'before_after.png',dpi=160);plt.close(fig)
    result={'selection':plan,'alignment':aligned,'geometry_checks':geometry_checks,'rows':summaries,'reference_power':pref,
            'missing_path_coverage_unchanged':True,'calibration_not_used_as_evaluation':True,
            'interpretation':'global delay alignment cannot invent absent scattered paths; unchanged/worse cases retained',
            'source_sha256':{str(file):hashlib.sha256(file.read_bytes()).hexdigest() for records_ in records.values() for file,*_ in records_}}
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'alignment':aligned,'results':summaries},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
