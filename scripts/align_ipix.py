#!/usr/bin/env python3
"""Receiver-centering audit and frozen-vs-prefix-conditioned IPIX prediction."""
import argparse,csv,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from radar_platform.ipix_dataset import load_ipix,IQCorrection
from radar_platform.distributed_clutter import ClutterRecipe,fit_clutter_recipe,simulate_clutter_iq,rd_blocks
from radar_platform.repeat_metrics import common_safe_gates,distribution
from radar_platform.comparison import compare_power,representative_indices
from validate_common_real import plot_maps
from matplotlib import font_manager
import matplotlib.pyplot as plt


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mean_blocks(iq,prf):
    rd,f=rd_blocks(iq,prf)
    # Every scored window contains the same 16 CPIs, and power is averaged
    # before the logarithm. All currently selected records have PRF=1000Hz.
    count=len(rd)//16
    p=np.mean(abs(rd[:count*16].reshape(count,16,512,iq.shape[1]).astype(np.complex128))**2,axis=1)
    return p,f


def mean_simulation(recipe,samples):
    def one(seed):return mean_blocks(simulate_clutter_iq(recipe,samples,seed),recipe.prf_hz)[0]
    with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(one,range(42,50)))
    return np.mean(values,axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    specs=json.loads(a.selection.read_text())['records']
    source=json.loads((a.root/'processed/validation_20260930_final/recipe.json').read_text())
    frozen=ClutterRecipe(**source['model']);original_correction=IQCorrection(**source['iq_correction'])
    protocol={'calibration_sweeps':[0,32768],'gap_sweeps':[32768,65536],'evaluation_sweeps':[65536,131072],
              'preprocessing':'record-prefix I/Q means; fixed #17 I/Q scales and phase correction for ALL model comparisons',
              'absolute_receiver_gain_calibrated':False,'models':['frozen_reference_transfer','prefix_conditioned'],
              'prefix_conditioned_is_not_zero_shot':True,'seeds':list(range(42,50)),'window_cpis':16,
              'synthesis_model_unchanged':True,'calibration_does_not_read_evaluation_tail':True}
    dump(a.output/'protocol.json',protocol)
    rows=[];record_rows=[];excluded=[];plot_cases=[];per_record_figures=[]
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9})
    for spec in specs:
        file=a.root/'raw'/spec['filename']
        try:
            record=load_ipix(file,'vv')
            if len(record.iq)!=131072 or record.prf_per_polarization_hz!=1000:raise ValueError('timing differs from prespecified protocol')
        except Exception as exc:excluded.append(dict(spec,reason=str(exc)));print(spec['record_id'],'excluded',str(exc),flush=True);continue
        nrange=record.iq.shape[1]
        mask=common_safe_gates(nrange,source['excluded_target_bins_one_based'],spec['excluded_target_gates'])
        prefix=record.iq[:32768]
        means=prefix[:,mask].mean(axis=0).mean()
        correction=replace(original_correction,mean_i=float(means.real),mean_q=float(means.imag))
        calibration=correction.apply(prefix);observed=correction.apply(record.iq[65536:])
        # Keep receiver scale/phase fixed. Local powers absorb receiver+sea state
        # jointly; do NOT label them independently identified hardware gains.
        conditional=fit_clutter_recipe(calibration,record.range_near_edge_m,record.carrier_hz,record.prf_per_polarization_hz,record.pulse_width_s,mask)
        baseline=replace(frozen,range_near_edge_m=tuple(record.range_near_edge_m),prf_hz=record.prf_per_polarization_hz,pulse_width_s=record.pulse_width_s)
        cal_rd,f=mean_blocks(calibration,1000);real,f=mean_blocks(observed,1000)
        valid=np.broadcast_to(mask,(512,nrange));reference=float(np.mean(cal_rd,axis=0)[valid].max())
        if reference<=0:raise ValueError('calibration has no positive power')
        legacy_real,_=mean_blocks(original_correction.apply(record.iq[65536:]),1000)
        zero=abs(f)<=1000/512
        zero_fraction=lambda power:float(power[:,zero,:][:,:,mask].sum()/power[:,:,mask].sum())
        recmeta={'record_id':spec['record_id'],'role':spec['role'],'file':str(file),'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                 'receiver_controls':record.metadata['receiver_controls_volts'],
                 'range_near_edge_m':record.range_near_edge_m.tolist(),'scoring_gates':(np.flatnonzero(mask)+1).tolist(),
                 'zero_band_fraction_old_centering':zero_fraction(legacy_real),'zero_band_fraction_prefix_centering':zero_fraction(real),
                 'iq_correction':asdict(correction),'frozen_model':asdict(baseline),'conditional_model':asdict(conditional),'reference_power':reference,
                 'scope':'conditional statistical prediction; prefix centering assumes no stationary land return in scored gates'}
        # Save all parameters before inspecting future-segment model scores.
        dump(a.output/f'record_{spec["record_id"]}_parameters.json',recmeta)
        versions={}
        for model,recipe in [('frozen_reference_transfer',baseline),('prefix_conditioned',conditional)]:
            predicted=mean_simulation(recipe,len(observed));versions[model]=predicted
            for index,(m,s) in enumerate(zip(real,predicted)):
                comp=compare_power(m,s,reference_power=reference,valid_mask=valid)
                # Every record also gets a shape-only score; never rank it as amplitude fidelity.
                shape=compare_power(m/m[valid].mean(),s/s[valid].mean(),reference_power=1,valid_mask=valid)
                rows.append({'record_id':spec['record_id'],'role':spec['role'],'model':model,'window':index,'start_s':65.536+8.192*index,
                             'mae_db':comp.metrics['mae_db'],'shape_only_mae_db':shape.metrics['mae_db'],
                             'calibrated_domain_mean_power_error_db':float(10*np.log10(s[valid].mean()/m[valid].mean())),
                             'same_range_grid_as_reference':bool(np.allclose(record.range_near_edge_m,frozen.range_near_edge_m,atol=.002,rtol=0))})
                if model=='prefix_conditioned':plot_cases.append({'name':f'#{spec["record_id"]} 窗 {index}','comparison':comp,'mask':valid,
                    'extent':[record.range_near_edge_m[0],record.range_near_edge_m[-1],f[0],f[-1]],'ylabel':'Doppler / Hz','ylim':(-150,150)})
            group=[x for x in rows if x['record_id']==spec['record_id'] and x['model']==model]
            record_rows.append({'record_id':spec['record_id'],'role':spec['role'],'model':model,'windows':len(group),
                                'mae_db':distribution([r['mae_db'] for r in group]),'shape_only_mae_db':distribution([r['shape_only_mae_db'] for r in group]),
                                'mean_power_error_db':distribution([r['calibrated_domain_mean_power_error_db'] for r in group]),
                                'reference_range_grid_matches':group[0]['same_range_grid_as_reference']})
        np.savez_compressed(a.output/f'record_{spec["record_id"]}_average_rd.npz',measured=real,before=versions['frozen_reference_transfer'],after=versions['prefix_conditioned'],
                            range_m=record.range_near_edge_m,doppler_hz=f,valid_mask=valid,reference_power=reference)
        print('IPIX',spec['record_id'],'prefix-calibrated evaluation complete',flush=True)
        csv_write(a.output/'window_metrics.partial.csv',rows)
    csv_write(a.output/'window_metrics.csv',rows)
    selected=representative_indices([x['comparison'].metrics['mae_db'] for x in plot_cases])
    plot_maps(a.output/'conditional_examples.png',[plot_cases[i] for i in selected],
              'IPIX：当前记录前段提供统计参数，预测未用于拟合的后段平均 RD\n实测 / 条件仿真 / 差值；不是无校准跨海况迁移或逐海浪预测')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    ids=sorted({r['record_id'] for r in record_rows})
    for ax,key,title in zip(axes,['mae_db','shape_only_mae_db'],['同一接收域：平均 RD MAE / dB','单独形状比较：MAE / dB']):
        for model,label in [('frozen_reference_transfer','固定 #17 配方'),('prefix_conditioned','当前记录前段校准')]:
            vals=[next(r[key]['mean'] for r in record_rows if r['record_id']==i and r['model']==model) for i in ids]
            ax.plot(range(len(ids)),vals,'o-',label=label)
        ax.set_xticks(range(len(ids)),[str(i) for i in ids]);ax.set(xlabel='记录号',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('模型结构未改：区分冻结迁移与额外前段输入带来的条件预测改进')
    fig.savefig(a.output/'before_after.png',dpi=160);plt.close(fig)
    dump(a.output/'summary.json',{'protocol':protocol,'records':record_rows,'excluded_records':excluded,'eligible_records':len(ids),
         'measured_evaluation_cpis':len(ids)*128,'comparison_windows':len(rows),'seeds_per_model':8,
         'new_record_ids':[i for i in ids if i not in [17,18,19,25]],'selected_examples':[plot_cases[i]['name'] for i in selected],
         'warning':'frozen range-index profile at different absolute ranges is only transfer stress, not a validated range attenuation law; no independent gain calibration'})


if __name__=='__main__':main()
