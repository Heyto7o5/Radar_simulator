#!/usr/bin/env python3
"""Frozen semi-empirical IQ model vs original IPIX temporal/record holdouts."""
import argparse,csv,hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from radar_platform.ipix_dataset import load_ipix,IQCorrection
from radar_platform.distributed_clutter import fit_clutter_recipe,simulate_clutter_iq,rd_blocks


def style():
    path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(path);plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10})


def describe(iq,prf,mask,normalization):
    rd,f=rd_blocks(iq,prf);power=np.mean(abs(rd)**2,axis=0)
    psd=power[:,mask].mean(axis=1);psd=psd/psd.sum()
    center=float(f@psd);width=float(np.sqrt(((f-center)**2)@psd))
    intensity=(abs(iq[:,mask])**2/normalization[mask]).ravel()
    # Deterministic thinning for distribution metrics; no cherry-picking traces.
    intensity=intensity[::max(1,len(intensity)//100000)]
    return {'rd':rd,'mean_rd_power':power,'f':f,'psd':psd,'center':center,'width':width,
            'power':float(np.mean(abs(iq[:,mask])**2)),'intensity':intensity,
            'q99':float(np.quantile(intensity,.99))}


def scores(ref,pred):
    return {'doppler_w1_hz':float(wasserstein_distance(ref['f'],pred['f'],u_weights=ref['psd'],v_weights=pred['psd'])),
            'psd_correlation':float(np.corrcoef(ref['psd'],pred['psd'])[0,1]),
            'centroid_error_hz':abs(ref['center']-pred['center']),
            'rms_width_error_hz':abs(ref['width']-pred['width']),
            'mean_power_error_db':float(10*np.log10(pred['power']/ref['power'])),
            'intensity_w1_cal_units':float(wasserstein_distance(ref['intensity'],pred['intensity'])),
            'q99_intensity_error_cal_units':abs(ref['q99']-pred['q99'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise SystemExit('choose a new output directory')
    first=load_ipix(args.root/'raw/19931107_135603_starea.cdf','vv')
    second=load_ipix(args.root/'raw/19931107_141630_starea.cdf','vv')
    if not np.allclose(first.range_near_edge_m,second.range_near_edge_m):raise ValueError('acquisition range gates differ')
    mask=np.ones(14,dtype=bool);mask[7:11]=False
    correction=IQCorrection.fit(first.iq[:32768,mask])
    train=correction.apply(first.iq[:32768])
    config=fit_clutter_recipe(train,first.range_near_edge_m,first.carrier_hz,first.prf_per_polarization_hz,first.pulse_width_s,mask)
    args.output.mkdir(parents=True)
    recipe={'model':asdict(config),'iq_correction':asdict(correction),'calibration':'record #17 sweeps 0:32768, only bins outside 8:11 (one-based)',
            'within_record_validation':'record #17 sweeps 65536:131072','external_record_validation':'record #18 sweeps 0:65536',
            'excluded_target_bins_one_based':[8,9,10,11],'polarization':'vv','seeds':list(range(42,50)),
            'scope':'statistical clutter; no exact sea geometry or sphere trajectory supplied by source'}
    (args.output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    refs={
        'within_record':describe(correction.apply(first.iq[65536:]),config.prf_hz,mask,np.array(config.mean_power)),
        'external_record':describe(correction.apply(second.iq[:65536]),config.prf_hz,mask,np.array(config.mean_power)),
    }
    models={};saved={}
    for name,baseline in [('iid_gaussian',True),('parametric_clutter',False)]:
        models[name]=[]
        for seed in range(42,50):
            sim=simulate_clutter_iq(config,65536,seed,white_baseline=baseline)
            models[name].append(describe(sim,config.prf_hz,mask,np.array(config.mean_power)))
            if seed==42:saved[name]=sim
    rows=[]
    for split,ref in refs.items():
        for model,predictions in models.items():
            values=[scores(ref,x) for x in predictions]
            row={'dataset':'IPIX Dartmouth','split':split,'model':model}
            for key in values[0]:
                row[key]=float(np.mean([v[key] for v in values]));row[key+'_seed_std']=float(np.std([v[key] for v in values]))
            row['measured_centroid_hz']=ref['center'];row['measured_rms_width_hz']=ref['width'];rows.append(row)
    with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(args.output/'signal_samples.npz',simulated_iq=saved['parametric_clutter'],white_iq=saved['iid_gaussian'],
        measured_iq=correction.apply(first.iq[65536:]),range_near_edge_m=first.range_near_edge_m,
        prf_hz=config.prf_hz,clutter_mask=mask)
    result={'recipe':recipe,'rows':rows,'source_metadata':first.metadata,'external_record_metadata':second.metadata,
        'cross_record_warning':'RF attenuation control changes from 0 to 2.4975586 V; no volts-to-dB calibration is provided. External record is an instrument+scene shift stress test, not same-gain validation.',
        'record_files_sha256':{file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in (args.root/'raw').glob('*.cdf')},
        'new_features':['NetCDF unsigned I/Q adapter','train-only I/Q correction','range-gated complex stochastic echo generator',
                        'Doppler Gaussian+white PSD model','gamma texture','pulse-width-derived gate overlap covariance','temporal and acquisition holdout'],
        'limitations':['no hydrodynamic wave geometry','no observed sphere trajectory','excluded target gates','VV-only evaluated',
                       'fixed gamma texture time scale','no scene-specific multipath','do not treat one seeded RD as pixelwise ground-truth reconstruction']}
    (args.output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    style();fig,axes=plt.subplots(2,3,figsize=(16,8.5),layout='constrained')
    ref=refs['within_record'];pred=models['parametric_clutter'][0];freq=ref['f'];r=first.range_near_edge_m
    reference=float(np.max(describe(train,config.prf_hz,mask,np.array(config.mean_power))['mean_rd_power']))
    db=lambda a:10*np.log10(np.maximum(a,1e-15)/reference)
    extent=[r[0],r[-1],freq[0],freq[-1]]
    for ax,power,title in [(axes[0,0],abs(ref['rd'][0])**2,'实测：未用于校准的单 CPI RD'),(axes[0,1],abs(pred['rd'][0])**2,'平台：独立随机 IQ → 单 CPI RD')]:
        im=ax.imshow(db(power),origin='lower',aspect='auto',extent=extent,vmin=-45,vmax=10,cmap='turbo')
        ax.axvspan(r[7]-7.5,r[10]+7.5,color='gray',alpha=.35,hatch='//')
        ax.set(xlabel='距离门近端 / m',ylabel='Doppler / Hz',ylim=(-150,150),title=title);fig.colorbar(im,ax=ax,label='dB')
    measured_mean=db(ref['mean_rd_power']);sim_mean=db(np.mean([x['mean_rd_power'] for x in models['parametric_clutter']],axis=0))
    diff=abs(measured_mean-sim_mean);diff[:,~mask]=np.nan
    ax=axes[0,2];im=ax.imshow(diff,origin='lower',aspect='auto',extent=extent,vmin=0,vmax=20,cmap='magma')
    ax.set(xlabel='距离门近端 / m',ylabel='Doppler / Hz',ylim=(-150,150),title='时间平均 RD 差异（靶区不评分）');fig.colorbar(im,ax=ax,label='dB')
    ax=axes[1,0]
    ax.plot(freq,10*np.log10(ref['psd']+1e-15),color='black',label='实测验证段')
    ax.plot(freq,10*np.log10(np.mean([x['psd'] for x in models['parametric_clutter']],axis=0)+1e-15),label='参数化杂波')
    ax.plot(freq,10*np.log10(np.mean([x['psd'] for x in models['iid_gaussian']],axis=0)+1e-15),label='独立白噪声基线')
    ax.set(xlabel='Doppler / Hz',ylabel='归一化 PSD / dB',xlim=(-150,150),title='Doppler 谱形');ax.legend(fontsize=8)
    ax=axes[1,1]
    for x,label in [(ref,'实测'),(models['parametric_clutter'][0],'参数化杂波'),(models['iid_gaussian'][0],'白噪声基线')]:
        values=np.sort(x['intensity']);survival=1-np.arange(len(values))/len(values)
        ax.semilogy(values,survival,label=label)
    ax.set(xlabel='强度 / 校准期平均功率',ylabel='超过该强度的概率',xlim=(0,35),ylim=(1e-4,1),title='强回波尾部：不是只有均值');ax.legend(fontsize=8)
    ax=axes[1,2];positions=np.arange(2);width=.35
    for offset,model,label in [(-width/2,'iid_gaussian','白噪声'),(width/2,'parametric_clutter','参数化杂波')]:
        vals=[next(x for x in rows if x['split']==s and x['model']==model)['doppler_w1_hz'] for s in refs]
        ax.bar(positions+offset,vals,width,label=label)
        for i,v in enumerate(vals):ax.text(i+offset,v,f'{v:.1f}',ha='center',va='bottom')
    ax.set_xticks(positions,['同记录未来段','另次采集\nRF 衰减改变']);ax.set(ylabel='Doppler W1 / Hz，低更好',title='冻结参数后的差异（包含仪器变化）');ax.legend(fontsize=8)
    fig.suptitle('IPIX 9.39 GHz：从原始 I/Q 到真实 RD 与平台统计回波对比\n灰带为公开标注球靶可能出现的距离门，本次不捏造其运动真值',fontsize=14)
    fig.savefig(args.output/'ipix_comparison.png',dpi=180);fig.savefig(args.output/'ipix_comparison.pdf');plt.close(fig)
    print(json.dumps({'rows':rows,'output':str(args.output)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
