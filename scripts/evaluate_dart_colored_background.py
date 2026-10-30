#!/usr/bin/env python3
"""Development follow-up: calibrate a fixed fast-time PSD, do not fit check frames."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.colored_adc_background import colored_adc_noise,expected_windowed_range_power
from evaluate_dart_frozen_batch import load,align,masks,DATA,OUT as PREVIOUS
from dart_batch_metrics import mae_db,db_power

OUT=PREVIOUS.parent/'colored_background_followup'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((PREVIOUS/'frozen_calibration.json').read_text())
    protocol=json.loads((PREVIOUS/'evaluation_protocol.json').read_text())
    gain=frozen['gain'];reference=frozen['shared_power_reference'];samples=[]
    for index in protocol['calibration_valid_indices']:
        request,adc,r,v,real=load(index);sim=align(adc,request,r,v);region,_=masks(request,r,v,sim)
        columns=region['off_kinematic'][0]
        if not columns.any():raise ValueError('calibration has no background support')
        samples.append(real[:,columns])
    target=np.median(np.concatenate(samples,axis=1),axis=1)
    wave=request.waveform
    native_r=np.arange(wave.samples)*wave.sample_rate_hz/wave.samples*wave.wavelength_m*wave.carrier_hz/(2*wave.slope_hz_s)
    conversion=8*np.sum(np.hanning(512)**2)*np.sum(np.hanning(256)**2)
    # Direct PSD estimate, not a claimed exact deconvolution of the range window.
    # Unknown high-range PSD stays at the ORIGINAL white-calibration variance.
    q=np.interp(native_r,r,target/(gain*conversion),left=target[0]/(gain*conversion),right=frozen['adc_noise_variance'])
    predicted=gain*expected_windowed_range_power(q,np.hanning(512),np.hanning(256),8)
    prediction=np.interp(r,native_r,predicted)
    model={'calibration_indices':protocol['calibration_valid_indices'],'spectral_power':q.tolist(),
           'existing_gain_unchanged':gain,'shared_reference_unchanged':reference,
           'estimator':'per-range median pooled over calibration off-kinematic samples; linear interpolation to native fast FFT bins',
           'range_window_convolution_is_not_inverted':True,
           'unobserved_high_range_psd':'original fixed white-noise variance, not measured',
           'kernel_forecast_calibration_profile_mae_db':float(np.mean(abs(10*np.log10(np.maximum(prediction,1e-30)/np.maximum(target,1e-30))))),
           'slow_time_model':'independent Gaussian; no transient or Doppler ridge modeling',
           'not_thermal_noise_truth':True,'development_followup_on_previously_inspected_frames':True,
           'independent_holdout_scenes':0,'seed_list':protocol['noise_seed_list']}
    path=OUT/'frozen_colored_model.json'
    if path.exists() and json.loads(path.read_text())!=model:raise ValueError('follow-up model already frozen')
    path.write_text(json.dumps(model,indent=2))
    rows=[];cache=DATA/'processed/colored_background_followup';cache.mkdir(exist_ok=True)
    for index in protocol['temporal_check_valid_indices']:
        request,adc,r,v,real=load(index);surface=align(adc,request,r,v);region,_=masks(request,r,v,surface)
        for seed in protocol['noise_seed_list']:
            noise=colored_adc_noise(adc.shape,q,np.random.SeedSequence([seed,index]))
            for name,x in [('colored_noise_only',noise),('surface_plus_colored_noise',adc+noise)]:
                p=gain*align(x,request,r,v)
                row={'valid_frame_index':index,'seed':seed,'model':name}
                for label,mask in region.items():row[f'{label}_mae_db']=mae_db(real,p,reference,mask) if mask.any() else None
                rows.append(row)
                if name=='surface_plus_colored_noise' and seed==protocol['noise_seed_list'][0]:
                    np.savez_compressed(cache/f'f{index:04d}.npz',real_power=real,simulated_power=p,range_m=r,velocity_mps=v)
        print('colored background scored',index,flush=True)
    with (OUT/'per_frame_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    previous=json.loads((PREVIOUS/'summary.json').read_text())
    summary={'scene_count':1,'temporal_check_frames':8,'independent_holdout_scenes':0,'models':{}}
    for name in ['colored_noise_only','surface_plus_colored_noise']:
        summary['models'][name]={}
        for label in ['primary','geometry','off_kinematic']:
            values=[np.mean([row[f'{label}_mae_db'] for row in rows if row['model']==name and row['valid_frame_index']==index]) for index in protocol['temporal_check_valid_indices']]
            summary['models'][name][label]={'mean_mae_db':float(np.mean(values)),'median_mae_db':float(np.median(values)),'p90_mae_db':float(np.quantile(values,.9))}
    summary['white_surface_reference']=previous['models']['surface_plus_noise']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(r,db_power(target,reference),label='Calibration measured profile')
    axes[0].plot(r,db_power(prediction,reference),label='Window-aware colored prediction')
    axes[0].axhline(float(db_power(frozen['measured_floor_proxy'],reference)),c='gray',ls='--',label='Original white floor')
    axes[0].set(xlabel='Range (m)',ylabel='dB, same reference',title='Calibration profile (NOT validation)');axes[0].legend(fontsize=8)
    x=np.arange(3);labels=['primary','geometry','off_kinematic']
    axes[1].bar(x-.18,[previous['models']['surface_plus_noise'][k]['mean_mae_db'] for k in labels],width=.36,label='Surface + white')
    axes[1].bar(x+.18,[summary['models']['surface_plus_colored_noise'][k]['mean_mae_db'] for k in labels],width=.36,label='Surface + colored')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='MAE (dB)',title='Same 8 development frames, 3 seeds');axes[1].legend()
    fig.savefig(OUT/'colored_background_comparison.png',dpi=160);plt.close(fig)
    (OUT/'距离相关高斯背景对照.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}</style>
<h1>从均匀白噪声到距离相关的高斯背景</h1><p>不改变面目标、运动、幅度比例和显示参考，只改变 ADC 背景的快时间频谱。参数仍仅使用原三个校准帧；检查帧、随机种子与上一批相同。</p><img src="colored_background_comparison.png">
<p>左图是校准曲线，不是验证结果。ADC 频谱经过距离窗后会扩散到相邻距离格，程序显式计算这一影响；当前直接估计频谱，没有把窗卷积反演掉。右图为之前已查看的八帧上的开发对照，不是新的独立留出验证。</p>
<p>模型只能表达不同距离的背景强弱和快时间相关性，仍假设慢时间独立。它不能替代真实地物散射、时间突变、相位跳变或多径，也不能被解释为已知接收机的热噪声参数。</p>
<p><a href="summary.json">数值汇总</a> · <a href="per_frame_metrics.csv">完整逐帧数据</a> · <a href="frozen_colored_model.json">冻结参数与限制</a></p></html>''')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
