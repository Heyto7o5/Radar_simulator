#!/usr/bin/env python3
"""Predict held-out metal-plate measurements using geometry + one reference."""
import argparse,csv,hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from radar_platform.maqu_dataset import load_background_subtracted
from radar_platform.physical_optics import rectangular_plate,simulate_po_surface,sfcw_range_profile,range_gate
from radar_platform.models import C0
from radar_platform.instrument_response import SmoothInstrumentResponse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/maqu_scatterometer'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise SystemExit('choose a new output directory')
    f,reference,ref_files=load_background_subtracted(args.root,'20180815',13,42)
    # These geometries come from the data README, never from validation peaks.
    configs=[('same_day_27m_vertical','20180815',14,27.7,90.,42),
             ('next_day_36m_horizontal','20180816',1,36.3,0.,13),
             ('next_day_36m_vertical','20180816',7,36.3,90.,13)]
    geometry=rectangular_plate(27.7,.85,.65)
    reference_model=simulate_po_surface(geometry,f)
    band=(f>=9e9)&(f<=9.875e9)
    transfers={};delays={};response_models={}
    for pol in ('11','22'):
        rp,ranges=sfcw_range_profile(reference[pol],f)
        # Instrument delay estimated only from the known 27.7m reference.
        search=(ranges>27.7)&(ranges<34.)
        peak=ranges[np.flatnonzero(search)[np.argmax(abs(rp[search]))]]
        delays[pol]=float(peak-27.7)
        calibrated=range_gate(reference[pol],f,peak,.65,.2)
        response_model=SmoothInstrumentResponse.fit(f,calibrated,reference_model,(f>=1.5e9)&(f<=9.875e9))
        response_models[pol]=asdict(response_model)
        transfers[pol]=response_model.evaluate(f)
    args.output.mkdir(parents=True)
    recipe={'reference_measurement':'20180815 meas13, horizontal plate, R=27.7 m; subtract same-day mast meas42',
        'validation_geometries':[{'name':n,'date':d,'measurement':i,'range_m':R,'rotation_deg':rot,'background':bg} for n,d,i,R,rot,bg in configs],
        'width_m':.85,'height_m':.65,'tx_ned_m':[0,-.2,0],'rx_ned_m':[0,.2,0],
        'forward_model':'scalar finite-area PEC physical optics with exact spherical lengths',
        'calibration':'six real coefficients per co-polar channel: cubic log-magnitude plus linear phase, fitted only to reference plate (1.5–9.875GHz)',
        'instrument_response_models':response_models,
        'instrument_equivalent_range_offset_m':delays,'validation_gate_half_width_m':.8,
        'waveform':'SFCW, 3201 frequencies 0.75–10.25 GHz; no slow-time',
        'evaluated_band_hz':[9e9,9.875e9],
        'limitations':['recorded plate distances used literally','target tilt and phase-center position not measured in full',
                       'transfer is reference-target calibrated, not a blind receiver simulation',
                       'single-bounce co-polar scalar model','source grassland subset not yet simulated']}
    (args.output/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2)+'\n')
    rows=[];results={};source_files=set(ref_files)
    for name,date,index,R,rotation,bg in configs:
        freq,measured,files=load_background_subtracted(args.root,date,index,bg)
        if not np.array_equal(freq,f):raise ValueError('frequency axes differ')
        source_files.update(files)
        target=rectangular_plate(R,.85,.65,rotation)
        model=simulate_po_surface(target,f)
        results[name]={}
        for pol in ('11','22'):
            pred=model*transfers[pol]
            real=range_gate(measured[pol],f,R+delays[pol],.8,.25)
            mr,r=sfcw_range_profile(real,f);pr,_=sfcw_range_profile(pred,f)
            within=(abs(r-R-delays[pol])<1.)
            sim_peak=float(r[np.flatnonzero(within)[np.argmax(abs(pr[within]))]])
            measured_peak=float(r[np.flatnonzero(within)[np.argmax(abs(mr[within]))]])
            s=pred[band];m=real[band]
            corr=float(abs(np.vdot(s,m))/(np.linalg.norm(s)*np.linalg.norm(m)))
            nmse=float(np.sum(abs(s-m)**2)/np.sum(abs(m)**2))
            magnitude_error=float(np.mean(abs(20*np.log10(np.maximum(abs(s),1e-15)/np.maximum(abs(m),1e-15)))))
            # Geometry-only 1/R baseline, using the SAME reference as the PO model.
            Lref=2*np.sqrt(27.7**2+.2**2);Lt=2*np.sqrt(R**2+.2**2)
            go=transfers[pol]*reference_model*(Lref/Lt)*np.exp(-2j*np.pi*f*(Lt-Lref)/C0)
            row={'dataset':'Maqu reference plates','case':name,'polarization':{'11':'VV','22':'HH'}[pol],
                 'nominal_range_m':R,'rotation_deg':rotation,
                 'measured_instrument_corrected_peak_m':measured_peak-delays[pol],
                 'simulated_instrument_corrected_peak_m':sim_peak-delays[pol],
                 'peak_error_m':abs(sim_peak-measured_peak),'xband_complex_coherence':corr,'xband_complex_nmse':nmse,
                 'xband_magnitude_mae_db':magnitude_error,
                 'xband_power_error_db':float(10*np.log10(np.sum(abs(s)**2)/np.sum(abs(m)**2))),
                 'inverse_range_baseline_power_error_db':float(10*np.log10(np.sum(abs(go[band])**2)/np.sum(abs(m)**2)))}
            rows.append(row)
            results[name][pol]={'measured':real,'simulated':pred,'model_raw':model,'range_m':r-delays[pol],
                                'measured_profile':mr,'simulated_profile':pr}
        # Direct platform API-compatible request, no measured transfer included.
        request={'method':'physical_optics','max_bounces':1,'quadrature_order':24,'frequencies_hz':f.tolist(),
            'tx_ned_m':[[0,-.2,0]],'rx_ned_m':[[0,.2,0]],'vertices_ned_m':target.vertices_m.tolist(),'faces':target.faces.tolist()}
        (args.output/f'{name}_api_request.json').write_text(json.dumps(request,indent=2)+'\n')
    with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    output={'recipe':recipe,'rows':rows,'measured_files_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(source_files)},
            'interpretation':'Within-session rotation agrees, but cross-session delay/range mismatch remains about 0.46m. Do not fit away validation offsets.',
            'source':'https://essd.copernicus.org/articles/13/2819/2021/index.html','dataset_doi':'10.17026/DANS-ZFB-QEGY'}
    (args.output/'comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    save={'frequencies_hz':f,'transfer_VV':transfers['11'],'transfer_HH':transfers['22']}
    for case,item in results.items():
        for pol,values in item.items():
            for key,val in values.items():save[f'{case}_{pol}_{key}']=val
    np.savez_compressed(args.output/'comparison_arrays.npz',**save)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(16,8.5),layout='constrained')
    titles=['同日：27.7 m 板旋转 90°','次日：36.3 m 水平板','次日：36.3 m 竖直板']
    for col,((case,date,idx,R,rot,bg),title) in enumerate(zip(configs,titles)):
        item=results[case]['11'];ax=axes[0,col]
        measured=item['measured_profile'];simulated=item['simulated_profile'];rp=item['range_m']
        normalization=max(float(abs(simulated).max()),1e-15)
        ax.plot(rp,20*np.log10(np.maximum(abs(measured)/normalization,1e-8)),c='black',label='实测（背景扣除/距离门）')
        ax.plot(rp,20*np.log10(np.maximum(abs(simulated)/normalization,1e-8)),c='#e88425',label='平台 PO + 冻结传递函数')
        ax.axvline(R,c='gray',ls='--',label='记录标称距离')
        ax.set(xlim=(R-1,R+1),ylim=(-55,5),xlabel='仅参考板校正后的距离 / m',ylabel='相对功率 / dB',title=title);ax.legend(fontsize=7)
        ax=axes[1,col]
        ax.plot(f[band]/1e9,20*np.log10(np.maximum(abs(item['measured'][band]),1e-15)),c='black',label='实测 VV')
        ax.plot(f[band]/1e9,20*np.log10(np.maximum(abs(item['simulated'][band]),1e-15)),c='#e88425',label='平台预测 VV')
        ax.set(xlabel='频率 / GHz',ylabel='S 参数幅度 / dB',title='9–9.875 GHz 幅度预测（统一校准）');ax.legend(fontsize=8)
    fig.suptitle('玛曲原始标定靶：0.85 × 0.65 m 金属板 → 有限面相干积分 → 实测比较\n只用 27.7 m 水平板拟合每极化 6 个仪器参数；次日距离偏移保留为失败',fontsize=14)
    fig.savefig(args.output/'maqu_comparison.png',dpi=180);fig.savefig(args.output/'maqu_comparison.pdf');plt.close(fig)
    print(json.dumps({'rows':rows,'output':str(args.output)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
