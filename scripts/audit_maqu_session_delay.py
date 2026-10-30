#!/usr/bin/env python3
"""Diagnose Maqu cross-session discrepancy using measured-only controls.

No fitted validation range correction is applied; all old validation remains.
"""
import base64,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from radar_platform.maqu_dataset import calibration_file,load_citi,load_background_subtracted
from radar_platform.physical_optics import sfcw_range_profile
from radar_platform.delay_diagnostics import profile_shift
from radar_platform.models import C0
from render_scene import configure_font

P=Path(__file__).resolve().parents[1]
ROOT=Path('/nas/datasets/hyt/datasets/radar_open/maqu_scatterometer')
OUT=P/'docs/validation_diagnostics/2026-10-15'

def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    for p in [OUT/'images',OUT/'metrics']:p.mkdir(parents=True,exist_ok=True)
    configure_font()
    protocol={'post_hoc_diagnostic':True,'new_fidelity_validation_claimed':False,
        'background_windows_before_target_comparison_m':[[1,8],[8,16]],'background_lag_limit_m':1.,
        'bands_hz':{'full':[.75e9,10.25e9],'X':[9e9,9.875e9]},
        'target_peak_search':'[nominal R, nominal R+8 m], same rule for every plate/polarization',
        'source_geometry':'README: 27.7 m on 2018-08-15; 36.3 m on 2018-08-16',
        'calibration_or_simulator_modified':False,'do_not_apply_validation_peak_shift':True}
    dump(OUT/'metrics/maqu_protocol.json',protocol)
    files=[calibration_file(ROOT,'20180815',42),calibration_file(ROOT,'20180816',13)]
    f,b15=load_citi(files[0]);g,b16=load_citi(files[1]);np.testing.assert_array_equal(f,g)
    bg=[];display={}
    for pol in ['11','22','12','21']:
        for name,(lo,hi) in protocol['bands_hz'].items():
            band=(f>=lo)&(f<=hi);a,r=sfcw_range_profile(b15[pol][band],f[band]);b,_=sfcw_range_profile(b16[pol][band],f[band])
            if name=='full':display[pol]=(a,b,r)
            for win in protocol['background_windows_before_target_comparison_m']:
                m=profile_shift(a,b,r,win,1.)
                bg.append({'channel':pol,'band':name,'window_start_m':win[0],'window_end_m':win[1],**m})
    csvout(OUT/'metrics/background_delay.csv',bg)
    peaks=[];profiles={}
    for day,number,background,R in [('20180815',13,42,27.7),('20180815',14,42,27.7),('20180816',1,13,36.3),('20180816',7,13,36.3)]:
        ff,x,used=load_background_subtracted(ROOT,day,number,background);np.testing.assert_array_equal(f,ff)
        rawf,raw=load_citi(calibration_file(ROOT,day,number));np.testing.assert_array_equal(rawf,f)
        files.extend(Path(p) for p in used)
        for pol in ['11','22']:
            a,r=sfcw_range_profile(x[pol],f);mask=(r>=R)&(r<=R+8)
            k=np.flatnonzero(mask)[np.argmax(abs(a[mask]))]
            raw_profile,_=sfcw_range_profile(raw[pol],f)
            raw_k=np.flatnonzero(mask)[np.argmax(abs(raw_profile[mask]))]
            peaks.append({'date':day,'measurement':number,'polarization':pol,'nominal_range_m':R,'raw_delay_peak_m':float(r[k]),
                'peak_before_background_subtraction_m':float(r[raw_k]),'background_subtraction_peak_shift_m':float(r[k]-r[raw_k])})
            profiles[(day,number,pol)]=(a,r)
    pairs=[]
    for pol in ['11','22']:
        for before in [x for x in peaks if x['date']=='20180815' and x['polarization']==pol]:
            for after in [x for x in peaks if x['date']=='20180816' and x['polarization']==pol]:
                measured=after['raw_delay_peak_m']-before['raw_delay_peak_m'];nominal=after['nominal_range_m']-before['nominal_range_m']
                pairs.append({'polarization':pol,'reference_measurement':before['measurement'],'next_day_measurement':after['measurement'],
                    'measured_peak_increment_m':measured,'nominal_increment_m':nominal,'increment_discrepancy_m':measured-nominal})
    csvout(OUT/'metrics/measured_plate_peaks.csv',peaks);csvout(OUT/'metrics/measured_only_increments.csv',pairs)
    # Independent parameter-scale checks: these do NOT fit the measured offset.
    sep=.4;radius=36.3
    scale_checks={'exact_C0_to_3e8_effect_on_8p6m_increment_m':abs(8.6*(3e8/C0-1)),
        'symmetric_0p4m_bistatic_vs_monostatic_range_m':float(np.sqrt(radius**2+(sep/2)**2)-radius),
        'documented_plate_tilt_bound_path_projection_m':float(.425*np.sin(np.deg2rad(.26))+.325*np.sin(np.deg2rad(.10))),
        'tilt_basis':'paper E3.1 gives approximately 0.16 deg normal offset and +/-0.10 deg angular uncertainty; bound is diagnostic, not an exact pose truth',
        'not_a_correction':True}
    near=[x for x in bg if x['window_start_m']==1]
    summary={'background_window_tests':len(bg),'plate_measurements':4,'co_polar_profiles':8,'paired_increments':len(pairs),
        'background_subtraction_max_peak_shift_m':max(abs(x['background_subtraction_peak_shift_m']) for x in peaks),
        'frequency_step_hz_from_actual_files':float(f[1]-f[0]),
        'measured_increment_m':[min(x['measured_peak_increment_m'] for x in pairs),max(x['measured_peak_increment_m'] for x in pairs)],
        'nominal_increment_m':8.6,'near_background_shift_max_abs_m':max(abs(x['shift_m']) for x in near),
        'near_background_correlation_min':min(x['normalized_complex_correlation'] for x in near),
        'interpretation':'the ~0.46 m discrepancy exists in measured-only cross-session increments; stable early background disfavors one global receiver delay shift, but does not uniquely identify geometry or target-specific path errors',
        'uncertainty':'early returns are not independently identified hardware paths; no unique causal diagnosis',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},'scale_checks':scale_checks}
    dump(OUT/'metrics/maqu_delay_summary.json',summary)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,pol in enumerate(['11','22']):
        a,b,r=display[pol];keep=(r>=1)&(r<=8);ref=max(abs(a[keep]).max(),abs(b[keep]).max())
        axes[0,col].plot(r,20*np.log10(np.maximum(abs(a)/ref,1e-6)),label='8月15日空目标背景')
        axes[0,col].plot(r,20*np.log10(np.maximum(abs(b)/ref,1e-6)),label='8月16日空目标背景',alpha=.8)
        axes[0,col].set(xlim=(1,8),ylim=(-60,3),xlabel='未经去嵌的等效距离 / m',ylabel='共同参考 / dB',title=f'{pol}：早期背景位置基本不变');axes[0,col].legend(fontsize=8)
        for day,num,R,label in [('20180815',13,27.7,'15日水平板'),('20180815',14,27.7,'15日竖直板'),('20180816',1,36.3,'16日水平板'),('20180816',7,36.3,'16日竖直板')]:
            signal,r=profiles[(day,num,pol)];window=(r>=R)&(r<=R+8);reference=abs(signal[window]).max()
            axes[1,col].plot(r-R,20*np.log10(np.maximum(abs(signal)/reference,1e-6)),label=label)
        axes[1,col].set(xlim=(2.8,4.2),ylim=(-45,2),xlabel='原始延迟峰位置 − 记录名义距离 / m',ylabel='各曲线峰值归一 / dB',title=f'{pol}：目标相关偏差约 0.46 m');axes[1,col].legend(fontsize=8)
    fig.suptitle('实测对实测的诊断：未使用仿真峰，也没有对测试目标平移校正\n上图同参考比较背景；下图各曲线归一只比较峰的位置，不评价幅度保真',fontsize=12)
    fig.savefig(OUT/'images/maqu_measured_only_diagnostic.png',dpi=160);plt.close(fig)
    image='data:image/png;base64,'+base64.b64encode((OUT/'images/maqu_measured_only_diagnostic.png').read_bytes()).decode()
    body=f'<h1>玛曲跨日距离偏差：实测对实测复核</h1><p>2026-10-15 · 后验诊断，不是新增保真通过结果；仿真模型与旧验证结果未修改。</p><img src="{image}">'
    body+='<table><tr><th>检查</th><th>结果</th><th>能说明什么</th></tr>'
    for row in [
        ['名义距离增量','36.3 − 27.7 = 8.60 m','来自数据说明'],
        ['实测目标峰增量',f"{summary['measured_increment_m'][0]:.3f}–{summary['measured_increment_m'][1]:.3f} m",'差异在实测之间已存在，不是仿真才生成'],
        ['早期背景相对位移',f"全频带接近 0；所有近窗对照最大约 {summary['near_background_shift_max_abs_m']*1000:.1f} mm",'不支持把所有通道统一挪动 0.46 m 的简单修正'],
        ['是否由背景扣除引起',f"8 个目标谱的峰移动最大 {summary['background_subtraction_max_peak_shift_m']:.3f} m",'该偏差在扣除背景之前已存在'],
        ['频率轴',f"使用原文件 {summary['frequency_step_hz_from_actual_files']:g} Hz 步长",'没有把论文中约 3 MHz 的描述当成精确步长'],
        ['光速取值 / 双站路径 / 小角度倾斜','本次尺度检查均为毫米量级','不足以单独解释约 0.46 m'],
        ['仍未唯一确定','名义距离参考面、目标安装位置或目标特定路径等','需要测量几何或实验日志，不应调整测试真值来追求好看结果']]:
        body+='<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>'
    body+='</table><p>“早期背景”没有被独立证明是某条唯一硬件路径，因此这里只能排查简单解释，不能宣布找到唯一根因。相关峰位置也不等于毫米级测距精度。下一步应优先核对两次测量的距离定义与目标位置记录。</p><p>论文中的小角度与仪器几何依据：<a href="https://essd.copernicus.org/articles/13/2819/2021/">玛曲原始论文，B2 与 E3.1</a>。没有加入多 PRF，没有修改原始数据，也没有用测试峰进行模型标定。</p>'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>玛曲偏差诊断</title><style>body{max-width:1150px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;color:#24364b}img{width:100%}table{border-collapse:collapse;width:100%}th,td{padding:12px;border:1px solid #ccd5df;text-align:left}th{background:#eaf2f6}</style>'+body+'</html>'
    (OUT/'玛曲偏差诊断.html').write_text(doc)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
