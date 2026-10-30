#!/usr/bin/env python3
"""Civil/laboratory plate campaign: dynamic analytic checks + honest static measured boundary."""
import base64
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radar_platform.unified.schema import Simulation
from radar_platform.unified.engine import simulate_frame
from radar_platform.comparison import power_db
from render_scene import configure_font

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'docs/surface_extension/2026-10-15'
DATA=Path('/nas/datasets/hyt/datasets/radar_open')
STATIC=DATA/'maqu_scatterometer/processed/review_20261015_scalar_po'
CACHE=DATA/'layered_validation/dynamic_surface_20261015'

def load(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    global OUT, CACHE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--low-order',type=int,default=8)
    parser.add_argument('--high-order',type=int,default=16)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();OUT=args.output.resolve()
    CACHE=CACHE/OUT.name
    if not 2<=args.low_order<args.high_order<=48:raise ValueError('ordered quadrature levels in 2..48 required')
    for p in [OUT/'images',OUT/'metrics',OUT/'configs',CACHE]:p.mkdir(parents=True,exist_ok=True)
    configure_font()
    base=load(PROJECT/'configs/lab_moving_plate_po.json')
    cases=list(itertools.product(['pulsed_lfm','fmcw'],[20.,30.,40.],[-3.,0.,3.],[0.,1.]))
    write(OUT/'metrics/protocol.json',{'cases':cases,'selection_before_results':True,'independent_real_motion_records':0,
        'static_comparison_is_not_RD':True,'quadrature_orders':[args.low_order,args.high_order],
        'refinement_note':'orders above 8/16 are a follow-up to the retained initial 35/36 result; not new independent scenes',
        'analytic_threshold':'one native distance bin and one native Doppler bin; complex ADC refinement difference <0.2%',
        'representative_selection':'30 m, zero yaw; all three velocities and both waveforms',
        'default_model_unchanged':'centroid remains default; scalar_po must be explicitly selected',
        'unvalidated':'finite conductivity, edge diffraction, vector polarization, environmental multipath/occlusion, receiver and material absolute gain'})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');rows=[]
    if args.render_only:
        with (OUT/'metrics/dynamic_cases.csv').open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                rows.append({k:(v=='True' if k=='passed' else v if k=='waveform' else float(v)) for k,v in r.items()})
    for index,(kind,radius,closing,yaw) in enumerate([] if args.render_only else cases):
        d=json.loads(json.dumps(base));d['name']=f'{kind}_{radius}_{closing}_{yaw}'
        if kind=='fmcw':
            d['waveform']={'kind':kind,'start_hz':9e9,'slope_hz_s':1e12,'ramp_s':60e-6,'slot_period_s':200e-6,
                          'adc_start_s':2e-6,'samples':128,'sample_rate_hz':4e6,'loops':64}
        d['scene']['max_scatterers']=8192
        obj=d['scene']['objects'][0];obj.update(position_ned_m=[radius,0,0],velocity_ned_mps=[-closing,0,0],yaw_deg=yaw,quadrature_order=args.low_order)
        low=simulate_frame(Simulation.model_validate(d))
        obj['quadrature_order']=args.high_order;request=Simulation.model_validate(d);high=simulate_frame(request)
        p=high.products;ri,di=np.unravel_index(np.argmax(p.rd_power),p.rd_power.shape)
        epoch=high.metadata['truth_epoch_s'];true_range=radius-closing*epoch
        complex_error=float(np.linalg.norm(low.adc-high.adc)/np.linalg.norm(high.adc))
        er=abs(float(p.range_m[ri])-true_range);ev=abs(float(p.velocity_mps[di])-closing)
        row={'case':index,'waveform':kind,'range_m':radius,'closing_mps':closing,'yaw_deg':yaw,
             'range_error_m':er,'velocity_error_mps':ev,'range_error_bins':er/np.diff(p.range_m)[0],
             'velocity_error_bins':ev/p.metadata['native_velocity_cell_mps'],
             'complex_adc_quadrature_relative_difference':complex_error,
             'passed':bool(er<np.diff(p.range_m)[0] and ev<p.metadata['native_velocity_cell_mps'] and complex_error<.002)}
        rows.append(row)
        write(OUT/'configs'/f'case_{index:03d}.json',request.model_dump(mode='json'))
        if radius==30 and yaw==0:
            rr=0 if kind=='pulsed_lfm' else 1;col=[-3.,0.,3.].index(closing);ax=axes[rr,col]
            im=ax.pcolormesh(p.range_m,p.velocity_mps,power_db(p.rd_power,float(p.rd_power.max())).T,
                shading='auto',cmap='turbo',vmin=-50,vmax=0,rasterized=True)
            ax.scatter([true_range],[closing],marker='x',c='white',s=65,label='解析中心真值')
            ax.set(xlim=(5,65),ylim=(-10,10),xlabel='距离 / m',ylabel='接近速度 / (m/s)',
                   title=f'{"脉冲 LFM" if rr==0 else "线性 FMCW"}：{closing:g} m/s')
            ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='各图峰值参考 / dB',shrink=.8)
            np.savez_compressed(CACHE/f'case_{index:03d}.npz',adc=high.adc,rd_channels=p.rd_channels,
                range_m=p.range_m,velocity_mps=p.velocity_mps)
        print(index+1,'/',len(cases),row['passed'],complex_error,flush=True)
    fig.suptitle('新增动态面目标：整块 85 × 65 cm 金属板的面积积分，而非只取三角面中心\n纯理想解析检查，未冒充玛曲动态实测；每图独立峰值归一，仅展示位置与谱形',fontsize=12)
    if not args.render_only:fig.savefig(OUT/'images/dynamic_surface_rd.png',dpi=155)
    plt.close(fig)
    csvout(OUT/'metrics/dynamic_cases.csv',rows)
    summary={'cases':len(rows),'passed':sum(r['passed'] for r in rows),'max_range_error_bins':max(r['range_error_bins'] for r in rows),
             'max_velocity_error_bins':max(r['velocity_error_bins'] for r in rows),
             'max_quadrature_relative_error':max(r['complex_adc_quadrature_relative_difference'] for r in rows)}
    write(OUT/'metrics/dynamic_summary.json',summary)
    # Reuse the NEW rerun, not old scores labeled as a new experiment.
    static=load(STATIC/'comparison.json')
    shutil.copy2(STATIC/'maqu_comparison.png',OUT/'images/maqu_static.png')
    shutil.copy2(STATIC/'metrics.csv',OUT/'metrics/maqu_static.csv')
    shutil.copy2(STATIC/'recipe.json',OUT/'metrics/maqu_recipe.json')
    # Expand frequency-band evaluation without inventing extra independent records.
    spectral=[]
    with np.load(STATIC/'comparison_arrays.npz') as z:
        f=z['frequencies_hz']
        for case in static['recipe']['validation_geometries']:
            for pol in ['11','22']:
                m=z[f"{case['name']}_{pol}_measured"];s=z[f"{case['name']}_{pol}_simulated"]
                for name,lo,hi in [('L',1.5e9,1.75e9),('S',2.5e9,3e9),('C',4.5e9,5e9),('X',9e9,9.875e9)]:
                    mask=(f>=lo)&(f<=hi);a,b=m[mask],s[mask]
                    spectral.append({'case':case['name'],'polarization':pol,'band':name,'frequency_points':int(mask.sum()),
                        'magnitude_mae_db':float(np.mean(abs(20*np.log10(np.maximum(abs(a),1e-15)/np.maximum(abs(b),1e-15))))),
                        'complex_coherence':float(abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))})
    csvout(OUT/'metrics/maqu_four_bands.csv',spectral)
    noise=load(PROJECT/'docs/alignment/2026-10-05/metrics/ramp_noise_control_v1/summary.json')
    nrows=[r for r in noise['results'] if r['role']=='heldout_record']
    write(OUT/'metrics/prior_noise_control.json',{'origin':'2026-10-05 historical frozen experiment; inspected, NOT rerun in this stage','results':nrows})
    files=[*PROJECT.glob('backend/radar_platform/unified/*.py'),PROJECT/'backend/radar_platform/physical_optics.py',
           PROJECT/'tests/test_dynamic_surface_po.py',Path(__file__),PROJECT/'configs/lab_moving_plate_po.json',
           PROJECT/'configs/scattering_data_catalog.json']
    write(OUT/'metrics/provenance.json',{'source_sha256':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'real_static_input_sha256':static['measured_files_sha256'],'raw_and_complex_cache':str(CACHE),
        'old_projects_modified':False,'real_motion_validation_available':False})
    def table(head,rows):
        return '<table><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/'images'/name).read_bytes()).decode()+'">'
    body='<h1>面目标扩展：第一阶段结果</h1><p>2026-10-15 · 通用地面实验室场景 · 前端未启动</p>'
    body+='<h2>本次实际增加了什么</h2>'+table(['原有','新增','仍未完成'],[
        ['三角面中心点，或独立的静态 PO 积分','动态链路显式选择 scalar_po，按整块表面做相干积分','不是完整电磁求解；材料、极化、衍射和多径仍缺失'],
        ['脉冲 / FMCW 点目标运动','两种波形下的匀速平移金属板；保持跨帧相位','没有实测运动板轨迹，动态部分只能做解析自检'],
        ['只看一张图是否相似',f"36 组波形/距离/速度/姿态对照，{summary['passed']} 组满足预定阈值",'批量参数组合不等于独立真实场景']])
    body+=image('dynamic_surface_rd.png')
    body+='<p>静态复频响桥接、面积守恒、积分收敛、正反面、跨帧相位，以及独立矩形孔径 sinc 解析曲线均有自动化检查。新模型必须显式开启；默认中心点模型保持兼容。动态 PO 只允许带宽不超过载频的 5%，因为响应幅度使用载频近似；这不是模型精度保证。</p>'
    body+=f'<p>本轮比较积分阶数 {args.low_order} 与 {args.high_order}，最大复数 ADC 相对差为 {summary["max_quadrature_relative_error"]:.3%}。初轮 8/16 阶曾有 1/36 组未达到 0.2% 阈值，已保留初轮结果；后续提高阶数是数值收敛复查，不是放宽阈值，也不是新增实测。</p>'
    body+='<h2>玛曲：真实验证的是静态面响应，不是动态 RD</h2>'+image('maqu_static.png')
    body+=table(['实测条件','极化','距离峰偏差 / m','频谱幅度 MAE / dB'],[
        [r['case'],r['polarization'],f"{r['peak_error_m']:.3f}",f"{r['xband_magnitude_mae_db']:.2f}"] for r in static['rows']])
    body+='<p>只有 4 次几何明确的矩形板测量：1 次用于参考标定，3 次用于对照；两个极化和四个频带不能冒充 24 个独立实验。全部可用矩形板对照已覆盖，二面角反射器和朝天测量没有混作矩形板。标定只用参考记录，未用跨日测试峰平移校正。</p>'
    body+='<p><b>同日主峰接近，但跨日约 0.46 m 偏差仍然存在。</b>面积积分接入动态链路，不会自动修复记录几何、仪器时延或场景配准问题。图中实测经过背景扣除和固定距离门，不能称为完整原始背景复现。</p>'
    body+='<h2>高斯噪声：已有实验证据，不能重复当作新增成果</h2>'+table(['历史对照（180 帧）','全图 MAE / dB','标签区强回波 MAE / dB'],[
        [r['model'],f"{r['rd_mae_db']['mean']:.2f}",f"{r['strong_10db_mae_db']['mean']:.2f}"] for r in nrows])
    body+='<p>只有噪声也能得到很低的全图 MAE，说明总分受大量背景像素主导。目标加噪声略有改善，但尚未复现真实车体。下一阶段需要比较噪声方差、相关性、谱形和强回波，不逐帧调参追求图像相似。</p>'
    body+='<h2>剩余工作：目标仍在进行</h2>'+table(['方向','本阶段状态','后续验证'],[
        ['更多散射模型','现有点/中心面元/新面积积分、独立海杂波模型；尚未完成统一扩展','分别接入粗糙地表与统计背景，明确参数来源和适用范围'],
        ['更多公开数据','已核查候选，新增原始数据尚未全部取得','草地、海面、云雨等分开选；没有原始谱的数据不当作 RD 真值'],
        ['更真实的背景','已复核高斯噪声历史对照；尚未完成新的背景模型实验','冻结标定、跨记录评价，并保留噪声-only 对照'],
        ['面目标真实动态验证','本次只有静态实测与动态解析验证，不能合并为真实 RD 保真通过','需要有已知几何和运动记录的配套动态实测']])
    catalog=load(PROJECT/'configs/scattering_data_catalog.json')
    write(OUT/'metrics/scattering_data_catalog.json',catalog)
    body+='<h2>散射数据资料卡：已获得与仅候选分开</h2>'+table(['资源','类别','资料状态','边界'],[
        ['玛曲金属板','金属面','已取得原始复频响与记录','静态响应，不是动态 RD'],
        ['玛曲草地','土壤 / 植被','已有角度相关散射系数；原始时序尚未补齐','仅覆盖指定频段、极化和观测角'],
        ['IPIX','海面','已有多条 I/Q 记录','统计海面，不是实际海浪三维重建'],
        ['<a href="https://www.arm.gov/publications/tech_reports/handbooks/kazr_handbook.pdf">ARM KAZR</a>','云雨体散射','文档候选；文件未取得','需选谱数据，不能用平均速度代替完整 Doppler 谱'],
        ['<a href="https://essd.copernicus.org/articles/14/3915/2022/">SnowSAR</a>','积雪 / 林地','已查论文；文件未取得','只考虑散射强度约束，不开展 SAR 成像'],
        ['<a href="https://arxiv.org/abs/2010.09076">RADIATE</a>','城市场景 / 恶劣天气','已查论文；文件未取得','信号表示与许可待核查，不称为已接入 RD 基准']])
    body+='<p>来源：<a href="https://essd.copernicus.org/articles/13/2819/2021/">玛曲论文与数据说明</a>。参数配置、逐样本 CSV、来源哈希随报告保存。原始数据不放入源码仓库。</p>'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>面目标第一阶段</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.65 system-ui;color:#23354b}img{width:100%;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd6df;padding:10px;text-align:left}th{background:#e9f2f6}h2{margin-top:32px}a{color:#176b91}</style>'+body+'</html>'
    (OUT/'面目标扩展与验证.html').write_text(doc)
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(PROJECT))
    with zipfile.ZipFile(OUT.parent/f'面目标第一阶段_{OUT.name}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if summary['passed']!=len(rows):raise SystemExit('some prespecified dynamic cases failed; kept all results')

if __name__=='__main__':main()
