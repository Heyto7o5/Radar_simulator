#!/usr/bin/env python3
"""Stage-aware plate/background/noise demonstration plus real RAMP noise controls."""
import argparse,base64,csv,hashlib,json,shutil,zipfile
from dataclasses import asdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from radar_platform.unified.schema import Simulation
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.processing import range_axis,selected_range_indices,range_compress,process_range_iq
from radar_platform.unified.environment import prepare_environment
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.storage import save_frame
from radar_platform.statistical_background import BackgroundRecipe,SpectrumComponent
from radar_platform.comparison import compare_power,power_db
from radar_platform.ramp_alignment import read_ramp_adc
from radar_platform.ramp_comparison import _class_rcs_scale
from verify_unified_ramp import acquisition
from render_scene import configure_font

P=Path(__file__).resolve().parents[1]
DATA=Path('/nas/datasets/hyt/datasets/radar_open')
OUT=P/'docs/composite_extension/2026-10-15'
CACHE=DATA/'layered_validation/composite_20261015'

def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def demo_request(kind,tx,rx,mode):
    d=load(P/'configs/lab_moving_plate_po.json');d['scene']['objects'][0]['quadrature_order']=16
    d['array']={'layout':'siso' if tx*rx==1 else 'ula','num_tx':tx,'num_rx':rx}
    if kind=='fmcw':d['waveform']={'kind':kind,'start_hz':9e9,'slope_hz_s':1e12,'ramp_s':60e-6,'slot_period_s':200e-6,
        'adc_start_s':2e-6,'samples':128,'sample_rate_hz':4e6,'loops':64}
    d['sequence']={'frame_count':3,'frame_interval_s':.04}
    d['processing'].update(azimuth_deg=list(np.arange(-40,41)),elevation_deg=[0.])
    req=Simulation.model_validate(d);layout=build_layout(req.array,req.waveform,req.platform)
    ranges=range_axis(req)[selected_range_indices(req)];n=len(ranges);prf=1/layout.loop_period_s
    sign=1 if kind=='pulsed_lfm' else -1
    recipe=BackgroundRecipe(prf,tuple(ranges),tuple(np.full(n,1e-5)),(SpectrumComponent(1.,sign*prf/8,prf/128),),
        .05,3.,.05,tuple(map(tuple,np.eye(n))),tuple(map(tuple,np.zeros((n,n)))),'compound_gamma')
    d['receiver_noise']={'variance':1e-3,'seed':77}
    d['range_background']={'model':asdict(recipe),'power_scale':1.,'provenance':'synthetic laboratory background; not a transplanted sea/land measured calibration',
        'seed':42,'channel_mode':mode,'azimuth_deg':20 if mode=='steered' else 0}
    return Simulation.model_validate(d)

def laboratory():
    rows=[]
    for kind in ['pulsed_lfm','fmcw']:
        for tx,rx in [(1,1),(1,4),(2,4)]:
            for mode in ['independent','steered']:
                request=demo_request(kind,tx,rx,mode);name=f'{kind}_{tx}Tx{rx}Rx_{mode}'
                dump(OUT/'configs'/f'{name}.json',request.model_dump(mode='json'))
                environment=prepare_environment(request);layout=build_layout(request.array,request.waveform,request.platform)
                for index in range(3):
                    frame=simulate_frame(request,index,environment=environment);p=frame.products;c=frame.components
                    replay=process_range_iq(range_compress(frame.adc,request)+c['background_range_iq'],request,layout)
                    error=float(np.max(abs(replay.rd_channels-p.rd_channels))/max(float(np.max(abs(p.rd_channels))),1e-30))
                    row={'case':name,'frame':index,'waveform':kind,'tx':tx,'rx':rx,'background_channel_mode':mode,
                        'component_sum_relative_error':p.metadata['component_sum_relative_error'],'saved_stage_replay_relative_error':error,
                        'passed':bool(error<1e-6 and p.metadata['component_sum_relative_error']<2e-5)}
                    rows.append(row)
                    if tx==2 and rx==4 and mode=='steered':save_frame(CACHE/name,frame,True)
                    if tx==2 and rx==4 and mode=='steered' and index==0:
                        images=[np.mean(abs(c[k].astype(complex))**2,axis=(2,3)) for k in ['clean_rd_channels','receiver_noise_rd_channels','background_rd_channels']]+[p.rd_power]
                        reference=max(float(a.max()) for a in images)
                        fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
                        for ax,a,title in zip(axes,images,['有限面积金属板','ADC 接收机噪声经处理','距离处理后加入的统计背景','复数合成后的 RD']):
                            im=ax.pcolormesh(p.range_m,p.velocity_mps,power_db(a,reference).T,shading='auto',cmap='turbo',vmin=-60,vmax=0,rasterized=True)
                            ax.set(xlabel='距离 / m',ylabel='接近速度 / (m/s)',ylim=(-10,10),title=title)
                            fig.colorbar(im,ax=ax,label='同一参考 / dB',shrink=.8)
                        fig.suptitle(f'{kind} · 2Tx4Rx · 实验室组合场景\n先叠加复数信号，再取功率；所有幅度为声明的相对单位，不是实测背景复现',fontsize=12)
                        fig.savefig(OUT/'images'/f'components_{kind}.png',dpi=150);plt.close(fig)
                print('lab',name,flush=True)
    csvout(OUT/'metrics/laboratory.csv',rows)
    result={'frames':len(rows),'passed':sum(r['passed'] for r in rows),
        'max_component_sum_relative_error':max(r['component_sum_relative_error'] for r in rows),
        'max_saved_stage_replay_relative_error':max(r['saved_stage_replay_relative_error'] for r in rows),
        'is_measured_validation':False}
    dump(OUT/'metrics/laboratory_summary.json',result)
    if result['passed']!=len(rows):raise AssertionError('composite laboratory checks failed')
    return result

def ramp():
    old=DATA/'alignment_campaign/20261005'
    items=sorted([x for x in load(old/'ramp_v3/simulation_inputs.json') if x['role']=='heldout_record'],key=lambda x:x['frame_id'])
    noise=load(old/'ramp_noise_trial_v1/frozen_calibration.json');power=load(old/'ramp_v3/power_alignment.json')
    variance=np.array(noise['noise_variance_ADC_units_tx_major']).reshape(2,4).T
    gain=power['gains']['physical_smooth_roi_gain'];reference=power['reference_RD_power']
    selections=[]
    for seq in sorted({x['frame_id'].split('/')[1] for x in items}):
        group=[x for x in items if x['frame_id'].split('/')[1]==seq];selections.append(group[len(group)//2]['frame_id'])
    dump(OUT/'metrics/ramp_protocol.json',{'frames':[x['frame_id'] for x in items],'independent_records':3,
        'noise_calibration_source':str(old/'ramp_noise_trial_v1/frozen_calibration.json'),
        'noise_calibration_sha256':sha(old/'ramp_noise_trial_v1/frozen_calibration.json'),
        'power_calibration_sha256':sha(old/'ramp_v3/power_alignment.json'),
        'variance_per_rx_tx':variance.tolist(),'power_gain':gain,'reference_power':reference,
        'seeds':[42,142,242],'seed_rule':'base_seed+fixed_frame_index; receiver frame_index=0',
        'no_evaluation_refit':True,'selected_before_scoring':selections,
        'scene_model':'one ideal point per annotation; velocities from smoothed labels, zero placeholder when unavailable',
        'range_background_added':False,'reason':'IPIX sea background cannot be transferred to automotive ADC without validated units/scene mapping',
        'scope':'unified raw-receiver-noise integration recheck, not new untouched blind data or complete car fidelity'})
    cfg=acquisition();cfg.platform.beam_half_angle_deg=89;cfg.processing.azimuth_deg=[0.]
    layout=build_layout(cfg.array,cfg.waveform,cfg.platform);rows=[]
    for index,item in enumerate(items):
        adc=read_ramp_adc(DATA/'ramp_adc_77ghz/alignment_subset_20261005'/item['frame_id'])
        real=process_range_iq(range_compress(adc,cfg),cfg,layout)
        objects=[{'kind':'point','id':str(l['uid']),'position_ned_m':[l['y_m'],l['x_m'],0.],
            'velocity_ned_mps':l['velocity_ned_mps'],'amplitude':_class_rcs_scale(int(l['class_id']))*np.sqrt(gain)} for l in item['smoothed_labels']]
        d={**cfg.model_dump(),'scene':{'objects':objects,'amplitude_reference_range_m':10.},
            'sequence':{'frame_count':1},'receiver_noise':{'variance_per_rx_tx':variance.tolist(),'seed':42+index}}
        roi=np.zeros(real.rd_power.shape,bool)
        for l in item['raw_labels']:
            extent=.5*(abs(l['x_m']/l['range_m'])*l['width_m']+abs(l['y_m']/l['range_m'])*l['length_m'])
            roi|=(abs(real.range_m-l['range_m'])<=1.5+extent)[:,None]
        from radar_platform.unified.processing import window
        wr=window(cfg.processing.range_window,cfg.waveform.samples);wd=window(cfg.processing.doppler_window,cfg.waveform.loops)
        expected_floor=float(variance.mean()*np.sum(wr**2)/wr.sum()**2*np.sum(wd**2)/wd.sum()**2)
        strong=roi&(real.rd_power>=expected_floor*10)
        for seed in [42,142,242]:
            d['receiver_noise']['seed']=seed+index;req=Simulation.model_validate(d);frame=simulate_frame(req)
            c=frame.components
            versions={'target_only':np.mean(abs(c['clean_rd_channels'].astype(complex))**2,axis=(2,3)),
                'noise_only':np.mean(abs(c['receiver_noise_rd_channels'].astype(complex))**2,axis=(2,3)),
                'target_plus_noise':frame.products.rd_power}
            for label,signal in versions.items():
                comp=compare_power(real.rd_power,signal,reference_power=reference,target_mask=roi)
                rows.append({'frame_id':item['frame_id'],'sequence':item['frame_id'].split('/')[1],'model':label,'seed':seed+index,
                    'mae_db':comp.metrics['mae_db'],'roi_mae_db':comp.metrics['target_mae_db'],
                    'strong_mae_db':float(np.mean(abs(comp.difference_db[strong]))) if strong.any() else None,
                    'component_sum_relative_error':frame.products.metadata['component_sum_relative_error']})
            if item['frame_id'] in selections and seed==42:
                example=selections.index(item['frame_id'])+1
                dump(OUT/'configs'/f'ramp_example_{example}.json',req.model_dump(mode='json'))
                np.savez_compressed(CACHE/f'ramp_{example}.npz',measured_power=real.rd_power,**versions,
                    noisy_adc=frame.adc,noise_adc=c['receiver_noise_adc'],range_m=real.range_m,velocity_mps=real.velocity_mps)
                comp=compare_power(real.rd_power,versions['target_plus_noise'],reference_power=reference)
                fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
                arrays=[comp.measured_db,power_db(versions['target_only'],reference),comp.simulated_db,comp.difference_db]
                for k,(ax,a,title) in enumerate(zip(axes,arrays,['实测 RD','仿真目标（无噪声）','统一链路：目标＋ADC 噪声','差值：仿真 − 实测'])):
                    im=ax.pcolormesh(real.range_m,real.velocity_mps,a.T,shading='auto',cmap='RdBu_r' if k==3 else 'turbo',vmin=-30 if k==3 else -60,vmax=30 if k==3 else 10,rasterized=True)
                    ax.set(xlabel='距离 / m',ylabel='接近速度 / (m/s)',title=title);fig.colorbar(im,ax=ax,label='dB',shrink=.8,extend='both')
                fig.suptitle(item['frame_id']+'\n冻结历史噪声方差与幅度增益；没有向汽车场景强加海杂波',fontsize=11)
                fig.savefig(OUT/'images'/f'ramp_{example}.png',dpi=150);plt.close(fig)
                shutil.copy2(P/f'docs/weekly/2026-10-10/images/ramp_scene_{example}.png',OUT/'images'/f'ramp_scene_{example}.png')
        if index%20==0:print('RAMP composite',index+1,'/',len(items),flush=True)
    csvout(OUT/'metrics/ramp_frames.csv',rows);summary=[]
    for model in ['target_only','noise_only','target_plus_noise']:
        rr=[r for r in rows if r['model']==model]
        summary.append({'model':model,'frames':180,'records':3,'seeds':3,
            'mae_db':float(np.mean([r['mae_db'] for r in rr])),
            'strong_mae_db':float(np.mean([r['strong_mae_db'] for r in rr if r['strong_mae_db'] is not None])),
            'max_component_error':max(r['component_sum_relative_error'] for r in rr)})
    dump(OUT/'metrics/ramp_summary.json',summary)
    return summary

def report(lab,real):
    transfer=load(DATA/'layered_validation/composite_20261015_real_transfer/summary.json')
    dump(OUT/'metrics/real_processing_compatibility.json',transfer)
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/'images'/name).read_bytes()).decode()+'">'
    def table(h,rows):return '<table><tr>'+''.join('<th>'+v+'</th>' for v in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'
    body='<h1>面目标、背景与噪声：统一处理链</h1><p>2026-10-15 · 通用实验室场景与公开汽车雷达对照</p>'
    body+=table(['组成','加入的位置','不能混淆的事情'],[
        ['有限面积目标','生成原始复数 ADC','物理模型仍是近似，不是完整材料/多径数字孪生'],
        ['接收机高斯噪声','原始 ADC 层，随后参与距离与 Doppler 处理','方差按复采样功率给出，不按当前目标峰值自动凑 SNR'],
        ['统计环境背景','距离处理之后、Doppler 加窗之前','不是原始 ADC，必须另存复数背景；不直接加到 ADC 或 dB 图'],
        ['最终 RD / RA','先叠加复数信号，再统一处理与取功率','单帧功率不等于各分量功率逐像素相加']])
    body+='<h2>实验室有限金属板：逐项展示，而非图像叠加</h2>'+image('components_pulsed_lfm.png')+image('components_fmcw.png')
    body+=table(['检查','结果'],[
        ['两种波形 × SISO/SIMO/MIMO × 两种背景通道假设 × 3 帧',f"{lab['passed']} / {lab['frames']}"],
        ['复数分量相加与最终 RD 的最大相对差',f"{lab['max_component_sum_relative_error']:.3g}"],
        ['用已保存的 ADC 和距离域背景重放 RD',f"最大相对差 {lab['max_saved_stage_replay_relative_error']:.3g}"],
        ['真实 RAMP 处理兼容性',f"{transfer['passed']} / {transfer['frames']} 帧；不是保真通过率"]])
    body+='<p>背景可明确选择“各通道独立”或“单一已知方向的相干场”。后者考虑 Tx 时隙偏移与空间相位，但只是声明的角度假设，不代表真实海面/道路的角度分布。背景距离轴必须精确匹配处理网格，PRF 必须匹配同一 Tx 的采样节奏；不匹配会报错，不静默插值。</p>'
    body+='<p>为避免重复计算底噪，组合场景默认从背景配方中移除其接收机式白噪声分量，再单独加入 ADC 噪声；不会把被移除的功率重新分配给杂波。若要求同时保留该分量并添加 ADC 噪声，接口会报错。这是明确的统计分量处理约定，不等于已独立测出了硬件热噪声。</p>'
    body+='<h2>RAMP：新接收机噪声接口在真实数据上复核</h2>'+table(['模型','180 帧 × 3 种子平均 MAE','标签区强回波 MAE'],[
        [r['model'],f"{r['mae_db']:.2f} dB",f"{r['strong_mae_db']:.2f} dB"] for r in real])
    body+='<p>3 段独立采集，沿用历史标签与冻结噪声标定，不是全新盲测。强回波定义为标签距离区中高于校准噪声均值 10 dB 的实测单元，并非独立精确分割。保留“只有噪声”对照，避免把背景底噪相似当作车体已复现。</p>'
    for i in [1,2,3]:body+=f'<details {"open" if i==1 else ""}><summary>固定样例 {i}：输入场景与结果</summary>'+image(f'ramp_scene_{i}.png')+image(f'ramp_{i}.png')+'</details>'
    body+='<h2>输出与使用边界</h2><p>沿用 /api/v1/scene-simulations，新增可选 receiver_noise、range_background。默认均关闭，旧理想场景数值保持兼容。保存 clean_rd_channels、receiver_noise_rd_channels、background_rd_channels 与 background_range_iq，便于逐项复核。原始 ADC 只包含目标和可选接收机噪声，不含后加的统计背景。</p>'
    body+=table(['接口','可直接使用的功能','输出边界'],[
        ['/api/v1/scene-simulations','点 / 有限面积面目标 + 可选噪声 / 统计背景','复数分量、RD、RA；默认理想模型不变'],
        ['/api/v1/background-simulations','独立高斯、相关高斯、复合起伏背景','距离门 I/Q，不是原始 ADC'],
        ['/api/v1/scattering/ground-response','受频率、极化、地类、角度约束的经验地表系数','散射系数与等效面积响应，不伪装成动态 RD']])
    body+='<p>仍未完成的保真问题包括：玛曲跨日距离偏差、完整材料与环境多径、真实背景的角度与空间共同起伏。草地经验系数尚未自动转换成有标定依据的地形 RD。不同频段、不同场景的实测统计参数不能自动迁移。新增 MRR 数据的物理坐标与体散射验证也仍需推进。</p>'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>统一面目标背景噪声链路</title><style>body{max-width:1250px;margin:30px auto;padding:20px;font:16px/1.65 system-ui;color:#26374c}img{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #cad5df;overflow-wrap:anywhere}th{background:#e9f3f6}h2{margin-top:35px}summary{padding:12px;background:#edf4f7;cursor:pointer}</style>'+body+'</html>'
    (OUT/'统一面目标背景噪声链路.html').write_text(doc)
    files=[*P.glob('backend/radar_platform/unified/*.py'),P/'backend/radar_platform/ground_api.py',
           P/'backend/radar_platform/api.py',P/'tests/test_composite_environment.py',P/'tests/test_empirical_ground.py',Path(__file__)]
    dump(OUT/'metrics/provenance.json',{'source_sha256':{str(f.relative_to(P)):sha(f) for f in files},
        'raw_and_components_cache':str(CACHE),'laboratory_is_real':False,'ramp_is_real':True,'RAMP_scene_is_point_approximation':True})
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.relative_to(P))
    with zipfile.ZipFile(OUT.parent/'统一链路_2026-10-15.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in OUT.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(OUT))

def main():
    global OUT,CACHE
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    OUT=args.output.resolve();CACHE=args.cache.resolve()
    for p in [OUT/'images',OUT/'configs',OUT/'metrics',CACHE]:p.mkdir(parents=True,exist_ok=True)
    configure_font()
    dump(OUT/'metrics/selection_before_run.json',{'laboratory_waveforms':['pulsed_lfm','fmcw'],'arrays':[[1,1],[1,4],[2,4]],
        'background_modes':['independent','steered'],'frames_per_case':3,'real_ramp_frames':'all 180 prior heldout_record items','noise_seeds':[42,142,242]})
    if args.render_only:
        lab=load(OUT/'metrics/laboratory_summary.json');real=load(OUT/'metrics/ramp_summary.json')
    else:lab=laboratory();real=ramp()
    report(lab,real)
    print(json.dumps({'laboratory':lab,'ramp':real},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
