#!/usr/bin/env python3
"""Public access audit and explicitly NON-dataset-matched analytic prerequisites.

No gated downloads, no guessed ADC layout, no claimed real-data comparison.
"""
import hashlib,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.unified.schema import Simulation
from radar_platform.unified.engine import simulate_frame

PROJECT=Path(__file__).resolve().parents[1]
DATA=Path('/nas/datasets/hyt/datasets/radar_open/sim2radar')
OUT=PROJECT/'docs/sim2radar/2026-10-16'

def main():
    for folder in [DATA/'raw',DATA/'extracted',DATA/'processed',OUT/'images',OUT/'configs']:folder.mkdir(parents=True,exist_ok=True)
    url='https://huggingface.co/api/datasets/AnonymousAuthors0018/sim2radar_artifacts?expand%5B%5D=gated&expand%5B%5D=sha'
    with urllib.request.urlopen(url,timeout=25) as response:metadata=json.load(response)
    manifest={'checked_utc':datetime.now(timezone.utc).isoformat(),'repository':metadata['id'],'revision':metadata['sha'],
              'gated':metadata.get('gated'),'metadata_source':url,'raw_data_acquired':False,'real_comparison_performed':False,
              'dataset_card':'https://huggingface.co/datasets/AnonymousAuthors0018/sim2radar_artifacts/blob/main/README.md',
              'public_card_facts':{'radar':'TI IWR1843BOOST + DCA1000EVM','band_hz':77e9,
                  'rover_speed_mps':.275,'rover_travel_m':1.,'rover_passes':4,'rover_start_distances_m':[.5,1.,2.],
                  'microbench_objects':['corner reflector','soda can'],
                  'microbench_motions':['turntable clockwise/counterclockwise','forward/back','lateral','diagonal','static'],
                  'rover_raw_frame_bytes':131072},
              'missing_for_alignment':['approved ADC/IQ files','per-record chirp/profile configuration',
                  'active Tx/Rx and channel ordering','object dimensions and rotation-axis offset',
                  'time-resolved translation/rotation reference and timestamps','camera/radar extrinsics when used'],
              'do_not_infer_cube_shape_from_byte_count':True,
              'similarly_named_paper':'https://arxiv.org/abs/2602.13314',
              'paper_dataset_linkage_verified':False,
              'paper_caution':'public paper evaluates IFR cascade radar; do not substitute its parameters for this gated IWR1843 repository'}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'access_and_inputs.json').write_text(json.dumps(manifest,indent=2))
    rows=[];pictures=[]
    for distance in [.5,1.,2.]:
        for mode in ['static','radar_receding','target_receding']:
            # Deliberately declared test configuration, not reverse-engineered
            # from the official file size and not borrowed as dataset truth.
            request=Simulation.model_validate(dict(name=f'Sim2Radar preparation ONLY: {mode} {distance}m',
                waveform=dict(kind='fmcw',start_hz=77e9,slope_hz_s=20e12,ramp_s=40e-6,slot_period_s=.001,
                              adc_start_s=2e-6,sample_rate_hz=8e6,samples=256,loops=128),
                platform=dict(position_ned_m=[0,0,0],velocity_ned_mps=[-.275 if mode=='radar_receding' else 0,0,0],
                              boresight_ned=[1,0,0],right_hint_ned=[0,1,0],beam_half_angle_deg=30),
                array=dict(layout='siso',num_tx=1,num_rx=1),
                scene=dict(objects=[dict(kind='point',id='ideal_point_not_measured_reflector',position_ned_m=[distance,0,0],
                                         velocity_ned_mps=[.275 if mode=='target_receding' else 0,0,0])],amplitude_reference_range_m=1.),
                processing=dict(range_limits_m=[0,3],azimuth_deg=[0.],elevation_deg=[0.])))
            frame=simulate_frame(request);products=frame.products
            peak=np.unravel_index(products.rd_power.argmax(),products.rd_power.shape)
            epoch=frame.metadata['truth_epoch_s'];moving=mode!='static'
            truth_r=distance+(.275*epoch if moving else 0);truth_v=-.275 if moving else 0
            dr=float(np.diff(products.range_m)[0]);dv=float(np.diff(products.velocity_mps)[0])
            row={'case':f'{mode}_{distance:g}m','is_real':False,'dataset_configuration_matched':False,
                 'motion_speed_source':'rover-card nominal speed; target-motion case uses same speed only for analytic relative-motion check',
                 'target':'ideal point, NOT actual corner geometry','true_range_m':truth_r,'detected_range_m':float(products.range_m[peak[0]]),
                 'true_closing_velocity_mps':truth_v,'detected_closing_velocity_mps':float(products.velocity_mps[peak[1]]),
                 'range_error_bins':abs(float(products.range_m[peak[0]])-truth_r)/dr,
                 'velocity_error_bins':abs(float(products.velocity_mps[peak[1]])-truth_v)/dv}
            row['analytic_pass']=row['range_error_bins']<=1 and row['velocity_error_bins']<=1
            rows.append(row)
            (OUT/'configs'/f'{row["case"]}.json').write_text(request.model_dump_json(indent=2))
            np.savez_compressed(DATA/'processed'/f'ideal_{row["case"]}.npz',adc=frame.adc,rd=products.rd_channels,
                                range_m=products.range_m,velocity_mps=products.velocity_mps)
            if distance==1.:pictures.append((mode,products))
    reference=max(float(p.rd_power.max()) for _,p in pictures)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,(name,p) in zip(axes,pictures):
        image=ax.pcolormesh(p.velocity_mps,p.range_m,10*np.log10(np.maximum(p.rd_power/reference,1e-6)),
                           shading='auto',vmin=-60,vmax=0,cmap='viridis')
        ax.set(title=name,xlabel='Closing velocity (m/s)',ylabel='Range (m)')
        fig.colorbar(image,ax=ax,label='dB, shared synthetic reference')
    fig.suptitle('SYNTHETIC ANALYTIC CHECKS ONLY — NOT Sim2Radar measurement/configuration')
    fig.savefig(OUT/'images/ideal_motion_prerequisites.png',dpi=150);plt.close(fig)
    (OUT/'analytic_checks.json').write_text(json.dumps({'is_real':False,'dataset_aligned':False,'cases':rows},indent=2))
    (OUT/'Sim2Radar对齐准备.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}.note{background:#fff0ce;padding:15px}</style>
<h1>Sim2Radar 对齐准备</h1><p class="note">仓库需要人工批准，当前没有获得真实 ADC。以下仅为理想运动自检；chirp 参数和目标形状未与数据集对齐，没有真实图、差值图或实测 MAE。</p>
<img src="images/ideal_motion_prerequisites.png"><p>左：平台和理想点均静止。中：雷达远离静止点。右：点远离静止雷达。后两种相对运动相同，应用于检查距离与 Doppler 符号。采用公开 rover 标称速度作自检，但不能视为 microbench 转台或目标运动真值。</p>
<p>获取数据后先核对每条记录的配置、通道排列、目标尺寸、旋转轴和运动时序。优先做简单目标平移，再加入刚体旋转；噪声和多径单独验证，不从实测图反推参数后冒充独立预测。</p>
<p>同名 arXiv 论文使用 IFR 级联雷达，与此仓库卡片的 IWR1843 配置不同，关联尚未核实，暂不混用参数。131,072 字节不能唯一确定 ADC 数据立方体形状。</p>
<p><a href="access_and_inputs.json">访问状态、已知输入与缺口</a> · <a href="analytic_checks.json">九组解析检查</a> · <a href="https://huggingface.co/datasets/AnonymousAuthors0018/sim2radar_artifacts">数据访问申请</a></p></html>''')
    print(json.dumps({'gated':manifest['gated'],'cases':len(rows),'passed':sum(r['analytic_pass'] for r in rows),
                      'real_data_acquired':False,'dataset_aligned':False,'report':str(OUT/'Sim2Radar对齐准备.html')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
