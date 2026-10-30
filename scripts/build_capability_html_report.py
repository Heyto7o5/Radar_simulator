#!/usr/bin/env python3
"""Offline pictorial report; new paired examples use the current unified engine.

No simulation model is modified. Three record-midpoint frames are chosen before
scoring, with a previously frozen calibration gain. Original ADC stays on NAS.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from radar_platform.comparison import compare_power
from radar_platform.ramp_alignment import read_ramp_adc
from radar_platform.ramp_comparison import _class_rcs_scale
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.processing import process_adc
from radar_platform.unified.schema import Simulation
from radar_platform.unified.storage import write_json
from verify_unified_ramp import acquisition
from render_scene import configure_font

PROJECT=Path(__file__).resolve().parents[1]
CAMPAIGN=Path('/nas/datasets/hyt/datasets/radar_open/alignment_campaign/20261005')
DATA=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005')
DERIVED=Path('/nas/datasets/hyt/datasets/radar_open/alignment_campaign/report_20261008')
CLASS_NAMES={0:'行人',2:'汽车',3:'摩托车',5:'公交车',7:'卡车',80:'骑行者'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):return json.loads(path.read_text())


def triptych(path,comparison,ranges,vertical,title,ylabel,mask=None,ylim=None):
    fig,axes=plt.subplots(1,3,figsize=(13.8,3.9),layout='constrained')
    extent=[ranges[0],ranges[-1],vertical[0],vertical[-1]]
    arrays=[comparison.measured_db,comparison.simulated_db,comparison.difference_db]
    for i,(ax,array,label) in enumerate(zip(axes,arrays,['实测 RD','仿真 RD','差值：仿真 − 实测'])):
        if mask is not None:array=np.where(mask,array,np.nan)
        image=ax.imshow(array,origin='lower',aspect='auto',extent=extent,interpolation='nearest',
            cmap='RdBu_r' if i==2 else 'turbo',vmin=-40 if i==2 else -60,vmax=40 if i==2 else 20)
        ax.set(xlabel='距离 / m',ylabel=ylabel,title=label)
        if ylim:ax.set_ylim(*ylim)
        fig.colorbar(image,ax=ax,label='dB',shrink=.84,extend='both')
    fig.suptitle(title,fontsize=12)
    fig.savefig(path,dpi=165);plt.close(fig)


def prepare_ramp(output):
    inputs=load(CAMPAIGN/'ramp_v3/simulation_inputs.json')
    power=load(CAMPAIGN/'ramp_v3/power_alignment.json')
    chosen=[]
    for sequence in sorted({x['frame_id'].split('/')[1] for x in inputs if x['role']=='heldout_record'}):
        frames=[x for x in inputs if x['frame_id'].split('/')[1]==sequence]
        chosen.append(frames[len(frames)//2])
    DERIVED.mkdir(parents=True,exist_ok=True)
    files=[*PROJECT.glob('backend/radar_platform/unified/*.py'),Path(__file__).resolve()]
    protocol={'client_report_date':'2026-10-08','selection':'middle item of each of the three pre-existing cross-record frame lists, selected before scoring',
        'frames':[x['frame_id'] for x in chosen], 'calibration_origin':str(CAMPAIGN/'ramp_v3/power_alignment.json'),
        'calibration_sha256':sha(CAMPAIGN/'ramp_v3/power_alignment.json'),
        'power_gain':power['gains']['physical_smooth_roi_gain'],'reference_power':power['reference_RD_power'],
        'per_display_frame_fit':False,'simulation_engine':'radar_platform.unified.engine.simulate_frame',
        'model':'one ideal point per annotated object, annotation-smoothed Cartesian velocity, fixed class-relative amplitudes',
        'background_noise_or_multipath_added':False,'absolute_amplitude_calibrated':False,
        'scope':'three newly generated illustration pairs, NOT a new batch fidelity certification',
        'source_sha256':{str(f.relative_to(PROJECT)):sha(f) for f in files}}
    write_json(DERIVED/'selection_and_protocol.json',protocol)
    with zipfile.ZipFile(DERIVED/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(f,f.relative_to(PROJECT))
    rows=[]
    for index,item in enumerate(chosen,1):
        cfg=acquisition()
        cfg.platform.beam_half_angle_deg=89.  # Explicit broad ideal illumination, not measured antenna pattern.
        layout=build_layout(cfg.array,cfg.waveform,cfg.platform)
        real=process_adc(read_ramp_adc(DATA/item['frame_id']),cfg,layout)
        objects=[{'kind':'point','id':str(label['uid']),
            'position_ned_m':[label['y_m'],label['x_m'],0.],
            'velocity_ned_mps':label['velocity_ned_mps'],
            'amplitude':_class_rcs_scale(int(label['class_id']))*np.sqrt(protocol['power_gain'])}
            for label in item['smoothed_labels']]
        request=Simulation.model_validate({**cfg.model_dump(),'name':f'report paired example {index}',
            'scene':{'objects':objects,'amplitude_reference_range_m':10.},'sequence':{'frame_count':1}})
        frame=simulate_frame(request)
        np.testing.assert_array_equal(real.range_m,frame.products.range_m)
        np.testing.assert_array_equal(real.velocity_mps,frame.products.velocity_mps)
        roi=np.zeros_like(real.rd_power,dtype=bool)
        for label in item['raw_labels']:
            radius=label['range_m']
            extent=.5*(abs(label['x_m']/radius)*label['width_m']+abs(label['y_m']/radius)*label['length_m'])
            roi|=(abs(real.range_m-radius)<=1.5+extent)[:,None]
        comparison=compare_power(real.rd_power.T,frame.products.rd_power.T,
            reference_power=protocol['reference_power'],target_mask=roi.T)
        name=f'ramp_pair_{index}'
        classes='、'.join(CLASS_NAMES.get(c,str(c)) for c in sorted({int(l['class_id']) for l in item['raw_labels']}))
        sequence=item['frame_id'].split('/')[1]
        unknown=sum(l.get('radial_velocity_mps') is None for l in item['smoothed_labels'])
        title=f"样例 {index} · {classes} · 全图 MAE {comparison.metrics['mae_db']:.2f} dB / 标注距离区 MAE {comparison.metrics['target_mae_db']:.2f} dB"
        triptych(output/'images'/f'{name}.png',comparison,real.range_m,real.velocity_mps,title,'接近速度 / (m/s)')
        row={'example':index,'sequence':sequence,'frame_id':item['frame_id'],'classes':classes,'targets':len(objects),
            'unknown_velocity_targets':unknown,'mae_db':comparison.metrics['mae_db'],
            'target_mae_db':comparison.metrics['target_mae_db'],'joint_floor_fraction':comparison.metrics['joint_floor_fraction'],
            'source_adc_sha256':sha(DATA/item['frame_id']),'image':f'{name}.png','input_scene':f'{name}_request.json'}
        rows.append(row)
        write_json(output/'metrics'/f'{name}_request.json',request.model_dump(mode='json'))
        write_json(DERIVED/f'{name}_request.json',request.model_dump(mode='json'))
        # Numeric/complex data remain under the approved data root, not in the report ZIP.
        np.savez_compressed(DERIVED/f'{name}.npz',real_rd_power=real.rd_power,
            simulated_rd_power=frame.products.rd_power,simulated_adc=frame.adc,
            range_m=real.range_m,velocity_mps=real.velocity_mps,roi=roi,
            reference_power=protocol['reference_power'])
        print(json.dumps(row,ensure_ascii=False),flush=True)
    write_json(output/'metrics/ramp_examples.json',{'protocol':protocol,'examples':rows})
    with (output/'metrics/ramp_examples.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def prepare_ipix(output):
    path=CAMPAIGN/'ipix_v1/record_40_average_rd.npz'
    with np.load(path,allow_pickle=False) as z:
        comparison=compare_power(z['measured'][0],z['after'][0],reference_power=float(z['reference_power']),valid_mask=z['valid_mask'])
        triptych(output/'images/ipix_pair.png',comparison,z['range_m'],z['doppler_hz'],
            f"IPIX #40 · 后段第 1 窗 · 16 CPI 平均 RD · 全部有效单元 MAE {comparison.metrics['mae_db']:.2f} dB",
            'Doppler / Hz',mask=z['valid_mask'],ylim=(-150,150))
    summary=load(CAMPAIGN/'ipix_v1/summary.json')
    values={model:[r['mae_db']['mean'] for r in summary['records'] if r['role']=='new_heldout_record' and r['model']==model]
            for model in ['frozen_reference_transfer','prefix_conditioned']}
    result={'sample_mae_db':comparison.metrics['mae_db'],'record':40,'window':0,'source_npz_sha256':sha(path),
            'new_record_count':len(values['prefix_conditioned']),
            'before_mean_mae_db':float(np.mean(values['frozen_reference_transfer'])),
            'after_mean_mae_db':float(np.mean(values['prefix_conditioned'])),
            'scope':'previous statistical-clutter model; record-prefix-conditioned prediction, not a unified terrain-surface replay'}
    write_json(output/'metrics/ipix_example.json',result)
    return result


def compile_html(output):
    template=(PROJECT/'docs/templates/capability_report.html').read_text()
    examples=load(output/'metrics/ramp_examples.json')['examples']
    ipix=load(output/'metrics/ipix_example.json')
    analytic=load(PROJECT/'docs/unified/metrics/analytic_v2/summary.json')
    result_rows=''
    example_blocks=''
    for row in examples:
        velocity=f"{row['unknown_velocity_targets']} 个目标的速度未知，仿真用 0 作为占位，不是静止真值。" if row['unknown_velocity_targets'] else '速度来自标签轨迹估计，不是独立测量真值。'
        result_rows+=f"<tr><td>{row['example']}</td><td>{html.escape(row['classes'])}</td><td>{row['targets']}</td><td>{row['mae_db']:.2f} dB</td><td>{row['target_mae_db']:.2f} dB</td><td>{html.escape(velocity)}</td></tr>"
        example_blocks+=f'''<figure class="pair"><div class="figure-meta"><b>样例 {row['example']} · {html.escape(row['classes'])}</b><span>{html.escape(row['sequence'])} / 000451</span></div>
        <img src="@@IMAGE:{row['image']}@@" alt="样例 {row['example']} 的实测 RD、当前统一框架仿真 RD 和差值" width="2277" height="643">
        <figcaption>{html.escape(velocity)} 每个标注物体仅用一个理想散射点；没有车身网格、道路背景或噪声模型。</figcaption></figure>'''
    substitutions={'@@RAMP_ROWS@@':result_rows,'@@RAMP_EXAMPLES@@':example_blocks,
        '@@IPIX_MAE@@':f"{ipix['sample_mae_db']:.2f}",
        '@@IPIX_BEFORE@@':f"{ipix['before_mean_mae_db']:.2f}",'@@IPIX_AFTER@@':f"{ipix['after_mean_mae_db']:.2f}",
        '@@ANALYTIC_CASES@@':str(analytic['cases']),
        '@@RANGE_ERROR@@':f"{analytic['max_error']['range_error_bins']:.2f}",
        '@@VELOCITY_ERROR@@':f"{analytic['max_error']['velocity_error_bins']:.2f}"}
    for key,value in substitutions.items():template=template.replace(key,value)
    embedded={}
    for filename in re.findall(r'@@IMAGE:([^@]+)@@',template):
        path=output/'images'/filename
        if not path.exists():raise FileNotFoundError(path)
        embedded[filename]='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
    for filename,url in embedded.items():template=template.replace(f'@@IMAGE:{filename}@@',url)
    if '@@' in template:raise ValueError('unresolved template marker')
    (output/'雷达仿真平台_功能扩展与验证报告.html').write_text(template)
    # A second copy references separate image files, convenient for reuse/editing.
    editable=template
    for filename,url in embedded.items():editable=editable.replace(url,'images/'+filename)
    (output/'报告_分离图片版.html').write_text(editable)


def input_geometry(output):
    """Show exactly what is known from annotations, not an invented vehicle CAD."""
    rows=load(output/'metrics/ramp_examples.json')['examples']
    lookup={x['frame_id']:x for x in load(CAMPAIGN/'ramp_v3/simulation_inputs.json')}
    fig,axes=plt.subplots(1,len(rows),figsize=(13.8,3.8),layout='constrained')
    for ax,row in zip(axes,rows):
        item=lookup[row['frame_id']]
        for ordinal,(raw,smooth) in enumerate(zip(item['raw_labels'],item['smoothed_labels']),1):
            color='#c77e25' if smooth['radial_velocity_mps'] is None else '#245d88'
            ax.add_patch(Rectangle((raw['x_m']-raw['width_m']/2,raw['y_m']-raw['length_m']/2),
                raw['width_m'],raw['length_m'],fill=False,edgecolor=color,lw=1.4))
            ax.scatter(smooth['x_m'],smooth['y_m'],color=color,s=28)
            text=f'T{ordinal}'+('：速度未知' if smooth['radial_velocity_mps'] is None else '')
            ax.annotate(text,(smooth['x_m'],smooth['y_m']),xytext=(4,4),textcoords='offset points',fontsize=8,color=color)
            if smooth['radial_velocity_mps'] is not None:
                v=smooth['velocity_ned_mps']
                ax.arrow(smooth['x_m'],smooth['y_m'],v[1],v[0],color=color,width=.025,head_width=.25,length_includes_head=True)
        ax.scatter(0,0,marker='^',color='#172e3f',s=45)
        ax.set(xlim=(-16,12),ylim=(-2,27),xlabel='横向位置 / m',ylabel='前向位置 / m',title=f"样例 {row['example']}：{row['classes']} / {row['targets']} 个标注目标")
        ax.set_aspect('equal');ax.grid(alpha=.18)
    fig.suptitle('输入给仿真器的目标：公开标注框 + 中心散射点 + 标签估计速度\n矩形不是车辆表面模型；圆点才是当前参与回波生成的目标近似',fontsize=12)
    fig.savefig(output/'images/ramp_input_geometry.png',dpi=165);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'docs/reports/2026-10-08')
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    for name in ['images','metrics']: (output/name).mkdir(parents=True,exist_ok=True)
    configure_font();plt.rcParams['font.size']=10
    images={
        'docs/unified/images/scene.png':'scene.png','docs/unified/images/lfm_rd.png':'lfm_rd.png',
        'docs/unified/images/fmcw_rd.png':'fmcw_rd.png','docs/unified/images/array_and_angle.png':'array_and_angle.png',
        'docs/unified/images/mimo_resolution.png':'mimo_resolution.png','docs/unified/images/analytic_checks.png':'analytic_checks.png',
        'docs/alignment/2026-10-05/images/ipix_records.png':'ipix_records.png',
        'docs/alignment/2026-10-05/images/maroon_distance_examples.png':'maroon_distance_examples.png',
        'docs/alignment/2026-10-05/images/ramp_noise_control.png':'ramp_noise_control.png',
    }
    for source,dest in images.items():shutil.copy2(PROJECT/source,output/'images'/dest)
    if not args.render_only:
        prepare_ramp(output);prepare_ipix(output)
    input_geometry(output)
    for source,dest in [
        ('docs/unified/metrics/analytic_v2/summary.json','analytic_summary.json'),
        ('docs/unified/metrics/analytic_v2/cases.csv','analytic_cases.csv'),
        ('docs/unified/metrics/real_transfer_v2/summary.json','real_adc_transfer_summary.json'),
        ('docs/unified/metrics/real_transfer_v2/frames.csv','real_adc_transfer_frames.csv'),
        ('docs/unified/metrics/resolution.json','mimo_resolution.json'),
        ('docs/unified/provenance/completion_audit.json','framework_audit.json'),
        ('docs/alignment/2026-10-05/metrics/ipix_v1/summary.json','ipix_batch_summary.json'),
        ('docs/alignment/2026-10-05/metrics/maroon_rigid_v1/summary.json','maroon_static_summary.json'),
    ]:shutil.copy2(PROJECT/source,output/'metrics'/dest)
    compile_html(output)
    manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write_json(output/'manifest.json',{'report_date':'2026-10-08','generated_utc':datetime.now(timezone.utc).isoformat(),
        'derived_arrays_retained_at':str(DERIVED),'files':manifest,
        'scope':'offline report; no frontend/server started and no simulator source modified',
        'raw_measurements_packaged':False})
    archive_path=output.parent/'雷达仿真平台_HTML图表报告_2026-10-08.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file():archive.write(p,Path('雷达仿真平台报告')/p.relative_to(output))
        if archive.testzip():raise ValueError('archive CRC failure')
    print(json.dumps({'html':str(output/'雷达仿真平台_功能扩展与验证报告.html'),'zip':str(archive_path),
        'archive_bytes':archive_path.stat().st_size,'images':len(list((output/'images').glob('*.png')))},ensure_ascii=False))


if __name__=='__main__':main()
