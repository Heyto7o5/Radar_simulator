#!/usr/bin/env python3
"""Summarize only complete frozen campaigns; changes are not measured errors."""
import argparse,base64,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SITES=['alpine','valley','foothill'];SEEDS=[11,23,37]
VARIANTS=['uniform_0','hamming_0','hamming_8','tree_plus6','water_minus6']


def load_campaign(path,expected):
    status=json.loads((path/'campaign_status.json').read_text())
    if status['status']!='complete' or status['completed_cases']!=expected or status['expected_cases']!=expected:
        raise ValueError(f'campaign incomplete: {path}; no partial success summary allowed')
    frozen=json.loads((path/'campaign_inputs.json').read_text())
    for name,digest in frozen['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('current source differs from frozen campaign')
    outputs={}
    for row in status['results']:
        if row['status']!='complete':raise ValueError('failed case retained; cannot omit it')
        run=path/row['case'];manifest=json.loads((run/'run.json').read_text())
        if manifest['source_sha256']!=frozen['source_sha256']:raise ValueError('per-run source snapshot differs from campaign')
        meta=json.loads((run/'frame_0000/metadata.json').read_text());spec=meta['products']['rd_power']
        power=np.fromfile(run/'frame_0000'/spec['file'],dtype=spec['dtype']).reshape(spec['shape']).astype(float)
        if not np.isfinite(power).all() or power.max()<=0:raise ValueError('invalid or empty RD output')
        request=json.loads((run/'request.json').read_text())
        if request['scene']['seed']!=row['seed']:raise ValueError('seed differs from plan')
        outputs[(row['site'],row['variant'],row['seed'])]=(power,meta,request)
    return outputs,status


def db(power,reference):return 10*np.log10(np.maximum(power/reference,1e-6))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--scan-campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main_data,main_status=load_campaign(a.campaign,45);scan_data,scan_status=load_campaign(a.scan_campaign,6)
    a.output.mkdir(parents=True,exist_ok=False)
    reference=max(main_data[(s,'uniform_0',seed)][0].max() for s in SITES for seed in SEEDS)
    largest_db=max(float(10*np.log10(item[0].max()/reference)) for item in [*main_data.values(),*scan_data.values()])
    display_top=max(0.,float(np.ceil(largest_db/5)*5))
    rows=[]
    for site in SITES:
        for seed in SEEDS:
            baseline,meta,base_request=main_data[(site,'hamming_0',seed)]
            for variant in VARIANTS:
                power,m,r=main_data[(site,variant,seed)]
                for key in ['waveform','platform','array','sequence','processing','visibility']:
                    if r[key]!=base_request[key]:raise ValueError(f'unpaired acquisition changed: {key}')
                for key in ['north_m','east_m','elevation_up_m','land_cover_map']:
                    if r['scene']['terrain'][key]!=base_request['scene']['terrain'][key]:raise ValueError('input geometry or labels changed')
                if m['axes']!=meta['axes']:raise ValueError('physical axes differ')
                rows.append(dict(site=site,seed=seed,variant=variant,
                    mean_abs_db_change_vs_hamming0=float(np.mean(abs(db(power,reference)-db(baseline,reference)))),
                    integrated_power_change_db=float(10*np.log10(power.sum()/baseline.sum())),
                    elapsed_s=m['elapsed_s'],occluded_channel_slots=m['visibility']['occluded_scatterer_channel_slots']))
    with (a.output/'per_case_changes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for site in SITES:
        for variant in VARIANTS:
            selected=[r for r in rows if r['site']==site and r['variant']==variant]
            changes=np.array([r['mean_abs_db_change_vs_hamming0'] for r in selected])
            summaries.append(dict(site=site,variant=variant,seeds=len(selected),
                mean_abs_db_change_mean=float(changes.mean()),median=float(np.median(changes)),p90=float(np.quantile(changes,.9)),
                mean_integrated_power_change_db=float(np.mean([r['integrated_power_change_db'] for r in selected]))))
    pictures=[]
    for label,control,treatment in [('weighting','uniform_0','hamming_0'),('pointing','hamming_0','hamming_8'),
                                     ('tree_sensitivity','hamming_0','tree_plus6'),('water_sensitivity','hamming_0','water_minus6')]:
        fig,axes=plt.subplots(3,3,figsize=(15,10),layout='constrained')
        for i,site in enumerate(SITES):
            before=np.mean([main_data[(site,control,s)][0] for s in SEEDS],axis=0)
            after=np.mean([main_data[(site,treatment,s)][0] for s in SEEDS],axis=0)
            m=main_data[(site,control,11)][1];r=np.array(m['axes']['range_m']);v=np.array(m['axes']['velocity_mps'])
            extent=[r[0]-(r[1]-r[0])/2,r[-1]+(r[1]-r[0])/2,v[0]-(v[1]-v[0])/2,v[-1]+(v[1]-v[0])/2]
            bd,ad=db(before,reference),db(after,reference)
            for j,(value,title) in enumerate([(bd,control),(ad,treatment),(ad-bd,'treatment minus control')]):
                im=axes[i,j].imshow(value.T,origin='lower',aspect='auto',extent=extent,
                    cmap='RdBu_r' if j==2 else 'viridis',vmin=-30 if j==2 else -60,vmax=30 if j==2 else display_top)
                axes[i,j].set(title=f'{site}: {title}',xlabel='Slant range (m)',ylabel='Apparent closing velocity (m/s)');fig.colorbar(im,ax=axes[i,j],label='dB')
        fig.suptitle('Synthetic RD controls: mean of 3 seeded linear-power realizations, then dB — NOT measured fidelity')
        path=a.output/(label+'.png');fig.savefig(path,dpi=130);plt.close(fig);pictures.append(path)
    scan_rows=[];fig,axes=plt.subplots(3,3,figsize=(15,10),layout='constrained')
    for i,site in enumerate(SITES):
        held,meta,r0=scan_data[(site,'hold0_t20ms',11)];scanned,m,r1=scan_data[(site,'scan8_t20ms',11)]
        for key in ['waveform','platform','array','sequence','processing','visibility','scene']:
            if r0[key]!=r1[key]:raise ValueError('scan is not paired at identical geometry/time')
        r=np.array(meta['axes']['range_m']);v=np.array(meta['axes']['velocity_mps'])
        extent=[r[0]-(r[1]-r[0])/2,r[-1]+(r[1]-r[0])/2,v[0]-(v[1]-v[0])/2,v[-1]+(v[1]-v[0])/2]
        bd,ad=db(held,reference),db(scanned,reference)
        for j,(value,title) in enumerate([(bd,'hold 0 deg at 20 ms'),(ad,'scan to 8 deg at 20 ms'),(ad-bd,'scan minus held')]):
            im=axes[i,j].imshow(value.T,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r' if j==2 else 'viridis',vmin=-30 if j==2 else -60,vmax=30 if j==2 else display_top)
            axes[i,j].set(title=f'{site}: {title}',xlabel='Slant range (m)',ylabel='Apparent closing velocity (m/s)');fig.colorbar(im,ax=axes[i,j],label='dB')
        scan_rows.append(dict(site=site,simulation_time_s=.02,mean_abs_db_change=float(np.mean(abs(ad-bd))),
                              integrated_power_change_db=float(10*np.log10(scanned.sum()/held.sum()))))
    path=a.output/'scan_same_time.png';fig.savefig(path,dpi=130);plt.close(fig);pictures.append(path)
    summary={'real_sites':3,'main_runs':45,'paired_scan_runs':6,'seeds':[11,23,37],
        'power_reference':float(reference),'display_floor_db':-60,'display_ceiling_db':display_top,
        'difference_display_limits_db':[-30,30],'metrics_use_unclipped_differences':True,
        'db_power_floor_applied_to_change_metric':True,'integrated_power_metric_uses_unfloored_linear_power':True,
        'summaries':summaries,'scan':scan_rows,
        'measured_radar_fidelity_claimed':False,
        'metric_meaning':'absolute dB change and integrated power change inside the saved RD window between controlled simulations, not measured-radar error or total scene/radiated power',
        'inputs':'real Copernicus DSM and WorldCover classifications; NRCS values and ideal radiation models are declared assumptions',
        'sampling_scope':'three geographic patches; seeds are not extra real scenes; P90 of three seeds is descriptive only'}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    images=''.join('<img style="width:100%" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">' for p in pictures)
    table=''.join(f'<tr><td>{r["site"]}</td><td>{r["variant"]}</td><td>{r["mean_abs_db_change_mean"]:.3f}</td><td>{r["mean_integrated_power_change_db"]:.3f}</td></tr>' for r in summaries)
    page='<html lang="zh"><meta charset="utf-8"><title>真实地形第二批功能对照</title><body style="max-width:1500px;margin:30px auto;font:16px sans-serif"><h1>真实地形上的第二批功能对照</h1><p>3 个固定真实地形窗口，45 个组合运行及 6 个同一时刻的扫描对照全部保留。地形为 Copernicus DSM，类别来自 WorldCover 2021。NRCS 统一 −20 dB 基线及单类 ±6 dB 变化是明确假设，用于敏感性检查，不能宣称是真实材料反射。</p><p>前三类对照依次改变发射权重、电子指向或某一类别的反射参数，其他输入及相位种子保持一致。RD 图先在线性功率上平均三个种子，再转 dB；最后的扫描图为固定 seed 11，在同一个 20 ms 时刻比较，不把平台运动误当成扫描效果。</p><p>纵轴为单 PRF 表观速度，实际平台速度仍为 1200 m/s。没有实测 RD，也没有实测 MAE。下面的平均绝对 dB 变化只表示改动影响大小，不是保真误差，越小不一定越好。</p><table border="1" cellpadding="7"><tr><th>区域</th><th>对照</th><th>相对 Hamming0 的平均绝对 dB 变化</th><th>总功率变化 dB</th></tr>'+table+'</table>'+images+'<p>完整逐样本指标见 per_case_changes.csv。混合地物单元按主类别建模；原始分类占比仍保留。数据年份不完全一致、绝对功率未标定、有限地形以外的旁瓣杂波未建模，不能由这些对照推出厘米波实测保真。</p><p>produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved</p><p>© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium</p></body></html>'
    (a.output/'report.html').write_text(page);print(json.dumps({'main_runs':45,'scan_runs':6,'report':str(a.output/'report.html')}),flush=True)


if __name__=='__main__':main()
