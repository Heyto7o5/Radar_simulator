#!/usr/bin/env python3
"""Prespecified IPIX family ablation and Maqu land-coefficient transfer audit.

All available eligible prior IPIX records, fixed prefix/gap/tail split. No
evaluation fitting, per-image normalization or claims of independent frames.
"""
import argparse
import base64
import csv
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from radar_platform.statistical_background import BackgroundRecipe,SpectrumComponent,fit_background,simulate_background
from radar_platform.distributed_clutter import ClutterRecipe,simulate_clutter_iq,rd_blocks
from radar_platform.ipix_dataset import load_ipix,IQCorrection
from radar_platform.comparison import compare_power
from radar_platform.ground_scattering import load_maqu_sigma0,EmpiricalBackscatterCurve
from render_scene import configure_font

P=Path(__file__).resolve().parents[1]
ROOT=Path('/nas/datasets/hyt/datasets/radar_open')
OLD=ROOT/'alignment_campaign/20261005/ipix_v1'
LABELS={'white_gaussian':'独立高斯','correlated_gaussian':'相关高斯','compound_gamma':'相关场 + 起伏 + 独立底噪','legacy':'旧海杂波模型'}
MODELS=list(LABELS)

def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def averaged_rd(iq,prf):
    rd,f=rd_blocks(iq,prf,512)
    return np.mean(abs(rd.astype(np.complex128).reshape(-1,16,512,iq.shape[1]))**2,axis=1),f

def statistics(x,power,mask):
    y=x[:,mask]/np.sqrt(np.asarray(power)[mask])[None,:]
    intensity=abs(y.astype(complex))**2
    med=float(np.quantile(intensity,.5));tail=float(np.quantile(intensity,.999))
    centered=intensity-intensity.mean(0)
    variance=max(float(np.mean(centered**2)),1e-30)
    ilags=np.array([10,100,500,1000])
    ia=np.array([np.mean(centered[:-k]*centered[k:])/variance for k in ilags])
    clags=np.array([1,5,10,20])
    ca=np.array([abs(np.mean(y[:-k]*y[k:].conj()))/np.sqrt(np.mean(abs(y[:-k])**2)*np.mean(abs(y[k:])**2)) for k in clags])
    pairs=np.flatnonzero(mask[:-1]&mask[1:])
    range_corr=[]
    for k in pairs:
        a,b=x[:,k],x[:,k+1]
        range_corr.append(abs(np.mean(a*b.conj()))/np.sqrt(np.mean(abs(a)**2)*np.mean(abs(b)**2)))
    return {'tail_ratio':tail/max(med,1e-30),'intensity_acf':ia,'complex_acf':ca,
            'adjacent_range_corr':float(np.mean(range_corr)) if range_corr else 0.,
            'mean_power':float(np.mean(abs(x[:,mask])**2))}

def compare_stats(real,pred):
    return {'power_bias_db':float(10*np.log10(pred['mean_power']/real['mean_power'])),
        'tail_ratio_error_db':float(abs(10*np.log10(pred['tail_ratio']/real['tail_ratio']))),
        'intensity_acf_mae':float(np.mean(abs(pred['intensity_acf']-real['intensity_acf']))),
        'complex_acf_mae':float(np.mean(abs(pred['complex_acf']-real['complex_acf']))),
        'range_correlation_error':float(abs(pred['adjacent_range_corr']-real['adjacent_range_corr']))}

def ground(output,images):
    folder=ROOT/'maqu_scatterometer/raw/original/Scatterometer_Angular_Vars'
    train=folder/'sigma0_20170825';validation=folder/'sigma0_20180819'
    rows=[];excluded=[];fig,axes=plt.subplots(1,4,figsize=(14,3.7),layout='constrained')
    bands=['1500-1750','2500-3000','4500-5000','9000-9875']
    sources={}
    for file in sorted(train.glob('*_alpha0_azip00_*.dat')):
        other=validation/file.name
        if not other.exists():excluded.append({'file':str(other),'reason':'missing validation file'});continue
        obs=load_maqu_sigma0(file);test=load_maqu_sigma0(other);sources[str(file)]=sha(file);sources[str(other)]=sha(other)
        for mode in ['interpolate','cosine_power']:
            reason=None
            try:curve=EmpiricalBackscatterCurve(tuple(obs),mode)
            except ValueError as exc:curve=None;reason=str(exc)
            for o in test:
                p=None;status='ok'
                try:
                    if curve is None:raise ValueError(reason)
                    p=float(curve.predict_db((o.frequency_min_hz+o.frequency_max_hz)/2,o.incidence_center_deg,o.polarization))
                except ValueError as exc:
                    status='unsupported';excluded.append({'file':str(other),'mode':mode,'incidence_deg':o.incidence_center_deg,'reason':str(exc)})
                baseline=float(np.mean([x.sigma0_db for x in obs]))
                rows.append({'file':file.name,'mode':mode,'polarization':o.polarization,
                    'frequency_min_hz':o.frequency_min_hz,'frequency_max_hz':o.frequency_max_hz,
                    'incidence_deg':o.incidence_center_deg,'measured_sigma0_db':o.sigma0_db,'predicted_sigma0_db':p,
                    'absolute_error_db':abs(p-o.sigma0_db) if p is not None else None,
                    'constant_training_mean_error_db':abs(baseline-o.sigma0_db),'status':status})
        if obs[0].polarization=='vv':
            band=file.stem.rsplit('_',1)[-1];ax=axes[bands.index(band)]
            ax.plot([o.incidence_center_deg for o in obs],[o.sigma0_db for o in obs],'o-',label='2017-08-25 输入')
            ax.errorbar([o.incidence_center_deg for o in test],[o.sigma0_db for o in test],
                yerr=[[-o.lower_uncertainty_db for o in test],[o.upper_uncertainty_db for o in test]],fmt='s',label='2018-08-19 实测')
            ax.set(xlabel='足迹平均入射角 / °',ylabel='后向散射系数 / dB',title=band+' MHz');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('玛曲草地：实测系数形成有边界的经验模型；跨日期变化不能靠一个角度公式解释\n不是动态 RD，也没有把草地当作金属板；误差棒为来源提供的测量区间',fontsize=11)
    fig.savefig(images/'ground_transfer.png',dpi=160);plt.close(fig)
    csvwrite(output/'ground_transfer.csv',rows)
    result={'training_date':'20170825','evaluation_date':'20180819','independent_dates':2,
        'excluded_date':'20180629: source README notes cable interchange during this period; not treated as clean transfer truth without correction audit',
        'source_sha256':sources,'exclusions':excluded,'groups':[]}
    for mode in ['interpolate','cosine_power']:
        all_rows=[r for r in rows if r['mode']==mode];rr=[r for r in all_rows if r['status']=='ok']
        result['groups'].append({'mode':mode,'comparisons':len(rr),'total_comparisons':len(all_rows),'unsupported':len(all_rows)-len(rr),
            'mean_sigma0_mae_db':float(np.mean([r['absolute_error_db'] for r in rr])),
            'constant_training_mean_mae_db':float(np.mean([r['constant_training_mean_error_db'] for r in rr]))})
    write(output/'ground_summary.json',result)
    return result

def plot_pair(path,m,s,r,f,ref,mask,title):
    comparison=compare_power(m,s,reference_power=ref,valid_mask=mask)
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),layout='constrained')
    for i,(ax,a,label) in enumerate(zip(axes,[comparison.measured_db,comparison.simulated_db,comparison.difference_db],['实测平均 RD','仿真平均 RD','差值：仿真 − 实测'])):
        im=ax.pcolormesh(r,f,np.where(mask,a,np.nan),shading='auto',cmap='RdBu_r' if i==2 else 'turbo',
            vmin=-25 if i==2 else -60,vmax=25 if i==2 else 10,rasterized=True)
        ax.set(xlabel='距离门近边缘 / m',ylabel='Doppler / Hz',ylim=(-150,150),title=label)
        fig.colorbar(im,ax=ax,label='dB',shrink=.8,extend='both')
    fig.suptitle(title+f" · MAE {comparison.metrics['mae_db']:.2f} dB\n每幅先在线性功率平均 16 CPI；白色为排除靶区，不计分",fontsize=11)
    fig.savefig(path,dpi=155);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'background_campaign/20261015_v1')
    parser.add_argument('--report',type=Path,default=P/'docs/background_extension/2026-10-15')
    parser.add_argument('--finish-existing',action='store_true',help='finish report from completed saved IPIX scores, without rerunning or refitting')
    args=parser.parse_args();out=args.output.resolve();report=args.report.resolve()
    out.mkdir(parents=True,exist_ok=args.finish_existing)
    for path in [report/'images',report/'metrics',report/'configs']:path.mkdir(parents=True,exist_ok=True)
    configure_font()
    specs=[load(p) for p in sorted(OLD.glob('record_*_parameters.json'),key=lambda p:int(p.name.split('_')[1]))]
    ids=[s['record_id'] for s in specs];representative=[ids[int(round((len(ids)-1)*q))] for q in [.25,.5,.75]]
    protocol={'records':ids,'models':MODELS,'seeds':[42,43,44],'calibration':[0,32768],'unused_gap':[32768,65536],
        'evaluation':[65536,131072],'cpi_pulses':512,'cpis_per_average':16,
        'statistical_unit':'one acquisition record, not individual frames or random seeds',
        'comparison':'conditional prefix-to-tail prediction; NOT zero-shot or a new untouched blind set',
        'calibration_frozen_before_any_evaluation':True,'representative_records_before_scoring':representative,
        'worst_example':'extra highest compound_gamma MAE record after scoring',
        'same_IQ_correction_and_masks_as_prior_run':True,'target_gates_excluded':True,
        'source_sha256':{str(p.relative_to(P)):sha(p) for p in [Path(__file__),P/'backend/radar_platform/statistical_background.py',
            P/'backend/radar_platform/distributed_clutter.py',P/'backend/radar_platform/ground_scattering.py']}}
    rows=[];records=[];excluded=[];frozen=[]
    if args.finish_existing:
        protocol=load(out/'protocol.json')
        completed=load(out/'fitting_complete.json');excluded=completed['excluded']
        frozen=[(s,None,None) for s in specs if s['record_id'] in completed['records']]
        for filename,dest in [('window_metrics.csv',rows),('record_metrics.csv',records)]:
            with (out/filename).open(encoding='utf-8-sig') as h:
                for row in csv.DictReader(h):
                    dest.append({k:(v if k in ('role','model') else int(v) if k in ('record_id','seed','window','windows','seeds') else float(v)) for k,v in row.items()})
        write(out/'report_completion.json',{'evaluation_not_rerun':True,'report_source_sha256':sha(Path(__file__)),
              'original_evaluation_source_sha256':sha(out/'evaluation_source.py')})
    else:
        write(out/'protocol.json',protocol)
        import shutil
        shutil.copy2(Path(__file__),out/'evaluation_source.py')
    # Phase 1: fit every calibration prefix before any validation score is read.
    for spec in ([] if args.finish_existing else specs):
        try:
            file=Path(spec['file']);assert sha(file)==spec['source_sha256']
            record=load_ipix(file,'vv');assert len(record.iq)==131072 and record.prf_per_polarization_hz==1000
            mask=np.zeros(record.iq.shape[1],bool);mask[np.array(spec['scoring_gates'])-1]=True
            correction=IQCorrection(**spec['iq_correction']);prefix=correction.apply(record.iq[:32768])
            recipe,fitmeta=fit_background(prefix,record.range_near_edge_m,1000.,mask)
            info={'record_id':spec['record_id'],'source_sha256':spec['source_sha256'],'model':asdict(recipe),'fit_metadata':fitmeta,
                  'IQ_correction':spec['iq_correction'],'reference_power':spec['reference_power'],'scoring_gates':spec['scoring_gates']}
            write(out/f'record_{spec["record_id"]}_frozen.json',info)
            frozen.append((spec,recipe,mask))
            print('frozen',spec['record_id'],flush=True)
        except Exception as exc:excluded.append({'record_id':spec['record_id'],'reason':str(exc)})
    if not frozen:raise RuntimeError('no eligible records')
    if not args.finish_existing:write(out/'fitting_complete.json',{'records':[s['record_id'] for s,r,m in frozen],'excluded':excluded})
    # Phase 2: evaluate tails and retain every result, including regressions.
    for spec,recipe,mask in ([] if args.finish_existing else frozen):
        record=load_ipix(spec['file'],'vv');correction=IQCorrection(**spec['iq_correction'])
        observed=correction.apply(record.iq[65536:]);real,f=averaged_rd(observed,1000.)
        real_stats=statistics(observed,recipe.mean_power,mask)
        valid=np.broadcast_to(mask,real.shape[1:]);ref=spec['reference_power'];arrays={'measured':real,'doppler_hz':f,'range_m':record.range_near_edge_m,'valid_mask':valid,'reference_power':ref}
        for family in MODELS:
            powers=[];stat_rows=[]
            for seed in protocol['seeds']:
                sim=simulate_clutter_iq(ClutterRecipe(**spec['conditional_model']),len(observed),seed) if family=='legacy' else simulate_background(replace(recipe,family=family),len(observed),seed)
                predicted,_=averaged_rd(sim,1000.);powers.append(predicted)
                statistics_delta=compare_stats(real_stats,statistics(sim,recipe.mean_power,mask));stat_rows.append(statistics_delta)
                for wi,(m,s) in enumerate(zip(real,predicted)):
                    score=compare_power(m,s,reference_power=ref,valid_mask=valid)
                    rows.append({'record_id':spec['record_id'],'role':spec['role'],'model':family,'seed':seed,'window':wi,
                                 'mae_db':score.metrics['mae_db'],**statistics_delta})
            average=np.mean(powers,axis=0);arrays[family]=average
            # Match stochastic averaging fairly: each seed has same 16-CPI windows as measured.
            # Extra multi-seed average below is only an illustration, not the main MAE score.
            group=[r for r in rows if r['record_id']==spec['record_id'] and r['model']==family]
            entry={'record_id':spec['record_id'],'role':spec['role'],'model':family,'windows':len(real),'seeds':3,
                'mean_rd_mae_db':float(np.mean([r['mae_db'] for r in group])),
                **{k:float(np.mean([r[k] for r in stat_rows])) for k in stat_rows[0]}}
            records.append(entry)
        np.savez_compressed(out/f'record_{spec["record_id"]}_averages.npz',**arrays)
        if spec['record_id'] in representative:
            for model in MODELS:
                plot_pair(report/'images'/f'ipix_{spec["record_id"]}_{model}.png',real[0],arrays[model][0],record.range_near_edge_m,f,ref,valid,
                          f'#{spec["record_id"]} · {LABELS[model]}（仿真图另平均 3 个种子，仅展示）')
        csvwrite(out/'window_metrics.partial.csv',rows)
        print('evaluated',spec['record_id'],flush=True)
    csvwrite(out/'window_metrics.csv',rows);csvwrite(out/'record_metrics.csv',records)
    summaries=[]
    for model in MODELS:
        rr=[r for r in records if r['model']==model]
        entry={'model':model,'records':len(rr)}
        for key in ['mean_rd_mae_db','tail_ratio_error_db','intensity_acf_mae','complex_acf_mae','range_correlation_error']:
            values=[r[key] for r in rr];entry[key]={'mean':float(np.mean(values)),'median':float(np.median(values)),'p90':float(np.percentile(values,90))}
        summaries.append(entry)
    write(out/'summary.json',{'protocol':protocol,'models':summaries,'excluded':excluded,'record_count':len(frozen)})
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,key,title in zip(axes,['mean_rd_mae_db','tail_ratio_error_db','intensity_acf_mae'],
        ['平均 RD 全图 MAE / dB','偶发强回波相对强度差 / dB','强弱起伏的时间相关差（0–1尺度）']):
        for model in MODELS:
            rr=[r for r in records if r['model']==model]
            ax.plot([str(r['record_id']) for r in rr],[r[key] for r in rr],'o-',label=LABELS[model],ms=3)
        ax.set(title=title,xlabel='独立采集记录号');ax.tick_params(axis='x',rotation=55);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('所有合格记录均保留：复杂模型不保证每项都更好\n参数仅来自前段；每个点先平均 3 个随机种子，不把种子当独立实测',fontsize=12)
    fig.savefig(report/'images/background_metrics.png',dpi=160);plt.close(fig)
    ground_result=ground(out,report/'images')
    # Keep a worst case in addition to the fixed representative records.
    worst=max([r for r in records if r['model']=='compound_gamma'],key=lambda r:r['mean_rd_mae_db'])
    # Render one seed and exactly the same observation duration on both sides.
    # Preserve complex IQ/RD for these examples, separate from ensemble displays.
    for rid in sorted(set([*representative,worst['record_id']])):
        saved=load(out/f'record_{rid}_frozen.json');d=saved['model']
        d['components']=tuple(SpectrumComponent(**x) for x in d['components'])
        recipe=BackgroundRecipe(**d)
        spec=next(s for s in specs if s['record_id']==rid)
        with np.load(out/f'record_{rid}_averages.npz') as z:
            for family in MODELS:
                target=out/f'example_{rid}_{family}_seed42.npz'
                if not target.exists():
                    iq=simulate_clutter_iq(ClutterRecipe(**spec['conditional_model']),65536,42) if family=='legacy' else simulate_background(replace(recipe,family=family),65536,42)
                    rd,f=rd_blocks(iq,1000.)
                    mean=np.mean(abs(rd[:16].astype(np.complex128))**2,axis=0)
                    np.savez_compressed(target,iq=iq,rd=rd,average_first_window=mean,range_m=recipe.range_m,doppler_hz=f)
                with np.load(target) as example:mean=example['average_first_window']
                if rid in representative:
                    plot_pair(report/'images'/f'ipix_{rid}_{family}.png',z['measured'][0],mean,z['range_m'],z['doppler_hz'],float(z['reference_power']),z['valid_mask'],
                        f'#{rid} · {LABELS[family]} · 实测与仿真均为 16 CPI，固定种子 42')
                if rid==worst['record_id'] and family=='compound_gamma':
                    plot_pair(report/'images/ipix_worst.png',z['measured'][0],mean,z['range_m'],z['doppler_hz'],float(z['reference_power']),z['valid_mask'],
                        f'最差复合模型记录 #{rid}；展示首窗，排名按全记录；均为 16 CPI')
    import shutil
    for path in out.glob('*.json'):shutil.copy2(path,report/'metrics'/path.name)
    for name in ['record_metrics.csv','window_metrics.csv','ground_transfer.csv']:shutil.copy2(out/name,report/'metrics'/name)
    selected_model=load(out/f'record_{representative[0]}_frozen.json')['model']
    for family in MODELS[:-1]:
        model=dict(selected_model,family=family)
        write(report/'configs'/f'background_{family}.json',{'model':model,'samples':65536,'seed':42,'pulses_per_cpi':512})
    write(report/'configs/usage.json',{'POST_endpoint':'/api/v1/background-simulations',
        'input_files':'background_*.json','calibration_record_id':representative[0],
        'domain':'range_gated_iq, not raw ADC','front_end_started':False})
    shutil.copy2(P/'configs/scattering_data_catalog.json',report/'metrics/scattering_data_catalog.json')
    if (ROOT/'mrr_ace/manifest.json').exists():shutil.copy2(ROOT/'mrr_ace/manifest.json',report/'metrics/mrr_source_manifest.json')
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((report/'images'/name).read_bytes()).decode()+'">'
    def table(h,rr):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rr)+'</table>'
    body='<h1>背景模型与草地散射：第二阶段</h1><p>2026-10-15 · 通用环境统计模型；没有把合成背景伪装成真实数据。</p>'
    body+=table(['模型','新增／保留内容','验证边界'],[
        ['独立高斯','固定功率的独立复高斯样本','只是基线，不代表海面'],['相关高斯','少量参数描述谱形 + 距离门相关','参数从前段估计，不是真实海浪几何'],
        ['复合背景','相关场乘平稳 Gamma 起伏，再加独立高斯底噪','统计模型；噪声底不能随纹理一起放大'],['旧海杂波','保留原实现作为对照','不靠删除旧方法制造提升'],
        ['草地经验系数','实测角度插值 / 余弦幂经验曲线','不外推频段、极化、地类或入射角，不是草地动态 RD']])
    body+=f'<h2>IPIX：{len(frozen)} 条记录，先校准前段、再评价后段</h2>'+image('background_metrics.png')
    body+=table(['模型','记录平均 RD MAE','强亮点强度差','起伏时间相关差','相邻距离门相关差'],[
        [LABELS[s['model']],f"{s['mean_rd_mae_db']['mean']:.2f} dB",f"{s['tail_ratio_error_db']['mean']:.2f} dB",f"{s['intensity_acf_mae']['mean']:.3f}",f"{s['range_correlation_error']['mean']:.3f}"] for s in summaries])
    body+='<p><b>存在明确取舍：</b>复合起伏更接近强亮点和时间变化，但相邻距离门的相关性比相关高斯模型更差；当前各距离门纹理独立，不能把它写成所有指标全面提高。后续需要有依据的空间共同起伏模型。</p>'
    body+='<p>MAE 是同参考 dB 图逐像素绝对差。强亮点指标比较强度第 99.9 百分位相对中位数的比值；它补充“亮点出现得有多强”。时间相关差比较 10、100、500、1000 ms 间隔的强弱相关性；它补充“背景怎样随时间变化”。均越小越好，不能只挑最好的一项。</p>'
    body+='<p>前 32768 个采样用于参数估计，中间 32768 个不用，后 65536 个评价。沿用已有记录级分组，不是首次盲测，也不是无校准跨海况迁移。每种模型主指标使用与实测相同的 16 CPI 平均窗口，分别评分 3 个种子后汇总。</p>'
    for rid in representative:
        body+=f'<details><summary>固定样例 #{rid}：四种模型的实测—仿真—差值</summary>'+''.join(image(f'ipix_{rid}_{m}.png') for m in MODELS)+'</details>'
    body+=image('ipix_worst.png')
    body+='<h2>玛曲草地：新增的是有来源、有适用边界的经验系数模型</h2>'+image('ground_transfer.png')
    body+=table(['经验模型','有效 / 全部比较数','不支持数','散射系数 MAE','固定训练均值基线 MAE'],[
        [r['mode'],f"{r['comparisons']} / {r['total_comparisons']}",r['unsupported'],f"{r['mean_sigma0_mae_db']:.2f} dB",f"{r['constant_training_mean_mae_db']:.2f} dB"] for r in ground_result['groups']])
    body+='<p>L 波段交叉极化记录存在“相同平均入射角、不同足迹范围”的条目。当前只用平均角的曲线不能唯一表达它们，明确记为不支持并保留在 CSV，不合并成虚假真值，也不把它们算作通过。MAE 只针对可生成预测的子项，覆盖率另列。</p>'
    body+='<p>训练使用 2017-08-25，评价使用 2018-08-19；频带、极化、角度子项不是独立日期。原说明提示 2018 年 6–7 月曾有电缆交换，本阶段不把 2018-06-29 当作未经复核的干净验证数据。没有输入湿度、植被和粗糙度，因而不能解释跨日期全部差异。</p>'
    if (report/'metrics/mrr_ingestion.json').exists():
        mrr=load(report/'metrics/mrr_ingestion.json')
        body+='<h2>新增实际取得的数据：MRR-2 降水雷达谱</h2>'+image('mrr_real_examples.png')
        body+=table(['文件','实测数组维度','高度范围','当前状态'],[
            [Path(r['source_file']).name,str(r['shape']),str(r['height_range_m'])+' m','已读取真实谱；尚未完成物理模型验证'] for r in mrr['files']])
        body+='<p>来源为 <a href="https://zenodo.org/records/3929289">ACE 航次公开数据</a>，CC BY 4.0。只按压缩包目录预选三个小文件，没有下载整套大包。原始功率为工程单位，缺测保留为 NaN；不同文件的高度配置并不完全相同。已保存速度格点间距，但没有猜测零点和正负方向，因此先显示原生 Doppler bin。当前选中时刻的降水类别尚未与事件表逐条关联，不能把每张图都叫作降雨真值。</p>'
    body+='<h2>新增接口与尚未完成的部分</h2><p>新增 /api/v1/background-simulations：输入冻结模型参数，输出复数距离门 I/Q 和 RD；不要求输入真实波形。输出不是原始 ADC，不能直接加到原始 ADC 上。原有默认理想仿真不变。</p><p>尚未完成：把有材料来源的面散射、统计环境与接收机噪声在统一链路正确组合；更多独立数据接入；面目标动态实测验证。不能以本阶段统计改善宣称完整场景高保真。</p>'
    body+='<p>来源：<a href="https://soma.ece.mcmaster.ca/ipix/index.html">IPIX</a> · <a href="https://essd.copernicus.org/articles/13/2819/2021/">玛曲</a>。复数缓存位于 NAS；完整指标、冻结参数和来源校验随报告保存。</p>'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>背景与地表散射验证</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.65 system-ui;color:#24364a}img{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd5df;overflow-wrap:anywhere}th{background:#eaf2f6}h2{margin-top:35px}summary{cursor:pointer;background:#edf3f7;padding:12px}</style>'+body+'</html>'
    (report/'背景与地表散射验证.html').write_text(doc)
    with zipfile.ZipFile(report/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),P/'backend/radar_platform/statistical_background.py',P/'backend/radar_platform/background_api.py',P/'backend/radar_platform/ground_scattering.py',P/'backend/radar_platform/mrr_dataset.py',P/'tests/test_statistical_background.py',P/'tests/test_empirical_ground.py',P/'tests/test_mrr_dataset.py',P/'scripts/prepare_mrr_ace_subset.py',P/'scripts/inspect_mrr_subset.py',P/'configs/scattering_data_catalog.json']:
            z.write(f,f.relative_to(P))
    with zipfile.ZipFile(report.parent/'背景与地表散射_2026-10-15.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in report.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(report))
    print(json.dumps({'models':summaries,'ground':ground_result['groups']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
