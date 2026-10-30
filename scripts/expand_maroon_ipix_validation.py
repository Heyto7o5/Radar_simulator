#!/usr/bin/env python3
"""Frozen-model, multi-channel/frame/record validation; no evaluation-time fit.

MAROON has ONE static capture, not new temporal frames. IPIX random seeds are
simulation repetitions, not independent real scenes. Plots use fixed selections.
"""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from radar_platform.distributed_clutter import ClutterRecipe, simulate_clutter_iq, rd_blocks
from radar_platform.ipix_dataset import IQCorrection, load_ipix
from radar_platform.maroon_dataset import load_measurement
from radar_platform.models import C0
from radar_platform.repeat_metrics import common_safe_gates, distribution, peak_width
from radar_platform.surface_echo import range_response
from radar_platform.validation_metrics import complex_metrics
from validate_ipix_clutter import describe, scores

DATA = Path('/nas/datasets/hyt/datasets/radar_open')
PROJECT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style():
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10})


def validate_maroon(output, forward):
    root = DATA/'maroon_corner'
    old = root/'processed/validation_20260929_final'
    frozen = json.loads((old/'comparison.json').read_text())
    measured, frequencies, valid = load_measurement(root)
    with np.load(forward/'forward.npz') as z:
        signal = z['signal']
        assert np.array_equal(z['frequencies_hz'], frequencies)
    g = frozen['fitted_gains']['mesh']
    prediction = (complex(g['real'], g['imag'])*signal).astype(np.complex64)
    # Check that the re-run reproduces the already-frozen model, not a new fit.
    with np.load(old/'comparison.npz') as z:
        identical = np.array_equal(signal, z['simulated_spectrum'])
        assert identical
    calibration = valid & (np.arange(94)[:, None] % 5 == 0)
    evaluation = valid & ~calibration
    real_response, axis = range_response(measured, frequencies, 512)
    sim_response, _ = range_response(prediction, frequencies, 512)
    real_power, sim_power = abs(real_response)**2, abs(sim_response)**2
    roi = (axis >= .2) & (axis <= .6)
    resolution = C0/(2*(frequencies[-1]-frequencies[0]))

    def profile_stats(real, sim):
        a, b = peak_width(axis[roi], real[roi]), peak_width(axis[roi], sim[roi])
        outside = abs(axis-a['peak']) > 2*resolution
        rf = float(real[outside].sum()/real.sum())
        sf = float(sim[outside].sum()/sim.sum()) if sim.sum() > 0 else None
        return {
            'real_peak_m': a['peak'], 'sim_peak_m': b['peak'],
            'peak_error_m': abs(a['peak']-b['peak']) if b['peak'] is not None else None,
            'real_3db_width_m': a['width'], 'sim_3db_width_m': b['width'],
            'width_error_m': abs(a['width']-b['width']) if a['width'] is not None and b['width'] is not None else None,
            'real_offpeak_fraction': rf, 'sim_offpeak_fraction': sf,
            'offpeak_fraction_abs_error': abs(sf-rf) if sf is not None else None,
        }

    rows = []
    for tx, rx in zip(*np.where(evaluation)):
        m, s = measured[tx, rx].astype(complex), prediction[tx, rx].astype(complex)
        has_path = bool(np.any(abs(signal[tx, rx]) > 0))
        row = {'tx_zero_based': int(tx), 'rx_zero_based': int(rx), 'predicted_nonzero': has_path,
               'complex_nmse': float(np.sum(abs(s-m)**2)/np.sum(abs(m)**2)),
               'complex_coherence': float(abs(np.vdot(s, m))/np.sqrt(np.vdot(s,s).real*np.vdot(m,m).real)) if has_path else None,
               **profile_stats(real_power[tx,rx], sim_power[tx,rx])}
        rows.append(row)
    table(output/'maroon_channels.csv', rows)
    groups = []
    tx_groups = np.array_split(np.flatnonzero(np.arange(94) % 5 != 0), 6)
    reference = float(np.mean(real_power[calibration], axis=0).max())
    db = lambda p: 10*np.log10(np.maximum(p, 1e-30)/reference)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for i, (ax, txs) in enumerate(zip(axes.flat, tx_groups)):
        mask = evaluation & np.isin(np.arange(94), txs)[:,None]
        real, sim = real_power[mask].mean(axis=0), sim_power[mask].mean(axis=0)
        count = int(mask.sum())
        covered = int((mask & np.any(abs(signal)>0, axis=-1)).sum())
        row = {'group': i+1, 'tx_start_zero_based': int(txs[0]), 'tx_end_zero_based': int(txs[-1]),
               'channels': count, 'predicted_nonzero': covered, 'coverage': covered/count,
               **complex_metrics(prediction, measured, mask), **profile_stats(real, sim)}
        groups.append(row)
        ax.plot(axis, db(real), color='black', label='实测')
        ax.plot(axis, db(sim), color='#d87526', label='平台三次反射')
        peak_text = f"{row['peak_error_m']*1000:.1f}" if row['peak_error_m'] is not None else '不可用'
        ax.set(xlim=(.22,.57), ylim=(-55,8), xlabel='半双程等效距离 / m', ylabel='共用校准参考 / dB',
               title=f"组 {i+1}：Tx {txs[0]}–{txs[-1]}，{count} 对通道\n主峰差 {peak_text} mm；有回波 {covered/count:.1%}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('MAROON：按 Tx 顺序固定分为六组，全部留出通道计入\n同一静态场景的空间检查，不是六个独立场景；不逐图重新归一化', fontsize=14)
    fig.savefig(output/'maroon_groups.png', dpi=170)
    plt.close(fig)
    table(output/'maroon_groups.csv', groups)
    summary = {
        'capture_count': 1, 'evaluation_channels': len(rows),
        'predicted_nonzero_channels': sum(r['predicted_nonzero'] for r in rows),
        'zero_prediction_channels': sum(not r['predicted_nonzero'] for r in rows),
        'fixed_tx_group_count': 6, 'rerun_bitwise_equal': identical,
        'global_metrics': complex_metrics(prediction, measured, evaluation),
        'per_channel': {key: distribution([r[key] for r in rows]) for key in
                        ['peak_error_m','width_error_m','offpeak_fraction_abs_error','complex_nmse','complex_coherence']},
        'groups': groups,
        'offpeak_definition': 'fraction of full unambiguous range power outside measured main peak +/- 2*c/(2*B); includes secondary echoes/background, NOT pure sidelobe power or PSLR',
        'conditional_metrics': 'peak, width and offpeak errors exclude zero predictions; NMSE includes all 6605 valid evaluation channels',
        'calibration': 'reuse original single global complex gain, Tx%5==0; no new fitting',
        'resolution_m': resolution, 'interpolated_width_is_not_resolution_improvement': True,
        'source_sha256': {'signal.npz':sha(root/'raw/signal.npz'), 'frozen_comparison.json':sha(old/'comparison.json'),
                          'new_forward.npz':sha(forward/'forward.npz')},
    }
    dump(output/'maroon_summary.json', summary)
    return summary


def spectral_frame_metrics(real_rd, sim_rd, mask, frequencies):
    real = abs(real_rd[:,:,mask])**2
    sim = abs(sim_rd[:,:,mask])**2
    rp, sp = real.mean(axis=2), sim.mean(axis=2)
    rsum, ssum = rp.sum(axis=1), sp.sum(axis=1)
    rp, sp = rp/rsum[:,None], sp/ssum[:,None]
    rc, sc = rp-rp.mean(axis=1,keepdims=True), sp-sp.mean(axis=1,keepdims=True)
    corr = (rc*sc).sum(axis=1)/np.sqrt((rc*rc).sum(axis=1)*(sc*sc).sum(axis=1))
    w1 = abs(np.cumsum(rp-sp,axis=1)).sum(axis=1)*(frequencies[1]-frequencies[0])
    return corr, w1, 10*np.log10(ssum/rsum)


def validate_ipix(output):
    root = DATA/'ipix_dartmouth'
    recipe_path = root/'processed/validation_20260930_final/recipe.json'
    frozen = json.loads(recipe_path.read_text())
    recipe = ClutterRecipe(**frozen['model'])
    correction = IQCorrection(**frozen['iq_correction'])
    specs = [
        (17, '19931107_135603_starea.cdf', 65536, [8,9,10,11]),
        (18, '19931107_141630_starea.cdf', 0, [8,9,10,11]),
        (19, '19931107_145028_starea.cdf', 0, [7,8,9]),
        (25, '19931108_213827_starea.cdf', 0, [6,7,8]),
    ]
    summaries, window_rows, frame_rows, global_rows = [], [], [], []
    cache = {}
    fig_records, axs = plt.subplots(4,3,figsize=(15,14),layout='constrained')
    for record_id, filename, start, excluded in specs:
        record = load_ipix(root/'raw'/filename, 'vv')
        if not np.isclose(record.prf_per_polarization_hz, recipe.prf_hz) or not np.isclose(record.pulse_width_s, recipe.pulse_width_s, rtol=1e-6, atol=0):
            raise ValueError('new acquisition timing needs a separate model configuration')
        same_range = np.allclose(record.range_near_edge_m, recipe.range_near_edge_m, atol=.002, rtol=0)
        # Geometry metadata may change, but there is no calibrated range-power law.
        # For #19 only test normalized spectral/shape transfer, not absolute power.
        config = recipe if same_range else replace(recipe,
            range_near_edge_m=tuple(record.range_near_edge_m), mean_power=tuple(np.ones(14)))
        iq = correction.apply(record.iq[start:])
        samples = len(iq)
        mask = common_safe_gates(14, frozen['excluded_target_bins_one_based'], excluded)
        cache_key = (samples, same_range)
        if cache_key not in cache:
            cache[cache_key] = [simulate_clutter_iq(config, samples, seed) for seed in frozen['seeds']]
        generated = cache[cache_key]
        ref = describe(iq, config.prf_hz, mask, np.asarray(config.mean_power))
        frame_count = len(ref['rd'])
        window_count = samples//8192  # 16 consecutive CPIs; no temporal overlap.
        records_windows = []
        full_scores = []
        for seed, sim in zip(frozen['seeds'], generated):
            pred = describe(sim, config.prf_hz, mask, np.asarray(config.mean_power))
            metrics = scores(ref, pred)
            if not same_range:
                for key in ('mean_power_error_db', 'intensity_w1_cal_units', 'q99_intensity_error_cal_units'):
                    metrics[key] = None
            full_scores.append(metrics)
            global_rows.append({'record_id':record_id, 'seed':seed, **metrics})
            corr, w1, power = spectral_frame_metrics(ref['rd'],pred['rd'],mask,ref['f'])
            for index in range(frame_count):
                frame_rows.append({'record_id':record_id,'seed':seed,'cpi_zero_based':index,
                                   'absolute_start_s':(start+512*index)/config.prf_hz,
                                   'psd_correlation':float(corr[index]),'doppler_w1_hz':float(w1[index]),
                                   'mean_windowed_power_error_db':float(power[index]) if same_range else None})
            for index in range(window_count):
                sl = slice(index*8192,(index+1)*8192)
                real_window = describe(iq[sl], config.prf_hz,mask,np.asarray(config.mean_power))
                sim_window = describe(sim[sl],config.prf_hz,mask,np.asarray(config.mean_power))
                metric = scores(real_window,sim_window)
                if not same_range:
                    for key in ('mean_power_error_db', 'intensity_w1_cal_units', 'q99_intensity_error_cal_units'):
                        metric[key] = None
                row = {'record_id':record_id,'seed':seed,'window_zero_based':index,
                       'absolute_start_s':(start+index*8192)/config.prf_hz, **metric}
                window_rows.append(row)
                records_windows.append(row)
        # Aggregate seeds within each measured time window before describing its spread.
        window_means = []
        for index in range(window_count):
            selected = [r for r in records_windows if r['window_zero_based']==index]
            window_means.append({key:float(np.mean([r[key] for r in selected])) if selected[0][key] is not None else None
                                 for key in full_scores[0]})
        aggregate = {key:distribution([r[key] for r in full_scores]) for key in full_scores[0]}
        summary = {'record_id':record_id,'filename':filename,'start_s':start/config.prf_hz,
                   'measured_cpis':frame_count,'time_windows_8_192s':window_count,'simulation_seeds':frozen['seeds'],
                   'range_m':[float(record.range_near_edge_m[0]),float(record.range_near_edge_m[-1])],
                   'scored_gates_one_based':(np.flatnonzero(mask)+1).tolist(),
                   'absolute_power_evaluated':same_range,'source_metadata':record.metadata,
                   'full_record_seed_distribution':aggregate,
                   'time_window_seed_mean_distribution':{key:distribution([r[key] for r in window_means]) for key in full_scores[0]},
                   'notes': 'frozen calibration including receiver correction; receiver/scene shifts not corrected' if same_range else
                            'new range: only normalized spectral transfer; unit model gate power, no absolute amplitude or intensity-distribution score',
                   'sha256':sha(root/'raw'/filename)}
        summaries.append(summary)
        pred = describe(generated[0],config.prf_hz,mask,np.asarray(config.mean_power))
        # Keep actual complex inputs/outputs on NAS; the report packager excludes them.
        np.savez_compressed(output/f'ipix_record_{record_id}_samples.npz',
                            measured_iq=iq, simulated_iq_seed42=generated[0],
                            range_near_edge_m=record.range_near_edge_m, clutter_mask=mask,
                            prf_hz=config.prf_hz, start_sweep=start,
                            model_mean_power=np.asarray(config.mean_power),
                            absolute_power_evaluated=same_range)
        row_index = len(summaries)-1
        extent = [record.range_near_edge_m[0],record.range_near_edge_m[-1],ref['f'][0],ref['f'][-1]]
        # Mean-power normalization is ONLY for these shape plots, identically stated.
        # Absolute power errors remain separately reported without this normalization.
        for ax, power, title in [(axs[row_index,0],ref['mean_rd_power'],'实测时间平均 RD'),
                                 (axs[row_index,1],pred['mean_rd_power'],'平台时间平均 RD（seed 42）')]:
            power = power.copy()
            reference = power[:,mask].mean()
            power[:,~mask] = np.nan
            image = ax.imshow(10*np.log10(np.maximum(power,1e-15)/reference), origin='lower',aspect='auto',extent=extent,
                              vmin=-25,vmax=25,cmap='turbo')
            ax.set(ylim=(-150,150),xlabel='距离门近端 / m',ylabel='Doppler / Hz',title=f'#{record_id} {title}')
            fig_records.colorbar(image,ax=ax,label='各自平均功率归一化 / dB',shrink=.8)
        ax = axs[row_index,2]
        ax.plot(ref['f'],10*np.log10(ref['psd']+1e-15),c='black',label='实测')
        ax.plot(ref['f'],10*np.log10(pred['psd']+1e-15),c='#d87526',label='冻结模型')
        ax.set(xlim=(-200,200),ylim=(-65,-5),xlabel='Doppler / Hz',ylabel='归一化谱 / dB',
               title=f"#{record_id} 全 {frame_count} CPI：谱相关 {aggregate['psd_correlation']['mean']:.3f}\nW1 {aggregate['doppler_w1_hz']['mean']:.1f} Hz")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        if record_id==17:
            fig, axes = plt.subplots(2,4,figsize=(16,7),layout='constrained')
            indices = np.linspace(0,frame_count-1,4,dtype=int)
            refpower = float(np.max(ref['mean_rd_power'][:,mask]))
            for col,index in enumerate(indices):
                for row,rd,label in [(0,ref['rd'],'实测'),(1,pred['rd'],'独立随机仿真')]:
                    p = abs(rd[index])**2
                    p[:,~mask] = np.nan
                    im = axes[row,col].imshow(10*np.log10(np.maximum(p,1e-15)/refpower),origin='lower',aspect='auto',
                                             extent=extent,vmin=-45,vmax=15,cmap='turbo')
                    axes[row,col].set(ylim=(-150,150),xlabel='距离门近端 / m',ylabel='Doppler / Hz',
                                      title=f'{label}：留出 CPI {index}\n实测时刻 {(start+512*index)/1000:.3f} s')
            fig.colorbar(im,ax=axes.ravel().tolist(),label='共用实测留出段参考 / dB',shrink=.7)
            fig.suptitle('IPIX 同记录：按时间等间隔选四帧展示，全部 128 帧均评分\n空白列不评分；仿真与实测没有逐时刻纹理对应关系',fontsize=14)
            fig.savefig(output/'ipix_frames.png',dpi=170)
            plt.close(fig)
        print(f'IPIX #{record_id}: {frame_count} CPI, {window_count} windows, 8 seeds',flush=True)
    fig_records.suptitle('IPIX：四条采集记录的谱形迁移检查（不是按图重新拟合）\n显示各自归一化的形状，绝对功率偏差另列；#19 距离改变，仅作谱形检查',fontsize=14)
    fig_records.savefig(output/'ipix_records.png',dpi=150)
    plt.close(fig_records)
    table(output/'ipix_windows.csv',window_rows)
    table(output/'ipix_frames.csv',frame_rows)
    table(output/'ipix_full_records.csv',global_rows)
    result = {'real_records':4,'measured_cpis':sum(r['measured_cpis'] for r in summaries),
              'measured_windows':sum(r['time_windows_8_192s'] for r in summaries),
              'new_acquisitions_this_run':[19,25],'new_unseen_part_existing_record':'#18 second 65.536 seconds',
              'seeds':frozen['seeds'],'records':summaries,'calibration_frozen':True,'recipe_sha256':sha(recipe_path),
              'statistical_unit':'four acquisition records, not CPI x seed independent experiments; time windows are descriptive correlated samples',
              'receiver_correction_warning':'offset/scale/phase correction fitted on #17 is frozen; changed records can include correction drift as well as scene change',
              'no_pixelwise_or_sidelobe_claim':True}
    dump(output/'ipix_summary.json',result)
    fig,axes = plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,key,label in zip(axes,['psd_correlation','doppler_w1_hz','mean_power_error_db'],
                            ['谱相关：越高越好','Doppler W1 / Hz：越低越好','功率偏差 / dB：0 最好']):
        for j,record in enumerate(summaries):
            ws = []
            for index in range(record['time_windows_8_192s']):
                vals = [r[key] for r in window_rows if r['record_id']==record['record_id'] and r['window_zero_based']==index and r[key] is not None]
                if vals:
                    ws.append(float(np.mean(vals)))
            if ws:
                ax.scatter(np.full(len(ws),j)+np.linspace(-.1,.1,len(ws)),ws,s=22,alpha=.65,color='#3b7da1')
                ax.plot([j-.18,j+.18],[np.median(ws)]*2,c='#d87526',lw=3)
            else:
                ax.text(j,0,'不评分',ha='center',va='bottom')
        ax.set_xticks(range(4),['#17 后段','#18','#19 新距离','#25 次日'])
        ax.set_title(label)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('每个点 = 一个 8.192 s 实测时间窗，与 8 个仿真种子比较后的均值\n橙线为时间窗中位数；分布是波动范围，不是独立试验置信区间',fontsize=13)
    fig.savefig(output/'ipix_variation.png',dpi=170)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--forward',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('choose a new immutable output directory')
    args.output.mkdir(parents=True)
    style()
    maroon = validate_maroon(args.output,args.forward)
    ipix = validate_ipix(args.output)
    source_files = ['scripts/expand_maroon_ipix_validation.py','scripts/prepare_ipix_extension.py',
                    'scripts/validate_ipix_clutter.py','backend/radar_platform/repeat_metrics.py',
                    'backend/radar_platform/distributed_clutter.py','backend/radar_platform/ipix_dataset.py',
                    'backend/radar_platform/surface_echo.py','backend/radar_platform/maroon_dataset.py']
    dump(args.output/'provenance.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),
         'forward_directory':str(args.forward),'source_hashes':{s:sha(PROJECT/s) for s in source_files},
         'model_retuned':False,'new_physical_solver':False,'scope':'extended evaluation and two additional IPIX acquisitions',
         'maroon_channel_count':maroon['evaluation_channels'],'ipix_cpis':ipix['measured_cpis']})
    print(json.dumps({'output':str(args.output),'maroon':maroon['per_channel'],'ipix_records':ipix['real_records']},indent=2))


if __name__ == '__main__':
    main()
