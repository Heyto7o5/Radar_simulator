#!/usr/bin/env python3
"""Common physical axes, honest display-only normalization; no fidelity MAE."""
import hashlib,json,urllib.request
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from radar_platform.models import C0
from radar_platform.dart_dataset import load_dart_frame
from radar_platform.dart_processing import rover_magnitude,published_to_closing_axis

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    url='https://raw.githubusercontent.com/WiseLabCMU/dart/main/dart/pose.py'
    with urllib.request.urlopen(url,timeout=30) as response:source=response.read(100001)
    if len(source)>100000:raise ValueError('unexpected source length')
    (ROOT/'raw/source_snapshots/dart_pose.py').write_bytes(source)
    measured=load_dart_frame(ROOT/'extracted/lab-1',0)
    velocity,real=published_to_closing_axis(measured.published_velocity_mps,measured.rd_power)
    with np.load(ROOT/'processed/lab-1_surface_baseline.npz',allow_pickle=False) as f:
        magnitude=rover_magnitude(f['adc'])
    cfg=json.loads((OUT/'surface_baseline_request.json').read_text())
    w=cfg['waveform'];fc=w['start_hz']+w['slope_hz_s']*(w['adc_start_s']+w['samples']/(2*w['sample_rate_hz']))
    native_range=np.arange(512)*w['sample_rate_hz']/512*C0/(2*w['slope_hz_s'])
    raw_velocity=np.fft.fftshift(np.fft.fftfreq(256,.001))*C0/(2*fc)
    sim_velocity,sim=published_to_closing_axis(raw_velocity,np.mean(magnitude**2,axis=2))
    rr,vv=np.meshgrid(measured.range_m,velocity,indexing='ij')
    aligned=RegularGridInterpolator((native_range,sim_velocity),sim,bounds_error=False,fill_value=np.nan)(np.stack([rr,vv],axis=-1))
    real_ref=float(real.max());sim_ref=float(np.nanmax(aligned))
    real_db=10*np.log10(np.maximum(real/real_ref,1e-6))
    sim_db=10*np.log10(np.maximum(aligned/sim_ref,1e-6))
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,values,title in zip(axes,[real_db,sim_db],['MEASURED: published processed amplitude','SIMULATED: partial LiDAR surfaces, no added noise']):
        im=ax.pcolormesh(velocity,measured.range_m,values,shading='auto',cmap='viridis',vmin=-60,vmax=0)
        ax.set(xlabel='Closing velocity (m/s; approaching positive)',ylabel='Range (m)',title=title)
        fig.colorbar(im,ax=ax,label='dB relative to EACH image peak (shape preview ONLY)')
    fig.savefig(OUT/'dart_aligned_shape_preview.png',dpi=160);plt.close(fig)
    report={'is_final_validation':False,'mae':None,'reason_no_mae':'no independent common amplitude calibration; background processing still differs',
            'raw_frame_index':426,'trajectory_index':158,'scene_count':1,
            'velocity_mapping':'closing = -published d, justified by project_angle: direction dot velocity = -d',
            'velocity_source':url,'velocity_source_sha256':hashlib.sha256(source).hexdigest(),
            'range_mapping':'real: published corrected range axis; simulated: ideal exact-C0 FMCW range axis, no instrument offset added',
            'resampling':'linear interpolation of simulated LINEAR POWER onto real physical grid; out-of-support = NaN',
            'unsupported_bins':int(np.isnan(aligned).sum()),
            'normalization':'separate peak references, for shape preview ONLY, not amplitude agreement',
            'real_peak_reference':real_ref,'simulated_peak_reference':sim_ref,
            'zero_doppler_processing_aligned':False,'material_coefficients_calibrated':False,
            'hardware_IQ_sign_independently_verified':False,
            'processing_tests':'known positive frequency tone peaks at native range=20, Doppler=+5, angle=+2; angular mean-power Parseval; frozen background profile'}
    (OUT/'aligned_preview_audit.json').write_text(json.dumps(report,indent=2))
    (OUT/'DART同坐标实测仿真预览.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1250px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}.warn{background:#fff0c9;padding:15px}td,th{border:1px solid #bbb;padding:10px}table{border-collapse:collapse}</style>
<h1>实测与面目标仿真：先统一物理坐标</h1><p class="warn">这是结构预览，不是高保真通过结果。两张图分别按自身峰值归一化，仅为了看回波分布；不能由颜色接近判断幅度一致，所以本页不报告差值或 MAE。</p>
<img src="dart_aligned_shape_preview.png"><table><tr><th>已对齐</th><th>仍未对齐</th></tr>
<tr><td>速度统一为接近为正。变换依据官方几何公式，不按实测主峰挑选方向。</td><td>材料与绝对/相对通道增益未独立标定，不能比较绝对强度。</td></tr>
<tr><td>仿真 ADC 使用 Hann 距离/慢时间窗、距离 FFT、Doppler FFT 和八通道角度 FFT，再平方平均；已做解析测试。</td><td>实测已有零速度背景扣除，当前仿真未用配套校准序列复现；噪声、环境多径和遮挡未加入。</td></tr>
<tr><td>实测采用公开距离轴，仿真采用理想物理距离轴；在线性功率域插值到相同网格，不平移主峰。</td><td>实测距离零点来自作者标定，仍无记录级原始 ADC 可独立重做。几何覆盖和面片离散仍有误差。</td></tr></table>
<p>下一步：冻结校准/开发/保留记录清单，统一背景处理和功率参考，增加轨迹驱动与多帧测试，再给出差值及 MAE。此帧已用于开发，不作为最终保留样本。</p>
<p><a href="aligned_preview_audit.json">坐标与处理审计</a> · <a href="surface_baseline_request.json">完整仿真输入</a> · <a href="DART面目标输入核查.html">实际几何与实测帧</a></p></html>''')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
