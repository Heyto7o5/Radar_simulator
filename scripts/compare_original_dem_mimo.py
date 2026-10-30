"""Paired SISO / 2Tx2Rx TDM example: fixed DEM, total CPI and pulse slots."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from radar_platform.unified.schema import Simulation
from radar_platform.unified.session import SimulationSession
from radar_platform.unified.geometry import build_layout


def main():
    source=ROOT/'.runtime/original_dem_comparison_20261023'
    out=source/'mimo_2tx2rx'
    render_only='--render-only' in sys.argv
    out.mkdir(exist_ok=render_only)
    base=json.loads((source/'new_config.json').read_text())
    raw=copy.deepcopy(base)
    raw['name']='Original DEM: paired 2Tx2Rx TDM, same 128 physical slots'
    raw['array']={'layout':'ula','num_tx':2,'num_rx':2}
    raw['waveform']['loops']=64
    raw['input_provenance']+=['Only array and loops changed: 2Tx2Rx, 64 loops = 128 physical transmit slots; same total CPI as 128-slot SISO. Single-Tx velocity aliasing is retained, not resolved.']
    request=Simulation.model_validate(raw)
    session=SimulationSession(request)
    session.save_inputs(out)
    layout=build_layout(request.array,request.waveform,request.platform)
    (out/'estimate.json').write_text(json.dumps(session.estimate,ensure_ascii=False,indent=2))
    (out/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'backend/radar_platform').rglob('*.py')},indent=2))
    print(json.dumps({'status':'starting','scatterers':len(session.scene.amplitudes),'array':layout.metadata()}),flush=True)
    if render_only:
        from types import SimpleNamespace
        from radar_platform.unified.processing import process_adc
        meta=json.loads((out/'frame_0000/metadata.json').read_text())
        adc=np.fromfile(out/'frame_0000/adc.bin',dtype='<c8').reshape(meta['products']['adc']['shape'])
        f=SimpleNamespace(adc=adc,products=process_adc(adc,request,layout))
    else:
        f=session.frame(0);session.save_frame(out,f)
    s=np.load(source/'comparison_products.npz')
    sp=abs(s['new_rd'].astype(complex))**2
    ch=abs(f.products.rd_channels.astype(complex))**2
    mp=ch.mean(axis=(2,3));single=ch[:,:,0,0]
    reference=float(sp.max())
    def db(p):return 10*np.log10(np.maximum(p/reference,1e-6))
    def peak(p,r,v):
        i,j=np.unravel_index(p.argmax(),p.shape)
        return {'range_m':float(r[i]),'apparent_velocity_mps':float(v[j])}
    metrics={'cases':1,'same_dem':raw['scene']==base['scene'],'same_motion':raw['platform']==base['platform'],
        'same_waveform_except_loop_count':{k:v for k,v in raw['waveform'].items() if k!='loops'}=={k:v for k,v in base['waveform'].items() if k!='loops'},
        'total_slots':128,'total_cpi_s':128*request.waveform.slot_period_s,
        'siso_peak':peak(sp,s['new_range'],s['new_velocity']),
        'mimo_mean_peak':peak(mp,f.products.range_m,f.products.velocity_mps),
        'siso_unambiguous_speed_mps':request.waveform.wavelength_m/(4*request.waveform.slot_period_s),
        'mimo_unambiguous_speed_mps':f.products.metadata['max_unambiguous_velocity_mps'],
        'native_velocity_bin_mps':f.products.metadata['native_velocity_cell_mps'],
        'layout':layout.metadata(),'mimo_summed_normalized_rd_power_ratio_to_siso_db':float(10*np.log10(mp.sum()/sp.sum())),
        'all_finite':bool(np.isfinite(f.adc).all() and np.isfinite(mp).all()),
        'note':'No MAE across unequal velocity ambiguity intervals. Sum of |FFT/N|^2 equals mean slow-time signal power (Parseval); no extra loop-count factor. No coherent SNR gain claim. No noise, no measured truth, no ambiguity resolution. Array boresight follows original rapidly varying trajectory; no validated motion-corrected RA claimed.'}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    vmax=max(0.,float(np.ceil(max(db(sp).max(),db(mp).max(),db(single).max())/5)*5))
    panels=[(sp,s['new_range'],s['new_velocity'],'SISO：1Tx × 1Rx'),
        (single,f.products.range_m,f.products.velocity_mps,'MIMO：Tx0–Rx0 单通道'),
        (mp,f.products.range_m,f.products.velocity_mps,'MIMO：4 通道非相干功率平均')]
    for row in range(2):
        for col,(p,r,v,title) in enumerate(panels):
            ax=axs[row,col]
            im=ax.pcolormesh(v,r,db(p),shading='nearest',cmap='turbo',vmin=-60,vmax=vmax)
            ax.set(title=title+('｜完整有效速度轴' if row==0 else '｜主要回波放大'),xlabel='表观径向速度 (m/s)，接近为正',ylabel='斜距 (m)')
            if row:
                ax.set_ylim(2400,3100);ax.set_xlim((900,1400) if col==0 else (-1125,-750))
            fig.colorbar(im,ax=ax,label='共同 SISO 参考功率 (dB)')
    fig.suptitle('原 DEM、1 GHz、相同运动与总 CPI：SISO → 2Tx2Rx TDM-MIMO\n轮流发射使同一 Tx 的采样间隔加倍，约 1200 m/s 回波发生折叠；不是目标反向运动',fontsize=14)
    fig.savefig(out/'siso_mimo_rd.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.7),layout='constrained')
    tx=np.asarray(layout.tx_local_m)[:,1];rx=np.asarray(layout.rx_local_m)[:,1];virt=layout.virtual_local_m[:,1]
    for y,x,label,color in [(2,tx,'实际 Tx','tab:red'),(1,rx,'实际 Rx','tab:blue'),(0,virt,'虚拟坐标 Tx+Rx','tab:green')]:
        ax.scatter(x,np.full(len(x),y),s=100,c=color)
        for i,a in enumerate(x):ax.text(a,y+.15,f'{i}: {a:.3f} m',ha='center')
    ax.set(yticks=[0,1,2],yticklabels=['4 个虚拟位置','2 个实际接收阵元','2 个实际发射阵元'],ylim=(-.4,2.5),xlabel='传感器局部横向坐标 (m)',title='2Tx2Rx 线阵：相邻虚拟位置约 λ/2 = 0.150 m；不是四根实际接收天线')
    ax.grid(axis='x',alpha=.2);fig.savefig(out/'array_layout.png',dpi=160);plt.close(fig)
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">'
    html='<!doctype html><meta charset="utf-8"><title>原 DEM 的 MIMO RD</title><style>body{font:17px sans-serif;max-width:1450px;margin:30px auto;padding:20px;line-height:1.7}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd}</style><h1>同一区域：SISO 与 2Tx2Rx TDM-MIMO</h1><p>复用前一轮统一平台 SISO，DEM、随机散射相位、轨迹、波形及处理窗不变，仅修改阵列和每个 Tx 的脉冲数。未启动前端。</p>'
    html+='<table><tr><th>项目</th><th>SISO</th><th>MIMO</th></tr><tr><td>实际收发</td><td>1Tx1Rx</td><td>2Tx2Rx，轮流发射</td></tr><tr><td>物理发射时隙</td><td>128</td><td>128＝2×64</td></tr><tr><td>总 CPI</td><td colspan="2">4.2624 ms，不延长观察时间</td></tr><tr><td>同一 Tx 间隔</td><td>33.3 μs</td><td>66.6 μs</td></tr><tr><td>速度无模糊范围</td><td>约 ±2251 m/s</td><td>约 ±1125 m/s</td></tr><tr><td>速度单元宽度</td><td colspan="2">约 35.17 m/s；本例未提高距离或速度分辨率</td></tr></table>'
    html+=image('array_layout.png')+image('siso_mimo_rd.png')
    html+='<p>四通道功率平均没有相干波束形成，不应称为更精准的 RD。阵列主要增加空间相位信息，后续可用于测角；单看平均 RD 不能验证测角收益。TDM 折叠约为 v−2250.69 m/s，因此原约 1200 m/s 对应约 −1051 m/s。分布式地形和原轨迹姿态变化会使谱峰偏离这一单速度示意值。没有做多 PRF 或其他解模糊。</p><p>本例保留前一轮的回波截断问题、矩形窗和原复杂运动。两种模式速度轴的无模糊范围不同，不计算误导性的逐像素 MAE；也不宣称高保真或真实雷达实验证明。</p><pre>'+json.dumps(metrics,ensure_ascii=False,indent=2)+'</pre>'
    (out/'report.html').write_text(html)
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
