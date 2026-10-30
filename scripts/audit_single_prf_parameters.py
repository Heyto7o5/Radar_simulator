"""Fixed 1200 m/s diagnostic sweeps; no multi-PRF fusion and no simulator edits."""
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from radar_platform.models import SimulationRequest
from radar_platform.unified.adapters import from_pulsed_request
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.schema import Simulation
from radar_platform.comparison import power_db
from render_scene import configure_font

P=Path(__file__).resolve().parents[1]
O=P/'docs/weekly/2026-10-10/单PRF参数复核'
C=299792458.

def main():
    O.mkdir(exist_ok=True);(O/'configs').mkdir(exist_ok=True);configure_font()
    base=from_pulsed_request(SimulationRequest.model_validate(json.loads((P/'configs/high_altitude_missile_10ghz.json').read_text())))
    base.sequence.frame_count=3;base.processing.range_limits_m=(4900,5220)
    # Predeclared comparisons, NOT optimizing frame appearance or real-data scores.
    cases=[('A 原报告',10e9,15000,20e-6),('B 仅改 PRF',10e9,16000,20e-6),
           ('C 缩短脉冲并提高 PRF',10e9,25000,5e-6),('D 仅降低载频',1e9,15000,20e-6),
           ('E 旧频段的无模糊对照',1e9,20000,5e-6)]
    config_rows=[];metrics=[];truthrows=[];fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    views={}
    for index,(name,fc,prf,pw) in enumerate(cases):
        d=base.model_dump();d['name']=name
        d['waveform'].update(carrier_hz=fc,slot_period_s=1/prf,pulse_width_s=pw,samples=int(np.floor(150e6/prf)))
        req=Simulation.model_validate(d)
        (O/'configs'/f'{index+1}.json').write_text(json.dumps(req.model_dump(mode='json'),ensure_ascii=False,indent=2))
        lam=C/fc;vperiod=lam*prf/2;vmax=vperiod/2
        config_rows.append({'case':name,'carrier_GHz':fc/1e9,'PRF_kHz':prf/1000,'pulse_us':pw*1e6,
            'speed_mps':float(np.linalg.norm(req.platform.velocity_ned_mps)),'height_m':5000.,
            'v_unambiguous_mps':vmax,'range_unambiguous_m':C/(2*prf)})
        for fi in range(3):
            fr=simulate_frame(req,fi);p=fr.products;t=fr.metadata['truth_epoch_s'];pv=np.array(req.platform.velocity_ned_mps)
            truth=[]
            for obj in req.scene.objects:
                delta=np.array(obj.position_ned_m)-np.array(req.platform.position_ned_m)-pv*t
                radius=np.linalg.norm(delta);vr=pv@delta/radius
                order=int(np.floor((vr+vmax)/vperiod));apparent=vr-order*vperiod
                full=2*radius/C+pw<1/prf
                truth.append((radius,vr,apparent,order));assert full
            remaining=p.rd_power.copy();peaks=[]
            for _ in range(3):
                ri,di=np.unravel_index(np.argmax(remaining),remaining.shape)
                peaks.append((p.range_m[ri],p.velocity_mps[di]));remaining[max(0,ri-15):ri+16,:]=0
            for n,(tr,peak) in enumerate(zip(sorted(truth),sorted(peaks))):
                radius,vr,apparent,order=tr
                # Circular velocity difference across FFT interval endpoints.
                error=abs((float(peak[1])-apparent+vmax)%vperiod-vmax)
                truthrows.append({'case':name,'frame':fi,'target':f'T{n+1}','true_range_m':float(radius),
                    'true_velocity_mps':float(vr),'apparent_velocity_mps':float(apparent),'ambiguity_order':order,
                    'detected_range_m':float(peak[0]),'detected_velocity_mps':float(peak[1]),
                    'velocity_error_bins':float(error/p.metadata['native_velocity_cell_mps']),
                    'receive_gate_contains_complete_echo':True})
            edge=abs(p.velocity_mps)>.9*vmax
            folded=sum(tr[3]!=0 for tr in truth)
            row={'case':name,'frame':fi,'folded_target_fraction':folded/3,
                 'edge_power_fraction':float(p.rd_power[:,edge].sum()/p.rd_power.sum()),
                 'all_echoes_in_PRI':True}
            metrics.append(row)
            if fi==0:
                views[name]=(p,truth)
                ax=axes.flat[index];db=power_db(p.rd_power,float(p.rd_power.max()))
                im=ax.pcolormesh(p.range_m,p.velocity_mps,db.T,shading='auto',cmap='turbo',vmin=-45,vmax=0,rasterized=True)
                ax.scatter([r[0] for r in truth],[r[2] for r in truth],marker='x',s=40,c='white')
                ax.set(xlabel='距离 / m',ylabel='表观接近速度 / (m/s)',
                    title=f'{name}：{fc/1e9:g} GHz / {prf/1000:g} kHz / {pw*1e6:g} μs\n折叠目标 {folded}/3；速度边缘功率 {row["edge_power_fraction"]:.2%}')
                fig.colorbar(im,ax=ax,label='本图峰值参考 / dB',shrink=.8)
    ax=axes.flat[-1];ax.axis('off')
    ax.text(.02,.93,'所有组固定：\n1200 m/s、5 km、80°、相同三个点\n60 MHz 带宽、150 MHz ADC、64 pulses\n\nA→B：图的峰移到零速附近，但仍全部折叠。\nA→C：提高 PRF，仍全部折叠。\nD→E：在 1 GHz 下才实现本组无模糊。\n\n1 GHz 是分米波，不是厘米波。\n本图只比较谱的位置和形状，\n不同载频的幅度不代表材料散射保真。',va='top',fontsize=11)
    fig.suptitle('固定 1200 m/s 的单 PRF 对照：图看起来集中，不代表真实速度没有折叠\n白叉为独立计算的表观速度；图中不执行多 PRF 融合或速度解模糊',fontsize=13)
    fig.savefig(O/'固定1200参数对照.png',dpi=160,bbox_inches='tight');plt.close(fig)
    for name,rows in [('参数清单.csv',config_rows),('逐帧指标.csv',metrics),('目标真值与检测.csv',truthrows)]:
        with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    # Fully analytical feasibility sweep, including configurations the current one-PRI gate cannot run.
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    prfs=np.linspace(5e3,180e3,1200);rmax=max(r[0] for r in views[cases[0][0]][1])
    for fc in [1e9,3e9,10e9]:axs[0].plot(prfs/1e3,C/(4*fc)*prfs,label=f'{fc/1e9:g} GHz')
    axs[0].axhline(1200,c='black',ls='--',label='1200 m/s')
    axs[0].axvline(C/(2*rmax)/1000,c='red',ls=':',label='距离无模糊上限（不计脉宽）')
    axs[0].set(xlabel='PRF / kHz',ylabel='正负对称无模糊速度上限 / (m/s)',ylim=(0,4000),title='厘米波 + 当前距离：零中心无模糊区不够宽');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    p,truth=views[cases[0][0]]
    angles=np.linspace(0,20,400);span=1200*(1-np.cos(np.radians(angles)))
    axs[1].plot(angles,span);axs[1].axvline(4,c='gray',ls='--');axs[1].scatter([4],[1200*(1-np.cos(np.radians(4)))],c='orange')
    axs[1].set(xlabel='视线相对速度方向的夹角 / °',ylabel='相对中心径向速度的变化 / (m/s)',title='窄波束内速度很接近：即使整体折叠仍可集中');axs[1].grid(alpha=.2)
    fig.suptitle('两件事要区分：无模糊速度范围，与图上回波分布宽度',fontsize=12)
    fig.savefig(O/'参数约束与谱宽.png',dpi=160,bbox_inches='tight');plt.close(fig)
    old=Path('/home/heyiting/projects/radar_simulator_complex')
    files=[old/'configs/config_complex_motion.yml',old/'configs/config_near_vertical.yml',old/'scripts/gen_complex_motion_dataset.py',old/'scripts/gen_mamba_dataset.py',old/'archive/visualize_rd_comparison.py',old/'echo.py',old/'signal_processing.py']
    maximum=max(r['velocity_error_bins'] for r in truthrows)
    summary={'speed_fixed_mps':1200,'height_fixed_m':5000,'single_PRF_only':True,'configurations':config_rows,
        'frames':15,'target_checks':len(truthrows),'max_error_to_folded_velocity_bins':maximum,
        'all_checks_within_one_velocity_bin':maximum<=1,
        'old_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'historical_metadata_limit':'saved old NPZ files do not embed carrier frequency; current config cannot prove every old run used 1 GHz',
        'scope':'current unified point model sweep, not an exact replay of old DEM scattering or proof of historical runs'}
    (O/'复核摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    assert maximum<=1
    rows=''.join('<tr>'+''.join('<td>'+x+'</td>' for x in [r['case'],f"{r['carrier_GHz']:g}",f"{r['PRF_kHz']:g}",f"{r['pulse_us']:g}",f"±{r['v_unambiguous_mps']:.1f}",'无速度折叠' if r['case'].startswith('E') else '全部目标仍折叠'])+'</tr>' for r in config_rows)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定1200 m/s单PRF参数复核</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;color:#243449}img{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4de;padding:10px}th{background:#eef3f6}.note{background:#fff4dc;padding:15px}h2{margin-top:35px}</style><h1>固定 1200 m/s：为什么旧图看起来模糊少？</h1><p>2026-10-10 · 本次不降低速度、不加入多 PRF，不修改旧工程或仿真引擎。</p>
<h2>先核对旧工程</h2><table><tr><th>证据</th><th>发现</th><th>解释边界</th></tr>
<tr><td>复杂运动与 Mamba 生成入口</td><td>当前都读取 config_complex_motion.yml：1 GHz、PRI 33.3 μs、128 pulses</td><td>这组无模糊速度约 ±2251 m/s，1200 m/s 不必折叠；1 GHz 是分米波。</td></tr>
<tr><td>另一个近垂直配置</td><td>10 GHz、PRI 100 μs、64 pulses</td><td>这是另一组设置，不能与 1 GHz 的结果混为一谈。</td></tr>
<tr><td>历史绘图脚本</td><td>部分图仅显示 Doppler bin；另一个写死 35 m/s/bin、30 m/bin</td><td>35 m/s 接近 1 GHz/30 kHz/128 脉冲的速度格点；但当前 10 MHz ADC 对应约 15 m 距离采样，30 m 接近带宽分辨率，存在混淆风险。不能照搬这些轴。</td></tr>
<tr><td>旧回波处理</td><td>按距离单元取整后与 chirp 卷积；相位符号、窗及地面散射模型与新引擎不同</td><td>拖尾、旁瓣、散斑的外观差异不能全归因于速度折叠。</td></tr>
<tr><td>旧距离门外处理</td><td>echo.py 将超出终止距离门的散射回波直接清零</td><td>没有完整模拟跨 PRI 的距离折叠叠加；旧图没有出现某些模糊回波，也可能是模型未生成它们。</td></tr>
<tr><td>历史存档</td><td>读取的 NPZ 未完整保存载频元数据</td><td>当前代码能解释一类旧图，不能证明你指的每张旧图都采用了 1 GHz。</td></tr></table>
<h2>同场景参数对照</h2><p>所有组为 1200 m/s、5 km、80°、同样三个点。每组 3 帧，共 45 个目标–帧检查，检测速度与折叠解析速度误差均不超过一个原生速度单元。改变载频与 PRF 的候选仅用于诊断，没有替换周报主场景。</p>
<table><tr><th>对照组</th><th>载频/GHz</th><th>PRF/kHz</th><th>脉宽/μs</th><th>无模糊速度 m/s</th><th>结果</th></tr>ROWS</table>IMAGE1
<p class="note">A→B 图上看起来从约 75 m/s 变到约 0 m/s，但模糊阶数仍为 5：回波没有跨图像上下边缘，不等于没有折叠。单 PRF 可以把一个已知窄速度区间放到合适频带显示，但仅凭该观测仍不能独立确定绝对模糊阶数。</p>
<h2>为什么窄条带看起来“不模糊”</h2>IMAGE2
<p>近垂直、窄波束时，各目标径向速度接近，真实速度可集中在 1200 m/s 附近；整体减去相同模糊周期后仍然集中。图的清晰度或是否跨上下边界，不是判断速度是否正确的标准。</p>
<h2>目前能下的结论</h2><p>旧工程常用入口当前是 1 GHz，不是报告的 10 GHz，足以解释很大一部分差异。1 GHz / 20 kHz / 5 μs 在本次 5 km 点目标对照中可无模糊测量 1200 m/s，但不是厘米波方案。保持 10 GHz、5 km 和近垂直 1200 m/s 时，当前零中心单 PRF 脉冲模型不能仅调 PRF 同时获得全局距离、速度无模糊；调 PRF 可改变折叠位置，不会消除绝对速度歧义。</p><p>仍需你指出具体旧图或结果文件，才能把其采集配置与坐标完全对应。未重新拟合目标或给速度轴加上已知真值。</p></html>'''
    text=text.replace('ROWS',rows)
    for marker,name in [('IMAGE1','固定1200参数对照.png'),('IMAGE2','参数约束与谱宽.png')]:
        text=text.replace(marker,'<img src="data:image/png;base64,'+base64.b64encode((O/name).read_bytes()).decode()+'">')
    (O/'固定1200单PRF复核.html').write_text(text)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
