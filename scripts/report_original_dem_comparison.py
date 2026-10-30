"""Render original/new simulation differences, not measured-fidelity scores."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main(root):
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
    old=np.load(root/'original.npz');new=np.load(root/'comparison_products.npz')
    provenance=json.loads((root/'original_provenance.json').read_text())
    estimate=json.loads((root/'new_estimate.json').read_text())
    # Only known FFT/energy normalization is removed; no peak/least-squares fitting.
    gain=provenance['old_num_fft']/old['rd'].shape[0]
    vo=old['velocity_mps'];order=np.argsort(vo);vo=vo[order]
    ro=old['range_m'];po=abs(old['rd'][order].T.astype(complex)*gain)**2
    rn=new['new_range'];vn=new['new_velocity'];pn=abs(new['new_rd'].astype(complex))**2
    keep_r=(ro>=rn.min())&(ro<=rn.max());keep_v=(vo>=vn.min())&(vo<=vn.max())
    r=ro[keep_r];v=vo[keep_v];rr,vv=np.meshgrid(r,v,indexing='ij')
    a=po[np.ix_(keep_r,keep_v)]
    b=RegularGridInterpolator((rn,vn),pn)(np.stack([rr,vv],axis=-1))
    ref=float(a.max());ad=10*np.log10(np.maximum(a/ref,1e-6));bd=10*np.log10(np.maximum(b/ref,1e-6));diff=bd-ad
    roi=(rr>=2400)&(rr<=3100)&(vv>=900)&(vv<=1400)
    def peak(power):
        i,j=np.unravel_index(power.argmax(),power.shape)
        return {'range_m':float(r[i]),'closing_velocity_mps':float(v[j])}
    metrics={'interpretation':'model-to-model difference, NOT error against measured radar',
        'cases':1,'common_range_bins':len(r),'common_velocity_bins':len(v),
        'old_fixed_amplitude_normalization':gain,'reference_power':ref,
        'normalization':'old num_fft/loops; new native unit-window gain; common old power maximum; no data-fitted gain',
        'floor_db':-60,'full_mae_db':float(abs(diff).mean()),'fixed_roi_mae_db':float(abs(diff[roi]).mean()),
        'roi_range_m':[2400,3100],'roi_velocity_mps':[900,1400],
        'window_power_ratio_db':float(10*np.log10(b.sum()/a.sum())),
        'old_peak_common_grid':peak(a),'new_peak_common_grid':peak(b),
        'native_dem_spacing_m':12.5,'new_scatterers':estimate['scatterers'],
        'old_input_hashes_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in provenance['source_sha256'].items()),
        'finite_products':bool(np.isfinite(a).all() and np.isfinite(b).all()),
        'limitations':['single first CPI','no measured radar ground truth','not an isolated single-factor ablation',
            'original received pulse truncated','old/new light-speed constants differ',
            'new diffuse phases and triangular integration differ from old coherent grid-node integration']}
    (root/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    np.savez_compressed(root/'aligned_comparison.npz',range_m=r,velocity_mps=v,old_power=a,new_power=b,difference_db=diff)
    fig,axs=plt.subplots(1,2,figsize=(14,5.6),layout='constrained')
    for ax,z,n,e,title in [(axs[0],old['full_elevation'],old['full_north'],old['full_east'],'原 DEM 全图：未替换、未缩放高程'),
                          (axs[1],old['elevation'],old['north'],old['east'],'本次照射区域：保留原生 12.5 m 网格')]:
        im=ax.imshow(z,extent=[n.min(),n.max(),e.min(),e.max()],origin='lower',cmap='terrain',vmin=324,vmax=454,aspect='equal')
        ax.set(xlabel='原工程 x / 新平台 north (m)',ylabel='原工程 y / 新平台 east (m)',title=title)
        fig.colorbar(im,ax=ax,label='DEM 高程 (m)')
    axs[0].add_patch(Rectangle((old['north'].min(),old['east'].min()),np.ptp(old['north']),np.ptp(old['east']),fill=False,edgecolor='red',lw=1.5))
    for i,color,label in [(0,'black','CPI 开始波束轴落点'),(127,'magenta','CPI 结束波束轴落点')]:
        p=old['position'][i];f=old['boresight'][i];t=(-float(np.median(old['elevation']))-p[2])/f[2]
        xy=p[:2]+t*f[:2];axs[1].scatter(*xy,c=color,s=45,label=label)
    axs[1].legend(fontsize=9)
    fig.suptitle('输入目标就是原来的 DEM；波束轴落点仅按区域中位高程示意',fontsize=14)
    fig.savefig(root/'input_dem.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    vmax=max(0.,float(np.ceil(bd.max()/5)*5))
    for row in range(2):
        for col,z,title in [(0,ad,'原工程 RD'),(1,bd,'统一平台 RD'),(2,diff,'差值：新 − 旧（不是实测误差）')]:
            ax=axs[row,col];im=ax.pcolormesh(v,r,z,shading='nearest',cmap='RdBu_r' if col==2 else 'turbo',
                vmin=-30 if col==2 else -60,vmax=30 if col==2 else vmax,rasterized=True)
            ax.set(xlabel='接近为正的径向速度 (m/s)',ylabel='斜距 (m)',title=title+('｜全范围' if row==0 else '｜固定局部放大'))
            if row:ax.set_xlim(900,1400);ax.set_ylim(2400,3100)
            fig.colorbar(im,ax=ax,label='dB 差值（显示限 ±30）' if col==2 else '相对功率 dB（共用旧图参考）')
    fig.suptitle(f'原参数 / 原 DEM / 相同逐脉冲运动输入｜1 GHz · 5 MHz · 128 pulses\n全图差异 MAE {metrics["full_mae_db"]:.2f} dB；固定局部 {metrics["fixed_roi_mae_db"]:.2f} dB；单 CPI，不代表批量保真',fontsize=14)
    fig.savefig(root/'rd_before_after.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for power,label in [(a,'原工程'),(b,'统一平台')]:
        axs[0].plot(r,10*np.log10(np.maximum(power.sum(axis=1)/ref,1e-6)),label=label)
        axs[1].plot(v,10*np.log10(np.maximum(power.sum(axis=0)/ref,1e-6)),label=label)
    axs[0].set(xlim=(2400,3100),xlabel='斜距 (m)',ylabel='沿速度维求和的相对功率 (dB)',title='距离能量分布')
    axs[1].set(xlim=(900,1400),xlabel='接近为正的径向速度 (m/s)',ylabel='沿距离维求和的相对功率 (dB)',title='速度能量分布')
    for ax in axs:ax.legend();ax.grid(alpha=.25)
    fig.savefig(root/'profiles.png',dpi=160);plt.close(fig)
    rows=[('输入 DEM','AP_01759_FBS_F0570_RT1_3.mat；1000×1000，12.5 m；高程 324–454 m'),
        ('实际初始状态','位置 (0,0,−3000) m；速度 (208,0,1180) m/s，模长 1198.19 m/s；不是精确 1200'),
        ('共同雷达配置','1 GHz；B=5 MHz；fs=10 MHz；脉宽 20 μs；PRI=33.3 μs；CPI=128；波束半角 10°'),
        ('共同运动输入','直接复用旧程序生成的逐脉冲位置、速度、波束方向；保留旧复杂运动的行为，不冒称已修正'),
        ('共同设置','SISO；不加噪声、不加环境多径；双方关闭遮挡；距离与 Doppler 均使用矩形窗'),
        ('地形范围处理',f'旧工程使用完整 DEM；新工程只裁剪全部逐脉冲照射节点外加两格边缘，{old["elevation"].shape}；无降采样、无高程缩放'),
        ('回波截断','完整回波要求 R ≤ c(PRI−脉宽)/2，约 2 km；实际照射斜距约 2.57–2.96 km，因此发生截断。原参数保留，不偷偷修复'),
        ('多普勒模糊','约 ±2251 m/s 无模糊范围；本例未发现照射点速度折叠，无多 PRF 解模糊'),
        ('数值实现差异','旧 c=3e8，新 c=299792458；旧浮点 ceil 得到 334 个接收样本，新严格 PRI 内 333；旧 chirp linspace 含终点，新按采样时钟'),
        ('比较方法','先统一接近为正符号，再将新线性功率插值到重叠旧物理坐标；只去除可解析的 FFT 增益，不做峰值拟合或分别归一化'),
        ('强度/相位模型差异','旧：网格节点、确定性相位、整数距离分箱；新：三角面中心、固定种子随机散射相位、分数延时与脉内运动。新侧使用旧散射公式的有界角度查表近似'),
        ('波束细节','匹配旧高斯包络；旧程序按 12 脉冲批次取波束并集，新程序逐脉冲硬边界；因此不是完全相同的有效照射集合'),
        ('结论边界','这是两个模型的迁移对照，不是实测验证；差值大小不能证明新系统更准。只完成第一 CPI，不是长期复杂运动验收')]
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((root/name).read_bytes()).decode()+'">'
    html='<!doctype html><html lang="zh"><meta charset="utf-8"><title>原工程与统一平台对照</title><style>body{font:16px sans-serif;max-width:1500px;margin:32px auto;padding:20px;color:#172334}td,th{padding:12px;border:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}img{width:100%}p{line-height:1.7}.warn{background:#fff3d6;padding:16px}</style><h1>原参数、原 DEM：旧工程与统一平台对照</h1>'
    html+='<p class="warn">原工程早已具备地形面元回波，不能把“点到面”算作相对原工程的新功能。本次不改原文件，以第一 CPI 对照；不是实测保真证明。图片已内嵌，可直接下载离线打开。</p>'
    html+=image('input_dem.png')+'<table>'+''.join('<tr><th>'+k+'</th><td>'+v+'</td></tr>' for k,v in rows)+'</table>'
    html+=image('rd_before_after.png')+image('profiles.png')
    html+='<h2>怎样理解差异</h2><p>MAE 是固定参考下每个格子 dB 差值的绝对值平均，越大表示两套输出越不一样；不是“离真实雷达越远”。新增功能是统一可配置运动、阵列、材料表、天线和风险检查，而不是旧图自动变成真实图。本例的面元位置、散射相位和数值积分均变化，因此需下一步逐项消融，才能分清每项贡献。</p>'
    html+='<table><tr><th>指标</th><th>本次结果</th></tr>'+''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in metrics.items())+'</table></html>'
    (root/'report.html').write_text(html)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
