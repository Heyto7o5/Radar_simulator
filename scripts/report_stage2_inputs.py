#!/usr/bin/env python3
"""Document actual classification inputs and analytic radiation controls."""
import argparse,base64,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from scipy.signal import find_peaks
from radar_platform.worldcover import CLASSES
from radar_platform.unified.radiation import Radiation,directions,aperture_voltage,element_voltage

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#000000','#006400','#ffbb22','#ffff4c','#f096ff','#fa0000','#b4b4b4','#f0f0f0','#0064c8','#0096a0','#00cf75','#fae6a0']


def beam_metrics(angles,power):
    peak=int(np.argmax(power));level=power[peak]/2
    left=np.flatnonzero(power[:peak]<=level)[-1];right=peak+np.flatnonzero(power[peak:]<=level)[0]
    left_angle=float(np.interp(level,power[left:left+2],angles[left:left+2]))
    right_angle=float(np.interp(level,power[right-1:right+1][::-1],angles[right-1:right+1][::-1]))
    minima=find_peaks(-power)[0];lo=int(minima[minima<peak][-1]);hi=int(minima[minima>peak][0])
    sidelobe=float(max(power[:lo].max(),power[hi+1:].max()))
    return dict(peak_angle_deg=float(angles[peak]),half_power_width_deg=right_angle-left_angle,
                peak_relative_drive_power_db=float(10*np.log10(power[peak])),
                peak_sidelobe_relative_to_own_main_db=float(10*np.log10(sidelobe/power[peak])),
                main_null_boundaries_deg=[float(angles[lo]),float(angles[hi])])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--configs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=Path(os.environ.get('RADAR_DATA_ROOT','/nas/datasets/hyt/datasets/radar_open'))/'worldcover_2021'
    codes=list(CLASSES);cmap=ListedColormap(COLORS);norm=BoundaryNorm(np.arange(len(codes)+1)-.5,len(codes))
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');stats=[]
    for ax,site in zip(axes,['alpine','valley','foothill']):
        with np.load(data/f'processed/fixed_dem_cells_20261022/{site}_30m_cells.npz',allow_pickle=False) as f:
            values=f['dominant_code'];meta=json.loads(str(f['metadata_json']))
        center=values.shape[0]//2;crop=values[center-32:center+32,center-32:center+32]
        display=np.zeros_like(crop)
        for i,code in enumerate(codes):display[crop==code]=i
        ax.imshow(display.T,origin='lower',extent=[-960,960,-960,960],interpolation='nearest',cmap=cmap,norm=norm)
        ax.set(title=site,xlabel='East (m)',ylabel='North (m)')
        stats.append({'site':site,**meta})
    present=set().union(*[set(int(x) for x in r['dominant_class_counts']) for r in stats])
    fig.legend(handles=[Patch(color=COLORS[codes.index(c)],label=CLASSES[c]) for c in codes if c in present],loc='outside lower center',ncol=5)
    fig.suptitle('Real WorldCover 2021 classes on DEM cells — not radar scattering coefficients')
    fig.savefig(a.output/'land_cover.png',dpi=160);plt.close(fig)
    angles=np.linspace(-80,80,16001);u=directions(angles,np.zeros_like(angles))
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');metrics=[]
    for variant in ['uniform_0','hamming_0','hamming_8']:
        raw=json.loads((a.configs/f'alpine_{variant}_s11.json').read_text());spec=Radiation.model_validate(raw['radiation'])
        gain=aperture_voltage(u,np.zeros_like(angles),spec.transmit_aperture,raw['waveform']['carrier_hz'])
        gain*=element_voltage(u[:,0],spec.tx_element)*element_voltage(u[:,0],spec.rx_element)
        power=abs(gain)**2;db=10*np.log10(np.maximum(power,1e-8))
        for ax in axes:ax.plot(angles,db,label=variant)
        metrics.append({'variant':variant,**beam_metrics(angles,power)})
    for ax in axes:
        ax.set(xlabel='Sensor-local azimuth (deg)',ylabel='Relative response (dB, same drive-power reference)',ylim=(-60,14));ax.grid(alpha=.2);ax.legend()
    axes[0].set(title='Full modeled scalar response',xlim=(-80,80));axes[1].set(title='Mainlobe and nearby sidelobes',xlim=(-25,25))
    fig.savefig(a.output/'radiation_controls.png',dpi=160);plt.close(fig)
    (a.output/'input_statistics.json').write_text(json.dumps({'classes':stats,'radiation':metrics},ensure_ascii=False,indent=2))
    images=''.join('<img style="width:100%" src="data:image/png;base64,'+base64.b64encode((a.output/name).read_bytes()).decode()+'">' for name in ['land_cover.png','radiation_controls.png'])
    rows=''.join(f'<tr><td>{r["variant"]}</td><td>{r["peak_angle_deg"]:.2f}</td><td>{r["half_power_width_deg"]:.2f}</td><td>{r["peak_sidelobe_relative_to_own_main_db"]:.2f}</td><td>{r["peak_relative_drive_power_db"]:.2f}</td></tr>' for r in metrics)
    page='<html lang="zh"><meta charset="utf-8"><title>地物输入与波束控制</title><body style="max-width:1400px;margin:30px auto;font:16px sans-serif"><h1>真实类别输入与解析波束控制</h1><p>上图使用 ESA WorldCover 2021 v200 分类，重投影单独类别掩膜后按面积占比取主类别，不对类别编号做双线性插值。混合格子仍保存类别占比。它不是逐地块现场真值，也不提供雷达反射系数。</p>'+images+'<table border="1" cellpadding="7"><tr><th>控制</th><th>峰值角 °</th><th>半功率宽度 °</th><th>峰值旁瓣/本身主瓣 dB</th><th>峰值/共同驱动功率参考 dB</th></tr>'+rows+'</table><p>方向图是模型的解析检查，不是实测方向图。所有权重按相同总阵元驱动功率归一化，未分别把峰值拉成同样高度。加窗会降低旁瓣，但会展宽主瓣并降低峰值；电子扫描不是移动天线。</p><p>真实 DEM 组合实验中的 NRCS 统一基线与 ±6 dB 改变量都是预先声明的假设，只检验模型敏感性，不解释成地物真实反射强度或保真改善。</p><p>© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium</p><p><a href="https://esa-worldcover.org/en/data-access">官方数据、许可与分类精度说明</a></p></body></html>'
    (a.output/'report.html').write_text(page);print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
