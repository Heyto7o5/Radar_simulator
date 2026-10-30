#!/usr/bin/env python3
"""Validate newly acquired real spectral files; no simulation or physical-fit claim."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.mrr_dataset import load_mrr_spectra
from render_scene import configure_font

def main():
    root=Path('/nas/datasets/hyt/datasets/radar_open/mrr_ace')
    out=Path(__file__).resolve().parents[1]/'docs/background_extension/2026-10-15'
    out.mkdir(parents=True,exist_ok=True);configure_font()
    paths=sorted((root/'extracted').rglob('*.nc'));rows=[]
    fig,axes=plt.subplots(1,len(paths),figsize=(14,4.3),layout='constrained')
    for ax,path in zip(np.atleast_1d(axes),paths):
        d=load_mrr_spectra(path);eligible=np.flatnonzero((d.quality_percent>=50)&(d.good_spectra_count>0)&np.any(np.isfinite(d.power),axis=(1,2)))
        # Fixed middle quality-eligible sample, not the strongest or prettiest echo.
        index=int(eligible[len(eligible)//2]) if len(eligible) else None
        item={**d.metadata,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'shape':list(d.power.shape),'height_range_m':[float(d.height_m.min()),float(d.height_m.max())],
              'first_utc':d.utc[0],'last_utc':d.utc[-1],'velocity_bin_spacing_mps':d.velocity_bin_spacing_mps,
              'display_quality_threshold':50,'display_sample_index':index,'eligible_samples':len(eligible),
              'precipitation_class_at_selected_sample':'not assigned; requires joining the published event intervals',
              'not_simulation_validation':True}
        if index is not None:
            keep=d.height_m>0;values=d.power[index,keep];positive=values[np.isfinite(values)&(values>0)]
            reference=float(positive.max()) if len(positive) else 1.
            db=10*np.log10(np.maximum(values/reference,1e-6))
            im=ax.pcolormesh(np.arange(values.shape[1]),d.height_m[keep],db,shading='auto',cmap='turbo',vmin=-60,vmax=0)
            fig.colorbar(im,ax=ax,label='本样例峰值参考 / dB',shrink=.8)
            ax.set(title=d.utc[index],xlabel='原生 Doppler bin（不猜测正负速度）',ylabel='相对仪器高度 / m')
        else:ax.text(.5,.5,'没有满足显示条件的样本',ha='center');ax.set_axis_off()
        rows.append(item)
    fig.suptitle('新增公开实测：MRR-2 降水雷达功率谱，三个文件的固定质量合格样例\n仅完成数据接入与坐标审查；不是模拟结果，也未独立恢复雨滴几何、船体运动或垂直风',fontsize=11)
    (out/'images').mkdir(exist_ok=True);(out/'metrics').mkdir(exist_ok=True)
    fig.savefig(out/'images/mrr_real_examples.png',dpi=160);plt.close(fig)
    (out/'metrics/mrr_ingestion.json').write_text(json.dumps({'source':'https://zenodo.org/records/3929289','license':'CC BY 4.0','files':rows},ensure_ascii=False,indent=2))
    print(json.dumps(rows,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
