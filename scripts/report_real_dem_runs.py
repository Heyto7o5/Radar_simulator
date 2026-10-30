#!/usr/bin/env python3
"""Show real input geometry alongside explicitly synthetic RD, not a fake real/real pair."""
import argparse,base64,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    folder=ROOT/'.runtime/real_dem_20261022';items=[]
    for name in ['alpine','valley','foothill']:
        run=folder/f'{name}_30m_10ghz_run'
        manifest=json.loads((run/'run.json').read_text());assert manifest['status']=='complete'
        request=json.loads((run/'request.json').read_text());meta=json.loads((run/'frame_0000/metadata.json').read_text())
        spec=meta['products']['rd_power'];power=np.fromfile(run/'frame_0000'/spec['file'],dtype=spec['dtype']).reshape(spec['shape'])
        assert np.isfinite(power).all() and power.max()>0
        items.append((name,request,meta,power))
    reference=max(float(x[3].max()) for x in items)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    rows=[]
    for col,(name,r,m,power) in enumerate(items):
        t=r['scene']['terrain'];n=np.array(t['north_m']);e=np.array(t['east_m']);h=np.array(t['elevation_up_m'])
        ns=abs(n)<=960;es=abs(e)<=960
        image=axes[0,col].imshow(h[np.ix_(es,ns)].T,origin='lower',extent=[-975,975,-975,975],cmap='terrain',aspect='equal')
        axes[0,col].set(title=f'{name}: real DSM input',xlabel='East (m)',ylabel='North (m)')
        fig.colorbar(image,ax=axes[0,col],label='Height relative to center (m)',shrink=.8)
        ranges=np.array(m['axes']['range_m']);velocity=np.array(m['axes']['velocity_mps'])
        dr=ranges[1]-ranges[0];dv=velocity[1]-velocity[0]
        db=10*np.log10(np.maximum(power/reference,1e-6))
        im=axes[1,col].imshow(db.T,origin='lower',aspect='auto',extent=[ranges[0]-dr/2,ranges[-1]+dr/2,velocity[0]-dv/2,velocity[-1]+dv/2],vmin=-60,vmax=0,cmap='viridis')
        axes[1,col].set(title=f'{name}: simulated RD ONLY',xlabel='Slant range (m)',ylabel='Apparent closing velocity (m/s)')
        fig.colorbar(im,ax=axes[1,col],label='Relative power (shared reference, dB)')
        c=m['visibility']['coverage']
        row={'site':name,'scatterers':len(c['object_index']),'raw_adc_shape':m['products']['adc']['shape'],
             'illuminated_scatterer_slots':m['visibility']['illuminated_scatterer_slots'],
             'occluded_scatterer_channel_slots':m['visibility']['occluded_scatterer_channel_slots'],
             'bvh_work':m['visibility']['bvh_work'],'elapsed_s':m['elapsed_s'],'warnings':m['warnings'],
             'source_run':str((folder/f'{name}_30m_10ghz_run').relative_to(ROOT))}
        rows.append(row)
    fig.suptitle('Real DSM geometry → assumed relative surface scattering → synthetic 10 GHz RD')
    fig.savefig(out/'real_dem_and_simulated_rd.png',dpi=160);plt.close(fig)
    summary={'sites':rows,'carrier_hz':10e9,'platform_speed_mps':1200,'height_above_patch_center_m':5000,
        'projected_grid_spacing_m':30,'full_context_width_m':5760,'figure_geometry_crop_width_m':1920,
        'measured_rd_available':False,'absolute_power_calibrated':False,
        'power_display_reference':reference,'scattering':'uniform relative coefficient and seeded phase, not measured land-cover response'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    encoded=base64.b64encode((out/'real_dem_and_simulated_rd.png').read_bytes()).decode()
    page='<html lang="zh"><meta charset="utf-8"><title>真实地形输入与仿真 RD</title><body style="font:16px sans-serif;max-width:1500px;margin:30px auto"><h1>真实地形输入与仿真 RD</h1><p>三个预先固定的 Copernicus GLO-30 窗口，未改变地形高度。上排只放大显示中心 1.92 km 区域；回波与遮挡使用完整 5.76 km 上下文、30 m 投影网格，共 73728 个三角面。</p><p>10 GHz、1200 m/s、相对地形中心高度 5 km、离天底角 10°。单 PRF 存在 Doppler 模糊，纵轴明确为表观速度；没有降低真实速度或执行解模糊。三幅 RD 共用显示功率参考。</p><img style="width:100%" src="data:image/png;base64,'+encoded+'"><p><strong>真实的是地形几何，不是下排的雷达回波。</strong>下排由统一平台生成，反射系数与散射相位仍为假设；没有配套实测 RD，不能报告实测 MAE 或宣称厘米波地物保真。</p><p>produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved</p><p><a href="https://registry.opendata.aws/copernicus-dem/">官方来源</a></p></body></html>'
    (out/'report.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
