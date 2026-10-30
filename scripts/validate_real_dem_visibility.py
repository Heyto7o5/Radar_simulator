#!/usr/bin/env python3
"""Fixed real DSM patches: independent LOS and resampling-error checks."""
import argparse,base64,csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.dem_import import import_copernicus_patch
from radar_platform.dem_los_reference import reference_los,height_at
from radar_platform.surface_scene import mesh_from_dem
from radar_platform.unified.visibility_bvh import TriangleBVH

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'configs/real_dem_validation_plan.json').read_text())
    data=Path(os.environ.get('RADAR_DATA_ROOT','/nas/datasets/hyt/datasets/radar_open'))/'copernicus_glo30'
    manifest=json.loads((data/'manifest.json').read_text());source=data/'raw'/plan['tile']/(plan['tile']+'.tif')
    patches=data/'processed'/'fixed_context_20261022';patches.mkdir(parents=True,exist_ok=True)
    rows=[];height_errors=[];all_metadata=[];figure,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    context_width=5760. # Covers all fixed observer projections plus ROI; never pad unavailable terrain.
    q=np.linspace(-plan['patch_width_m']/2,plan['patch_width_m']/2,17)
    qn,qe=np.meshgrid(q,q);query=np.c_[qn.ravel(),qe.ravel()]
    for k,location in enumerate(plan['locations']):
        native=None;native_masks={};native_heights=None
        for spacing in plan['comparison_grid_spacing_m']:
            patch=import_copernicus_patch(source,location['latitude_deg'],location['longitude_deg'],width_m=context_width,spacing_m=spacing)
            file=patches/f'{location["id"]}_{spacing}m_context.npz'
            if not file.exists():patch.save(file)
            else:
                with np.load(file,allow_pickle=False) as saved:
                    if not np.array_equal(saved['elevation_up_m'],patch.elevation_up_m):raise ValueError('existing context differs; preserve and version inputs')
            all_metadata.append(patch.metadata)
            mesh=mesh_from_dem(patch.north_m,patch.east_m,patch.elevation_up_m);tree=TriangleBVH(mesh.triangles)
            heights=height_at(patch.north_m,patch.east_m,patch.elevation_up_m,query[:,0],query[:,1])
            endpoints=np.c_[query,-heights]
            if spacing==30:
                native=patch;native_heights=heights
                mask=(abs(patch.north_m)<=960);e_mask=(abs(patch.east_m)<=960)
                surface=patch.elevation_up_m[np.ix_(e_mask,mask)]+patch.metadata['origin_orthometric_height_m']
                im=axes[k].imshow(surface.T,origin='lower',extent=[-960,960,-960,960],cmap='terrain',aspect='equal')
                axes[k].set(title=f'{location["id"]}: {location["latitude_deg"]:.2f}N {location["longitude_deg"]:.2f}E',xlabel='East (m)',ylabel='North (m)')
                figure.colorbar(im,ax=axes[k],label='EGM2008 height (m)',shrink=.8)
            fine_n,fine_e=np.meshgrid(native.north_m,native.east_m)
            roi=(abs(fine_n)<=960)&(abs(fine_e)<=960)
            interpolated=height_at(patch.north_m,patch.east_m,patch.elevation_up_m,fine_n[roi],fine_e[roi])
            error=interpolated-native.elevation_up_m[roi]
            height_errors.append(dict(location=location['id'],spacing_m=spacing,
                rmse_m=float(np.sqrt(np.mean(error**2))),p90_abs_m=float(np.quantile(abs(error),.9)),max_abs_m=float(abs(error).max())))
            for off in plan['off_nadir_angles_deg']:
                for az in plan['view_azimuth_deg']:
                    offset=plan['altitude_above_patch_center_m']*np.tan(np.deg2rad(off))
                    origin=np.array([-offset*np.cos(np.deg2rad(az)),-offset*np.sin(np.deg2rad(az)),-plan['altitude_above_patch_center_m']])
                    reference=reference_los(patch.north_m,patch.east_m,patch.elevation_up_m,origin,endpoints)
                    if not all(x['covered'] for x in reference):raise ValueError('occlusion context does not cover entire rays')
                    work={'aabb_tests':0,'triangle_tests':0}
                    blocked=tree.blocked(origin,endpoints,work=work,work_limit=100_000_000)
                    ambiguous=np.array([x['near_tangent'] for x in reference]);expected=np.array([x['blocked'] for x in reference])
                    mismatch=int(((blocked!=expected)&~ambiguous).sum())
                    if spacing==30:native_masks[(off,az)]=blocked
                    row=dict(location=location['id'],spacing_m=spacing,off_nadir_deg=off,azimuth_deg=az,
                        rays=len(endpoints),near_tangent=int(ambiguous.sum()),independent_mismatches=mismatch,
                        blocked=int(blocked.sum()),visible=int((~blocked).sum()),
                        native_resolution_mask_differences=int((blocked!=native_masks[(off,az)]).sum()),**work)
                    rows.append(row);print(json.dumps(row),flush=True)
                    np.savez_compressed(out/f'{location["id"]}_{spacing}m_off{off}_az{az}.npz',
                        observer_ned_m=origin,endpoints_ned_m=endpoints,bvh_blocked=blocked,
                        reference_blocked=expected,reference_near_tangent=ambiguous)
        print('Finished '+location['id'],flush=True)
    figure.suptitle('Real Copernicus GLO-30 DSM — fixed sites, no height exaggeration')
    figure.savefig(out/'real_elevation.png',dpi=160);plt.close(figure)
    for name,values in [('visibility.csv',rows),('height_resampling.csv',height_errors)]:
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    summary={'status':'pass' if not any(r['independent_mismatches'] for r in rows) else 'mismatch_requires_investigation',
        'patches':3,'parameter_combinations':len(rows),'unique_viewpoints_per_resolution':15,
        'rays_checked':sum(r['rays'] for r in rows),'independent_mismatches':sum(r['independent_mismatches'] for r in rows),
        'near_tangent_rays':sum(r['near_tangent'] for r in rows),'blocked_ray_evaluations':sum(r['blocked'] for r in rows),
        'context_width_m':context_width,'height_scaling':1,'source_manifest':manifest,
        'selection_plan':plan,'height_errors':height_errors,
        'limits':['Independent geometry cross-check, not independent measured radar truth',
                  'All-visible cases retained; site names are selection labels, not measured land-cover classes',
                  'Coarse-grid comparison re-elevates each fixed horizontal site onto its own terrain surface',
                  'Local flat projected geometry neglects curvature; vertical reference is orthometric, not ellipsoidal']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'input_metadata.json').write_text(json.dumps(all_metadata,ensure_ascii=False,indent=2))
    image=base64.b64encode((out/'real_elevation.png').read_bytes()).decode()
    table=''.join(f'<tr><td>{r["location"]}</td><td>{r["spacing_m"]}</td><td>{r["rmse_m"]:.3f}</td><td>{r["p90_abs_m"]:.3f}</td><td>{r["max_abs_m"]:.3f}</td></tr>' for r in height_errors)
    page=f'<html lang="zh"><meta charset="utf-8"><title>真实 DEM 几何验证</title><body style="max-width:1400px;margin:30px auto;font:16px sans-serif"><h1>真实 DEM：先验证几何，不制造遮挡效果</h1><p>固定三个地点、30/60/120 m 三种网格、5 km 相对中心高度、0/10/20° 离天底角。完整射线路径使用 5.76 km 上下文地形，不能把区域外当成无遮挡。无高度放大，无人为山脊。</p><img style="width:100%" src="data:image/png;base64,{image}"><p>独立栅格高度剖面与三角面 BVH 比较：{summary["rays_checked"]} 次射线评价，非切线不一致 {summary["independent_mismatches"]}，近切线待定 {summary["near_tangent_rays"]}，遮挡评价 {summary["blocked_ray_evaluations"]}。重复分辨率和方向不计作独立真实场景。</p><h2>降采样造成的高程差异</h2><table border="1" cellpadding="8"><tr><th>区域</th><th>网格 m</th><th>RMSE m</th><th>P90绝对误差 m</th><th>最大绝对误差 m</th></tr>{table}</table><p>这里是相对同源 30 m 重投影网格的差异，不是相对真实地面的测量精度。本数据为 DSM，不提供材料、粗糙度、散射系数或配套雷达回波。</p><p>{manifest["derived_attribution"]}</p><p><a href="https://registry.opendata.aws/copernicus-dem/">官方数据来源</a></p></body></html>'
    (out/'report.html').write_text(page)
    if summary['independent_mismatches']:raise SystemExit(1)


if __name__=='__main__':main()
