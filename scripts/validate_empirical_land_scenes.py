#!/usr/bin/env python3
"""Measured coefficients -> synthetic controlled ground scene -> RD, with scope kept explicit."""
import argparse,base64,csv,hashlib,itertools,json,shutil,zipfile
from dataclasses import asdict,replace
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from radar_platform.ground_scattering import load_maqu_sigma0,EmpiricalBackscatterCurve
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.storage import save_frame
from radar_platform.comparison import power_db
from render_scene import configure_font

P=Path(__file__).resolve().parents[1]
DATA=Path('/nas/datasets/hyt/datasets/radar_open')
SOURCE=DATA/'maqu_scatterometer/raw/original/Scatterometer_Angular_Vars/sigma0_20170825/sigma0_vv_alpha0_azip00_9000-9875.dat'
OUT=P/'docs/land_extension/2026-10-15'
CACHE=DATA/'layered_validation/land_scene_20261015'

def load(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def make_request(kind,mode,speed,tx,rx,seed):
    obs=load_maqu_sigma0(SOURCE)
    if mode=='constant':
        average=float(np.mean([o.sigma0_db for o in obs]))
        obs=[replace(o,sigma0_db=average,source='derived_constant_training_mean:'+o.source) for o in obs]
    w=({'kind':kind,'start_hz':9.4e9,'slope_hz_s':4e12,'ramp_s':40e-6,'slot_period_s':1e-3,'adc_start_s':2e-6,
        'sample_rate_hz':4e6,'samples':128,'loops':64} if kind=='fmcw' else
       {'kind':kind,'carrier_hz':9.4e9,'bandwidth_hz':80e6,'sample_rate_hz':200e6,'pulse_width_s':.5e-6,
        'slot_period_s':1e-3,'adc_start_s':0.,'samples':256,'loops':64})
    return Simulation.model_validate({'name':f'controlled_ground_{kind}_{mode}_{speed}_{tx}Tx{rx}Rx',
        'waveform':w,'platform':{'position_ned_m':[0,0,-6],'velocity_ned_mps':[speed,0,0],
            'boresight_ned':[1,0,1],'right_hint_ned':[0,1,0],'beam_half_angle_deg':30.},
        'array':{'layout':'siso' if tx==rx==1 else 'ula','num_tx':tx,'num_rx':rx},
        'scene':{'terrain':{'kind':'flat','north_bounds_m':[5,11],'east_bounds_m':[-2,2],
            'spacing_m':1.,'relative_reflectivity':1.,'land_scattering':{
                'observations':[asdict(o) for o in obs],'mode':'interpolate' if mode=='constant' else mode,
                'polarization':'vv','land_cover':'alpine_meadow'}},'seed':seed,'amplitude_reference_range_m':10.},
        'processing':{'range_limits_m':[3,20],'azimuth_deg':list(np.arange(-30,31,2)) if rx>1 else [0.],'elevation_deg':[0.]},
        'sequence':{'frame_count':3,'frame_interval_s':.2}})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    for p in [OUT/'images',OUT/'configs',OUT/'metrics',CACHE]:p.mkdir(parents=True,exist_ok=True)
    configure_font()
    cases=list(itertools.product(['pulsed_lfm','fmcw'],['constant','interpolate','cosine_power'],[-1.,0.,1.],[(1,1),(2,4)],[42,43,44]))
    dump(OUT/'metrics/protocol.json',{'cases':cases,'selection_before_scoring':True,'frames_per_case':3,
        'coefficient_source':str(SOURCE),'coefficient_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'geometry_source':'synthetic controlled planar patch, NOT a measured Maqu DEM',
        'receiver_gain':'relative scale 1, reference range 10 m; not absolute instrument calibration',
        'random_phases':'world-fixed synthetic independent surface-cell phases, not measured roughness',
        'validation_scope':'coefficient transfer audited separately; these RD checks are numerical/kinematic only',
        'peak_test':'global RD maximum within geometry support plus 2 distance-resolution units and 1 native Doppler bin',
        'physical_limits':['footprint-average sigma0 treated as local coefficient','near-monostatic approximation <=5 degrees',
                           'no measured terrain/vegetation phase or correlation map','no receiver duplexing/leakage model'],
        'display_selection':'FMCW and pulsed LFM, 1 m/s, 1Tx1Rx, seed 42, first frame'})
    rows=[];selected={}
    if args.render_only:
        from types import SimpleNamespace
        with (OUT/'metrics/cases.csv').open(encoding='utf-8-sig') as h:
            for row in csv.DictReader(h):
                rows.append({k:(v=='True' if k=='passed' else v if k in ('waveform','model') else float(v)) for k,v in row.items()})
        for kind,mode in itertools.product(['pulsed_lfm','fmcw'],['constant','interpolate','cosine_power']):
            folder=CACHE/f'{kind}_{mode}/frame_0000';meta=json.loads((folder/'metadata.json').read_text())
            info=meta['products']['rd_power']
            selected[(kind,mode)]=SimpleNamespace(rd_power=np.fromfile(folder/info['file'],dtype=info['dtype']).reshape(info['shape']),
                range_m=np.array(meta['axes']['range_m']),velocity_mps=np.array(meta['axes']['velocity_mps']))
    for number,(kind,mode,speed,(tx,rx),seed) in enumerate([] if args.render_only else cases):
        r=make_request(kind,mode,speed,tx,rx,seed);scene=compile_scene(r.scene)
        if seed==42:dump(OUT/'configs'/f'case_{number:03d}.json',r.model_dump(mode='json'))
        for index in range(3):
            f=simulate_frame(r,index,compiled=scene);p=f.products
            t=f.metadata['truth_epoch_s'];radar=np.array(r.platform.position_ned_m)+np.array(r.platform.velocity_ned_mps)*t
            delta=scene.positions_ned_m-radar;radius=np.linalg.norm(delta,axis=1)
            velocity=delta@np.array(r.platform.velocity_ned_mps)/radius
            i,j=np.unravel_index(np.argmax(p.rd_power),p.rd_power.shape)
            dr=p.metadata['range_resolution_m'];dv=p.metadata['native_velocity_cell_mps']
            passed=bool(radius.min()-2*dr<=p.range_m[i]<=radius.max()+2*dr and velocity.min()-dv<=p.velocity_mps[j]<=velocity.max()+dv)
            rows.append({'case':number,'waveform':kind,'model':mode,'radar_speed_mps':speed,'tx':tx,'rx':rx,'seed':seed,'frame':index,
                'range_min_m':float(radius.min()),'range_max_m':float(radius.max()),'closing_min_mps':float(velocity.min()),'closing_max_mps':float(velocity.max()),
                'global_peak_range_m':float(p.range_m[i]),'global_peak_closing_mps':float(p.velocity_mps[j]),
                'RD_mean_power_relative':float(p.rd_power.mean()),'ground_source_objects':f.metadata['scene']['empirical_ground_objects'],'passed':passed})
            if speed==1 and tx==rx==1 and seed==42 and index==0:
                selected[(kind,mode)]=p
                save_frame(CACHE/f'{kind}_{mode}',f,True)
        if number%12==0:print('land scene',number+1,'/',len(cases),flush=True)
    csvout(OUT/'metrics/cases.csv',rows)
    result={'configurations_including_seeds':len(cases),'independent_geometry':1,'frames':len(rows),'passed':sum(r['passed'] for r in rows),
            'real_ground_RD_records':0,'has_measured_coefficient_source':True,'grid_single_realization_convergence_claimed':False}
    dump(OUT/'metrics/summary.json',result)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,kind in enumerate(['pulsed_lfm','fmcw']):
        reference=max(float(selected[(kind,m)].rd_power.max()) for m in ['constant','interpolate','cosine_power'])
        for col,(mode,title) in enumerate([('constant','固定训练均值'),('interpolate','实测角度插值'),('cosine_power','余弦幂经验曲线')]):
            p=selected[(kind,mode)];ax=axes[row,col]
            im=ax.pcolormesh(p.range_m,p.velocity_mps,power_db(p.rd_power,reference).T,shading='auto',cmap='turbo',vmin=-45,vmax=0,rasterized=True)
            ax.set(xlabel='距离 / m',ylabel='接近速度 / (m/s)',ylim=(-2,2),title=f'{kind} · {title}')
            fig.colorbar(im,ax=ax,label='同一行共同参考 / dB',shrink=.8)
    fig.suptitle('实测散射系数进入面元回波计算：几何与随机相位相同，仅改变角度强度模型\n控制场景的模拟 RD，不是玛曲实测 RD；两种波形不要求逐像素一致',fontsize=12)
    fig.savefig(OUT/'images/ground_rd.png',dpi=155);plt.close(fig)
    r=make_request('fmcw','interpolate',1,1,1,42);s=compile_scene(r.scene)
    obs=load_maqu_sigma0(SOURCE);curve=EmpiricalBackscatterCurve(tuple(obs))
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    axes[0].plot([0,12],[0,0],c='#886c45');axes[0].plot([5,11],[0,0],lw=5,c='#619144');axes[0].scatter(0,6,marker='v',s=70)
    axes[0].arrow(0,6,1,0,head_width=.18,color='#29698b',length_includes_head=True)
    axes[0].plot([0,6],[6,0],'--',c='gray');axes[0].set(xlim=(-1,13),ylim=(-1,7),xlabel='North / m',ylabel='高度 / m',title='合成控制场景：固定地面，雷达水平 1 m/s')
    axes[1].plot([o.incidence_center_deg for o in obs],[o.sigma0_db for o in obs],'o-',label='玛曲 2017-08-25 VV')
    angles=np.linspace(34,66,100);smooth=EmpiricalBackscatterCurve(tuple(obs),'cosine_power')
    axes[1].plot(angles,smooth.predict_db(9.4e9,angles,'vv'),label='余弦幂经验拟合')
    axes[1].set(xlabel='足迹平均入射角 / °',ylabel='散射系数 / dB',title='真实输入系数：9–9.875 GHz');axes[1].legend(fontsize=8)
    delta=s.positions_ned_m-np.array(r.platform.position_ned_m);dist=np.linalg.norm(delta,axis=1)
    inc=np.degrees(np.arccos(-np.sum(delta*s.normals_ned,axis=1)/dist))
    sigma=curve.predict_db(r.waveform.carrier_hz,inc,'vv')
    im=axes[2].scatter(s.positions_ned_m[:,0],s.positions_ned_m[:,1],c=sigma,cmap='viridis',marker='s',s=70)
    fig.colorbar(im,ax=axes[2],label='本地赋值 σ0 / dB',shrink=.8)
    axes[2].set(xlabel='North / m',ylabel='East / m',title='面元采用的系数（局部近似，非实测地图）');axes[2].set_aspect('equal')
    fig.suptitle('溯源边界：只有中图的系数来自实测；左、右图是明确声明的数值场景',fontsize=12)
    fig.savefig(OUT/'images/ground_input.png',dpi=155);plt.close(fig)
    old=load(P/'docs/background_extension/2026-10-15/metrics/ground_summary.json')
    shutil.copy2(P/'docs/background_extension/2026-10-15/metrics/ground_transfer.csv',OUT/'metrics/coefficient_transfer.csv')
    dump(OUT/'metrics/coefficient_validation_summary.json',old)
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/'images'/name).read_bytes()).decode()+'">'
    def table(head,data):return '<table><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in data)+'</table>'
    body='<h1>经验地表系数 → 面元 → RD</h1><p>2026-10-15 · 通用地面雷达控制实验；没有用合成地形冒充实测场景。</p>'+image('ground_input.png')
    body+=table(['原来','本次新增','仍不能证明'],[
        ['经验系数只在独立接口返回数值','可以直接在地形、平板或显式网格上选择 land_scattering，并影响复数回波','没有配套实测地形 RD，不能宣称已复现真实草地纹理'],
        ['面元强度统一、随几何余弦衰减','按实际视线角查询实测覆盖内的系数；不重复乘余弦','足迹平均系数局部化是近似，不是逐面元实测 RCS'],
        ['只检查是否生成图片',f"{len(rows)} 个数值帧全部保留，{result['passed']} 个满足预定运动/几何范围检查",'种子、相邻帧和参数组合不算独立真实场景']])
    body+=image('ground_rd.png')
    body+='<h2>新增防错与检查</h2><p>整个发射频带必须落入测量频带；极化、地类必须匹配；任何被照亮、朝向雷达的面元超出角度覆盖会报错，不静默夹紧或删除。大的双站角不按单站系数处理。金属导体 PO 与草地扩散系数不能同时套用。</p><p>合成检查包括：系数加 10 dB 对应回波功率加 10 dB；细分网格后面积散射权重守恒；256 个相位种子的平均功率与单元功率和一致；固定世界坐标与跨帧相位连续。随机相位没有实测粗糙度依据，不能要求网格改变后单次散斑逐像素不变。</p>'
    body+='<h2>实测证据仍单独列出</h2>'+table(['模型','有效 / 全部跨日比较','散射系数 MAE','固定均值基线'],[
        [g['mode'],f"{g['comparisons']} / {g['total_comparisons']}",f"{g['mean_sigma0_mae_db']:.2f} dB",f"{g['constant_training_mean_mae_db']:.2f} dB"] for g in old['groups']])
    body+='<p>上述系数结果沿用上一阶段实测复核，不伪装成新动态 RD 对齐。部分 L 波段交叉极化记录存在相同平均角、不同足迹，已列为不支持并保留明细。这里使用资料完整的 X 波段 VV 子集演示接入。</p>'
    body+='<h2>使用与边界</h2><p>通过已有 /api/v1/scene-simulations 提交 configs 中的请求。land_scattering 可附在 terrain 或地面网格上；不开启时旧模型不变。relative_reflectivity 和参考距离仍是声明的相对标度，不是经过仪器功率标定的绝对回波。</p><p>该控制平面不是旧工程 DEM，也不是玛曲实测地形。把实测足迹系数用于局部网格，是清楚标注的经验近似；真实地物类别、空间相关性、土壤湿度、植被结构、遮挡和多径仍未完整重建。近垂直等超出观测范围的场景不会自动获得该数据集的保真背书。</p>'
    if (OUT/'metrics/mrr_coordinate_event_audit.json').exists():
        mrr=json.loads((OUT/'metrics/mrr_coordinate_event_audit.json').read_text())
        body+='<h2>新增降水资料：坐标和事件口径复核</h2>'+image('mrr_coordinates_events.png')
        body+=table(['文件','实测谱记录数','事件分类计数','速度坐标'],[
            [Path(x['file']).name,x['records'],str(x['event_counts_all_records']),f"0–{x['native_velocity_mps'][1]:.3f} m/s（仪器表观值）"] for x in mrr['files']])
        body+='<p>仪器原生坐标约定由厂家说明和原始处理论文核对，格点间距使用文件自带值，不照搬其他设备的默认值。这只是坐标解释，不是解模糊，也不是消除了垂直风、船体运动的雨滴真实速度。事件表按明确的 UTC 假设关联；未标注不等于无降水。源文件有一条分隔符异常的事件记录，保留原文并列入异常，不擅自修复成真值。降水前向散射验证尚未完成。</p>'
        body+='<p>MRR 依据：<a href="https://metek.de/faqs/">METEK 厂家说明</a> · <a href="https://amt.copernicus.org/articles/5/2661/2012/">Maahn 与 Kollias 的原始处理论文</a>。</p>'
    body+='<p>来源：<a href="https://essd.copernicus.org/articles/13/2819/2021/">玛曲数据论文</a>；原始系数路径、哈希与完整请求保存在包内。旧工程和原始数据保持不变。</p>'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>经验地表RD链路</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.65 system-ui;color:#24374a}img{width:100%;margin:18px 0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd5df;padding:10px;overflow-wrap:anywhere}th{background:#eaf2f5}h2{margin-top:32px}</style>'+body+'</html>'
    (OUT/'经验地表RD链路.html').write_text(doc)
    files=[*P.glob('backend/radar_platform/unified/*.py'),P/'backend/radar_platform/ground_scattering.py',
           P/'backend/radar_platform/mrr_dataset.py',P/'tests/test_land_scene.py',P/'tests/test_mrr_dataset.py',
           P/'scripts/audit_mrr_coordinates_events.py',Path(__file__)]
    dump(OUT/'metrics/provenance.json',{'source_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'cache':str(CACHE),'real_dynamic_ground_truth_available':False})
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(P))
    with zipfile.ZipFile(OUT.parent/'经验地表RD链路_2026-10-15.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if result['passed']!=len(rows):raise SystemExit('some prespecified checks failed; results retained')

if __name__=='__main__':main()
