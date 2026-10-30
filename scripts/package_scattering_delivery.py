#!/usr/bin/env python3
"""Evidence audit and consolidated delivery for the three requested extensions.

Completion here means implemented extensions and bounded validation/exploration,
not universal scattering fidelity or a claim of measured dynamic ground truth.
"""
import base64,csv,hashlib,json,shutil,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
import numpy as np
from radar_platform.api import app
from radar_platform.unified.schema import Simulation
from radar_platform.background_api import BackgroundRequest
from radar_platform.ground_api import GroundResponseRequest

P=Path(__file__).resolve().parents[1]
OUT=P/'docs/deliveries/2026-10-15'
ROOT=Path('/nas/datasets/hyt/datasets/radar_open')
STAGES={
    'face':P/'docs/surface_extension/2026-10-15_refined',
    'background':P/'docs/background_extension/2026-10-15',
    'composite':P/'docs/composite_extension/2026-10-15_floor_checked',
    'land':P/'docs/land_extension/2026-10-15'}

def load(p):return json.loads(p.read_text())
def rows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    for p in [OUT/'details',OUT/'images',OUT/'metrics',OUT/'configs']:p.mkdir(parents=True,exist_ok=True)
    face=load(STAGES['face']/'metrics/dynamic_summary.json');land=load(STAGES['land']/'metrics/summary.json')
    composite=load(STAGES['composite']/'metrics/laboratory_summary.json')
    background=load(STAGES['background']/'metrics/summary.json')
    ground=load(STAGES['background']/'metrics/ground_summary.json')
    ramp=load(STAGES['composite']/'metrics/ramp_summary.json')
    transfer=load(STAGES['composite']/'metrics/real_processing_compatibility.json')
    plate=rows(STAGES['face']/'metrics/maqu_static.csv')
    assert face['passed']==face['cases']==36 and face['max_quadrature_relative_error']<.002
    assert land['passed']==land['frames']==324 and land['real_ground_RD_records']==0
    assert composite['passed']==composite['frames']==36 and composite['max_saved_stage_replay_relative_error']==0
    assert transfer['passed']==transfer['frames']==540 and transfer['sequences']==8
    assert len(plate)==6 and len({r['case'] for r in plate})==3
    # This real-data failure MUST remain visible, not fitted away or filtered out.
    cross=[float(r['peak_error_m']) for r in plate if r['case'].startswith('next_day')]
    assert all(.4<x<.55 for x in cross)
    br=rows(STAGES['background']/'metrics/window_metrics.csv')
    assert len(br)==14*4*3*8 and background['record_count']==14
    assert len({(x['record_id'],x['model'],x['seed'],x['window']) for x in br})==len(br)
    for m in background['models']:
        values=[float(r['mae_db']) for r in br if r['model']==m['model']]
        assert np.isclose(np.mean(values),m['mean_rd_mae_db']['mean'])
    for g in ground['groups']:assert g['comparisons']==42 and g['total_comparisons']==48 and g['unsupported']==6
    cr=rows(STAGES['composite']/'metrics/ramp_frames.csv')
    assert len(cr)==180*3*3 and len({x['frame_id'] for x in cr})==180 and len({x['sequence'] for x in cr})==3
    mrr=load(ROOT/'mrr_ace/manifest.json')
    spectra=[x for x in mrr['files'] if 'source_member' in x]
    assert len(spectra)==3 and not mrr['errors']
    for f in spectra:assert sha(ROOT/'mrr_ace'/f['file'])==f['sha256']
    events=load(STAGES['land']/'metrics/mrr_coordinate_event_audit.json')
    assert len(events['malformed_source_rows_retained'])==1
    suites=list(ET.parse(P/'.runtime/land_scene_20261015_tests.xml').iter('testsuite'))
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==0 for s in suites)
    test_count=sum(int(s.attrib['tests'])-int(s.attrib.get('skipped',0)) for s in suites)
    assert test_count>=234
    paths=app.openapi()['paths']
    for endpoint in ['/api/v1/scene-simulations','/api/v1/background-simulations','/api/v1/scattering/ground-response']:
        assert 'post' in paths[endpoint]
    validated=0
    for stage,folder in STAGES.items():
        for f in (folder/'configs').glob('*.json'):
            data=load(f)
            if 'waveform' in data and 'scene' in data:Simulation.model_validate(data);validated+=1
            elif 'observations' in data and 'patch_area_m2' in data:GroundResponseRequest.model_validate(data);validated+=1
            elif 'model' in data and data['model'].get('domain')=='range_gated_iq':BackgroundRequest.model_validate(data);validated+=1
    source_files=list((P/'backend/radar_platform/unified').glob('*.py'))+[
        P/'backend/radar_platform/statistical_background.py',P/'backend/radar_platform/ground_scattering.py',P/'backend/radar_platform/mrr_dataset.py']
    audit={
        'date':'2026-10-15','scope':'requested feature expansion and transparent validation/exploration, not universal high-fidelity certification',
        'goal_completion_proven':False,
        'requirements':[
            {'requirement':'extend point simulations to finite surface RD and evaluate with Maqu',
             'status':'implemented; real validation covers static response, not full dynamic RD',
             'implementation':'explicit scalar PO area integration in pulsed/FMCW dynamic chain; API and complex products',
             'evidence':['face/metrics/dynamic_summary.json','face/metrics/maqu_static.csv','tests.xml'],
             'boundary':'Maqu tests static frequency/range response; dynamic surface RD has analytic/numerical evidence, not measured dynamic ground truth'},
            {'requirement':'increase scattering models and collect additional categories of open data',
             'status':'multiple models implemented and data acquired; additional candidate/precipitation forward validation remains partial',
             'implementation':'empirical angle-dependent diffuse land cells; correlated Gaussian and Gamma-textured range-gated backgrounds; MRR spectra adapter and acquired samples',
             'evidence':['land/metrics/cases.csv','background/metrics/ground_summary.json','background/metrics/summary.json','mrr_manifest.json','data_catalog.json'],
             'boundary':'some sources are candidates only; cloud/rain forward electromagnetic model not implemented; source coverage cannot be extrapolated'},
            {'requirement':'evaluate Gaussian noise and alternative backgrounds for realism with verifiable metrics',
             'status':'implemented and conditionally evaluated; broader scene fidelity not proven',
             'implementation':'frozen prefix-to-tail IPIX controls; frozen RAMP receiver-noise controls; stage-correct composition with replayable complex components and duplicate-floor guard',
             'evidence':['background/metrics/record_metrics.csv','composite/metrics/ramp_summary.json','composite/metrics/laboratory_summary.json','tests.xml'],
             'boundary':'conditional statistical improvements, not absolute receiver gain/material calibration; target and spatial-correlation errors remain'}],
        'tests_passed':test_count,'saved_request_configs_validated':validated,
        'source_sha256':{str(p.relative_to(P)):sha(p) for p in source_files},
        'real_data_fidelity_globally_passed':False,
        'preserved_failures':['Maqu cross-session range mismatch ~0.46 m','initial PO quadrature 35/36 before refinement',
            '12 ground model-comparisons unsupported rather than silently dropped','compound texture spatial correlation worse than correlated Gaussian',
            'RAMP strong-return MAE remains ~19.65 dB','one malformed MRR event row retained'],
        'no_old_project_or_raw_source_modifications':True,'frontend_started':False}
    dump(OUT/'metrics/requirement_audit.json',audit)
    shutil.copy2(P/'.runtime/land_scene_20261015_tests.xml',OUT/'metrics/tests.xml')
    shutil.copy2(P/'configs/scattering_data_catalog.json',OUT/'metrics/data_catalog.json')
    shutil.copy2(ROOT/'mrr_ace/manifest.json',OUT/'metrics/mrr_manifest.json')
    for name,folder in STAGES.items():
        shutil.copy2(next(folder.glob('*.html')),OUT/'details'/f'{name}.html')
        for sub in ['metrics','configs']:
            dest=OUT/name/sub;dest.mkdir(parents=True,exist_ok=True)
            for f in (folder/sub).iterdir():
                if f.is_file():shutil.copy2(f,dest/f.name)
    image_files={'face':STAGES['face']/'images/dynamic_surface_rd.png',
        'ground':STAGES['land']/'images/ground_input.png',
        'background':STAGES['background']/'images/background_metrics.png',
        'composite':STAGES['composite']/'images/components_fmcw.png'}
    for name,p in image_files.items():shutil.copy2(p,OUT/'images'/f'{name}.png')
    def img(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/'images'/f'{name}.png').read_bytes()).decode()+'">'
    def table(h,rs):return '<div class="table"><table><tr>'+''.join('<th>'+x+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rs)+'</table></div>'
    body='<header><small>2026-10-15 · 本轮扩展交付总览</small><h1>面目标、散射模型与真实背景</h1><p>已交付功能扩展、公开数据对照和可复现输入。<br><b>不是“所有场景已高保真”的认证。</b></p></header>'
    body+=table(['需求','交付内容','证据边界'],[
        ['点 → 面的 RD','有限面积积分、两种波形、运动与阵列；玛曲静态实测复核','动态 RD 解析验证与静态实测分开，不伪造玛曲动态真值'],
        ['更多散射类型与数据','金属面、经验地表、相关/复合背景；新增实际取得的降水谱','经验系数受频段/角度/极化约束；部分其他资料仅为候选'],
        ['背景是否更真实','白噪声与相关背景对照；噪声、目标、背景分层接入','报告强回波、时间变化及空间相关的改善与退化，不只看全图 MAE']])
    body+='<section><h2>1. 面目标 RD 已接入</h2>'+img('face')
    body+=table(['检查','结果'],[['动态金属板解析/收敛检查','36/36；初轮低阶积分失败保留，未放宽阈值'],
        ['玛曲同日板测量','部分距离主峰接近实测'],['玛曲跨日板测量','约 0.46 m 偏差仍未解决'],
        ['地表系数进入面元 RD','324/324 数值帧满足声明的几何/运动检查；不是 324 个独立实测场景']])
    body+='<p><a href="details/face.html">面目标完整报告</a> · <a href="details/land.html">经验地表 RD 与新增降水资料报告</a></p></section>'
    body+='<section><h2>2. 地表输入与仿真假设分开</h2>'+img('ground')
    body+='<p>实测的是足迹平均散射系数。把它赋给局部面元是一种明确的经验近似；相位、控制平面和接收机增益不是实测三维场景。超出资料覆盖直接报错，不用厘米波近垂直场景替代已测角度。</p>'
    body+=table(['模型 / 数据','当前状态'],[['草地角度插值 / 余弦幂','可用于受约束的地面面元，也提供独立系数接口'],
        ['IPIX 海杂波','14 条实测记录，前段校准、后段评价，保留旧模型'],
        ['ACE MRR-2 降水','3 个原始谱文件已取得；解析坐标与事件，未声称恢复真实雨滴速度或几何'],
        ['云雷达、积雪、恶劣天气资料','已记录来源与缺口；未下载/未验证者明确标注']])+'</section>'
    body+='<section><h2>3. 高斯噪声并不等于完整真实背景</h2>'+img('background')
    body+=table(['IPIX 模型（同口径 14 条记录）','平均 RD MAE'],[
        ['独立高斯','15.50 dB'],['相关高斯','2.61 dB'],['相关场 + 随机起伏 + 独立底噪','2.69 dB'],['旧模型','3.50 dB']])
    body+='<p>复合模型的 MAE 不是最低，但强亮点与时间起伏更接近实测；空间相关性仍有不足。RAMP 中加噪后全图 MAE 约 2.59 dB，而只有噪声也约 2.61 dB，强回波区域仍约 19.65 dB，因此不能把总分降低等同于车体复现。</p><p><a href="details/background.html">背景与地表完整报告</a></p></section>'
    body+='<section><h2>4. 统一处理链能逐项重放</h2>'+img('composite')
    body+='<p>目标生成原始 ADC；接收机噪声在 ADC 层加入；统计背景在距离处理后加入；随后统一做 Doppler/角度处理。保留复数分量，禁止直接叠加 dB 图。背景配方已有的接收机式底噪与 ADC 噪声不能重复计算。</p>'
    body+=table(['验收项目','结果'],[['组合场景分量一致性','36/36；保存数据重放 RD 的相对差为 0'],
        ['真实 ADC 处理兼容性','540/540 帧；这不是实测保真通过率'],['当前全套自动化检查',f'{test_count} 项通过'],
        ['可直接提交的请求配置',f'{validated} 份完成当前接口校验']])
    body+='<p><a href="details/composite.html">组合链路完整报告</a></p></section>'
    body+='<section><h2>下一阶段仍需解决的保真问题</h2>'+table(['问题','不能夸大的结论'],[
        ['真实动态面目标缺少配套观测','当前没有完成真实动态面 RD 的全链路保真证明'],
        ['材料、遮挡、多径、方向图与硬件标定不完整','不宣称绝对功率或完整场景数字孪生'],
        ['实测足迹系数用于局部网格','不等于每个地形面元都有实测材料参数'],
        ['海况与场景变化','本轮条件统计预测不等于无需校准的跨场景迁移'],
        ['降水谱与雨滴物理','资料接入和坐标解释完成，不等于降水电磁模型已验证']])+'</section>'
    body+='<footer>交付包含详细报告、图像、请求配置、逐样本指标、来源与检查记录；原始实测数据仍在 NAS，不进入源码交付包。旧工程不改动，前端未启动。</footer>'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>雷达散射扩展总交付</title><style>body{max-width:1250px;margin:25px auto;padding:20px;font:16px/1.65 system-ui;color:#24364b}img{width:100%;margin:18px 0}h1{font-size:32px}h2{border-left:4px solid #278392;padding-left:12px;margin-top:30px}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #ccd5df;text-align:left}th{background:#eaf2f6}.table{overflow:auto}a{color:#176b94}footer,header{padding:18px;background:#f0f5f7}section{margin:28px 0}@media(max-width:700px){body{padding:10px}table{min-width:560px}}</style>'+body+'</html>'
    (OUT/'本轮扩展交付总览.html').write_text(doc)
    # Source only: no .env, raw data, runtime outputs, environments or node_modules.
    with zipfile.ZipFile(OUT/'代码与测试.zip','w',zipfile.ZIP_DEFLATED) as z:
        for top in ['backend','scripts','tests']:
            for f in (P/top).rglob('*.py'):z.write(f,f.relative_to(P))
        for f in (P/'scripts').glob('*.sh'):z.write(f,f.relative_to(P))
        for f in (P/'configs').glob('*.json'):z.write(f,f.relative_to(P))
        for top in ['frontend/src','frontend/public']:
            for f in (P/top).rglob('*'):
                if f.is_file():z.write(f,f.relative_to(P))
        for f in (P/'frontend').iterdir():
            if f.is_file() and f.suffix in ('.json','.ts','.html'):z.write(f,f.relative_to(P))
        z.write(P/'pyproject.toml','pyproject.toml')
        for name in ['README.md','.gitignore']:
            if (P/name).is_file():z.write(P/name,name)
    with zipfile.ZipFile(OUT.parent/'雷达散射扩展交付_2026-10-15.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in OUT.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(OUT))
    print(json.dumps({'evidence_checks_passed':True,'goal_completion_proven':False,'tests':test_count,'validated_configs':validated,'output':str(OUT)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
