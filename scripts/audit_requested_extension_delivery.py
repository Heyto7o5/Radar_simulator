#!/usr/bin/env python3
"""Audit the three requested extensions/evaluations, not universal fidelity."""
import csv,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT=Path(__file__).resolve().parents[1];DOC=PROJECT/'docs'
OUT=DOC/'deliveries/current_extension_evidence'

def read(path):return json.loads((DOC/path).read_text())

def main():
    world=DOC/'validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry'
    state=json.loads((world/'state.json').read_text());integrity=json.loads((world/'output_integrity_audit.json').read_text())
    if state['status']!='complete' or integrity['partial_audit'] or integrity['checked_completed_frames']!=11:raise ValueError('world comparison incomplete')
    rows=list(csv.DictReader((world/'per_frame_metrics.csv').open()))
    if len(rows)!=64 or len({r['valid_frame_index'] for r in rows})!=8:raise ValueError('world comparison rows missing')
    suite=ET.parse(PROJECT/'.runtime/extension_final_tests.xml').getroot()
    tests=sum(int(x.get('tests','0')) for x in suite.iter('testsuite'))
    failures=sum(int(x.get('failures','0'))+int(x.get('errors','0')) for x in suite.iter('testsuite'))
    if failures:raise ValueError('automated tests failed')
    surface=read('surface_extension/2026-10-15_refined/metrics/dynamic_summary.json')
    if surface['passed']!=surface['cases'] or surface['cases']!=36:raise ValueError('surface numerical evidence incomplete')
    with (DOC/'surface_extension/2026-10-15_refined/metrics/maqu_static.csv').open(encoding='utf-8-sig',newline='') as f:maqu=list(csv.DictReader(f))
    if not maqu or not any(abs(float(row['peak_error_m']))>.4 for row in maqu):raise ValueError('Maqu measured comparison or preserved failure missing')
    ground=read('background_extension/2026-10-15/metrics/ground_summary.json')
    sea=read('background_extension/2026-10-15/metrics/summary.json')
    if len(sea['protocol']['records'])!=14 or len(ground['groups'])<2:raise ValueError('land/sea evidence missing')
    colored=read('validation_diagnostics/2026-10-15/dart/colored_background_followup/summary.json')
    if colored['temporal_check_frames']!=8:raise ValueError('colored background comparison incomplete')
    datum=Path('/nas/datasets/hyt/datasets/radar_open/mrr_ace');manifest=json.loads((datum/'manifest.json').read_text())
    precipitation_files=[row for row in manifest['files'] if row['file'].endswith('.nc')]
    for row in precipitation_files:
        p=datum/row['file']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('precipitation sample missing or changed')
    if not precipitation_files:raise ValueError('additional measured category not acquired')
    required=['validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry/扩展世界几何RD对照.html',
              'validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry/images/frame_0072.png',
              'deliveries/DART扩展世界几何对照_2026-10-15.zip',
              'background_extension/2026-10-15/背景与地表散射验证.html',
              'surface_extension/2026-10-15_refined/面目标扩展与验证.html']
    for name in required:
        if not (DOC/name).is_file():raise FileNotFoundError(name)
    audit={'requested_extensions_and_evaluations_delivered':True,'universal_high_fidelity_certification':False,
           'requirements':[
               {'request':'扩展到面的 RD 仿真，并使用玛曲等数据验证',
                'evidence':['scalar PO and diffuse surface implementation','36 numerical surface cases','Maqu measured static comparisons with failures retained','DART measured RAD + LiDAR input, two 11-frame simulation batches and 8-frame evaluation'],
                'boundary':'Maqu validates static response, DART is one conditional development scene; no absolute or high-altitude fidelity claim'},
               {'request':'增加散射模型，搜集海面、地表等数据类别',
                'evidence':['finite-area scalar PO','empirical land sigma0 interpolation and cosine-power modes','IPIX correlated Gaussian and Gamma-textured statistical backgrounds','DART geometry/trajectory/real spectra and acquired MRR precipitation samples'],
                'boundary':'statistical background is distinct from electromagnetic scattering; rain/snow forward scattering not implemented or claimed'},
               {'request':'评估高斯噪声及其他背景方法，使改进可验证',
                'evidence':['14-record IPIX controls','DART white/noise-only/combined comparisons','calibration-only colored Gaussian follow-up','shared power references and fixed masks; complex ADC injection and replay'],
                'boundary':'conditional background improvement demonstrated; strong target and structural errors remain'}],
           'automated_tests':tests,'automated_failures':failures,'maqu_static_rows':len(maqu),
           'precipitation_sample_files':len(precipitation_files),'world_frames_integrity_checked':11,
           'important_unresolved_results':['Maqu cross-day distance bias about .46m','world geometry with old gain worsens primary MAE from 7.64 to 7.74dB',
                                           'world geometry after calibration-only gain refit yields primary 6.98dB and same-region 11.24dB; not full fidelity',
                                           'material coefficients, occlusion, general multipath, independent scenes and high-altitude measured certification remain future work'],
           'completion_meaning':'the requested implementation, data expansion and quantitative exploration are delivered; not the final high-fidelity radar-platform vision'}
    (OUT/'requested_delivery_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (OUT/'tests.xml').write_bytes((PROJECT/'.runtime/extension_final_tests.xml').read_bytes())
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
