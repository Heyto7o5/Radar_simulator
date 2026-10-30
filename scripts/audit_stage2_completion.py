#!/usr/bin/env python3
"""Requirement-level completion audit; no inference from a status badge alone."""
import argparse,hashlib,importlib.metadata,json,platform,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('audit already exists')
    base=ROOT/'.runtime/stage2_20261023';evidence={};all_rows=[]
    for name,expected in [('campaign_v1',45),('scan_campaign_v1',6)]:
        folder=base/name;state=json.loads((folder/'campaign_status.json').read_text())
        frozen=json.loads((folder/'campaign_inputs.json').read_text())
        assert state['status']=='complete' and state['completed_cases']==expected and len(state['results'])==expected
        expected_ids={r['case'] for r in frozen['plan']['cases']}
        assert {r['case'] for r in state['results']}==expected_ids
        current={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'backend/radar_platform').rglob('*.py')}
        current['scripts/run_scene.py']=hashlib.sha256((ROOT/'scripts/run_scene.py').read_bytes()).hexdigest()
        assert current==frozen['source_sha256'],'solver source drift'
        for row in state['results']:
            assert row['status']=='complete'
            run=folder/row['case'];manifest=json.loads((run/'run.json').read_text())
            assert manifest['status']=='complete' and manifest['source_sha256']==current
            r=json.loads((run/'request.json').read_text());baseline=json.loads((ROOT/f'.runtime/real_dem_20261022/{row["site"]}_30m_10ghz.json').read_text())
            for key in ['north_m','east_m','elevation_up_m']:
                assert r['scene']['terrain'][key]==baseline['scene']['terrain'][key],'real geometry changed'
            assert abs(np.linalg.norm(r['platform']['velocity_ned_mps'])-1200)<1e-6
            assert r['waveform']['carrier_hz']==10e9
            assert all(t['evidence']=='assumed' for t in r['scene']['terrain']['land_scattering']['tables'])
            m=json.loads((run/'frame_0000/metadata.json').read_text())
            assert m['is_real'] is False
            assert m['array']['physical_tx_element_count']==16 and m['array']['logical_tx_channels']==1
            for product,spec in m['products'].items():
                file=run/'frame_0000'/spec['file'];dtype=np.dtype(spec['dtype'])
                assert file.stat().st_size==spec['bytes']==int(np.prod(spec['shape']))*dtype.itemsize
                array=np.memmap(file,dtype=dtype,mode='r',shape=tuple(spec['shape']))
                assert np.isfinite(array).all(),product
            all_rows.append(row['case'])
        evidence[name]={'complete_cases':expected,'all_native_products_finite':True,
                        'real_geometry_unchanged':True,'solver_source_matches_current':True}
    inputs=json.loads((base/'config_audit.json').read_text());assert inputs['all_declared_only'] and len(inputs['cases'])==45
    evidence['paired_input_changes_verified']=True
    cover=[]
    for site in ['alpine','valley','foothill']:
        r=json.loads((base/'coverage_audit'/f'{site}_t20ms.json').read_text())
        assert r['non_tangent_count_mismatches']==0 and r['checked_scatterers']==1024 and r['checked_slots']==32
        cover.append({k:r[k] for k in ['ray_evaluations','non_tangent_count_mismatches','near_tangent_scatterers','reference_blocked_ray_evaluations']})
    evidence['independent_height_grid_checks']=cover
    regressions=json.loads((base/'regression.json').read_text())
    assert len(regressions)==11 and all(r['bitwise_equal'] for r in regressions)
    evidence['unchanged_historical_adc_cases']=11
    suites=ET.parse(base/'tests_validation.xml').getroot()
    tests=sum(int(s.attrib['tests']) for s in suites.iter('testsuite'))
    assert tests>=368 and all(int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0 for s in suites.iter('testsuite'))
    evidence['automated_tests_passed']=tests
    summary=json.loads((a.results/'summary.json').read_text())
    assert summary['main_runs']==45 and summary['paired_scan_runs']==6 and summary['measured_radar_fidelity_claimed'] is False
    html=(a.results/'report.html').read_text();assert html.count('data:image/png;base64,')==5
    evidence['self_contained_report']=str(a.results/'report.html')
    data=Path('/nas/datasets/hyt/datasets/radar_open')
    sources={}
    for dataset in ['copernicus_glo30','worldcover_2021']:
        manifest=json.loads((data/dataset/'manifest.json').read_text())
        sources[dataset]=[]
        for entry in manifest['files']:
            file=data/dataset/entry['file'];actual=hashlib.sha256(file.read_bytes()).hexdigest()
            assert actual==entry['sha256'],'source data changed'
            sources[dataset].append({'file':entry['file'],'sha256':actual})
    evidence['source_data_files_verified']=sources
    evidence['versions_observed_at_audit']={'python':platform.python_version(),**{x:importlib.metadata.version(x) for x in ['numpy','scipy','pydantic','rasterio','pyproj','matplotlib']}}
    report={'status':'complete_for_requested_functional_scope','evidence':evidence,
            'not_claimed':['Measured X-band material or radar fidelity','Absolute radar power calibration','Full-wave or receiver nonideality modeling',
                           'Per-cell ground truth from remote land-cover classification','Real-time performance for this full-resolution CPU campaign'],
            'execution_notes':'Interrupted attempts were retained under .interrupted names and rerun with unchanged frozen inputs; they are not counted as extra independent scenes.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(all_rows),'tests':tests,'report':evidence['self_contained_report']},ensure_ascii=False))


if __name__=='__main__':main()
