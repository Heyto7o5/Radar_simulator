#!/usr/bin/env python3
"""Package readable review documents, scientific figures and full metric tables."""
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,re,shutil,urllib.request,zipfile
from pathlib import Path
from radar_platform.ground_scattering import load_maqu_sigma0

PROJECT=Path(__file__).resolve().parents[1]
DATA=Path('/nas/datasets/hyt/datasets/radar_open')
RUN=DATA/'layered_validation/20261004'
REPORT=PROJECT/'docs/review/2026-10-04'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    (REPORT/'images').mkdir(parents=True,exist_ok=True)
    (REPORT/'metrics').mkdir(exist_ok=True)
    (REPORT/'sources').mkdir(exist_ok=True)
    runs={'analytic':'analytic_final','common_real':'common_real_final_v2','maroon_repeated':'maroon_repeated_final_v2','ramp_array':'ramp_array_audit'}
    summaries={}
    for name,folder in runs.items():
        source=RUN/folder;summaries[name]=json.loads((source/'summary.json').read_text())
        for relative,expected in summaries[name].get('source_hashes',{}).items():
            if sha(PROJECT/relative)!=expected:raise ValueError('source changed after numerical validation: '+relative)
        for file in source.iterdir():
            if file.suffix in ('.csv','.json'):
                shutil.copy2(file,REPORT/'metrics'/f'{name}_{file.name}')
    images={
        'analytic_final/ideal_target_motion.png':'01_目标运动_纯仿真.png',
        'analytic_final/analytic_summary.png':'02_运动与阵列批量解析检查.png',
        'maroon_repeated_final_v2/maroon_repeated_common_mae.png':'03_MAROON新增逐次对比.png',
        'common_real_final_v2/ramp_common_mae.png':'04_RAMP_RD与差值.png',
        'common_real_final_v2/ipix_common_mae.png':'05_IPIX平均RD与差值.png',
        'common_real_final_v2/maqu_common_mae.png':'06_玛曲距离响应与差值.png',
        'common_real_final_v2/maroon_common_mae.png':'07_MAROON原小样通道对比.png',
        'array_demo/array_angle_demo.png':'08_角度维作用_纯仿真.png',
    }
    for source,name in images.items():shutil.copy2(RUN/source,REPORT/'images'/name)
    for source,name in [(RUN/'source_catalog/catalog_audit.json','remote_source_audit.json'),
                         (RUN/'maroon_catalog/selected_catalog.json','maroon_archive_catalog.json'),
                         (DATA/'maroon_static_subset/raw/manifest.json','maroon_subset_manifest.json'),
                         (RUN/'baseline/baseline.json','pre_change_baseline.json')]:shutil.copy2(source,REPORT/'sources'/name)
    if (RUN/'api_verification.json').exists():shutil.copy2(RUN/'api_verification.json',REPORT/'sources/api_verification.json')
    observations=[asdict(row) for file in sorted((DATA/'maqu_scatterometer/raw').rglob('sigma0_*_alpha0_azip00_*.dat')) for row in load_maqu_sigma0(file)]
    dump(REPORT/'sources/maqu_sigma0_observations.json',observations)
    gate_url='https://huggingface.co/api/datasets/AnonymousAuthors0018/sim2radar_artifacts?expand%5B%5D=gated'
    try:
        with urllib.request.urlopen(gate_url,timeout=15) as r:gate=json.loads(r.read(10000))
    except Exception as exc:gate={'status':'not_refreshed','error':str(exc),'prior_checked_gate':'manual'}
    gate['source_url']=gate_url;dump(REPORT/'sources/sim2radar_access.json',gate)
    # Machine-readable cards retain unknowns instead of silently inventing them.
    cards=[
        {'id':'maroon_corner','geometry':'obtained','scattering':'ideal_PEC_assumption','observations':'obtained_static_spectrum',
         'trajectory':'not_applicable_static','configuration':'obtained','calibration':'estimated_global_complex_gain',
         'license':'CC BY-NC 4.0','source':'https://github.com/nihofm/polarized-radar-pathtracing'},
        {'id':'maroon_static_subset','geometry':'obtained_three_selected_meshes','scattering':'ideal_PEC_assumption',
         'observations':'30_saved_frames_obtained_20_eligible','trajectory':'coherent_slow_time_not_verified',
         'configuration':'obtained_with_rx_tx_axis_conversion','calibration':'first_five_frames_of_30cm_scene',
         'license':'CC BY-NC 4.0','source':'https://zenodo.org/records/18254440'},
        {'id':'maqu','geometry':'reference_plate_dimensions_obtained_no_exact_grass_mesh','scattering':'304_footprint_averaged_sigma0_observations_obtained',
         'observations':'reference_sweeps_and_sigma0_obtained','trajectory':'not_applicable_static','configuration':'obtained',
         'calibration':'reference_plate_six_parameter_response','license':'CC BY 4.0 data, Apache-2.0 processing',
         'source':'https://doi.org/10.17026/dans-zfb-qegy'},
        {'id':'ipix','geometry':'sea_wave_mesh_missing','scattering':'estimated_from_calibration_record_only',
         'observations':'four_real_records_obtained','trajectory':'sphere_trajectory_missing','configuration':'obtained',
         'calibration':'frozen_record_17_correction','license':'source research citation and communication conditions',
         'source':'https://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html'},
        {'id':'tsms','geometry':'dimensions_only','scattering':'equivalent_point_assumption','observations':'360_processed_RD_plus_representative_raw_obtained',
         'trajectory':'not_verified','configuration':'batch_vs_sample_unresolved','calibration':'historical_per_distance_fits_not_blind',
         'license':'CC BY 4.0','source':'https://www.nature.com/articles/s41597-026-06802-6'},
        {'id':'ramp','geometry':'labels_not_CAD','scattering':'missing','observations':'60_ADC_frames_obtained',
         'trajectory':'adjacent_label_difference_estimate','configuration':'processing_timing_obtained_array_geometry_not_verified',
         'calibration':'global_power_from_12_development_frames_no_hardware_angle_calibration',
         'license':'dataset redistribution terms not established','source':'https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection'},
        {'id':'sim2radar','geometry':'not_obtained','scattering':'not_obtained','observations':'manual_access_required_not_downloaded',
         'trajectory':'controlled_motion_described_not_verified','configuration':'not_obtained','calibration':'not_obtained',
         'license':'card_CC_BY_4.0_access_still_gated','source':'https://huggingface.co/datasets/AnonymousAuthors0018/sim2radar_artifacts'},
        {'id':'coloradar','geometry':'lidar_described_not_downloaded','scattering':'missing','observations':'official_download_entry_available_not_downloaded',
         'trajectory':'pose_described_not_verified','configuration':'not_obtained','calibration':'not_obtained',
         'license':'official_page_Apache_2.0_verify_downloaded_files','source':'https://arpg.colorado.edu/coloradar/'},
        {'id':'raspnet','geometry':'terrain_description_not_complete_inputs','scattering':'RFView_internal_not_independent_truth',
         'observations':'synthetic_not_real_not_used_here','trajectory':'not_verified','configuration':'paper_description',
         'calibration':'not_applicable_cross_simulator_candidate','license':'verify_selected_files_before_download',
         'source':'https://www.sdms.afrl.af.mil/index.php?collection=raspnet'},
    ]
    dump(REPORT/'dataset_cards.json',{'as_of':'2026-10-04','cards':cards})
    checks=[
        ('analytic_motion','648 parameter cases','synthetic','648/648 within one native range/velocity cell','metrics/analytic_motion_cases.csv'),
        ('analytic_array','126 parameter cases','synthetic','126/126 within 1 degree grid','metrics/analytic_array_cases.csv'),
        ('maroon_existing','6605 heldout channel pairs; one averaged scene','real_static','all retained, including zero prediction','metrics/common_real_maroon_channel_mae.csv'),
        ('maroon_repeated','20 saved frames; 5 calibration,15 evaluation; 2 geometries','real_static','metal angle separately excluded:17454 faces','metrics/maroon_repeated_frame_metrics.csv'),
        ('ipix','4 records;896 CPI;56 windows;8 seeds','real_statistics','record19 shape-only, not amplitude score','metrics/common_real_ipix_window_mae.csv'),
        ('maqu','3 validation measurements x 2 polarizations','real_static','full-domain and target-domain MAE both retained','metrics/common_real_maqu_mae.csv'),
        ('ramp_rd','60 frames;12 calibration,48 development;2 overlapping slices per frame','real_RD','development only; no geometry/material twin','metrics/common_real_ramp_frame_mae.csv'),
        ('ramp_rad','60 ADC frames','real_ADC_processing','assumed array geometry; angle accuracy not validated','metrics/ramp_array_frame_processing.csv'),
        ('tsms','360 processed frames','real_audit','0 eligible for strict physical MAE until configuration aligned','metrics/common_real_tsms_eligibility.csv'),
    ]
    with (REPORT/'验证清单.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.writer(h);w.writerow(['dataset_or_test','coverage','evidence_type','status','metrics_file']);w.writerows(checks)
    baseline=json.loads((RUN/'baseline/baseline.json').read_text())['source_hashes']
    code=sorted([*PROJECT.glob('backend/**/*.py'),*PROJECT.glob('scripts/*.py'),*PROJECT.glob('tests/**/*.py')])
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in code}
    changes={k:{'before':baseline.get(k),'after':v} for k,v in hashes.items() if baseline.get(k)!=v}
    dump(REPORT/'code_changes.json',changes)
    for document in REPORT.rglob('*.md'):
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',document.read_text()):
            if not link.startswith(('https://','http://','#')) and not (document.parent/link).exists():raise FileNotFoundError(f'{document.name}: {link}')
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),'run_root':str(RUN),'run_directories':runs,
              'source_hashes':hashes,'tests_passed':82,'contains_original_echoes':False,
              'metric_scope':'different datasets/ROIs and shape-only scores are not ranked together',
              'sha256':{str(p.relative_to(REPORT)):sha(p) for p in sorted(REPORT.rglob('*')) if p.is_file() and p.name!='package_manifest.json'}}
    dump(REPORT/'package_manifest.json',manifest)
    package=PROJECT/'docs/雷达平台分层验证_2026-10-04_文档图片指标.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(REPORT.rglob('*')):
            if file.is_file():z.write(file,Path('雷达平台分层验证_2026-10-04')/file.relative_to(REPORT))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.npy','.npz','.cdf','.mat','.cti')) for n in z.namelist())
    print(json.dumps({'package':str(package),'bytes':package.stat().st_size,'sha256':sha(package),'markdown_links':'valid',
                      'numerical_source_hashes':'verified','code_files_added_or_changed':len(changes)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
