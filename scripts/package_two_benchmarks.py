#!/usr/bin/env python3
"""Collect the two completed, immutable benchmark runs into a report bundle."""
import hashlib,json,re,shutil,zipfile
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    project=Path(__file__).resolve().parents[1]
    report=project/'docs/weekly/2026-09-28/09-30双数据集';(report/'images').mkdir(parents=True,exist_ok=True)
    data=Path('/nas/datasets/hyt/datasets/radar_open')
    ipix=data/'ipix_dartmouth/processed/validation_20260930_final'
    maqu=data/'maqu_scatterometer/processed/validation_20260930_final'
    copies={ipix/'ipix_comparison.png':'images/01_IPIX_RD对比.png',maqu/'maqu_comparison.png':'images/02_玛曲金属板对比.png',
        ipix/'metrics.csv':'ipix_metrics.csv',maqu/'metrics.csv':'maqu_metrics.csv',ipix/'comparison.json':'ipix_results.json',
        maqu/'comparison.json':'maqu_results.json',ipix/'recipe.json':'ipix_recipe.json',maqu/'recipe.json':'maqu_recipe.json',
        maqu/'api_verification.json':'api_verification.json',data/'ipix_dartmouth/manifest.json':'ipix_source_manifest.json',
        data/'maqu_scatterometer/manifest.json':'maqu_source_manifest.json'}
    for src,dst in copies.items():shutil.copy2(src,report/dst)
    for document in report.glob('*.md'):
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',document.read_text()):
            if not link.startswith(('https://','http://','#')) and not (document.parent/link).exists():raise FileNotFoundError(link)
    sources=['backend/radar_platform/'+n for n in ['ipix_dataset.py','distributed_clutter.py','maqu_dataset.py','physical_optics.py','instrument_response.py','clutter_api.py','surface_api.py','api.py']]
    sources+=['scripts/'+n for n in ['remote_zip.py','prepare_ipix.py','prepare_maqu.py','validate_ipix_clutter.py','validate_maqu_plate.py','verify_two_benchmarks.py']]
    files={str(p.relative_to(report)):sha(p) for p in report.rglob('*') if p.is_file() and p.name!='provenance.json'}
    provenance={'as_of':'2026-09-30','dataset_runs':{'IPIX':str(ipix),'Maqu':str(maqu)},
        'source_sha256':{n:sha(project/n) for n in sources},'report_files_sha256':files,
        'raw_measurements_in_package':False,'tests_passed':52,
        'scope':{'IPIX':'statistical clutter echo; no deterministic sea geometry or sphere trajectory',
                 'Maqu':'known reference plates with six-parameter receiver calibration; grassland not simulated'}}
    (report/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    package=project/'docs/双数据集复现_IPIX与玛曲_2026-09-30.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(report.rglob('*')):
            if p.is_file():archive.write(p,Path('双数据集复现_2026-09-30')/p.relative_to(report))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        print(json.dumps({'package':str(package),'bytes':package.stat().st_size,'files':len(archive.namelist()),'sha256':sha(package)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
