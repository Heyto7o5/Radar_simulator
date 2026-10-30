#!/usr/bin/env python3
"""Package derived evidence only; raw measurements remain on NAS."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from repair_ipix_downloads import validate


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=Path('/nas/datasets/hyt/datasets/radar_open/alignment_campaign/20261005'))
    parser.add_argument('--docs', type=Path, default=Path('docs/alignment/2026-10-05'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    campaign, output = args.campaign, args.docs.resolve()
    assets, metrics, provenance = output / 'images', output / 'metrics', output / 'provenance'
    for folder in [assets, metrics, provenance]:
        folder.mkdir(parents=True, exist_ok=True)
    accepted = ['maroon_v1', 'maroon_rigid_v1', 'maroon_energy_v1', 'ipix_v1', 'ramp_v3',
                'ramp_peak_diagnostics_v1', 'ramp_curvature_v1', 'ramp_phase_trial_v1',
                'ramp_noise_trial_v1', 'ramp_noise_control_v1']
    for name in accepted:
        for source in sorted((campaign / name).glob('*')):
            # Exclude raw/derived complex arrays and duplicated partial output.
            if source.suffix not in {'.csv', '.json'} or '.partial.' in source.name or source.name == 'simulation_inputs.json':
                continue
            target = metrics / name / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    images = {
        'maroon_rigid_v1/aligned_comparison.png': 'maroon_distance_examples.png',
        'ipix_v1/before_after.png': 'ipix_records.png',
        'ipix_v1/conditional_examples.png': 'ipix_rd_examples.png',
        'ramp_v3/aligned_rd_examples.png': 'ramp_signal_rd_examples.png',
        'ramp_phase_trial_v1/aligned_ra_examples.png': 'ramp_ra_examples.png',
        'ramp_noise_trial_v1/rd_examples.png': 'ramp_noise_rd_examples.png',
        'ramp_noise_control_v1/noise_control.png': 'ramp_noise_control.png',
    }
    for source, target in images.items():
        shutil.copy2(campaign / source, assets / target)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'axes.unicode_minus': False})
    peaks = json.loads((campaign / 'ramp_peak_diagnostics_v1/summary.json').read_text())['results']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    for ax, key, title in zip(axes, ['range_peak_difference_m', 'doppler_peak_difference_mps', 'angle_peak_difference_deg'],
                            ['距离峰差 / m', '速度峰差 / (m/s)', '角度峰差 / °']):
        group = [next(r for r in peaks if r['role'] == 'heldout_record' and r['model'] == m)
                 for m in ['legacy_random', 'physical_smooth_track']]
        values = [r[key]['mean'] for r in group]
        bars = ax.bar(['修改前', '固定目标 + 连续轨迹'], values, color=['#8894a7', '#27857c'])
        ax.bar_label(bars, fmt='%.2f')
        ax.set_ylim(0, max(values) * 1.3)
        ax.set_title(title)
    fig.suptitle('RAMP 跨记录：100 个可关联目标–帧的实测/仿真峰值差\n使用标注距离区间关联，不是无真值辅助的检测精度；重叠和重复 UID 不参与此图', fontsize=11)
    fig.savefig(assets / 'ramp_peak_changes.png', dpi=160)
    plt.close(fig)

    # Revalidate the final IPIX files, rather than trusting the initial interrupted-download manifest.
    selection = json.loads((campaign / 'ipix_sources/selection.json').read_text())
    integrity = []
    for spec in selection['records']:
        path = Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth/raw') / spec['filename']
        params = json.loads((campaign / f'ipix_v1/record_{spec["record_id"]}_parameters.json').read_text())
        sha = digest(path.read_bytes())
        if sha != params['source_sha256']:
            raise ValueError('current IPIX input differs from scored file')
        integrity.append({**spec, 'bytes': path.stat().st_size, 'sha256': sha, **validate(path), 'matches_scored_input': True})
    dump(provenance / 'ipix_final_integrity.json', {'utc': datetime.now(timezone.utc).isoformat(), 'files': integrity,
        'supersedes_for_integrity_only': 'initial download manifest; partial bytes were retained in raw/incomplete, not scored'})
    shutil.copy2(campaign / 'ipix_sources/selection.json', provenance / 'ipix_selection.json')
    shutil.copy2(Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005/selection.json'), provenance / 'ramp_selection.json')

    # Reconstruct only bytes whose SHA-256 matches the hashes saved before RAMP v3 ran.
    expected = json.loads((campaign / 'ramp_v3/protocol.json').read_text())['source_sha256']
    recovery = []
    with zipfile.ZipFile(campaign / 'baseline/source_before.zip') as before, zipfile.ZipFile(provenance / 'ramp_v3_exact_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, sha in expected.items():
            current = (repo / name).read_bytes()
            candidates = [('current_unchanged', current)]
            if name in before.namelist():
                candidates.append(('pre_campaign_snapshot', before.read(name)))
            if name.endswith('alignment_metrics.py'):
                candidates.append(('remove_later_appended_helpers', current.split(b'def fit_relative_channel_response')[0].rstrip() + b'\n'))
            if name.endswith('ramp_comparison.py'):
                candidates.append(('remove_later_explanatory_docstring', re.sub(
                    rb'(def official_velocity_axis_mps\(\) -> np.ndarray:\n)    """.*?"""\n', rb'\1', current, count=1, flags=re.S)))
            match = next(((method, content) for method, content in candidates if digest(content) == sha), None)
            if match is None:
                raise ValueError(f'cannot recover exact run source: {name}')
            archive.writestr(name, match[1])
            recovery.append({'path': name, 'sha256': sha, 'recovery': match[0]})
    dump(provenance / 'ramp_v3_source_verification.json', {'all_hashes_match_original_run': True, 'files': recovery})
    files = sorted([*repo.glob('backend/**/*.py'), *repo.glob('scripts/*.py'), *repo.glob('tests/**/*.py'), repo / 'pyproject.toml'])
    with zipfile.ZipFile(provenance / 'source_after.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(repo))
    dump(provenance / 'source_after_hashes.json', {str(p.relative_to(repo)): digest(p.read_bytes()) for p in files})
    shutil.copy2(campaign / 'baseline/baseline.json', provenance / 'baseline.json')
    shutil.copy2(campaign / 'baseline/source_before.zip', provenance / 'source_before.zip')
    dump(provenance / 'run_status.json', {
        'accepted_runs': accepted,
        'ramp_v1': 'stopped on duplicate annotation identifiers before scoring',
        'ramp_v2': 'superseded: extreme display reference jointly floored ~99% of RD cells; not a fidelity result',
        'ramp_v3': 'main shared-grid run; exact source snapshot independently hash-verified',
        'later_trials': 'current source snapshot includes phase/noise controls; no per-evaluation parameter fitting',
        'test_set_status': 'record splits prevent direct parameter fitting, but repeated inspection makes these development evidence, not an untouched final benchmark',
        'data_distribution': 'no original ADC, meshes, images, or I/Q bundled; derived figures and metrics only',
    })
    test_path = repo / '.runtime/alignment_20261005_tests.xml'
    suites = ET.parse(test_path).getroot().findall('testsuite')
    checks = {key: sum(int(s.attrib[key]) for s in suites) for key in ['tests', 'errors', 'failures', 'skipped']}
    if checks['tests'] < 99 or checks['errors'] or checks['failures']:
        raise ValueError('current regression report is missing or failed')
    checks['command'] = "OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -o addopts='' -q -p no:cacheprovider --junitxml=.runtime/alignment_20261005_tests.xml"
    checks['scope'] = 'regression checks, not real-world fidelity certification'
    dump(provenance / 'regression_tests.json', checks)
    manifest = {str(p.relative_to(output)): digest(p.read_bytes()) for p in output.rglob('*') if p.is_file() and p.name != 'package_manifest.json'}
    dump(output / 'package_manifest.json', manifest)
    destination = repo / 'docs/三数据集对齐_2026-10-05_文档图片指标.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, Path('三数据集对齐_2026-10-05') / path.relative_to(output))
        if archive.testzip() is not None:
            raise ValueError('archive CRC failed')
    print(json.dumps({'package': str(destination), 'bytes': destination.stat().st_size, 'sha256': digest(destination.read_bytes()),
                      'documents_and_assets': len(manifest), 'exact_ramp_v3_source_files': len(recovery), 'ipix_validated_files': len(integrity)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
