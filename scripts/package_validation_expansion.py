#!/usr/bin/env python3
"""Copy only figures/metrics into the Markdown report, never original echoes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    target = project/'docs/weekly/2026-09-28/09-30扩展验证'
    provenance = json.loads((args.results/'provenance.json').read_text())
    for name,expected in provenance['source_hashes'].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'source changed since evaluation: {name}')
    images = {'maroon_groups.png':'01_MAROON六组对比.png','ipix_frames.png':'02_IPIX多帧RD.png',
              'ipix_records.png':'03_IPIX四记录对比.png','ipix_variation.png':'04_IPIX时间窗波动.png'}
    (target/'images').mkdir(parents=True,exist_ok=True)
    for source,name in images.items():
        shutil.copy2(args.results/source,target/'images'/name)
    for name in ['maroon_channels.csv','maroon_groups.csv','maroon_summary.json','ipix_frames.csv',
                 'ipix_windows.csv','ipix_full_records.csv','ipix_summary.json','provenance.json']:
        shutil.copy2(args.results/name,target/name)
    data = Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth')
    shutil.copy2(data/'extension_manifest.json',target/'ipix_extension_manifest.json')
    shutil.copy2(data/'processed/validation_20260930_final/recipe.json',target/'frozen_ipix_recipe.json')
    print(json.dumps({'destination':str(target),'raw_echoes_included':False,'source_snapshot':'matches'},ensure_ascii=False))


if __name__=='__main__':
    main()
