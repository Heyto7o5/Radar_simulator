#!/usr/bin/env python3
"""Freeze source and data inventory before a new implementation campaign."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[1]
    files=sorted([*project.glob('backend/**/*.py'),*project.glob('tests/**/*.py'),*project.glob('scripts/*.py')])
    hashes={str(f.relative_to(project)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    with zipfile.ZipFile(a.output/'source_before.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(f,f.relative_to(project))
    inventory={}
    root=Path('/nas/datasets/hyt/datasets/radar_open')
    for name in ['maroon_corner','maqu_scatterometer','ipix_dartmouth','tsms_drone_sample','ramp_adc_77ghz']:
        dataset=root/name
        inventory[name]={suffix:sum(1 for d in ('raw','extracted') for f in (dataset/d).rglob('*'+suffix))
                         for suffix in ('.cdf','.cti','.mat','.npz','.obj')}
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':hashes,'inventory':inventory,
            'scope':'pre-change audit; file counts are not independent experiment counts'}
    (a.output/'baseline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(inventory,indent=2))


if __name__=='__main__':main()
