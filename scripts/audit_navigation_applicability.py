#!/usr/bin/env python3
"""Correct legacy saved preflight metadata against each exact frame request.

RadarWorld reuses a compiled-scene estimate. Its diagnostic assessment is live,
but early save_inputs files inherited the base-pose assessment. Only these JSON
metadata files are corrected; binary echoes, observations and labels are untouched.
"""
import argparse
import hashlib
import json
import shutil

from radar_platform.navigation.io import local_path,read_json,write_json
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.applicability import assess_scene


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=local_path(args.root);rows=[]
    for request_file in sorted(root.rglob('request.json')):
        if any(part.startswith('pytest') for part in request_file.relative_to(root).parts):continue
        target=request_file.parent/'applicability.json'
        if not target.exists():continue
        request=Simulation.model_validate(read_json(request_file));actual=assess_scene(request,compile_scene(request.scene))
        previous=read_json(target);changed=previous!=actual
        if changed:
            original=request_file.parent/'applicability_original_template.json'
            if not original.exists():shutil.copyfile(target,original)
            old_sha=hashlib.sha256(target.read_bytes()).hexdigest();write_json(target,actual)
            write_json(request_file.parent/'applicability_metadata_correction.json',dict(
                reason='Replace cached base-pose applicability with assessment of this exact recorded frame trajectory.',
                request_sha256=hashlib.sha256(request_file.read_bytes()).hexdigest(),
                previous_sha256=old_sha,corrected_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                raw_echoes_observations_and_labels_changed=False))
        rows.append(dict(request=str(request_file.relative_to(root)),corrected_now=changed,status=actual['status'],
            correction_record_exists=(request_file.parent/'applicability_metadata_correction.json').exists()))
    write_json(args.output,dict(frames_checked=len(rows),metadata_corrections=sum(r['corrected_now'] for r in rows),
                              recorded_metadata_corrections=sum(r['correction_record_exists'] for r in rows),
                              raw_echoes_observations_and_labels_changed=False,rows=rows))
    print(json.dumps(dict(frames_checked=len(rows),metadata_corrections=sum(r['corrected_now'] for r in rows))))


if __name__=='__main__':main()
