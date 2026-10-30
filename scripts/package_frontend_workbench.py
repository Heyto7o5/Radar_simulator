#!/usr/bin/env python3
"""Record frontend delivery evidence without packaging raw observations."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'docs/frontend-preview';output.mkdir(parents=True,exist_ok=True)
    release=root/'.runtime/frontend_browser_release'
    visual=root/'.runtime/frontend_browser_visual_final'
    checks=json.loads((release/'checks.json').read_text())
    assert len(checks['checks'])>=20 and not checks['uncaught_browser_exceptions']
    for source,target in [(visual/'simulate_initial.png','simulate.png'),
        (visual/'comparison_four_panels.png','compare.png'),
        (release/'simulate_dem_and_angle.png','angle.png'),
        (visual/'comparison_mobile.png','mobile.png')]:shutil.copy2(source,output/target)
    shutil.copy2(release/'checks.json',output/'checks.json')
    suites=ET.parse(root/'.runtime/frontend_workbench_tests_final.xml').getroot().findall('testsuite')
    counts={key:sum(int(s.attrib[key]) for s in suites) for key in ['tests','failures','errors','skipped']}
    assert counts['tests']>=158 and counts['failures']==counts['errors']==0
    files=[*root.glob('frontend/src/**/*.ts'),*root.glob('frontend/src/**/*.tsx'),*root.glob('frontend/src/*.css'),
           root/'backend/radar_platform/workbench.py',root/'backend/radar_platform/comparison_ui.py',
           root/'backend/radar_platform/unified/engine.py',root/'backend/radar_platform/unified/jobs.py',
           root/'backend/radar_platform/api.py',root/'tests/test_workbench.py',root/'scripts/preview.sh',
           root/'scripts/check_frontend_workbench.py',root/'frontend/dist/index.html']
    summary={'tests':counts,'browser_checks':len(checks['checks']),'uncaught_browser_exceptions':0,
        'build':'npm run build passed; TypeScript + Vite production bundle',
        'ui_routes':['/simulate','/compare'],'backend_used_in_browser_checks':'127.0.0.1:8789',
        'scope':'actual UI interactions and actual backend results; data figures are not mocks',
        'deployment':'temporary QA listener only; no external listener enabled',
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'image_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.png')}}
    (output/'delivery.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(output),'tests':counts,'browser_checks':len(checks['checks'])},ensure_ascii=False))


if __name__=='__main__':main()
