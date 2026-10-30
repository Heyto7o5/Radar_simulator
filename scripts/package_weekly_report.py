#!/usr/bin/env python3
"""Validate local Markdown assets and package a self-contained Feishu report."""
import hashlib,json,re,zipfile
from pathlib import Path


def main():
    project=Path(__file__).resolve().parents[1]
    report=project/'docs/weekly/2026-09-28'
    evidence=json.loads((report/'run_provenance.json').read_text())
    for relative,expected in evidence['source_hashes'].items():
        actual=hashlib.sha256((project/relative).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'source changed since validation: {relative}')
    expansion=report/'09-30扩展验证/provenance.json'
    if expansion.exists():
        for relative,expected in json.loads(expansion.read_text())['source_hashes'].items():
            if hashlib.sha256((project/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'expanded validation source changed: {relative}')
    for document in report.rglob('*.md'):
        text=document.read_text()
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if not link.startswith(('https://','http://','#')) and not (document.parent/link).exists():
                raise FileNotFoundError(f'{document.name}: {link}')
    files=sorted(p for p in report.rglob('*') if p.is_file() and p.name!='package_manifest.json')
    checksums={str(p.relative_to(report)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest={'report_period':'2026-09-28 to 2026-10-04','as_of':'2026-09-30',
              'scope':'MAROON six channel groups, IPIX four acquisitions and 896 measured CPIs, plus prior Maqu reference plate validation',
              'contains_raw_measurements':False,'sha256':checksums}
    (report/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    package=project/'docs/雷达周报_2026-09-28_含新增说明与图片.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(report.rglob('*')):
            if p.is_file():z.write(p,Path('雷达周报_2026-09-28')/p.relative_to(report))
    with zipfile.ZipFile(package) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
    print(json.dumps({'package':str(package),'bytes':package.stat().st_size,
                      'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
                      'markdown_links':'valid','source_snapshot':'matches'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
