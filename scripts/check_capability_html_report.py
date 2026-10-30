#!/usr/bin/env python3
"""Render the offline HTML in an isolated headless Chrome profile and check resources."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import tempfile


class Inspect(HTMLParser):
    def __init__(self):super().__init__();self.root={};self.images=[];self.external=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='html':self.root=a
        if tag=='img':self.images.append(a)
        if tag in ('script','link','iframe') and (a.get('src','').startswith('http') or a.get('href','').startswith('http')):self.external.append(a)
        if a.get('id'):self.ids.add(a['id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--html',type=Path,default=Path('docs/reports/2026-10-08/雷达仿真平台_功能扩展与验证报告.html'))
    p.add_argument('--output',type=Path,default=Path('.runtime/report_20261008_qa'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Inspect();source.feed(args.html.read_text())
    assert not source.external and all(i['src'].startswith('data:image/png;base64,') and i.get('alt') for i in source.images)
    assert {'capabilities','synthetic','mimo','ramp','other','conclusions'}.issubset(source.ids)
    results=[]
    for label,width,height,anchor in [('desktop',1440,1100,''),('ramp',1440,1800,'#ramp'),('mobile',480,1000,'')]:
        with tempfile.TemporaryDirectory(prefix='radar-report-chrome-') as profile:
            command=['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
                     '--disable-background-networking','--no-first-run','--no-default-browser-check',
                     '--run-all-compositor-stages-before-draw',
                     '--force-device-scale-factor=1',f'--user-data-dir={profile}',
                     f'--window-size={width},{height}','--virtual-time-budget=5000',
                     '--dump-dom',f'--screenshot={str((args.output/(label+".png")).resolve())}',args.html.resolve().as_uri()+anchor]
            process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=55)
            if process.returncode:raise RuntimeError(process.stderr[-2000:])
            dom=Inspect();dom.feed(process.stdout)
            assert dom.root.get('data-report-ready')=='yes',label
            assert dom.root.get('data-images-ready')=='yes',label
            assert dom.root.get('data-page-overflow')=='no',label
            results.append({'view':label,'width':width,'height':height,'images':len(dom.images),
                            'all_images_loaded':True,'horizontal_page_overflow':False})
            print(json.dumps(results[-1]),flush=True)
    (args.output/'browser_checks.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
