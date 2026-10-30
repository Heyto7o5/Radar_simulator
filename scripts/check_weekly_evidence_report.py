#!/usr/bin/env python3
"""Read-only numerical/artifact checks, then offline Chrome rendering."""
import asyncio
import base64
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np
import websockets

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'docs/weekly/2026-10-10'

def load(p):return json.loads(p.read_text())

def check_numbers():
    with (ROOT/'metrics/ramp_frames.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==540 and len({r['frame_id'] for r in rows})==180
    assert len({r['sequence'] for r in rows})==3
    assert len({(r['frame_id'],r['mode']) for r in rows})==540
    for s in load(ROOT/'metrics/ramp_summary.json')['groups']:
        values=np.array([float(r['mae_db']) for r in rows if r['mode']==s['mode']])
        np.testing.assert_allclose([values.mean(),np.median(values),np.percentile(values,90)],
                                   [s['mean'],s['median'],s['p90']],rtol=1e-12)
        cfg=load(ROOT/'configs'/f'ramp_{s["mode"]}_acquisition.json')
        assert cfg['array']['loop_period_s']==120e-6
        assert cfg['waveform']['loops']==255 and cfg['waveform']['samples']==128
    for i in range(1,4):assert load(ROOT/'metrics'/f'ramp_example_{i}_source.json')['camera_sha256']
    mo=load(ROOT/'metrics/motion_summary.json');assert mo['comparisons']==24 and mo['passed']==24
    assert mo['all_unambiguous'] and mo['unambiguous_checks']==36
    assert abs(np.linalg.norm(mo['base_velocity_ned_mps'])-1200)<1e-10
    cfg=load(ROOT/'configs/motion_unambiguous_1ghz.json')
    assert cfg['waveform']['carrier_hz']==1e9 and cfg['waveform']['slot_period_s']==50e-6
    for row in load(ROOT/'metrics/motion_peak_checks.json'):
        assert row['velocity_unambiguous'] and row['full_echo_inside_PRI']
        assert row['velocity_error_mps']<=mo['native_velocity_cell_mps']
        assert row['range_error_m']<=299792458/(2*150e6)
    for r in load(ROOT/'metrics/terrain_sources.json'):
        assert r['source_vs_engine_envelope_difference_m']<1e-8
        assert not r['paired_measured_RD']
    assert load(ROOT/'metrics/angle_demo.json')[1]['resolved_by_demo_rule']
    print('Numbers / axes / provenance / exact matching: passed',flush=True)

async def browser():
    report=ROOT/'雷达仿真平台周报.html'
    output=PROJECT/'.runtime/weekly_20261010_qa';output.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='radar-weekly-chrome-') as profile:
        process=subprocess.Popen(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
            '--disable-background-networking','--no-first-run','--remote-debugging-address=127.0.0.1',
            '--remote-debugging-port=0','--remote-allow-origins=http://localhost',f'--user-data-dir={profile}','about:blank'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            marker=Path(profile)/'DevToolsActivePort'
            for _ in range(100):
                if marker.exists():break
                await asyncio.sleep(.1)
            port=marker.read_text().splitlines()[0]
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=5) as response:pages=json.load(response)
            page=next(p for p in pages if p['type']=='page')
            async with websockets.connect(page['webSocketDebuggerUrl'],origin='http://localhost',max_size=32_000_000) as ws:
                counter=0
                async def call(method,params=None):
                    nonlocal counter
                    counter+=1;n=counter
                    await ws.send(json.dumps({'id':n,'method':method,'params':params or {}}))
                    while True:
                        r=json.loads(await asyncio.wait_for(ws.recv(),timeout=20))
                        if r.get('id')==n:
                            if 'error' in r:raise RuntimeError(r['error'])
                            return r.get('result',{})
                await call('Page.enable');checks=[]
                for name,width,section in [('desktop',1440,0),('ramp',1440,1),('ra',1440,3),('mobile',480,0)]:
                    await call('Emulation.setDeviceMetricsOverride',{'width':width,'height':1050,'deviceScaleFactor':1,'mobile':False})
                    await call('Page.navigate',{'url':report.as_uri()})
                    for _ in range(100):
                        r=await call('Runtime.evaluate',{'expression':"document.documentElement.dataset.imagesReady==='yes'",'returnByValue':True})
                        if r.get('result',{}).get('value'):break
                        await asyncio.sleep(.1)
                    await call('Runtime.evaluate',{'expression':f"document.querySelectorAll('section')[{section}].scrollIntoView()" if name!='mobile' else 'window.scrollTo(0,0)'})
                    await asyncio.sleep(.2)
                    r=await call('Runtime.evaluate',{'expression':"JSON.stringify({images:document.images.length,allImagesLoaded:[...document.images].every(i=>i.complete&&i.naturalWidth>0),overflow:document.documentElement.scrollWidth>innerWidth+1,scrollY:scrollY})",'returnByValue':True})
                    data=json.loads(r['result']['value']);assert data['allImagesLoaded'] and not data['overflow']
                    capture=await call('Page.captureScreenshot',{'format':'png'})
                    (output/(name+'.png')).write_bytes(base64.b64decode(capture['data']))
                    checks.append({'viewport':name,**data})
                (ROOT/'metrics/browser_checks.json').write_text(json.dumps(checks,indent=2))
                print(json.dumps(checks),flush=True)
        finally:
            process.terminate()
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)

if __name__=='__main__':check_numbers();asyncio.run(browser())
