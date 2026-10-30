#!/usr/bin/env python3
"""Check/report viewport screenshots after actual scrolling and compositor paint."""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np
from PIL import Image
import websockets


async def inspect():
    report=Path('docs/reports/2026-10-08/雷达仿真平台_功能扩展与验证报告.html').resolve()
    output=Path('.runtime/report_20261008_qa');output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='radar-cdp-profile-') as profile:
        process=subprocess.Popen(['google-chrome','--headless','--no-sandbox','--disable-gpu',
            '--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check',
            '--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--remote-allow-origins=http://localhost',
            f'--user-data-dir={profile}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            marker=Path(profile)/'DevToolsActivePort'
            for _ in range(100):
                if marker.exists():break
                if process.poll() is not None:raise RuntimeError('isolated Chrome exited')
                await asyncio.sleep(.1)
            port=marker.read_text().splitlines()[0]
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=5) as response:
                pages=json.load(response)
            page=next(p for p in pages if p['type']=='page')
            async with websockets.connect(page['webSocketDebuggerUrl'],origin='http://localhost',max_size=32_000_000) as ws:
                counter=0
                async def call(method,params=None):
                    nonlocal counter
                    counter+=1;number=counter
                    await ws.send(json.dumps({'id':number,'method':method,'params':params or {}}))
                    while True:
                        reply=json.loads(await asyncio.wait_for(ws.recv(),timeout=20))
                        if reply.get('id')==number:
                            if 'error' in reply:raise RuntimeError(reply['error'])
                            return reply.get('result',{})
                await call('Page.enable')
                results=[]
                for label,width,height,anchor in [('desktop',1440,1100,''),('ramp',1440,1800,'ramp'),('mobile',480,1000,'')]:
                    await call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                    await call('Page.navigate',{'url':report.as_uri()})
                    for _ in range(100):
                        state=await call('Runtime.evaluate',{'expression':"document.readyState==='complete' && document.documentElement.dataset.imagesReady==='yes'",'returnByValue':True})
                        if state.get('result',{}).get('value'):break
                        await asyncio.sleep(.1)
                    else:raise AssertionError('report images did not finish loading')
                    expression=f"document.getElementById({json.dumps(anchor)}).scrollIntoView({{behavior:'instant',block:'start'}})" if anchor else 'window.scrollTo(0,0)'
                    await call('Runtime.evaluate',{'expression':expression})
                    await asyncio.sleep(.4)
                    measured=await call('Runtime.evaluate',{'expression':"JSON.stringify({scrollY:window.scrollY,width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth>innerWidth+1,images:Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0),rampTop:document.getElementById('ramp').getBoundingClientRect().top})",'returnByValue':True})
                    info=json.loads(measured['result']['value'])
                    assert info['images'] and not info['overflow']
                    if anchor:assert abs(info['rampTop'])<30
                    captured=await call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,
                        'clip':{'x':0,'y':info['scrollY'],'width':width,'height':height,'scale':1}})
                    screenshot=output/(label+'_cdp.png')
                    screenshot.write_bytes(base64.b64decode(captured['data']))
                    pixels=np.asarray(Image.open(screenshot).convert('RGB'))
                    if float(pixels.std(axis=(0,1)).max())<10:raise AssertionError('blank viewport screenshot')
                    results.append({'view':label,**info,'nonblank_screenshot':True})
                    print(json.dumps(results[-1]),flush=True)
                (output/'browser_cdp_checks.json').write_text(json.dumps(results,indent=2)+'\n')
        finally:
            process.terminate()
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)


if __name__=='__main__':asyncio.run(inspect())
