#!/usr/bin/env python3
"""Acceptance against the real loopback server and browser, including actual solver jobs."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request
import websockets
from check_frontend_workbench import Browser

async def exercise(b,base):
    checks=[]
    async def record(name,**data):checks.append({'check':name,**data});print(checks[-1],flush=True)
    async def go(path):
        await b.call('Page.navigate',{'url':base+path})
        await asyncio.sleep(.5)
    async def run(previous=None):
        await b.wait("!document.querySelector('[data-testid=run-simulation]')?.disabled",'run enabled')
        await b.click('[data-testid=run-simulation]')
        await b.wait("Boolean(Array.from(document.querySelectorAll('button')).find(e=>e.textContent.includes('理解风险')))||Boolean(document.querySelector('[data-testid=job-status]')?.dataset.jobId)",'preflight finished')
        await asyncio.sleep(.4)
        await b.value("Array.from(document.querySelectorAll('button')).find(e=>e.textContent.includes('理解风险'))?.click()")
        return await b.job(previous)
    await b.call('Page.enable');await b.call('Runtime.enable')
    await b.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':900,'deviceScaleFactor':1,'mobile':False})
    await go('/simulate?case=points')
    await b.wait("Boolean(document.querySelector('[data-testid=run-simulation]'))&&!document.querySelector('[data-testid=run-simulation]').disabled",'draft initialized')
    assert not await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId"),'default auto-ran'
    await b.wait("Boolean(document.querySelector('[data-testid=scene-3d] canvas'))",'WebGL scene')
    await record('manual default, WebGL scene, preserved draft')
    first=await run()
    await b.wait("document.querySelector('[data-testid=heatmap]')?.dataset.loaded==='true'",'RD display')
    await b.screenshot('workbench_1440',1440,900)
    await b.screenshot('workbench_1920',1920,1080)
    await record('actual point scene calculation and RD',job=first)
    request=await b.api(f'/api/v1/workbench/jobs/{first}/request')
    await b.number('[data-testid="height"]',5.1)
    await b.wait("Boolean(document.querySelector('[data-testid=stale-result]'))",'stale banner')
    assert await b.value("document.querySelector('[data-testid=heatmap]').dataset.loaded==='true'")
    assert await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")==first
    await record('edit preserves old result and explicitly marks ownership')
    second=await run(first)
    request2=await b.api(f'/api/v1/workbench/jobs/{second}/request')
    assert request2['platform']['position_ned_m'][2]==-5100
    assert request2['waveform']==request['waveform'],'silently changed sampling'
    await record('manual edit executes without silently modifying acquisition',job=second)
    await b.value("Array.from(document.querySelectorAll('button')).find(e=>e.textContent.trim()==='保存结果').click()")
    await b.wait("document.body.innerText.includes('已保存到实验归档')",'save experiment')
    await b.call('Page.reload',{'ignoreCache':True});await b.wait("Boolean(document.querySelector('[data-testid=run-simulation]'))",'reload')
    # A case URL explicitly opens its preset; restore a selected experiment by id instead.
    await go(f'/simulate?job={second}')
    await b.job();assert await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")==second
    await record('saved experiment restoration does not create a new task')
    await go(f'/experiments?a={first}&b={second}')
    await b.wait("document.querySelectorAll('[data-testid=heatmap][data-loaded=true]').length>=3",'A/B and linear difference')
    await b.value("document.querySelectorAll('.lab-compare-grid details').forEach(e=>e.open=true)")
    await b.wait("document.querySelectorAll('.lab-compare-grid [data-testid=scene-preview]').length===2",'both saved scene geometries')
    await b.screenshot('comparison',1440,1100)
    assert await b.value("document.body.innerText.includes('platform.position_ned_m')")
    await record('A/B shared reference, explicit configuration diff and linear-power difference')
    await go('/cases');await b.wait("document.querySelectorAll('.lab-case-card').length===6",'six cases')
    await b.screenshot('cases',1440,900)
    await record('six real case configurations and explicit saved-result entry')
    await go('/simulate?case=array');await b.wait("Boolean(document.querySelector('[data-testid=run-simulation]'))&&!document.querySelector('[data-testid=run-simulation]').disabled",'array setup')
    array=await run()
    await b.value("Array.from(document.querySelectorAll('button')).find(e=>e.textContent.trim()==='距离–方位').click()")
    await b.wait("document.querySelector('[data-testid=heatmap]')?.dataset.loaded==='true'",'RA real product')
    await record('FMCW array case and real RA product',job=array)
    await go(f'/experiments?a={first}&b={array}')
    await b.wait("document.body.innerText.includes('逐像素差分不可用')",'incompatible grid refusal')
    assert await b.value("document.querySelectorAll('[data-testid=heatmap]').length")==2
    await record('incompatible grids remain visible but difference is disabled')
    await go('/simulate?job=demo-land-cover');await b.job()
    await b.wait("Boolean(document.querySelector('[aria-label=场景图层] option[value=echo]:not([disabled])'))",'archived coverage ready',timeout=180)
    await b.select('[aria-label=场景图层]','classes');await asyncio.sleep(2);await b.screenshot('real_dem_classes',1440,900)
    await b.select('[aria-label=场景图层]','echo');await asyncio.sleep(2);await b.screenshot('real_dem_coverage',1440,900)
    await b.value("Array.from(document.querySelectorAll('button')).find(e=>e.textContent.trim()==='演示模式').click()")
    await asyncio.sleep(2)
    await b.screenshot('presentation',1920,1080)
    await record('historical real DEM, classification, actual coverage and normalized antenna response')
    await b.screenshot('narrow',700,950)
    await go('/compare');await b.wait("document.body.innerText.includes('实测')&&Boolean(document.querySelector('select'))",'real-data workspace',timeout=180)
    await b.job(comparison=True)
    await b.screenshot('measured',1440,1200)
    await record('existing measured-data workspace retained')
    assert not b.exceptions,b.exceptions
    return checks

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:8014');parser.add_argument('--output',type=Path,default=Path('.runtime/frontend_lab_browser'));args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='radar-lab-browser-') as profile:
        chrome=subprocess.Popen(['google-chrome','--headless','--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--remote-debugging-port=0','--remote-allow-origins=http://localhost',f'--user-data-dir={profile}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            marker=Path(profile)/'DevToolsActivePort'
            for _ in range(100):
                if marker.exists():break
                await asyncio.sleep(.1)
            port=marker.read_text().splitlines()[0]
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:pages=json.load(r)
            async with websockets.connect(next(p for p in pages if p['type']=='page')['webSocketDebuggerUrl'],origin='http://localhost',max_size=100_000_000) as ws:
                b=Browser(ws,args.output)
                try:checks=await exercise(b,args.base)
                except Exception:
                    await b.screenshot('failure',1440,1100);(args.output/'failure.txt').write_text(await b.value('document.body.innerText'));(args.output/'exceptions.json').write_text(json.dumps(b.exceptions,ensure_ascii=False,indent=2));raise
                (args.output/'checks.json').write_text(json.dumps({'checks':checks,'exceptions':b.exceptions},ensure_ascii=False,indent=2))
        finally:
            chrome.terminate()
            try:chrome.wait(timeout=8)
            except subprocess.TimeoutExpired:chrome.kill();chrome.wait()
if __name__=='__main__':asyncio.run(main())
