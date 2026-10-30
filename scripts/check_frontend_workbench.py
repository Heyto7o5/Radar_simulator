#!/usr/bin/env python3
"""Drive the real browser UI against a loopback backend; no mocked simulation results."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image
import websockets


class Browser:
    def __init__(self,ws,output):self.ws=ws;self.count=0;self.output=output;self.exceptions=[]
    async def call(self,method,params=None):
        self.count+=1;number=self.count
        await self.ws.send(json.dumps({'id':number,'method':method,'params':params or {}}))
        while True:
            reply=json.loads(await asyncio.wait_for(self.ws.recv(),timeout=30))
            if reply.get('method')=='Runtime.exceptionThrown':self.exceptions.append(reply['params'])
            if reply.get('id')==number:
                if 'error' in reply:raise RuntimeError(reply['error'])
                return reply.get('result',{})
    async def value(self,expression):
        response=await self.call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
        if 'exceptionDetails' in response:raise RuntimeError(response['exceptionDetails'])
        return response.get('result',{}).get('value')
    async def wait(self,expression,description,timeout=100):
        started=time.monotonic();last=started
        while time.monotonic()-started<timeout:
            if await self.value(expression):return
            if time.monotonic()-last>15:print('waiting:',description,flush=True);last=time.monotonic()
            await asyncio.sleep(.2)
        state=await self.value('document.body.innerText.slice(0,3500)')
        raise AssertionError(f'{description} timed out: {state}')
    async def click(self,selector):
        point=await self.value(f"(()=>{{const e=document.querySelector({json.dumps(selector)});if(!e)throw new Error('missing control');e.scrollIntoView({{block:'center',behavior:'instant'}});const r=e.getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}};}})()")
        await self.call('Input.dispatchMouseEvent',{'type':'mousePressed','button':'left','clickCount':1,**point})
        await self.call('Input.dispatchMouseEvent',{'type':'mouseReleased','button':'left','clickCount':1,**point})
        await asyncio.sleep(.05)
    async def select(self,selector,value):
        await self.value(f"(()=>{{const e=document.querySelector({json.dumps(selector)});if(!e)throw new Error('missing select');e.value={json.dumps(value)};e.dispatchEvent(new Event('change',{{bubbles:true}}));}})()")
    async def number(self,selector,value,blur=True):
        await self.click(selector)
        assert await self.value(f'document.activeElement===document.querySelector({json.dumps(selector)})'),selector
        await self.call('Input.dispatchKeyEvent',{'type':'rawKeyDown','key':'a','code':'KeyA','modifiers':2,'windowsVirtualKeyCode':65,'nativeVirtualKeyCode':65,'commands':['selectAll']})
        await self.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'a','code':'KeyA','modifiers':2,'windowsVirtualKeyCode':65,'nativeVirtualKeyCode':65})
        await self.call('Input.insertText',{'text':str(value)})
        await asyncio.sleep(.05)
        entered=await self.value(f'Number(document.querySelector({json.dumps(selector)}).value)')
        assert abs(entered-float(value))<1e-8,(selector,entered,value)
        if blur:
            await self.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'Tab','code':'Tab'})
            await self.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Tab','code':'Tab'})
    async def file(self,selector,path):
        document=await self.call('DOM.getDocument')
        node=await self.call('DOM.querySelector',{'nodeId':document['root']['nodeId'],'selector':selector})
        assert node['nodeId'],selector
        await self.call('DOM.setFileInputFiles',{'nodeId':node['nodeId'],'files':[str(path.resolve())]})
    async def job(self,previous=None,comparison=False):
        selector='[data-testid="comparison-status"]' if comparison else '[data-testid="job-status"]'
        needed=2 if comparison else 1
        condition=f"(()=>{{const e=document.querySelector({json.dumps(selector)});return e&&e.dataset.jobId&&e.dataset.jobId!=={json.dumps(previous)}&&e.textContent.includes('已完成')&&document.querySelectorAll('[data-testid=heatmap][data-loaded=true]').length>={needed};}})()"
        await self.wait(condition,'current real UI job completed')
        return await self.value(f"document.querySelector({json.dumps(selector)}).dataset.jobId")
    async def api(self,path):
        return await self.value(f"fetch({json.dumps(path)}).then(async r=>{{if(!r.ok)throw new Error('HTTP '+r.status);return r.json();}})")
    async def screenshot(self,name,width=1440,height=1100):
        await self.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
        await self.value('window.scrollTo(0,0)');await asyncio.sleep(.3)
        shot=await self.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,'clip':{'x':0,'y':0,'width':width,'height':height,'scale':1}})
        path=self.output/(name+'.png');path.write_bytes(base64.b64decode(shot['data']))
        assert np.asarray(Image.open(path).convert('RGB')).std()>10,'blank screenshot'
        assert not await self.value('document.documentElement.scrollWidth>innerWidth+1'),'horizontal page overflow'


async def exercise(browser,base,smoke):
    b=browser;results=[]
    await b.call('Page.enable');await b.call('Runtime.enable')
    await b.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
    await b.call('Page.navigate',{'url':base+'/simulate'})
    first=await b.job();request=await b.api(f'/api/v1/workbench/jobs/{first}/request')
    assert request['scene']['terrain'] is not None and request['waveform']['kind']=='pulsed_lfm'
    await b.screenshot('simulate_initial')
    results.append({'check':'initial live LFM + terrain + array renders RD','job':first});print(results[-1],flush=True)
    assert not await b.value("document.body.innerText.includes('解析真值验证')||document.body.innerText.includes('通过率')")
    if not smoke:
        await b.number('[data-testid="height"]',6,blur=False)
        updated=await b.job(first);request=await b.api(f'/api/v1/workbench/jobs/{updated}/request')
        assert request['platform']['position_ned_m'][2]==-6000
        assert request['waveform']['adc_start_s']>request['waveform']['pulse_width_s']
        results.append({'check':'typing height without blur or Enter automatically updates simulation and gate','job':updated});print(results[-1],flush=True)
        for name in ['terrain-toggle','array-toggle','platform-motion','target-motion']:
            await b.click(f'[data-testid="{name}"]')
        latest=await b.job(updated);request=await b.api(f'/api/v1/workbench/jobs/{latest}/request')
        assert request['scene']['terrain'] is None
        assert request['array']['num_tx']==request['array']['num_rx']==1
        assert request['platform']['velocity_ned_mps']==[0,0,0]
        assert all(o['velocity_ned_mps']==[0,0,0] for o in request['scene']['objects'])
        state=await b.api(f'/api/v1/scene-simulations/{latest}')
        assert abs(state['frames'][0]['peak_candidates'][0]['apparent_closing_velocity_mps'])<1e-10
        results.append({'check':'four feature toggles alter actual backend configuration and Doppler','job':latest});print(results[-1],flush=True)
        await b.select('select[aria-label="波形"]','fmcw')
        fmcw=await b.job(latest);request=await b.api(f'/api/v1/workbench/jobs/{fmcw}/request')
        assert request['waveform']['kind']=='fmcw'
        results.append({'check':'same editable scene switches to FMCW and renders','job':fmcw});print(results[-1],flush=True)
        await b.call('Page.reload',{'ignoreCache':True});resumed=await b.job()
        assert resumed==fmcw,'refresh created a duplicate job rather than restoring it'
        results.append({'check':'refresh restores the existing result/job','job':resumed});print(results[-1],flush=True)
        await b.screenshot('simulate_controls')
        await b.number('[data-testid="frame-count"]',100)
        await b.wait("(()=>{const e=document.querySelector('[data-testid=job-status]');return e?.dataset.jobId&&e.textContent.includes('计算中')&&document.querySelector('[data-testid=heatmap]')?.dataset.loaded==='true'})()",'first frame is displayed before the long sequence finishes')
        older_id=await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")
        await b.number('[data-testid="height"]',6.1)
        await b.number('[data-testid="height"]',6.2)
        await b.wait(f"(()=>{{const e=document.querySelector('[data-testid=job-status]');return e?.dataset.jobId&&e.dataset.jobId!=={json.dumps(older_id)}&&e.textContent.includes('计算中')}})()",'latest edited parameters replace the running job')
        stopped_id=await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")
        latest=await b.api(f'/api/v1/workbench/jobs/{stopped_id}/request')
        assert latest['platform']['position_ned_m'][2]==-6200
        older=await b.api(f'/api/v1/scene-simulations/{older_id}')
        assert older['status']=='cancelled'
        results.append({'check':'rapid edits while computing cancel the old job and only submit the latest draft','old_job':older_id,'new_job':stopped_id});print(results[-1],flush=True)
        await b.value("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='停止').click()")
        await b.wait("document.querySelector('[data-testid=job-status]')?.textContent.includes('已停止')",'visible cancelled state')
        stopped=await b.api(f'/api/v1/scene-simulations/{stopped_id}')
        assert stopped['status']=='cancelled' and len(stopped['frames'])<100
        assert not await b.value("document.querySelector('[aria-label=自动更新仿真]').checked")
        results.append({'check':'stop cancels real computation and disables automatic restart','job':stopped_id,'completed_frames':len(stopped['frames'])});print(results[-1],flush=True)
        await b.number('[data-testid="frame-count"]',1)
        await b.wait("!document.querySelector('[data-testid=run-simulation]').disabled",'valid manual configuration')
        await asyncio.sleep(1)
        assert not await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")
        assert await b.value("document.querySelector('[data-testid=heatmap]').dataset.loaded==='false'")
        await b.click('[data-testid="run-simulation"]')
        manual=await b.job()
        results.append({'check':'manual mode does not auto-run and never displays the previous parameter result','job':manual});print(results[-1],flush=True)
        await b.number('input[aria-label="发射间隔"]',10)
        await b.wait("Boolean(document.querySelector('[role=alert]'))",'invalid physical timing visibly rejected')
        assert await b.value("document.querySelector('[data-testid=run-simulation]').disabled")
        await b.number('input[aria-label="发射间隔"]',300)
        await b.wait("!document.querySelector('[data-testid=run-simulation]').disabled&&!document.querySelector('[role=alert]')",'valid parameters recover')
        await b.click('[data-testid="run-simulation"]');recovered=await b.job(manual)
        results.append({'check':'invalid acquisition timing shows an error and recovers after correction','job':recovered});print(results[-1],flush=True)
        downloads=(b.output/'downloads').resolve();downloads.mkdir(exist_ok=True)
        await b.call('Browser.setDownloadBehavior',{'behavior':'allow','downloadPath':str(downloads)})
        await b.click('button[aria-label="导出场景配置"]')
        path=downloads/'radar-scene.json'
        for _ in range(60):
            if path.exists():break
            await asyncio.sleep(.1)
        exported=json.loads(path.read_text());assert exported['waveform']['kind']=='fmcw'
        exported['platform']['position_ned_m'][2]=-5500
        changed_file=b.output/'import_scene.json';changed_file.write_text(json.dumps(exported))
        await b.file('input[type=file][hidden]',changed_file)
        await b.wait("Number(document.querySelector('[data-testid=height]')?.value)===5.5&&!document.querySelector('[data-testid=run-simulation]').disabled",'configuration file import')
        await b.click('[data-testid="run-simulation"]');imported=await b.job(recovered)
        request=await b.api(f'/api/v1/workbench/jobs/{imported}/request');assert request['platform']['position_ned_m'][2]==-5500
        results.append({'check':'configuration export/import roundtrip changes actual simulation','job':imported});print(results[-1],flush=True)
        await b.click('[data-testid="terrain-toggle"]')
        dem=b.output/'dem.json';dem.write_text(json.dumps({'north_m':[-100,100],'east_m':[-100,100],'elevation_up_m':[[0,15],[10,25]]}))
        await b.file('.wb-file-button input[type=file]',dem)
        await b.wait("document.querySelector('select[aria-label=地形输入]')?.value==='dem'&&!document.querySelector('[data-testid=run-simulation]').disabled",'real DEM input accepted')
        await b.click('[data-testid="array-toggle"]')
        await b.select('select[aria-label="阵列布局"]','ura')
        await b.number('input[aria-label="Tx 数量"]',2)
        await b.number('input[aria-label="Rx 数量"]',4)
        await b.wait("!document.querySelector('[data-testid=run-simulation]').disabled",'planar array configuration')
        await b.click('[data-testid="run-simulation"]');planar=await b.job(imported)
        request=await b.api(f'/api/v1/workbench/jobs/{planar}/request')
        assert request['scene']['terrain']['kind']=='dem' and request['array']['layout']=='ura'
        assert request['array']['num_tx']==2 and request['array']['num_rx']==4
        await b.value("Array.from(document.querySelectorAll('.wb-tabs button')).find(b=>b.textContent==='方位–俯仰').click()")
        await b.wait("(()=>{const e=document.querySelector('[data-testid=heatmap]');return e?.dataset.loaded==='true'&&e.dataset.rows==='21'&&e.dataset.cols==='21'})()",'2D angle data visible')
        await b.screenshot('simulate_dem_and_angle')
        results.append({'check':'DEM import and planar MIMO generate an actual azimuth/elevation view','job':planar});print(results[-1],flush=True)
        await b.select('select[aria-label="场景预设"]','standard_pulsed_lfm')
        await b.number('[data-testid="frame-count"]',1)
        await b.wait("!document.querySelector('[data-testid=run-simulation]').disabled",'standard-quality scene ready')
        await b.click('[data-testid="run-simulation"]');standard=await b.job(planar)
        request=await b.api(f'/api/v1/workbench/jobs/{standard}/request')
        assert request['waveform']['bandwidth_hz']==20e6 and request['waveform']['loops']==64
        assert request['array']['num_tx']==4 and request['array']['num_rx']==4
        assert request['scene']['objects'][0]['divisions']==3
        results.append({'check':'standard high-altitude configuration remains runnable; interactive defaults do not replace full capabilities','job':standard});print(results[-1],flush=True)
        await b.number('input[aria-label="下视角"]',0)
        await b.wait("!document.querySelector('[data-testid=run-simulation]').disabled",'out-of-beam scenario ready')
        await b.click('[data-testid="run-simulation"]');empty=await b.job(standard)
        await b.wait("Boolean(document.querySelector('.wb-no-signal'))",'valid but zero-echo data explicitly distinguished from loading')
        results.append({'check':'zero-echo condition is explicitly shown, not a blank/uninitialized chart','job':empty});print(results[-1],flush=True)
        invalid={'schema_version':'1.0','name':'runtime geometry error check',
            'waveform':{'kind':'pulsed_lfm','carrier_hz':10e9,'bandwidth_hz':5e6,'sample_rate_hz':10e6,
                        'pulse_width_s':4e-6,'slot_period_s':100e-6,'adc_start_s':0,'samples':128,'loops':16},
            'platform':{'position_ned_m':[0,0,0],'velocity_ned_mps':[0,0,0],'boresight_ned':[1,0,0],'right_hint_ned':[0,1,0],'beam_half_angle_deg':15},
            'array':{'layout':'custom','num_tx':1,'num_rx':1,'spacing_m':None,'tx_local_m':[[0,0,0]],'rx_local_m':[[.01,0,0]],'tx_offsets_s':[0],'loop_period_s':100e-6},
            'scene':{'objects':[{'kind':'point','id':'collision','position_ned_m':[.01,0,0],'velocity_ned_mps':[0,0,0],'amplitude':1}]},
            'processing':{'range_limits_m':[0,100],'azimuth_deg':[0],'elevation_deg':[0]},'sequence':{'frame_count':1,'frame_interval_s':.1,'start_time_s':0}}
        failure_file=b.output/'runtime_error_scene.json';failure_file.write_text(json.dumps(invalid))
        await b.file('input[type=file][hidden]',failure_file)
        await b.wait("document.querySelector('[aria-label=场景预设]')?.value==='custom'&&!document.querySelector('[data-testid=run-simulation]').disabled",'custom single-channel geometry imported without silent collapse')
        await b.click('[data-testid="run-simulation"]')
        await b.wait("document.querySelector('[data-testid=job-status]')?.textContent.includes('计算失败')&&document.querySelector('[role=alert]')?.textContent.includes('天线')",'real backend computation failure is visible')
        failed_id=await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")
        failed_request=await b.api(f'/api/v1/workbench/jobs/{failed_id}/request')
        assert failed_request['array']['rx_local_m']==[[.01,0,0]]
        results.append({'check':'custom SISO geometry preserved and genuine runtime failure surfaced clearly','job':failed_id});print(results[-1],flush=True)
        await b.click('button[aria-label="恢复预设"]')
        await b.wait("!document.querySelector('[data-testid=run-simulation]').disabled&&!document.querySelector('[role=alert]')",'recovery from runtime failure')
        await b.click('[data-testid="run-simulation"]');working_again=await b.job(failed_id)
        results.append({'check':'workbench recovers after a failed backend job','job':working_again});print(results[-1],flush=True)
    await b.call('Page.navigate',{'url':base+'/compare'})
    pair=await b.job(comparison=True)
    await b.wait("Boolean(document.querySelector('[data-testid=real-camera]')?.naturalWidth)",'real paired camera loaded')
    saved=await b.value("JSON.parse(localStorage.getItem('radar.comparison.v1'))")
    assert saved['recordId'] and saved['request']['scene']['objects']
    refs=await b.value("Array.from(document.querySelectorAll('[data-testid=heatmap]')).map(e=>({ref:e.dataset.reference,rows:e.dataset.rows,cols:e.dataset.cols}))")
    assert refs[0]==refs[1]
    assert await b.value("document.querySelector('[data-testid=real-source]').textContent.includes('真实')")
    results.append({'check':'all four comparison panels use a real camera, labels, real ADC and unified simulation','job':pair,'record':saved['recordId']});print(results[-1],flush=True)
    await b.screenshot('comparison_four_panels',1440,1200)
    if not smoke:
        await b.select('select[aria-label="对比显示通道"]','tx0_rx0')
        await b.wait("document.querySelectorAll('[data-testid=heatmap][data-loaded=true]').length===2",'both channel-specific images loaded')
        refs=await b.value("Array.from(document.querySelectorAll('[data-testid=heatmap]')).map(e=>e.dataset.reference)")
        assert refs[0]==refs[1]
        results.append({'check':'channel selection updates both sides with one power reference'});print(results[-1],flush=True)
        await b.value("Array.from(document.querySelectorAll('.wb-tabs button')).find(b=>b.textContent==='编辑输入').click()")
        await b.click('.wb-input-fields details.wb-object summary')
        amplitude=await b.value("Number(document.querySelector('.wb-input-fields input[aria-label$=相对幅度]').value)")
        await b.number('.wb-input-fields input[aria-label$="相对幅度"]',amplitude*1.5)
        changed=await b.job(pair,comparison=True)
        request=await b.api(f'/api/v1/workbench/jobs/{changed}/request')
        assert abs(request['scene']['objects'][0]['amplitude']-amplitude*1.5)<1e-5
        assert request['waveform']['start_hz']==77e9
        results.append({'check':'editing simulation input launches paired current simulation; measured acquisition remains locked','job':changed});print(results[-1],flush=True)
        await b.click('button[aria-label="下一原始帧"]')
        other=await b.job(changed,comparison=True)
        newer=await b.value("JSON.parse(localStorage.getItem('radar.comparison.v1'))")
        assert newer['frameId']!=saved['frameId'] and newer['recordId']!=saved['recordId']
        await b.wait("Boolean(document.querySelector('[data-testid=real-camera]')?.naturalWidth)",'new frame camera loaded')
        results.append({'check':'changing raw frame replaces camera, inputs and both RD results','job':other});print(results[-1],flush=True)
        await b.call('Network.enable')
        await b.call('Network.setCacheDisabled',{'cacheDisabled':True})
        await b.call('Network.setBlockedURLs',{'urls':['*comparisons/ramp/records/*/camera*']})
        await b.call('Page.reload',{'ignoreCache':True})
        await b.wait("document.body.innerText.includes('此帧无可用相机')&&!document.querySelector('[data-testid=real-camera]')",'missing camera explicitly falls back to labelled geometry, not fake image')
        await b.value("Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='重试相机').click()")
        await b.wait("document.body.innerText.includes('此帧无可用相机')&&!document.querySelector('[data-testid=real-camera]')",'failed camera retry keeps truthful caption')
        await b.call('Network.setBlockedURLs',{'urls':[]})
        await b.value("Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='重试相机').click()")
        await b.wait("Boolean(document.querySelector('[data-testid=real-camera]')?.naturalWidth)",'real camera retry recovers')
        results.append({'check':'missing camera is not mislabeled, and retry restores the actual photo'});print(results[-1],flush=True)
    await b.screenshot('comparison_mobile',480,1050)
    assert not b.exceptions,b.exceptions
    return results


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8789')
    parser.add_argument('--output',type=Path,default=Path('.runtime/frontend_browser_checks'))
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='radar-frontend-browser-') as profile:
        chrome=subprocess.Popen(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
            '--disable-background-networking','--no-first-run','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',
            '--remote-allow-origins=http://localhost',f'--user-data-dir={profile}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            marker=Path(profile)/'DevToolsActivePort'
            for _ in range(100):
                if marker.exists():break
                await asyncio.sleep(.1)
            port=marker.read_text().splitlines()[0]
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=5) as response:pages=json.load(response)
            page=next(p for p in pages if p['type']=='page')
            async with websockets.connect(page['webSocketDebuggerUrl'],origin='http://localhost',max_size=40_000_000) as ws:
                browser=Browser(ws,args.output)
                try:results=await exercise(browser,args.base,args.smoke)
                except Exception:
                    await browser.screenshot('failure',1440,1200)
                    (args.output/'failure_body.txt').write_text(await browser.value('document.body.innerText'))
                    raise
                (args.output/'checks.json').write_text(json.dumps({'base':args.base,'checks':results,'uncaught_browser_exceptions':browser.exceptions},ensure_ascii=False,indent=2)+'\n')
        finally:
            chrome.terminate()
            try:chrome.wait(timeout=8)
            except subprocess.TimeoutExpired:chrome.kill();chrome.wait(timeout=5)


if __name__=='__main__':asyncio.run(main())
