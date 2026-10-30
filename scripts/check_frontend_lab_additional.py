#!/usr/bin/env python3
"""Additional real-browser checks for state transitions, exports and measured data."""
import asyncio
import json
from pathlib import Path
import check_frontend_lab as harness

async def extra(b,base):
    checks=[]
    async def record(name,**data):checks.append(dict(check=name,**data));print(checks[-1],flush=True)
    async def go(path):
        await b.call('Page.navigate',{'url':base+path});await asyncio.sleep(.4)
    async def click_text(text):
        await b.value(f"Array.from(document.querySelectorAll('button')).find(e=>e.textContent.trim()==={json.dumps(text)}).click()")
    await b.call('Page.enable');await b.call('Runtime.enable');await b.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':900,'deviceScaleFactor':1,'mobile':False})
    await go('/simulate?job=demo-land-cover');await b.job()
    await b.wait("Boolean(document.querySelector('[aria-label=场景图层] option[value=echo]:not([disabled])'))",'archive auxiliary')
    await asyncio.sleep(3)
    # Inspect the actual chart model, not a substitute render.
    info=await b.value("(()=>{const e=document.querySelector('[_echarts_instance_]');let f=e[Object.keys(e).find(k=>k.startsWith('__reactFiber'))];while(f&&!f.stateNode?.getEchartsInstance)f=f.return;if(!f)return null;const c=f.stateNode.getEchartsInstance();window.chartUnderTest=c;const d=c.getModel().getSeriesByIndex(0).getData();return {count:d.count(),dimensions:d.dimensions,samples:[0,8000,16000].map(i=>({raw:d.getRawDataItem(i),visual:d.getItemVisual(i,'style'),layout:d.getItemLayout(i)})),option:c.getOption().visualMap};})()")
    print('chart model:',json.dumps(info),flush=True)
    await b.screenshot('complete_dem_chart',1440,900)
    await record('dense RD render and archive metadata',chart_count=info['count'] if info else None)
    await go('/simulate?case=points');await b.wait("!document.querySelector('[data-testid=run-simulation]')?.disabled&&Boolean(document.querySelector('[data-testid=height]'))",'point config')
    downloads=(b.output/'downloads').resolve();downloads.mkdir(exist_ok=True)
    await b.call('Browser.setDownloadBehavior',{'behavior':'allow','downloadPath':str(downloads)})
    await b.click('[aria-label="导出场景配置"]')
    for _ in range(50):
        if (downloads/'radar-scene.json').exists():break
        await asyncio.sleep(.1)
    config=json.loads((downloads/'radar-scene.json').read_text());config['visibility']['report_coverage']=True
    config['input_provenance']=['UI roundtrip acceptance'];config['receiver_noise']={'variance':1e-8,'seed':82,'provenance':'test relative noise'}
    config['sequence']['frame_count']=100;config['waveform']['loops']=256
    file=b.output/'long_request.json';file.write_text(json.dumps(config))
    await b.file('input[type=file][hidden]',file)
    await b.wait("Number(document.querySelector('[data-testid=frame-count]')?.value)===100&&!document.querySelector('[data-testid=run-simulation]').disabled",'imported config validated')
    await b.click('[data-testid=run-simulation]')
    await b.wait("Boolean(Array.from(document.querySelectorAll('button')).find(e=>e.textContent.includes('理解风险')))||Boolean(document.querySelector('[data-testid=job-status]')?.dataset.jobId)",'check before long run')
    await b.value("Array.from(document.querySelectorAll('button')).find(e=>e.textContent.includes('理解风险'))?.click()")
    await b.wait("document.querySelector('[data-testid=job-status]')?.textContent.includes('计算中')&&document.querySelector('[data-testid=heatmap]')?.dataset.loaded==='true'",'partial results during computation')
    job=await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")
    await click_text('停止');await b.wait("document.querySelector('[data-testid=job-status]')?.textContent.includes('已停止')",'cancelled state')
    saved=await b.api(f'/api/v1/workbench/jobs/{job}/request');state=await b.api(f'/api/v1/scene-simulations/{job}')
    assert 0<len(state['frames'])<100
    assert saved['input_provenance']==config['input_provenance'] and saved['receiver_noise']=={**config['receiver_noise'],'variance_per_rx_tx':None,'fast_time_spectral_power':None}
    assert saved['waveform']==config['waveform'] and saved['scene']['seed']==config['scene']['seed']
    await record('advanced config roundtrip, real cancellation, completed partial frames retained',job=job,frames=len(state['frames']))
    await click_text('导出图像')
    await b.value("Array.from(document.querySelectorAll('a')).find(e=>e.textContent.trim()==='导出复现包').click()")
    for _ in range(150):
        if list(downloads.glob('experiment-*.zip')) and (downloads/'radar-result.png').exists():break
        await asyncio.sleep(.1)
    assert list(downloads.glob('experiment-*.zip')) and (downloads/'radar-result.png').exists()
    import zipfile
    with zipfile.ZipFile(next(downloads.glob('experiment-*.zip'))) as z:
        assert 'source.zip' in z.namelist() and 'source_manifest.json' in z.namelist()
        assert json.loads(z.read('experiment.json'))['partial']
    await record('PNG and partial reproduction ZIP downloaded with source provenance')
    await b.number('[data-testid=frame-count]',3);await b.number('input[aria-label="CPI 循环数"]',32)
    await b.number('input[aria-label="发射 PRF"]',100)
    await b.wait("Boolean(document.querySelector('[role=alert]'))&&document.querySelector('[data-testid=run-simulation]').disabled",'invalid timing blocks submit')
    assert await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")==job
    await b.number('input[aria-label="发射 PRF"]',10)
    await b.wait("!document.querySelector('[data-testid=run-simulation]').disabled",'recovered parameters')
    await record('invalid timing is visible, previous result retained, correction recovers')
    assert await b.value("document.querySelector('[aria-label=自动更新仿真]').disabled")
    await go('/simulate?case=points')
    await b.wait("Boolean(document.querySelector('[aria-label=自动更新仿真]'))&&!document.querySelector('[aria-label=自动更新仿真]').disabled",'known light preset permits auto updates')
    await b.click('[aria-label="自动更新仿真"]')
    await b.number('[data-testid=height]',5.12,blur=False);await b.number('[data-testid=height]',5.15,blur=False)
    await b.wait("Boolean(Array.from(document.querySelectorAll('button')).find(e=>e.textContent.includes('理解风险')))||Boolean(document.querySelector('[data-testid=job-status]')?.dataset.jobId)",'automatic validation completes')
    await b.value("Array.from(document.querySelectorAll('button')).find(e=>e.textContent.includes('理解风险'))?.click()")
    newest=await b.job(job);current=await b.api(f'/api/v1/workbench/jobs/{newest}/request')
    assert current['platform']['position_ned_m'][2]==-5150
    await record('automatic light-scene updates coalesce to latest edited request',job=newest)
    await b.click('[aria-label="自动更新仿真"]')
    await b.call('Page.reload',{'ignoreCache':True});await b.job()
    assert await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")==newest
    await record('ordinary refresh restores current job without a new calculation')
    await b.select('[aria-label="仿真显示通道"]','tx1_rx3');await b.wait("document.querySelector('[data-testid=heatmap]')?.dataset.loaded==='true'",'single channel')
    await b.select('[aria-label="显示动态范围"]','100' if False else '-100')
    assert await b.value("document.querySelector('[data-testid=job-status]').dataset.jobId")==newest
    await click_text('运动曲线');await b.wait("Boolean(document.querySelector('[aria-label=运动曲线]'))",'curves');await b.select('[aria-label=运动曲线]','closing')
    await b.screenshot('curves',1440,900)
    await record('channel, color scale and motion curves do not resubmit physics')
    await go('/compare');real_job=await b.job(comparison=True)
    await b.wait("document.querySelectorAll('[data-testid=heatmap][data-loaded=true]').length===2",'both measured and simulated RD',timeout=180)
    await b.screenshot('measured_complete',1440,1200)
    await record('RAMP camera/labels and real/simulated RD complete',job=real_job)
    assert not b.exceptions,b.exceptions
    return checks

harness.exercise=extra
if __name__=='__main__':asyncio.run(harness.main())
