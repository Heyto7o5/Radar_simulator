#!/usr/bin/env python3
"""Official-source access audit for the user-authorized ColoRadar fallback.

No authentication bypass, mirrors, credentials, or bulk data downloads.
"""
import hashlib,json,urllib.request,urllib.error
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
DATA=Path('/nas/datasets/hyt/datasets/radar_open/coloradar')
OUT=PROJECT/'docs/coloradar/2026-10-16'

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.append(dict(attrs).get('href',''))

def fetch(url,limit=200000):
    with urllib.request.urlopen(url,timeout=25) as r:
        data=r.read(limit+1)
        if len(data)>limit:raise ValueError('metadata limit exceeded')
        return data

def main():
    for p in [DATA/'raw/source_metadata',DATA/'extracted',DATA/'processed',OUT]:p.mkdir(parents=True,exist_ok=True)
    source='https://arpg.colorado.edu/coloradar/'
    page=fetch(source);parser=Links();parser.feed(page.decode())
    share=next(u for u in parser.links if 'sharepoint.com' in u)
    access={'original_dataset_page':source,'original_download_link':share}
    try:
        with urllib.request.urlopen(share,timeout=25) as r:
            sample=r.read(16000).decode(errors='replace')
            access['original_download']={'http_status':r.status,'final_host':r.url.split('/')[2],
                'login_text_seen':any(s in sample.lower() for s in ['sign in','login.microsoft']),
                'raw_data_acquired':False}
    except urllib.error.HTTPError as e:
        access['original_download']={'http_status':e.code,'raw_data_acquired':False,
            'interpretation':'access denied from this server; login requirement, expiry and network policy not distinguished'}
    plus='https://arpg.colorado.edu/coloradarplus/'
    parser=Links();parser.feed(fetch(plus).decode())
    globus=next(u for u in parser.links if 'app.globus.org' in u)
    access['plus_dataset_page']=plus;access['plus_download_link']=globus
    endpoint='https://transfer.api.globus.org/v0.10/endpoint/986f22f1-394e-4dc8-b7c5-77eb7eceb3d4'
    try:
        access['plus_metadata']=json.loads(fetch(endpoint))
    except urllib.error.HTTPError as e:
        body=e.read(4000).decode(errors='replace')
        access['plus_metadata']={'http_status':e.code,'error':json.loads(body) if body.lstrip().startswith('{') else 'request failed'}
    repo='https://api.github.com/repos/arpg/coloradar-library/git/trees/main?recursive=1'
    tree=json.loads(fetch(repo,2_000_000));revision=tree['sha'];sources=[]
    for name in ['README.md','include/configs/radar_configs.h','src/configs/radar_configs.cpp',
                 'src/run/coloradar_run.cpp','src/run/coloradar_plus_run.cpp']:
        url=f'https://raw.githubusercontent.com/arpg/coloradar-library/{revision}/{name}'
        content=fetch(url);target=DATA/'raw/source_metadata'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        sources.append({'source':url,'file':str(target.relative_to(DATA)),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    manifest={'checked_utc':datetime.now(timezone.utc).isoformat(),
              'target':'ColoRadar platform-motion validation; explicitly authorized fallback from gated Sim2Radar',
              'raw_data_acquired':False,'simulation_aligned_to_dataset':False,'access':access,
              'official_library_revision':revision,'source_metadata':sources,
              'preferred_original_sequence':'2_24_2021_aspen_run9 (83s, motion-capture environment); availability not confirmed',
              'alternative':'a short indoor ColoRadar+ sequence, only with its own matching calibration and pose files',
              'required_files':['one sequence of raw ADC with timestamps','matching radar waveform/antenna/channel calibration',
                                'ground-truth pose with timestamps and reference frame','radar-to-body/IMU/LiDAR extrinsics',
                                'LiDAR scans and timestamps for static-scene geometry'],
              'do_not_mix_original_and_plus_configs':True,
              'source_review_caution':'current official ColoradarRun implementation assigns singleChipScansDirPath_ to cascade; verify actual archive structure rather than silently trusting this path',
              'auth_action':'user login/download or user-provided local directory; do not send passwords/tokens in chat'}
    (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (OUT/'access_and_input_requirements.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (OUT/'ColoRadar接入状态.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1000px;margin:30px auto;font:16px/1.7 sans-serif}td,th{border:1px solid #bbb;padding:10px}table{border-collapse:collapse}</style>
<h1>ColoRadar：平台运动验证接入</h1><p>依据用户选择，从需审批的 Sim2Radar 切换到 ColoRadar。当前只取得官方资料和处理库源码，尚未取得真实 ADC，不生成虚假的对齐图或实测 MAE。</p>
<table><tr><th>入口</th><th>本次服务器检查</th></tr><tr><td>原版 ColoRadar 官方 SharePoint</td><td>返回 HTTP 403；具体是登录、链接失效还是网络限制，尚不能判定。</td></tr><tr><td>ColoRadar+ 官方 Globus</td><td>元数据接口提示未提供凭据，需要正常登录授权。</td></tr><tr><td>官方处理库</td><td>已记录版本并保存配置与读取代码，用于后续核对格式；未部署容器或启动前端。</td></tr></table>
<p>最小输入：一个短序列的 ADC 和时间戳、配套 calib 标定、位姿及时间戳、传感器外参、LiDAR。优先核查原版 Aspen 动捕场景；若只可获取 Plus，选其短室内序列并使用对应版本参数。</p>
<p>先处理实测 ADC 得到 RD，再接平台轨迹、静态几何和同配置仿真。位姿或配置缺失时保留缺口，不把控制指令、假设参数或处理后的点云冒充真值。</p>
<p><a href="https://arpg.colorado.edu/coloradar/">原版官方入口</a> · <a href="https://arpg.colorado.edu/coloradarplus/">Plus 官方入口</a> · <a href="https://github.com/arpg/coloradar-library">官方处理库</a> · <a href="access_and_input_requirements.json">详细访问记录与输入清单</a></p></html>''')
    print(json.dumps({'raw_data_acquired':False,'original_http_status':access['original_download']['http_status'],
                      'plus_metadata_status':access['plus_metadata'].get('http_status'),'source_files':len(sources),
                      'report':str(OUT/'ColoRadar接入状态.html')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
