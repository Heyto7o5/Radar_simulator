#!/usr/bin/env python3
"""Archive completed world-geometry comparison with source and related evidence."""
import hashlib,json,zipfile
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
DART=PROJECT/'docs/validation_diagnostics/2026-10-15/dart'
WORLD=DART/'batch_v2_world_geometry'

def main():
    if json.loads((WORLD/'state.json').read_text())['status']!='complete':raise ValueError('world batch is not complete')
    for name in ['扩展世界几何RD对照.html','summary.json','per_frame_metrics.csv','report_selection.json','output_integrity_audit.json']:
        if not (WORLD/name).is_file():raise FileNotFoundError(name)
    integrity=json.loads((WORLD/'output_integrity_audit.json').read_text())
    if integrity['partial_audit'] or integrity['checked_completed_frames']!=11:raise ValueError('all-frame integrity audit required')
    target=PROJECT/'docs/deliveries/DART扩展世界几何对照_2026-10-15.zip'
    files=[]
    for folder in ['backend','scripts','tests']:
        files.extend((p,Path('source')/p.relative_to(PROJECT)) for p in (PROJECT/folder).rglob('*.py'))
    files.extend((PROJECT/name,Path('source')/name) for name in ['pyproject.toml','README.md'])
    files.extend((p,Path('source')/p.relative_to(PROJECT)) for p in DART.rglob('*') if p.is_file() and p.suffix in {'.html','.json','.png','.csv'})
    inventory=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in sorted(files,key=lambda item:str(item[1])):
            content=path.read_bytes();z.writestr(str(name),content)
            inventory.append({'file':str(name),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        z.writestr('index.html','''<!doctype html><html lang="zh"><meta charset="utf-8"><h1>DART 扩展世界几何对照</h1>
<p><a href="source/docs/validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry/扩展世界几何RD对照.html">打开主报告</a></p>
<p>报告、配图、参数、逐帧指标与源码已打包。原始数据、ADC/RD 计算缓存和依赖环境不包含在此包内；复算需要服务器的数据根目录与对应依赖，或按脚本重新准备数据和生成缓存。它不是无需数据即可运行的独立安装包。</p>
<p>保留了旧几何、白噪声与有色背景对照。不同版本的冻结代码检查应使用对应归档快照，不应修改校验值来绕过版本检查。</p></html>''')
        z.writestr('inventory.json',json.dumps({'raw_data_included':False,'computed_ADC_cache_included':False,'files':inventory},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC check failed')
    print(json.dumps({'zip':str(target),'bytes':target.stat().st_size,'files':len(inventory),'crc_verified':True},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
