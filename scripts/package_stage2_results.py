#!/usr/bin/env python3
"""Package portable figures/reports/evidence, never raw public tiles or ADC cubes."""
import argparse,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=ROOT/'.runtime/stage2_20261023'
    audit=json.loads((root/'completion_audit.json').read_text())
    if audit['status']!='complete_for_requested_functional_scope':raise ValueError('completed evidence audit required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for source,label in [(root/'final_report_v1','results'),(root/'input_report_v1','inputs'),(root/'scan_report_v1','scan')]:
        files.extend((x,Path(label)/x.name) for x in source.iterdir() if x.suffix in ('.html','.png','.json','.csv'))
    for name in ['completion_audit.json','config_audit.json','regression.json','tests_validation.xml']:
        files.append((root/name,Path('evidence')/name))
    files.extend((x,Path('evidence/coverage')/x.name) for x in (root/'coverage_audit').glob('*.json'))
    files.append((ROOT/'docs/worldcover_integrity_check.json',Path('evidence/worldcover_integrity_check.json')))
    for folder in ['campaign_v1','scan_campaign_v1']:
        for name in ['campaign_inputs.json','campaign_status.json']:
            files.append((root/folder/name,Path('evidence')/folder/name))
    with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED) as archive:
        for source,destination in files:archive.write(source,destination)
        archive.writestr('README.txt',
            '本包为真实DEM第二批功能的图表与验收记录，不是完整源码或原始数据包。\n'
            '先打开 results/report.html；图片已内嵌，可单独复制HTML。inputs/report.html说明地物输入与解析波束。\n'
            '真实输入为Copernicus DSM及WorldCover分类；NRCS控制值和理想天线不是实测材料/硬件标定。\n'
            '45个参数组合与6个同刻扫描对照来自3个地理窗口；随机种子不是新增真实场景。\n'
            '原始GeoTIFF保留在服务器NAS，复数ADC/RD和全部配置保留在项目运行目录。\n'
            '中断的未完成尝试保留在服务器的.interrupted目录，未计作额外样本。\n'
            '来源：https://registry.opendata.aws/copernicus-dem/  https://esa-worldcover.org/en/data-access\n')
    with zipfile.ZipFile(a.output) as archive:
        assert archive.testzip() is None
        count=len(archive.namelist())
    print(json.dumps({'zip':str(a.output),'files':count,'bytes':a.output.stat().st_size,'crc_passed':True}))


if __name__=='__main__':main()
