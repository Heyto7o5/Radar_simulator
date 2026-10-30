# 数据、结果与复现

## 数据不进入源码仓库

公开原始雷达数据、DEM/遥感栅格、抽取缓存、模型权重和大型结果应独立管理。用户研究服务器的约定根目录为 `/nas/datasets/hyt/datasets/radar_open`；启动服务前通过 `RADAR_DATA_ROOT` 明确设置。未设置时 API 仍可启动，RAMP 页面明确标示无真实数据或合成回退，不能伪装成实测。

```bash
export RADAR_DATA_ROOT=/nas/datasets/hyt/datasets/radar_open
python scripts/datasets.py status
python scripts/datasets.py download ramp_adc
python scripts/datasets.py verify ramp_adc
python scripts/datasets.py extract ramp_adc
python scripts/datasets.py prepare-demo ramp_adc
```

下载会产生外部写入和可能较大流量，不属于快速开始或测试的默认步骤。当前 RAMP 下载工具保留研究环境的 NAS 安全约束；其他环境需明确配置外部数据管理方式，不能默默写入源码目录。

## 适配器与公共引擎

- RAMP：`prepare_ramp_scene.py` 导出目标标签条件配置，再使用公共运行入口。
- DART：`prepare_dart_scene.py` 接收独立几何、轨迹和雷达参数，只准备配置，不将实测 RD 写入仿真结果。
- MAROON / 玛曲：静态几何、频率与观测对照，走静态表面会话或明确的散射资料分析。
- IPIX：距离门 I/Q 与统计模型；观测/校准段和验证段必须区分。
- 实际地形：`download_copernicus_dem.py` → `prepare_real_dem.py` → `prepare_real_dem_simulation.py` → 公共运行入口；土地覆盖另由 `download_worldcover.py` / `prepare_worldcover.py` 处理。

每个工具的完整输入使用 `python scripts/<工具>.py --help` 查询。部分历史研究脚本包含原实验目录，属于复现辅助而非通用入口；源码仓库不附带这些历史数据。

## 主要来源与许可

| 来源 | 官方入口 | 处理原则 |
|---|---|---|
| RAMP 原始 ADC | https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection | 原始数据独立下载，遵守提供方条款 |
| MAROON | https://github.com/vwirth/maroon | 几何、校准与雷达数据分别记录来源 |
| IPIX | https://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html | 记录采集文件和统计校准范围 |
| 玛曲 | https://essd.copernicus.org/articles/13/2819/2021/ | 使用测量覆盖的频率、极化和入射角 |
| Copernicus DEM | https://registry.opendata.aws/copernicus-dem/ | 保留原始许可和高程基准；DSM 不等于裸地 DTM |
| ESA WorldCover | https://esa-worldcover.org/en/data-access | 保留版本、类别与来源；类别不等于材料标定 |

本仓库并不转授第三方数据或模型的许可。研究者应在下载时核对当前条款。项目代码的开源许可由仓库所有者决定；没有 LICENSE 不意味着可无条件再分发。

## 运行产物

每个离线运行使用一个全新目录，禁止覆盖旧结果。主要文件：

- `request.json`：实际执行配置，包含所有默认值。
- `scene.json` / `scatterers.npz`：当前模型的几何与散射元信息；不是所有采集模型都有几何。
- `run.json`：状态、帧清单、耗时、配置与源码哈希。
- `source.zip`：运行时源代码快照；服务修改代码后应重启，避免磁盘与已加载模块不一致。
- `frame_0000/metadata.json`：物理轴、模型假设、产品形状、dtype、顺序和警告。
- `frame_0000/*.bin`：原生复数/功率数据，按元数据解释，不通过图片反推数值。

```python
import json
from pathlib import Path
import numpy as np

frame = Path('.runtime/example/frame_0000')
meta = json.loads((frame / 'metadata.json').read_text())
spec = meta['products']['rd_channels']
rd = np.fromfile(frame / spec['file'], dtype=spec['dtype']).reshape(spec['shape'])
```

`.runtime` 全部留在本地。论文或报告可单独发布经检查的结果包，但不能把敏感配置、会话历史、凭据或原始第三方数据混入源码提交。
