# Radar Simulator · 雷达回波与 RD / RA / RAD 仿真平台

面向高空对地等科研场景的可配置雷达仿真平台。将**场景、运动、波形、阵列、散射、背景和信号处理**串在同一套公共引擎中，同时提供浏览器工作台、命令行、Python 和 REST/WebSocket 接口。

本项目是通用雷达科研工具，不限定某一种载体或用途。**能够运行、解析测试通过、真实数据局部对照、整体实测保真是不同结论。** 当前不宣称已经复现高空厘米波真实地表的全部电磁效应。

## 一条完整工作流程

```text
选择/导入场景 → 设置雷达与运动 → 参数和模型适用性检查
                                      ↓
                           公共 Session / 回波引擎
                                      ↓
                 复数 ADC / 通道 RD / RA / RAD 与物理轴
                                      ↓
                  场景联动、剖面、A/B 比较、保存与复现
```

数据集适配器准备输入和实测观测，不按数据集名字选择另一套生产仿真器。已有兼容 API 和离线脚本最终调用公共会话；静态频谱和统计距离门 I/Q 保留各自物理含义，不强行包装成 ADC 或动态 RD。

## 已集成功能

| 能力 | 内容 | 主要限制 |
|---|---|---|
| 波形与采集 | 脉冲 LFM、线性调频 FMCW；静态 SFCW | LFM 是调频形式，FMCW 是连续波采集体制；不是两个同层级概念 |
| 平台与目标 | 双方三维匀速；采样平台位置、速度、四元数姿态；跨帧相位 | 时隙内一阶路径近似；复杂组合需适用性检查 |
| 点与面目标 | 点、点云、板、箱体、三角网格、平地/山丘、外部 DEM | 理想散射/标量物理光学，不是完整物体数字孪生 |
| 地形与建筑 | 实际 DSM 导入、地物类别、建筑几何、双路径遮挡与覆盖 | DSM 不等于裸地；无一般环境多径、衍射或穿透 |
| 阵列 | SISO、SIMO、TDM-MIMO；ULA、URA、自定义阵元；RA/RAD | 理想远场阵列；基本 TDM 补偿不等于速度解模糊 |
| 天线与扫描 | 阵元方向图、相干发射子阵、幅相加权、电子扫描 | 单位驱动归一化；无绝对增益、互耦等实测标定 |
| 地物散射 | 按频率、入射角、极化、类别配置反射强度表 | 资料不足处明确标为假设，禁止无提示外推 |
| 噪声与杂波 | ADC 白/有色高斯；相关高斯、复合纹理距离门背景 | 不是完整真实接收机，也不是海浪三维重建 |
| 公开数据对照 | RAMP、MAROON、IPIX、玛曲、TSMS、DART 等准备/评价工具 | 每套数据只验证它实际提供的几何、处理或统计信息 |
| 科研工作台 | 3D 场景、参数/JSON 编辑、预检、逐帧播放、剖面、A/B、导出 | 单用户、无鉴权；大型地形为 CPU 批处理而非实时物理仿真 |
| 可选应用研究 | RDA 特征、位置反演、导航实验、跨 DEM 学习 | 应用层消费公共引擎；实验结果不替代核心回波保真验证 |

不同功能不是任意组合都成立。运行前检查会返回 `usable / risk / unsupported`，并解释接收窗、模糊、频带、距离走动、几何与资源风险。`usable` 不是实测保真认证。

## 快速开始

需要 Python 3.10–3.12；前端需要 Node.js 20+ 与 npm。核心仿真不需要 GPU、外部数据或学习权重。

```bash
git clone https://github.com/Heyto7o5/Radar_simulator.git
cd Radar_simulator
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

先离线运行一个明确的理想场景：

```bash
radar-platform capabilities
radar-platform check --config configs/unified/lab_fmcw_mimo.json
radar-platform run --config configs/unified/lab_fmcw_mimo.json --output .runtime/example
python scripts/render_scene.py --run .runtime/example --output .runtime/example_figures
```

`radar-platform run`、`python -m radar_platform run` 与历史 `python scripts/run_scene.py` 调用同一个运行实现。输出目录必须是新目录，避免覆盖实验。

启动工作台：

```bash
cd frontend
npm ci
npm run build
cd ..
radar-platform serve
```

浏览器打开 `http://127.0.0.1:8000/simulate`。前端与 API 同源，页面还包括 `/experiments`、`/compare`、`/cases`。开发模式可以直接运行 `bash scripts/dev.sh`；需要内网访问时显式设置 `RADAR_API_HOST=0.0.0.0` 和 `RADAR_UI_HOST=0.0.0.0`。**服务没有鉴权，不应直接暴露到公网。**

首次启动没有历史缓存，案例中的真实 DEM 归档可能显示未就绪，这是正常状态。仓库不附带本机历史运行、原始数据或权重，也不把缺失真实数据替换后标成实测。

## 常用配置与入口

| 任务 | 配置 / 工具 |
|---|---|
| 高空脉冲 LFM 点目标 | `configs/unified/high_altitude_1ghz.json` |
| FMCW 2Tx4Rx 与角度 | `configs/unified/lab_fmcw_mimo.json` |
| 有限金属板动态回波 | `configs/lab_moving_plate_po.json` |
| 理想山脊遮挡检查 | `configs/unified/high_altitude_ridge_visibility.json`，明确是理想几何 |
| 发射子阵与扫描 | `configs/unified/phased_scan_10ghz.json` |
| 静态表面频率响应 | `configs/surface_plane.json` |
| 统计背景 | `configs/unified/statistical_background.json` |
| 历史近似复现 | `configs/unified/block_texture_background.json`、`legacy_separable_if.json` |
| 实际 DEM / 地物 | `prepare_real_dem*.py`、`prepare_worldcover.py`，随后公共 `run` |
| 建筑加入场景 | `scripts/add_dem_buildings.py --help` |
| RAMP / DART 输入准备 | `scripts/prepare_ramp_scene.py`、`scripts/prepare_dart_scene.py` |
| 运动误差筛查 | `radar-platform check --config <配置.json> --motion` |
| 学习和应用 | [应用研究入口与限制](docs/guide/applications.md) |

上述预设不是全部无模糊：高速或长 CPI 案例可能故意用于观察折叠和距离走动。不要为了消除警告而暗中降低速度、更换地形或改变载频。

Python 同样使用公共会话：

```python
import json
from pathlib import Path
from radar_platform.runner import run_configuration

config = json.loads(Path('configs/unified/lab_fmcw_mimo.json').read_text())
run_configuration(config, '.runtime/python_example')
```

核心 HTTP 接口：

```text
GET    /api/v1/health
GET    /api/v1/capabilities
GET    /api/v1/scene-presets
POST   /api/v1/scene-check
POST   /api/v1/scene-motion-check
POST   /api/v1/scene-simulations
GET    /api/v1/scene-simulations/{job_id}
DELETE /api/v1/scene-simulations/{job_id}
GET    /api/v1/scene-simulations/{job_id}/frames/{frame_index}/data/{product}
WS     /api/v1/scene-simulations/{job_id}/events
```

API 详细字段见运行后的 `/docs`；二进制产品按照元数据中的 shape/dtype/order 读取，不假设所有模型具有同样的轴。

## 安装可选能力

```bash
# 栅格和地理坐标导入
python -m pip install -e '.[dev,geo]'

# 学习、Numba 加速和应用研究；不启动训练
python -m pip install -e '.[dev,geo,navigation]'
```

`RADAR_DATA_ROOT` 指向外部数据根目录，`RADAR_RUNTIME_ROOT` 指向运行结果目录。用户研究环境使用 `/nas/datasets/hyt/datasets/radar_open`；其他机器需要自行准备合法可用的数据。未配置外部数据时核心仿真仍可运行。

## 测试与验证

```bash
python -m pytest
python scripts/verify_configuration_matrix.py --output .runtime/configuration_matrix
```

基础依赖未安装时，相应可选学习/Numba 测试会明确跳过；真实 DEM 集成检查没有本地数据时也会明确跳过。要做全能力验收，需要完整可选依赖和相应数据，不能把跳过项称为通过。

验证分为三层：解析/程序测试、独立数值或几何对照、真实数据迁移比较。真实地形、测试通过或应用模型收敛，都不能单独证明高空厘米波回波整体保真。当前尚缺同场景实测、绝对功率标定、通用环境多径及完整接收机非理想。

## 工程结构

```text
backend/radar_platform/
  cli.py, runner.py, configuration.py  公共入口
  unified/                            回波、运动、阵列、处理、会话、作业
  *_dataset.py, *_adapter.py           公开数据与输入准备
  workbench*.py                       工作台与结果服务
  navigation/, navigation_multidem/   可选应用研究
frontend/src/                         当前科研工作台
configs/                             可复现配置
scripts/                             准备、批处理、审计、报告辅助工具
tests/                               数值、接口与集成测试
docs/guide/                          当前维护文档
.runtime/                            本地结果、历史归档与快照，不提交
```

[文档索引](docs/README.md) · [架构与模块](docs/guide/architecture.md) · [功能与验证](docs/guide/features-and-validation.md) · [数据与复现](docs/guide/data-and-reproducibility.md) · [开发与发布](docs/guide/development.md)

历史原始工程不作为本项目的 Python import 路径。部分研究复现脚本需要用户提供原实验输入，默认运行链路不依赖这些历史文件。第三方数据遵守各自许可；项目代码许可由仓库所有者确认，未提供 LICENSE 不表示自动开放所有再分发权利。
