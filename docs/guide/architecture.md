# 同一平台，配置不同场景

## 主链路

```text
前端 / CLI / Python / 数据准备脚本
               ↓
配置、单位、坐标与预算检查
               ↓
session_from_config：按采集模型选择 Session
               ↓
场景几何 + 相对运动 + 波形 + 阵列 + 散射/背景
               ↓
物理域对应的复数产品 + 元数据 + 来源快照
               ↓
RD / RA / RAD、比较、重放或应用层学习
```

数据集名不参与公共物理引擎的分支选择。适配器把公开参数、标签、几何或实测阵列转换成明确的配置与观测；仿真与实测处理均保留物理轴和复数通道。

## 关键代码

| 模块 | 职责 |
|---|---|
| `backend/radar_platform/cli.py`、`runner.py` | 安装后的公共命令；保存完整配置、源代码哈希、逐帧产品与失败状态 |
| `unified/schema.py`、`applicability.py`、`motion_check.py` | 配置/时序/预算/近似风险检查 |
| `unified/session.py` | 动态采集、静态扫频、统计背景、历史近似的统一会话选择 |
| `unified/scene.py`、`trajectory.py` | 编译点/面/DEM与平台轨迹；世界固定散射场景 |
| `unified/engine.py` | 共用动态回波合成；逐发射时隙更新双程路径、可见性与振幅 |
| `unified/geometry.py`、`radiation.py` | 阵元、TDM 时序、方向图、发射子阵与电子扫描 |
| `unified/visibility*.py` | 不透明三角面遮挡；NumPy/BVH 与可选 Numba 加速 |
| `unified/processing.py` | 距离压缩/FFT、Doppler、保留通道的波束形成与 RA/RAD |
| `unified/environment.py`、`land_response.py` | 噪声、背景、地表散射表的明确注入位置 |
| `unified/sfcw.py`、`background.py`、`separable_if.py` | 不同物理域的专用会话；不将静态频谱伪装成动态 RD |
| `unified/jobs.py`、`storage.py`、`provenance.py` | 作业、取消、二进制产品、可复现快照 |
| `workbench*.py`、`frontend/src/pages/WorkbenchPage.tsx` | 场景编辑、3D 展示、播放、结果检查和 A/B 比较 |
| `*_dataset.py`、`*_adapter.py` | 数据准备与观测解析，不替代公共回波引擎 |
| `navigation/`、`navigation_multidem/`、`fertilizer_pilot.py`、`xyz_features.py` | 可选应用研究；消费公共回波/RDA，不作为平台核心保真证明 |

## 三类物理产品不能混淆

1. 动态脉冲 LFM / FMCW：`adc[sample, loop, rx, tx]` → `rd_channels[range, doppler, rx, tx]` → `rad[range, doppler, elevation, azimuth]`。
2. 静态 SFCW：复数频率响应 `spectrum`，可用于静态距离响应；没有观测到运动，不能声称生成动态 RD。
3. 统计背景：已经分距离门的 I/Q 与分 CPI 的 RD；不是原始 ADC。

后台复杂数组始终保留，前端功率平均和 dB 压缩只用于显示。各模型最终产品形状、坐标、dtype、顺序以每帧 `metadata.json` 为准。

## 兼容与历史代码

`scripts/run_scene.py` 是公共 `radar-platform run` 的兼容入口，不再维护另一份运行实现。原高空、数组、静态表面和杂波 API 保留已有接口协议，其生产计算调用公共会话。

`legacy_simulator_reference.py`、`legacy_array_reference.py` 以及历史可分离 IF 保留为回归参考，不接入当前生产 API 的旧回波分支。删除它们会丢失可复核的数值基线，因此不以“代码行减少”作为清理目标。

`scripts/` 中研究脚本分为 `prepare_*`、`run_* / simulate_*`、`validate_* / audit_*`、`report_* / package_*`。后两类可能需要本地历史产物，不能因为源码已公开就声称数据包也已包含。

## 应用实验边界

应用层的训练、定位和闭环实验单独安装依赖，单独保存数据、权重与评估。核心回波程序不读取待验证目标的预测结果来“修正”模拟信号。固定场景的学习结果不等于未知地形泛化；正在运行的研究实验也不被发布为已验收功能。
