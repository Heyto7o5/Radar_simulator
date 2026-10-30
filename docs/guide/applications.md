# 应用研究入口

这些模块是公共仿真平台的使用者，不是另一套回波引擎。它们不是工作台启动或普通仿真的必选依赖。

| 方向 | 主要入口 | 输入与边界 |
|---|---|---|
| 固定场景多帧 RDA 研究 | `run_fertilizer_rda_pilot.py`、`predict_fertilizer_correction.py`、`audit_fertilizer_rda_pilot.py` | 固定标定、数值 RDA 与物理轴；理想落点模型，不代表真实飞行/颗粒精度 |
| XYZ 离线反演 | `generate_xyz_learning_data.py`、`train_xyz_learning.py`、`predict_xyz_learning.py`、`audit_xyz_learning.py` | 独立的训练/验证/测试配置；固定场景误差不能外推到未知地图 |
| 场景与建筑 | `add_dem_buildings.py`、`prepare_navigation.py` | 建筑加入三维几何，接受遮挡与信号生成；不是直接在图上画亮点 |
| 单场景应用实验 | `generate_navigation_data.py`、`train_navigation_v2.py`、`run_navigation.py`、`audit_navigation.py` | 只允许按明确观测契约使用雷达图；实验与控制的假设单独记录 |
| 跨 DEM 研究 | `configs/navigation_multidem_v1.json` / `v2.json`、`prepare_navigation_multidem.py`、`run_navigation_multidem_training.py` | 地区划分、地物/几何输入、训练与保留地区严格分开；进行中的批次不算最终通过 |

先安装可选环境，再查看每个工具的帮助：

```bash
python -m pip install -e '.[dev,geo,navigation]'
python scripts/add_dem_buildings.py --help
python scripts/prepare_navigation.py --help
python scripts/run_navigation_multidem_training.py --help
```

部分早期应用入口默认指向原研究的 `.runtime` 配置。克隆仓库后不会自带这些数据；应显式准备来源资料与实验配置，再运行。绝不将缺失实测/真实地形静默换成理想场景。项目内路径约束用于避免应用脚本访问任意外部文件；只读外部来源的导入应先通过专门的数据准备工具完成。

使用公共 `SimulationSession` 或公共采集链路生成观测。原始配置、真值只在模拟器、离线标注和独立评估中使用；预测接口不应读取待测真实位置或 DEM 来修正答案。

完整应用验收必须同时包含：冻结模型、数据划分、独立测试、失败案例、输入隔离、可复现结果。代码在仓库中可用不意味着所有应用实验已完成验收。
