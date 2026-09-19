# hydrogeo-insar v0.1

第一版通用 InSAR–地下水水文测地分析框架。它不处理 InSAR 参考点、大气改正或轨道改正；输入应为已经完成校正的最终形变 GeoTIFF 时序。

## 1. InSAR 输入约定

文件名默认：

```text
geo_YYYYMMDD_YYYYMMDD.tif
```

程序使用第二个日期作为观测日期。所有 GeoTIFF 必须具有一致的 CRS、transform 和 shape。正负号通过 YAML 声明，内部默认推荐“正=抬升、负=沉降”。

## 2. 地下水输入

支持 CSV / Excel，支持 wide 和 long 两种布局。内部统一为：

```text
station_id,date,lon,lat,head_m,aquifer_class,well_depth_m,elevation_m
```

不允许根据井深推断含水层组；`aquifer_class` 由原始数据显式标签提供。

## 3. 当前实现的科学模块

1. GeoTIFF InSAR 时序标准化为 HDF5 canonical stack；
2. 地下水观测标准化；
3. 时间低秩 + 空间 Gaussian RBF 连续承压水头场；
4. InSAR `quadratic + annual harmonic` 分解；
5. 基于长期形变特征的 K-means 演化类型；
6. 地下水年周期谐波、区域 lag、有效 `Ske`；
7. 总储量 / 可恢复储量 / 不可恢复储量预算；
8. 任意层数砂层/黏土层厚度分析；
9. 单点或多点分层标深度贡献；
10. 按形变类型汇总含水系统响应。

## 4. 安装

```bash
pip install -e .
```

## 5. 运行

复制并修改：

```text
configs/example_project.yaml
```

查看阶段：

```bash
hydrogeo-insar list-stages
```

完整运行：

```bash
hydrogeo-insar run configs/example_project.yaml
```

从某阶段开始：

```bash
hydrogeo-insar run configs/example_project.yaml --from storage-budget
```

仅运行一个阶段：

```bash
hydrogeo-insar stage configs/example_project.yaml classify-deformation
```

## 6. 主要输出

```text
outputs/
├── canonical/
│   ├── insar_stack.h5
│   └── groundwater.csv
├── groundwater/
│   └── groundwater_field.h5
├── deformation/
├── regimes/
├── seasonal/
├── storage/
├── hydrostratigraphy/
├── extensometer/
└── synthesis/
```

## 7. 储量符号

内部统一：

```text
正位移 = 抬升
正水头变化 = 水头恢复
正储量变化 = 储量增加
```

并定义：

```text
V_total = V_recoverable + V_irreversible
```

因此负的 `V_irreversible` 表示不可恢复储量减少；输出同时提供正值形式的 `irreversible_storage_loss_m3`。

## 8. v0.1 的定位

该版本优先建立通用数据契约和完整科学计算链，不保留旧工程中的 release hash、固定期望值、参考点模块或大量验收 gate。`Ske` 当前采用局地二维谐波最小二乘，可通过 `support_sigma_pixels` 设置局地支持平滑；后续可在不改变上下游接口的情况下替换为 Laplacian / RBF 正则化反演。
