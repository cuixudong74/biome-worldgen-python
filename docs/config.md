# 配置项

所有配置均为不可变数据类（`frozen dataclass`），创建后不可修改，保证
同一份配置下生成结果始终一致。

## WorldGenConfig

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `chunk_size` | 16 | 区块边长（单元数），区块为正方形 |
| `sea_level` | 0.45 | 高度低于该值的单元为海洋（0..1） |
| `terrain_octaves` | 4 | 地形高度 fBm 叠加倍频数 |
| `base_temperature` | 26.0 | 赤道基准温度（°C） |
| `latitude_factor` | 0.07 | 每单位 z 距离的温度递减（°C） |
| `temperature_lapse` | 7.5 | 每单位地形高度的温度递减（°C） |
| `climate_octaves` | 3 | 温度/湿度噪声的 fBm 叠加倍频数 |
| `propagation_enabled` | True | 是否启用跨区块传播（关闭后植被不生长，仅用于调试） |
| `max_propagation_rounds` | 300 | 传播轮数硬上限（安全网，正常会提前收敛） |
| `convergence_epsilon` | 1e-12 | 收敛阈值：一轮中所有单元字段最大变化低于该值即停止 |
| `field_decimals` | 6 | 提交前字段取整的小数位数（保证远场扰动确定性舍入） |
| `moisture_diffusion` | 0.20 | 每轮与四邻均衡湿度的比例 |
| `moisture_advection` | 0.30 | 每轮被上风单元替换的湿度比例 |
| `wind_dx` | 1 | 风向 x 分量（上风方向为 `-wind_dx`） |
| `wind_dz` | 0 | 风向 z 分量 |
| `ocean_evaporation` | 0.15 | 海洋单元每轮向 1.0 松弛的速率 |
| `rain_shadow` | 0.12 | 陆地因地形抬升流失湿度的速率（地形雨） |
| `groundwater_recharge` | 0.30 | 陆地湿度向基线噪声松弛的速率（地下水） |
| `temperature_diffusion` | 0.04 | 每轮与四邻均衡温度的比例 |
| `temperature_equilibration` | 0.35 | 温度向基线温度松弛的速率（气候由地理决定） |
| `vegetation_spread` | 0.10 | 每轮与四邻均衡植被的比例（传播） |
| `vegetation_growth` | 0.40 | 植被向当地气候潜势生长的速率 |
| `worker_threads` | 4 | 基线生成的并行工作线程数 |
| `default_timeout` | 60.0 | 操作默认超时（秒） |

### 收敛性说明

传播是压缩映射：每轮每个字段的更新系数均为凸组合（系数和 ≤ 1），因此
迭代必然收敛到唯一不动点。收敛速度由最慢的松弛模决定（约
`1 - max(松弛速率)` 每轮）。默认配置下约 70–140 轮收敛，远低于
`max_propagation_rounds` 上限。即使风场有旋（`wind_dz != 0`）或区块链
很长，同步雅可比迭代也不存在轮内依赖环，不会死锁。

`max_propagation_rounds` 是安全上限：若因配置极端（如松弛速率过小）未
在上限内收敛，迭代会在达到上限后停止并返回 `converged=False`，结果仍然
有限且范围内，但可能未完全达到不动点。

## RegionParams

区域参数只影响表面字段（温度、湿度、植被潜势），不影响地形高度。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `temperature_offset` | 0.0 | 区域内每单元的温度偏移（°C） |
| `moisture_offset` | 0.0 | 区域内每单元的基线湿度偏移（0..1） |
| `fertility` | 1.0 | 植被生长乘数；0 表示完全不长植被 |
| `seed_salt` | 0 | 混入噪声的额外种子，使不同区域在相同参数下也能确定性地不同 |

## 调参建议

- **加快收敛**：增大 `groundwater_recharge` / `temperature_equilibration`
  / `vegetation_growth`，使局部松弛主导，横向传播成为小修正。
- **增强海洋性气候**：增大 `moisture_advection` 与 `ocean_evaporation`，
  减小 `groundwater_recharge`。
- **减弱植被**：把 `fertility` 设为 0，或减小 `vegetation_growth`。
- **调试可复现性**：关闭 `propagation_enabled` 可得到纯基线结果，用于
  排查传播相关问题。
