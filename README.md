# Biome Worldgen (Python)

可复现的程序化生态区域生成与演化库。世界由相邻的 **区块（chunk）** 拼成，
地形 / 气候 / 水源 / 植被以**世界坐标**为唯一依据确定性生成，并通过固定顺序的
松弛迭代跨区块接缝传播。支持大范围生成、局部改参后的增量重算、并发安全写入、
世界快照与版本回退、取消与超时。

## 安装与测试

```bash
uv sync
uv run pytest            # 运行全部自动化测试（tests/）
```

## 快速开始

```python
from worldgen import World, WorldConfig, RegionParams, ChunkCoord

# 1) 同一份种子与配置，任意次生成逐格一致
world = World(WorldConfig(seed=1337))
chunks = world.generate(rect=(0, 0, 8, 8))          # 区块坐标矩形 [cx0,cz0,cx1,cz1)
# 或指定区块集合
world.generate(coords={ChunkCoord(0, 0), ChunkCoord(1, 2)})

chunk = world.get_chunk(ChunkCoord(0, 0))           # 不可变、带版本号
cell  = chunk.cell_at(lx=0, lz=0)                   # 局部格坐标 0..15
print(cell.elevation, cell.temperature, cell.moisture, cell.water, cell.biome)

# 2) 局部改参 -> 自动增量重算（只重写受影响区块，结果与全量重建逐格一致）
world.apply_region_params(
    "north-forest",
    rect=(2, 2, 6, 6),                              # 区块网格矩形
    params=RegionParams(temperature_bias=0.2, moisture_bias=0.1),
)

# 3) 快照与回退
snap = world.snapshot("before-draining")
world.apply_region_params("dry", (2, 2, 6, 6), RegionParams(water_bias=-0.5))
world.rollback(snap.snapshot_id)                    # 区块 + 区域参数一并还原
```

## 七大能力对应

| 需求 | 实现要点 | 测试 |
| --- | --- | --- |
| 1. 种子+参数可复现 | SplitMix64 定点整数 PRNG；地形是 `(seed, 世界坐标)` 的纯函数；参数状态有稳定 SHA-256 摘要并写入每个区块 | `tests/test_determinism.py` |
| 2. 接缝连续、无矛盾 | 一格只属于一个区块；跨格松弛两侧读同一世界函数；区域边缘有世界连续的渐变带 `border_blend_cells` | `tests/test_boundary.py` |
| 3. 增量重算==全量重建 | Jacobi 松弛 + K 深 halo 模拟；只物化写集；未变内容跳过不产生新版本；传播必有界终止，轮数/容差可配 | `tests/test_incremental.py` |
| 4. 并发安全 | 区块不可变；按坐标序加锁（无死锁）+ 提交时版本 CAS，失败则用确定性结果重试；读者无锁且只见完整版本 | `tests/test_concurrency.py` |
| 5. 快照与回退 | 显式 `in_progress / committed / aborted` 状态；回退恢复区块与参数，之后继续生成与直接在该版本生成一致 | `tests/test_snapshot.py` |
| 6. 取消与超时 | 在阶段边界协作检查；提交前中止不写入任何区块；重新触发仍可复现 | `tests/test_cancellation.py` |
| 7. 自动化测试 + 文档 | `tests/` 共 40+ 用例；本文档与 `docs/generation_model.md` | `uv run pytest` |

## 配置项（`WorldConfig`）

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `seed` | `1337` | 世界种子；只影响确定性，不含任何平台/时间相关熵 |
| `max_propagation_rounds` | `4` | 气候/水源跨格松弛的**最大轮数（硬上限）**。K 轮后一格只依赖切比雪夫距离 ≤ K 的格；影响区块半径为 `ceil(K/16)`（默认 1 个区块）。必须 ≥ 0 |
| `convergence_tolerance` | `0.0` | 单轮最大变化量低于该值即视为收敛并提前停止。**默认 0 表示严格执行满 K 轮**，这是"增量与全量逐位一致"的推荐配置 |
| `non_converged_policy` | `"accept"` | 容差 > 0 且未在 K 轮内收敛时：`"accept"` 接受当前结果，或 `"raise"` 抛 `NonConvergedError`（此时不写入） |
| `lock_timeout_seconds` | `30.0` | 获取区块锁的超时（死锁兜底，不是正常控制流） |
| `max_commit_retries` | `16` | 提交时 CAS 冲突的最大确定性重试次数，超出抛 `ConcurrentModificationError` |
| `generation_timeout_seconds` | `None` | 单次生成/重算的墙钟超时；到时抛 `GenerationTimeout` |

## 区域参数（`RegionParams`）

区域是区块网格上的轴对齐矩形；重叠时**区域 id 字典序最大者生效**（与插入顺序无关）。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `temperature_bias` / `moisture_bias` / `water_bias` / `elevation_bias` | `0.0` | 叠加到世界连续基线上的偏置 |
| `border_blend_cells` | `8` | 区域边缘的**平滑渐变带宽度（格）**。偏置按到最近区域边界的距离做 smoothstep 渐变，因此改参不会在接缝处形成硬断崖；该渐变只依赖世界坐标，区块两侧判定完全一致 |

## 收敛与最大传播轮数（为什么不会死循环）

传播是同步 Jacobi 平均：每个新值都是上一轮自身与四邻的**凸组合（权重和为 1）**，
因此它在无穷范数下是压缩映射，差异每轮严格衰减。依赖图里即使有环或长链路也
只会让信息按轮数扩散，**不可能无限循环**；轮数被 `max_propagation_rounds`
硬性截断。提前停止条件（可选）是"上一轮所有格、所有场的最大绝对变化
≤ `convergence_tolerance`"。详见 `docs/generation_model.md`。

## 增量重算的一致性条件

重算会把写集向外扩 `ceil(K/CHUNK_SIZE)` 个区块并与**已存在**区块求交。
为得到与全量重建逐位相同的结果，请保证受影响边界区块之外还有至少 K 格的
已生成/可模拟空间（实践中世界比改动区域大一圈即可）；未生成的邻块按基准值
即时模拟，不会被提前物化，因此以后真正生成它们时结果仍然一致。

## 异常

`worldgen.errors`：`CancellationError`、`GenerationTimeout`、
`LockAcquireTimeout`、`ConcurrentModificationError`、`NonConvergedError`、
`SnapshotError`。
