# Biome World Generator

可复现的程序化生态区域生成与演化系统。世界由大量相邻区块拼成，气候、
水源与植被跨区块相互传播，支持大范围生成、局部增量重算、版本回退、
并发生成与取消超时。

## 特性

- **可复现**：地形/气候/植被是种子与坐标的纯函数（确定性哈希噪声），
  同种子同参数下任意次生成逐格一致。
- **边界连续**：跨区块传播采用同步雅可比迭代，接缝两侧判定连续，同一
  格不会出现矛盾结果。
- **增量重算**：局部参数变更后，受影响区块与全量重建完全对齐，未受
  影响区块不被改写（对象身份保留）。
- **并发安全**：区块不可变 + 引用交换，读取者永远看到完整版本；提交
  与生成均有锁保护，同一区块不会被交叉写入。
- **快照回退**：已提交/未完成/已中止快照状态可区分；回退后继续生成
  与直接在该版本上生成结果一致。
- **取消超时**：取消或超时不留下部分写入，重新触发后结果仍满足一致
  性与可复现要求。

## 安装

```bash
uv sync
```

## 快速开始

### 作为库使用

```python
from biome_worldgen import World, WorldGenConfig, RegionParams

world = World(seed=42, config=WorldGenConfig(chunk_size=16))

# 生成 4x4 区块区域
result = world.generate_region(0, 0, 3, 3)
print(result.converged, result.rounds)

# 读取区块
chunk = world.get_chunk(0, 0)
print(chunk.cell(0, 0))  # Cell(height, temperature, moisture, vegetation, ...)

# 局部参数变更后的增量重算
world.recompute({(1, 1): RegionParams(temperature_offset=5.0)})

# 快照与回退
snap = world.create_snapshot(label="v1")
world.generate_region(4, 0, 5, 3)
world.rollback(snap.id)
```

### 作为 HTTP 服务

```bash
python main.py
# 服务运行在 http://127.0.0.1:8000，API 文档见 /docs
```

API 示例：

```bash
# 创建世界
curl -X POST http://127.0.0.1:8000/api/worlds \
  -H 'Content-Type: application/json' \
  -d '{"seed": 42}'

# 生成区域
curl -X POST http://127.0.0.1:8000/api/worlds/{world_id}/generate \
  -H 'Content-Type: application/json' \
  -d '{"x0": 0, "z0": 0, "x1": 3, "z1": 3}'

# 查询操作状态
curl http://127.0.0.1:8000/api/worlds/{world_id}/operations/{op_id}

# 增量重算
curl -X POST http://127.0.0.1:8000/api/worlds/{world_id}/recompute \
  -H 'Content-Type: application/json' \
  -d '{"changes": {"1,1": {"temperature_offset": 5.0}}}'

# 快照与回退
curl -X POST http://127.0.0.1:8000/api/worlds/{world_id}/snapshots \
  -H 'Content-Type: application/json' -d '{"label": "v1"}'
curl -X POST http://127.0.0.1:8000/api/worlds/{world_id}/snapshots/{snap_id}/rollback

# 取消操作
curl -X POST http://127.0.0.1:8000/api/worlds/{world_id}/operations/{op_id}/cancel
```

## 文档

- [生成模型](docs/model.md)：地形、气候、湿度、植被、传播、收敛、
  并发与快照的设计说明。
- [配置项](docs/config.md)：所有 `WorldGenConfig` 与 `RegionParams`
  字段的含义、默认值与调参建议。

## 测试

```bash
python -m pytest tests/ -v
```

测试覆盖：确定性复现、边界连续性、增量与全量一致、并发生成与重算、
中断恢复、版本回退、传播收敛与无死锁、HTTP API。
