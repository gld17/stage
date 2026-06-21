# FlexET v1: Inference-Only Refactor + Physical Topology Support

## Overview

将 `stage`（Symbolic Tensor Graph → Chakra ET 编译器）重构为 **FlexET**（Flexible-topology Execution Trace Generator）。这是一个重大重构迭代，包含：

1. **全面重命名**：`stage` → `flexet`，`STAGE_*` → `FLEXET_*`
2. **推理-only 清理**：删除全部训练相关代码（backward、FSDP、training microbatch）
3. **物理拓扑输入**：支持 N×N 带宽矩阵，0 表示断链
4. **Placement + Validation**：role → physical_id 双射搜索，PP 直连验证，collective 诱导子图连通验证
5. **统一编号**：ET / comm_group / workload 文件名全部使用 physical_id

## Context

**基线代码库**：`/share/guolidong-nfs/SeeSpace/SCSP_v1/stage`

**上一迭代**：VLM Parallel Decoupling（summary.md 已完成，所有 AC Verified）

**基线关键状态**：
- `main.py` 支持 `--include_backward`、`--weight_sharded`、`MicroBatchReplicator`、`MicroBatchReplicatorPostProcess`
- 存在 `grad_updater.py`（GradUpdater、FSDP weight grad）
- 环境变量：`FLEXET_OPTIMIZED`、`STAGE_MICROBATCH_OPTIMIZE`、`FLEXET_MERGE_COMMS`、`FLEXET_LEGACY_ATTR`
- 占位符：`__FLEXET_SYMBOL_REPLACE_*__`
- Sharding CSV 含 `require_grads`、`grad_of` 列
- 存在 `logical_to_physicall_rank_mapper.py`（未接入主流程）
- `scsp_astra_bridge/` 含大量历史运行结果

## Current State

项目当前同时支持训练和推理双模式。生成 ET 时假设理想逻辑全连接，`LogicalToPhysicalRankMapper` 存在但未接入主流程。对外使用 logical rank 0…N-1 编号。

## Goals

- FlexET **仅**生成推理执行图，完全删除训练能力
- 引入物理拓扑输入，在生成 ET 时考虑物理连接性
- 实现 Placement + Validation 两阶段
- 对外统一使用 **physical_id** 一套编号
- 定义清晰的物理拓扑输入接口

## Non-Goals

- ❌ Collective 路由规划（ring/tree/多跳 relay）— 下游运行时职责
- ❌ 自动调整并行度（如 4PP → 2PP）— v1 明确不自动修改
- ❌ PP 多跳 / relay
- ❌ 大规模拓扑二进制/稀疏格式 — Phase 4 后续
- ❌ 推理 PP forward 流水 micro-batch 拆分器 — Phase 4 后续

## Terminology (Strict)

| 使用 | 不使用 |
|------|--------|
| **physical_id**（唯一节点编号） | logical rank 0…N-1、logical_id / physical_id 双编号 |
| **role** / **role 坐标** `(pp, dp, tp, sp, ep)` | logical rank 编号 |
| **placement**（role → physical_id 绑定） | logical_to_physical_rank 映射（旧称）|
| **FlexET**（产品名） | Stage（产品名） |
| **pipeline stage**（PP 的 stage，ML 术语）| 保留，不替换 |

## Phase 1: Rename + Inference-Only Cleanup

### Task 1.1: 目录与项目重命名

将 `flexet/` 目录重命名为 `flexet/`（保留在 `/share/guolidong-nfs/SeeSpace/SCSP_v1/` 下）。

**注意**：`scsp_astra_bridge/` 下的大量历史结果子目录（如 `qwen2_5_vl_3b_*`）是运行产物，重命名项目目录时**不移动**这些历史结果。只移动源代码目录本身。

#### AC-1
- [ ] `flexet/` 目录已重命名为 `flexet/`
- [ ] `README.md` 中所有 `flexet/` 路径指向 `flexet/`
- [ ] 文档中产品名均为 FlexET

### Task 1.2: 全面替换 STAGE_* → FLEXET_*

替换范围（**不**替换 pipeline stage 语义）：

| 类别 | 当前 | 改为 |
|------|------|------|
| 环境变量 | `FLEXET_OPTIMIZED`、`STAGE_MICROBATCH_OPTIMIZE`、`FLEXET_MERGE_COMMS`、`FLEXET_LEGACY_ATTR`、`STAGE_CTRL_DEPS` | `FLEXET_OPTIMIZED`、`删除`、`FLEXET_MERGE_COMMS`、`FLEXET_LEGACY_ATTR`、`删除` |
| 占位符 | `__FLEXET_SYMBOL_REPLACE_*__` | `__FLEXET_SYMBOL_REPLACE_*__` |
| 代码变量 | `_FLEXET_ROOT` | `_FLEXET_ROOT` |
| CLI 描述 | "FlexET STG generator..." | FlexET 相关 |
| JSON schema title | `"FlexET model configuration"` | FlexET 相关 |

**deliberately 不改的 "stage"**：
- `_stage_for_layer()`、`pipeline_tensor_map`、`num_stacks_each_stage`、`stage_offset`
- `pipeline_parallel.py` 中 layer/stage 相关命名
- `convert_chakra.py` 注释中的 pipeline stage 语义

#### AC-2
- [ ] 代码中无 `FLEXET_OPTIMIZED`、`FLEXET_MERGE_COMMS`、`FLEXET_LEGACY_ATTR` 残留（训练相关 `STAGE_MICROBATCH_*` 已在 Task 1.3 删除）
- [ ] `__FLEXET_SYMBOL_REPLACE_*__` 占位符正确替换
- [ ] `_FLEXET_ROOT` 变量正确
- [ ] pipeline stage 语义未被误替换

### Task 1.3: 删除训练相关代码

#### 删除清单

| 模块 / 能力 | 处理 |
|-------------|------|
| `--include_backward` CLI 及所有分支 | **删除** |
| `grad_updater.py` | **删除** |
| `graph_mode.py` 训练模式 | **删除**，内联为唯一 forward 路径 |
| 模型 builder 的 `include_backward` 参数及 backward 子图、loss | **删除** |
| `MicroBatchReplicator` / `MicroBatchReplicatorPostProcess` | **删除** |
| `--weight_sharded` / FSDP 相关逻辑 | **删除** |
| 环境变量 `STAGE_MICROBATCH_OPTIMIZE` | **删除**，不保留 `FLEXET_*` 替代 |
| 训练 backward 相关测试与文档 | **删除** |
| `sharding_spreadsheets/` 中 `fsdp` 相关目录和文件 | **删除** |

#### AC-3
- [ ] `main.py` 无 `--include_backward` CLI 参数及分支
- [ ] `main.py` 无 `--weight_sharded` CLI 参数及分支
- [ ] `grad_updater.py` 已删除
- [ ] `MicroBatchReplicator` / `MicroBatchReplicatorPostProcess` 引用已删除
- [ ] 模型 builder（`gpt_model.py`、`llama_model.py`、`moe_model.py`、`vlm.py`、`vlm_moe.py`、`vision_encoder.py`、`vlm_connector.py`）无 `include_backward` 参数
- [ ] `_apply_training_mode()` 函数已删除，forward-only 为主路径

### Task 1.4: 清理 Sharding CSV

删除所有 CSV/JSON 中的训练相关列和行：

**删除**：
- `require_grads` 列
- `grad_of` 列
- backward 相关 tensor 行（`.dy`、`.dw`、`loss` 等）
- FSDP / weight shard 相关符号与模板
- `sharding_spreadsheets/module/tpsp_fsdp/` 目录
- `sharding_spreadsheets/module/*_fsdp_merged.*` 文件

**保留**：
- `id`, `op_type`, `op_attr`
- `x1_shape`, `x2_shape`, `x1_hidden`, `x2_hidden`
- shape 中的并行符号：`Batch/dp`、`(Seq/cp)/tp`、`Dout/tp` 等

#### AC-4
- [ ] 所有 Sharding CSV 无 `require_grads`、`grad_of` 列
- [ ] 无 backward tensor 行（.dy、.dw、loss）
- [ ] `sharding_spreadsheets/` 下无 fsdp 相关目录/文件
- [ ] 保留的 forward 算子 CSV 仍能正确加载

### Task 1.5: 环境变量与推理-only micro-batch

- 删除 `STAGE_MICROBATCH_OPTIMIZE` 环境变量及所有相关分支
- 保留 `--micro_batch` CLI 参数，但**简化**：
  - 默认 `micro_batch = batch`（整批一次 forward，不做 PP 内拆分）
  - v1 **不实现** micro-batch PP 流水逻辑
  - 删除 `MicroBatchReplicator` 后，若后续需要该能力，重新实现轻量 forward-only 拆分器

保留的环境变量（更名后）：

| 变量 | 默认 | 作用 |
|------|------|------|
| `FLEXET_OPTIMIZED` | `1` | 分片 / Chakra 转换优化 |
| `FLEXET_MERGE_COMMS` | `0` | 合并相邻同类 collective |
| `FLEXET_LEGACY_ATTR` | `0` | Chakra 后端 legacy 属性 |

#### AC-5
- [ ] `main.py` 无 `STAGE_MICROBATCH_OPTIMIZE` 环境变量检查
- [ ] 保留 `FLEXET_OPTIMIZED`、`FLEXET_MERGE_COMMS`、`FLEXET_LEGACY_ATTR`
- [ ] `--micro_batch` 参数保留，但不做 PP 内拆分（简化处理）

### Task 1.6: 更新文档与项目配置

- `README.md`：更新产品名、路径、功能描述（推理-only）
- `environment.yml`：更新项目名称引用
- `requirements.txt`：如有训练特有依赖则删除
- `tasks/` 目录中的 `stage-model-configs/` → `flexet-model-configs/`
- `PHYSICAL_TOPOLOGY_REQUIREMENTS.md` → 重命名或更新

#### AC-6
- [ ] `README.md` 描述 FlexET 推理-only 定位
- [ ] `tasks/stage-model-configs/` 已重命名
- [ ] `requirements.txt` 无训练特有依赖

### Task 1.7: 回归验证（Phase 1 守门）

在**无物理拓扑**（identity placement，全连接假设）下，验证推理 ET 生成与 baseline 对齐。

**回归测试矩阵**：

| 模型 | 并行策略 | 期望 |
|------|----------|------|
| dense (LLaMA/GPT) | `--dp 2` | reading out 100% |
| dense | `--tp 2 --pp 2` | reading out 100% |
| MoE | `--ep 8 --tp 1 --dp 1 --pp 1` | reading out 100% |
| VLM | `--dp 2 --tp 1 --pp 1` | reading out 100% |
| VLM-MoE | `--ep 8 --tp 2 --dp 2 --pp 1` | reading out 100% |

#### AC-7
- [ ] Dense 推理 ET 生成通过（至少 `--dp 2` 和 `--tp 2 --pp 2`）
- [ ] MoE 推理 ET 生成通过
- [ ] VLM 推理 ET 生成通过
- [ ] VLM-MoE 推理 ET 生成通过

---

## Phase 2: Physical Topology Input + Data Model

### Task 2.1: 定义 `physical_topology.json` Schema

主格式：N×N 带宽矩阵 JSON。

```json
{
  "schema_version": "1.0",
  "num_npus": 8,
  "bandwidth_gbps": [
    [0,   100, 0,   50 ],
    [100, 0,   100, 0  ],
    [0,   100, 0,   100],
    [50,  0,   100, 0  ]
  ],
  "latency_ns": null,
  "symmetric": true
}
```

语义：
- `bandwidth_gbps[i][j] > 0`：physical_id `i` → `j` 直连可行
- `bandwidth_gbps[i][j] == 0`：断链
- 下标 `i` 即 physical_id

#### AC-8
- [ ] Schema 有明确版本号
- [ ] 提供 schema 校验函数（`validate_physical_topology(json_data)`）
- [ ] 校验检查：矩阵为 N×N、对角线为 0、非负元素

### Task 2.2: CLI 集成 `--physical-topology`

```bash
python main.py ... --physical-topology physical_topology.json
```

- 提供拓扑：执行后续 Placement + Validation
- 未提供拓扑（兼容模式）：identity placement（physical_id `i` 的角色 = 标准 Cartesian 顺序第 `i` 槽位），假定全连接

#### AC-9
- [ ] CLI 新增 `--physical-topology` 参数
- [ ] 未提供时默认 identity placement + 全连接假设
- [ ] 提供时加载并校验拓扑文件

### Task 2.3: 拓扑数据模型

新建 `flexet/topology.py`（或 `symbolic_tensor_graph/topology.py`）：

```python
class PhysicalTopology:
    """物理拓扑数据模型"""
    def __init__(self, bandwidth_gbps: List[List[float]], latency_ns: Optional[List[List[float]]] = None):
        ...
    
    @property
    def num_npus(self) -> int: ...
    
    def has_direct_link(self, src: int, dst: int) -> bool: ...
    
    def is_induced_subgraph_connected(self, node_set: Set[int]) -> bool: ...
```

#### AC-10
- [ ] `PhysicalTopology` 类实现
- [ ] `has_direct_link(src, dst)` 正确判断 bandwidth > 0
- [ ] `is_induced_subgraph_connected(node_set)` 正确判断诱导子图连通性

---

## Phase 3: Placement + Validation + ET Output

### Task 3.1: Placement 引擎

**输入**：物理拓扑矩阵、并行策略 `(DP, TP, PP, SP, EP)`、模型
**输出**：`placement.json` — 每个 role 坐标 → 一个 `physical_id` 的双射

实现策略：
- 基于现有 `logical_to_physicall_rank_mapper.py` 的搜索框架升级
- 考虑拓扑矩阵约束（PP 直连 + comm_group 诱导连通）
- 启发式搜索 + `--max-placement-retries` 上限

#### AC-11
- [ ] Placement 引擎产生 role → physical_id 双射
- [ ] `num_npus == DP × TP × PP × SP × EP` 校验
- [ ] 无重复 physical_id、无遗漏

### Task 3.2: Validation 引擎

对给定 placement 检查：

1. **双射完整性**：N 个 role 槽位 ↔ N 个 physical_id
2. **PP P2P 直连**：每条 PP 边（相邻 `pp`、相同其他维的 rank 对）映射后 `bandwidth[u][v] > 0`
3. **Collective 诱导子图连通**：每个 comm_group 映射后的 physical_id 集合，诱导子图连通

#### AC-12
- [ ] PP 直连验证：相邻 stage 间 bandwidth > 0，否则失败
- [ ] Collective 连通验证：诱导子图不连通则失败
- [ ] 验证失败返回具体原因（哪个 PP 边 / 哪个 comm_group）

### Task 3.3: Placement↔Validation 循环 + Fail-Fast

```
Placement → Validation
  ├─ 通过 → 生成 ET
  └─ 失败 → 在同一并行策略下换 placement 重试
              ├─ 仍有 candidate → 重新 Placement
              └─ 重试耗尽 → fail-fast 报错 + feasibility_report.json
```

CLI 参数：
- `--max-placement-retries N`：总重试次数硬上限（默认 1000）
- `--placement-timeout-s T`：wall-clock 超时（可选，与次数上限取先到达者）

**明确不做**：自动调整并行度

#### AC-13
- [ ] `--max-placement-retries` CLI 参数存在，默认 1000
- [ ] `--placement-timeout-s` CLI 参数存在（可选）
- [ ] 达到上限仍未找到 → 立即 fail-fast，不无限循环
- [ ] 失败时输出 `feasibility_report.json`

### Task 3.4: ET / comm_group 全部使用 physical_id

- `workload.{physical_id}.et` 文件名
- `comm_group.json` 成员为 physical_id 列表
- ET 中 `comm_src`、`comm_dst`（PP SEND/RECV）为 physical_id
- `COLL_COMM` 的 `comm_group` 成员为 physical_id
- **无** logical rank 第二套编号

#### AC-14
- [ ] 输出文件名 `workload.{physical_id}.et`
- [ ] `comm_group.json` 成员为 physical_id
- [ ] ET 节点中通信目标为 physical_id

### Task 3.5: `placement.json` 输出

```json
{
  "schema_version": "1.0",
  "num_npus": 8,
  "parallelism": {
    "dp": 2, "tp": 1, "pp": 4, "sp": 1, "ep": 1
  },
  "placement": [
    {
      "physical_id": 0,
      "pp": 0, "dp": 0, "tp": 0, "sp": 0, "ep": 0
    }
  ]
}
```

#### AC-15
- [ ] `placement.json` 符合上述 schema
- [ ] 包含 parallelism 配置
- [ ] 每个 entry 有 physical_id + 完整 role 坐标

### Task 3.6: `feasibility_report.json`（失败时）

包含：
- 失败原因类型（PP 直连失败 / comm_group 不连通 / 双射错误）
- 失败的 PP 边：physical_src, physical_dst, 对应 role
- 失败的 comm_group：维度、成员 physical_id 列表
- placement 重试次数与最终结论

#### AC-16
- [ ] feasibility_report.json 包含失败原因类型
- [ ] 包含失败的 PP 边详情
- [ ] 包含失败的 comm_group 详情
- [ ] 包含重试次数

### Task 3.7: 稀疏拓扑场景端到端测试

测试矩阵：

| 场景 | 拓扑 | 并行策略 | 期望 |
|------|------|----------|------|
| 线性拓扑 4 卡 | chain 0-1-2-3 | PP=4 | 通过（相邻直连）|
| 线性拓扑 4 卡 | chain 0-1-2-3 | TP=2 + PP=2 | TP group 可能失败（诱导子图不连通）|
| 环状拓扑 4 卡 | ring | TP=4 | 通过（诱导子图连通）|
| 断链拓扑 4 卡 | 0-1, 2-3 两孤岛 | PP=4 | fail-fast，feasibility_report 指明断链 |

#### AC-17
- [ ] 线性拓扑 PP=4 通过
- [ ] 断链拓扑 PP=4 正确 fail-fast
- [ ] feasibility_report 信息准确

### Task 3.8: 回归验证（Phase 3 守门）

全连接拓扑（identity placement）下，Phase 3 新增功能不破坏 Phase 1 的回归测试结果。

#### AC-18
- [ ] Dense / MoE / VLM / VLM-MoE 在全连接拓扑下仍通过（同 AC-7 矩阵）

---

## Summary of All ACs

| AC | 描述 | Phase |
|----|------|-------|
| AC-1 | 目录重命名 stage → flexet | 1 |
| AC-2 | STAGE_* → FLEXET_* 全面替换（保留 pipeline stage 语义）| 1 |
| AC-3 | 删除训练代码（backward、grad_updater、weight_sharded、MicroBatchReplicator）| 1 |
| AC-4 | Sharding CSV 清理 grad 列/行 | 1 |
| AC-5 | 环境变量清理，保留推理相关 | 1 |
| AC-6 | 文档与配置更新 | 1 |
| AC-7 | Phase 1 回归验证（Dense/MoE/VLM/VLM-MoE）| 1 |
| AC-8 | physical_topology.json Schema + 校验 | 2 |
| AC-9 | CLI --physical-topology | 2 |
| AC-10 | PhysicalTopology 数据模型 | 2 |
| AC-11 | Placement 引擎双射输出 | 3 |
| AC-12 | Validation（PP 直连 + collective 连通）| 3 |
| AC-13 | Placement↔Validation 循环 + fail-fast | 3 |
| AC-14 | ET / comm_group 全部 physical_id | 3 |
| AC-15 | placement.json 输出 | 3 |
| AC-16 | feasibility_report.json | 3 |
| AC-17 | 稀疏拓扑端到端测试 | 3 |
| AC-18 | Phase 3 回归验证 | 3 |

## Dependencies

- 外部拓扑模块按 `physical_topology.json` schema 提供数据
- Codex CLI 0.135.0 + PTY 模式执行
- `uv run --with pytest --with requests python -m pytest` 运行测试
