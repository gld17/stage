# FlexET 迭代需求与计划（讨论定稿）

> **项目名称**：FlexET（Flexible-topology Execution Trace Generator）  
> **定位**：大模型分布式**推理**执行图（Chakra ET）生成软件  
> **现状基线**：`/share/guolidong-nfs/SeeSpace/SCSP_v1/stage`（Symbolic Tensor Graph → Chakra ET）  
> **本文性质**：FlexET 项目本身的迭代需求与实施计划定稿；**不包含代码修改**。

---

## 一、项目背景与目标

### 1.1 现状

当前 `stage` 项目本质是 **Symbolic Tensor Graph (STG) → Chakra ET** 的编译器：

- 根据模型结构 + 并行策略（DP/TP/PP/SP/EP 等），为每个 rank 生成执行 trace（`workload.N.et`）和通信组配置（`comm_group.json`）
- 输出 Chakra ET 格式，供下游仿真或 profiling 工具消费
- **未**内置物理拓扑输入；生成 ET 时假设理想逻辑全连接
- `LogicalToPhysicalRankMapper` 存在但**未接入**主流程
- 同时支持训练（backward/FSDP/MicroBatch）与推理双模式

### 1.2 迭代目标

将项目重命名并重构为 **FlexET**，实现：

1. **仅推理**执行图生成，删除全部训练相关能力
2. 引入**物理拓扑输入**，在生成 ET 时考虑物理连接性
3. 实现 **Placement + Validation** 两阶段：在给定物理拓扑、模型、混合并行策略下，为每张卡分配并行角色并生成可行 ET
4. 对外统一使用 **一套节点编号（physical_id）**
5. 定义清晰的**物理拓扑输入接口**，供外部模块按约定格式供给拓扑数据

**不在 FlexET v1 范围内**：Collective 路由规划（ring/tree/多跳 relay 等），属于下游运行时或独立 Planner 的职责。

---

## 二、项目重命名：stage → FlexET

### 2.1 命名含义

- **FlexET** = Flexible-topology Execution Trace Generator
- 强调：执行图生成可支持更灵活的拓扑（异构、稀疏、断链），不限于理想全连接假设

### 2.2 重命名策略

- **全面替换**，**不保留**任何 `STAGE_*` 环境变量、常量、路径 alias
- 项目内文档、目录、代码引用一次性同步更新

### 2.3 需替换范围（产品名 / 工具名）

| 类别 | 当前 | 改为 |
|------|------|------|
| 目录 | `flexet/` | `flexet/` |
| 环境变量 | `FLEXET_OPTIMIZED`、`STAGE_MICROBATCH_OPTIMIZE`、`FLEXET_MERGE_COMMS`、`FLEXET_LEGACY_ATTR`、`STAGE_CTRL_DEPS` 等 | `FLEXET_*` |
| 内部占位符 | `__FLEXET_SYMBOL_REPLACE_*__` | `__FLEXET_SYMBOL_REPLACE_*__` |
| 代码变量 | 如 `_FLEXET_ROOT` | `_FLEXET_ROOT` |
| CLI 描述 | "FlexET STG generator..." | FlexET 相关描述 |
| JSON schema title | `"FlexET model configuration"` | FlexET 相关 |
| 文档 / 任务目录 | `tasks/stage-model-configs/`、`PHYSICAL_TOPOLOGY_REQUIREMENTS.md` 等 | `flexet-*` 命名 |
| README | 所有指向 `flexet/` 的路径与产品名 | 指向 `flexet/` |

### 2.4  deliberately 不改的 “stage” 用语

以下为 **Pipeline Parallel 的 pipeline stage**（ML 术语），**不得**因重命名而修改：

- `_stage_for_layer()`、`pipeline_tensor_map`、`num_stacks_each_stage`、`stage_offset`
- `pipeline_parallel.py` 中 layer/stage 相关命名
- `convert_chakra.py` 注释中的 pipeline stage 语义

### 2.5 术语规范（全文统一）

| 使用 | 不使用 |
|------|--------|
| **physical_id**（唯一节点编号） | logical rank 0…N-1、logical_id / physical_id 双编号 |
| **role** / **role 坐标** `(pp, dp, tp, sp, ep)` | logical rank 编号 |
| **placement**（role → physical_id 绑定） | logical_to_physical_rank 映射（旧称） |
| **FlexET**（产品名） | Stage（产品名） |

---

## 三、功能范围：仅推理（Inference-only）

### 3.1 原则

FlexET **只**生成大模型**推理**执行图；训练执行图能力**完全删除**，不保留兼容开关。

### 3.2 需删除的代码与能力

| 模块 / 能力 | 处理 |
|-------------|------|
| `--include_backward` CLI 及所有分支 | **删除** |
| `grad_updater.py`（GradUpdater、FSDP weight grad 等） | **删除** |
| `graph_mode.py` 训练模式 | **删除**或内联为唯一 forward 路径 |
| 模型 builder 的 `include_backward` 参数及 backward 子图、loss | **删除** |
| `MicroBatchReplicator` / `MicroBatchReplicatorPostProcess`（训练 PP 气泡） | **删除** |
| `--weight_sharded` / FSDP 相关逻辑 | **删除** |
| 训练 backward 相关测试与文档 | **删除** |
| VRAM 估算中的 grads 分量 | **删除**，仅保留推理相关（params + activations） |

### 3.3 Sharding CSV：删除全部 grad 字段

CSV 变为**纯 forward 算子 + 并行切分**描述，**不保留任何训练语义**：

**删除的列 / 概念：**

- `require_grads`
- `grad_of`
- backward 相关 tensor 行（`.dy`、`.dw`、`loss` 等）
- FSDP / weight shard 相关符号与模板

**保留的列 / 语义：**

- `id`, `op_type`, `op_attr`
- `x1_shape`, `x2_shape`, `x1_hidden`, `x2_hidden`
- shape 中的并行符号：`Batch/dp`、`(Seq/cp)/tp`、`Dout/tp` 等

**原则：** 代码与 CSV 均不含 backward/grad/FSDP；避免“删代码但 CSV 仍暗示可训练”的歧义。

### 3.4 推理 PP 与 micro-batch

- 推理-only 下**复用现有 `--micro_batch` 参数**，不新增 `--inference_micro_batch`
- 默认 **`micro_batch = batch`**（整批一次 forward，不做 PP 内拆分）
- 当 `micro_batch < batch` 时：表示将 batch 拆成多个 micro-batch，在 PP 各 stage 间做 **forward 流水**（仅 forward 语义，无训练气泡/backward）
- v1 **不实现** micro-batch PP 流水逻辑；删除训练版 `MicroBatchReplicator` 后，若后续需要该能力，应**重新实现轻量 forward-only 拆分器**，仍通过 `--micro_batch` 触发，不恢复训练路径

### 3.5 删除后的主数据流（概念）

```
模型 config + 并行策略 + 物理拓扑矩阵
  → 构建 forward-only TensorGraph
  → GraphDistributer 分片（内部仍可用 role 坐标 tuple 作 key）
  → Placement：role → physical_id
  → Validation
  → ConvertChakra（COMP + COLL_COMM + PP SEND/RECV）
  → workload.{physical_id}.et + comm_group.json + placement.json
```

---

## 四、核心概念与架构

### 4.1 两层拓扑

| 层次 | 含义 | FlexET v1 职责 |
|------|------|----------------|
| **逻辑层通信需求** | 混合并行策略隐含的通信：PP 相邻 stage 点对点、TP/DP/SP/EP 的 collective 组 | 由并行度 + 模型 sharding 规则推导 |
| **物理层通信拓扑** | 实际 NPU 间直连链路与带宽 | 用户/外部模块提供带宽矩阵；FlexET 读取并约束 placement |

当前 baseline：生成 ET 时不读物理拓扑；迭代后 FlexET **必须**支持物理拓扑输入，并在生成 ET 前完成 placement 与 validation。

### 4.2 节点编号：仅一套 physical_id

**对外唯一编号：`physical_id`（拓扑带宽矩阵的行/列下标）。**

- ET 中 `comm_src`、`comm_dst`、comm_group 成员、`workload.{id}.et` 文件名 → **全部 physical_id**
- **Logical 不是第二套数字 ID**，而是每张卡上的 **并行角色坐标** `(pp, dp, tp, sp, ep)`，记录在 `placement.json`
- 内部实现（如 GraphDistributer 用 `((pp,0),(dp,1),…)` 作 key）为编译期中间表示，**不对外暴露**第二套 numeric rank

**约束：**

- `num_npus == DP × TP × PP × SP × EP`（及 MoE 等扩展维的乘积）
- Placement 为 **双射**：每个 role 槽位 ↔ 唯一 physical_id

### 4.3 通信类型与物理约束（铁律）

| 通信类型 | ET 形式 | 典型并行维 | FlexET v1 物理约束 |
|----------|---------|------------|-------------------|
| **点名通信（P2P）** | `COMM_SEND` / `COMM_RECV`，固定 src/dst | **PP** | 映射后的 `(u,v)` 在拓扑矩阵上 **直连**：`bandwidth[u][v] > 0`；**不允许多跳** |
| **组通信（Collective）** | `COLL_COMM` + `comm_group` | **TP、DP、SP、EP** | comm_group 对应 physical_id 集合 **诱导子图连通**；**不要求**组内两两直连 |

**铁律（写入 spec）：**

1. **PP = 矩阵直连 `bandwidth > 0`**
2. **Collective = 诱导子图连通**（组内可无直连边，但整组在保留边下须连通）
3. **Placement 可搜索；并行度不可自动修改**

### 4.4 P2P 与诱导子图的区别（Validation 必区分）

**PP P2P（成对、直连）：**

- 检查对象：**两个** physical 节点 `u`, `v`
- 条件：`bandwidth[u][v] > 0`（非对称拓扑则按有向边定义）
- 全图存在 u→v 多跳路径 **不能** 替代直连

**Collective（集合、连通）：**

- 检查对象：comm_group 内全体 physical_id 集合 `P`
- **诱导子图**：仅保留两端均在 `P` 内的边
- 条件：诱导子图 **连通**（任意两点在诱导子图内有路径）
- 组内部分 pair 无直连但诱导子图连通 → **可行**（具体 collective 路径由下游运行时负责，FlexET 不规划）
- 诱导子图不连通 → **不可行**，Validation 失败

**EP / all-to-all：** v1 仍以诱导子图连通为**必要条件**；若后续发现 all-to-all 需更强条件，可对 EP 单独加 stricter 规则，不改变 PP/TP/DP 规则。

### 4.5 Placement 与 Validation（两阶段）

#### Placement

- **输入**：物理拓扑矩阵、并行策略 `(DP, TP, PP, SP, EP)`、模型
- **输出**：每个 role 坐标 → 一个 `physical_id` 的双射（`placement.json`）
- **含义**：在 N 张物理卡上，为每个并行角色槽位分配具体卡

**不是** “某张卡只做 PP、某张卡只做 DP”；每张卡同时拥有一个完整 role 坐标。

#### Validation

对给定 placement 检查：

1. **双射完整性**：N 个 role 槽位 ↔ N 个 physical_id，无遗漏、无重复
2. **PP P2P**：每条 PP 边（相邻 `pp`、相同其他维的 rank 对）映射后 **直连 bandwidth > 0**
3. **Collective**：每个 comm_group 映射后的 physical_id 集合，**诱导子图连通**

#### 失败后的调整策略（v1 固定）

```
Placement → Validation
  ├─ 通过 → 生成 ET
  └─ 失败 → 在同一并行策略下换 placement 重试
              ├─ 仍有 candidate → 重新 Placement
              └─ 重试耗尽 → fail-fast 报错 + feasibility_report.json
```

**v1 明确不做：**

- ❌ 自动调整并行度（如 4PP → 2PP）
- ❌ 自动调整 DP/TP/EP
- ❌ PP 多跳 / relay
- ❌ Collective 路由规划

**v1 必配项（避免搜索空间过大导致不收敛）：**

- **`--max-placement-retries N`**：Placement 搜索 + Validation 的**总重试次数硬上限**（建议默认如 `1000`，可 CLI 覆盖）
- 达到上限仍未找到可行 placement → **立即 fail-fast**，输出 `feasibility_report.json`，不无限循环
- 可选 **`--placement-timeout-s`**： wall-clock 超时，与次数上限取**先到达者**终止
- 并行策略本身不可行 → 由用户修改配置；策略搜索可作为**未来独立工具**，不在 FlexET v1 核心路径

**说明：** Validation 本身不重试，仅判定当前 placement 是否可行；重试发生在 **Placement 生成下一个 candidate → 再次 Validation** 的循环中，故上限约束的是该循环总次数。

**与现有 `LogicalToPhysicalRankMapper`：** 现有实现为维度质因数分解，**未考虑物理边**；需升级为基于拓扑矩阵、满足 PP 直连 + comm_group 诱导连通约束的 **Placement 引擎**（可借鉴其搜索框架，但约束不同）。

### 4.6 Collective 路由规划（不在 FlexET v1）

FlexET v1 职责止于：

- 推导逻辑通信需求（PP p2p、各维 comm_group）
- Placement + Validation
- 按 **physical_id** 输出 ET 与 comm_group

FlexET **不**规划 all-reduce / all-gather 的具体路径或算法；最多在 `feasibility_report.json` 中记录连通性失败信息。

---

## 五、物理拓扑输入格式

### 5.1 主格式：N×N 带宽矩阵

大节点数下避免边列表冗余；**矩阵为 FlexET 主输入**。

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

### 5.2 语义约定

| 元素 | 含义 |
|------|------|
| `bandwidth_gbps[i][j] > 0` | physical_id `i` → `j` **直连可行**，值为带宽（Gbps） |
| `bandwidth_gbps[i][j] == 0` | **断链**（无直连边） |
| 对角线 | 通常为 0（无自环） |
| 矩阵下标 `i` | 即 **physical_id**，与 ET rank 一致 |

**可选字段：**

- `latency_ns`：同维度矩阵；缺省则统一默认 latency，或由外部模块填好
- `symmetric`：若 true，文档约定对称性处理方式（如 undirected 边取 max 或要求上三角一致）

### 5.3 大规模扩展（后续）

- N 很大时 JSON 矩阵体积大，可扩展支持 `.npy`、压缩二进制、CSR 稀疏等
- **语义不变**：0 = 断链，>0 = 可行带宽；下标仍为 physical_id

### 5.4 外部拓扑供给（接口约定）

FlexET 只定义**输入契约**，不负责生成真实星座/集群拓扑：

- 外部模块按 `physical_topology.json` schema 提供 NPU 级 `bandwidth_gbps` 矩阵（不可行链路 → 0）
- FlexET 核心 **只认带宽矩阵**，不维护多种等价格式
- 可选文档说明：外部模块如何将自有拓扑表示（如距离矩阵、邻接表）**转换为** FlexET 矩阵（转换逻辑不在 FlexET 内实现）

### 5.5 CLI（建议）

```bash
python main.py ... --physical-topology physical_topology.json
```

- **提供拓扑**：执行 Placement + Validation + 按 physical_id 输出 ET
- **未提供拓扑**（可选兼容模式）：identity placement——physical_id `i` 的角色 = 标准 Cartesian 顺序第 `i` 槽位；假定全连接；对外编号仍为 physical_id

---

## 六、输出产物规范

### 6.1 主输出

| 文件 | 说明 |
|------|------|
| `workload.{physical_id}.et` | 该 physical 卡上的 Chakra 执行 trace |
| `comm_group.json` | 通信组 id → **physical_id** 列表 |
| `placement.json` | 并行策略 + 每张卡的 role 与 physical_id 绑定 |

### 6.2 ET 内容要求

- `comm_src` / `comm_dst`（PP SEND/RECV）：**physical_id**
- `COLL_COMM` 的 `comm_group` 成员：**physical_id**
- **无** logical rank 第二套编号

### 6.3 placement.json（示例结构）

```json
{
  "schema_version": "1.0",
  "num_npus": 8,
  "parallelism": {
    "dp": 2,
    "tp": 1,
    "pp": 4,
    "sp": 1,
    "ep": 1
  },
  "placement": [
    {
      "physical_id": 0,
      "pp": 0,
      "dp": 0,
      "tp": 0,
      "sp": 0,
      "ep": 0
    },
    {
      "physical_id": 2,
      "pp": 1,
      "dp": 0,
      "tp": 0,
      "sp": 0,
      "ep": 0
    }
  ]
}
```

可按实现需要附加 `comm_groups`（成员已为 physical_id），便于调试与下游消费。

### 6.4 feasibility_report.json（Validation 失败时）

建议包含：

- 失败原因类型（PP 直连失败 / comm_group 不连通 / 双射错误等）
- 失败的 PP 边：`physical_src`, `physical_dst`, 对应 role
- 失败的 comm_group：维度（dp/tp/sp/ep）、成员 physical_id 列表
- placement 重试次数与最终结论
- **不包含**自动并行度调整（v1 仅报错；可选 **文字提示** 用户自行调整策略，但不自动执行）

---

## 七、并行策略与通信需求（实现参考）

### 7.1 并行度与 NPU 数

```
num_npus = DP × TP × PP × SP × EP
```

（MoE 等模型按现有规则处理 EP；dense 模型 EP=1。）

### 7.2 各并行维通信形态

| 并行维 | 通信形态 | FlexET ET 节点 | Validation |
|--------|----------|----------------|------------|
| PP | P2P SEND/RECV | 相邻 pp stage 间 | 直连 bandwidth > 0 |
| TP | Collective（ALL_GATHER、REDUCE_SCATTER 等） | COLL_COMM + comm_group | 诱导子图连通 |
| DP | Collective（推理场景较少；若 ET 中有则同 TP） | COLL_COMM | 诱导子图连通 |
| SP | 同 TP | COLL_COMM | 诱导子图连通 |
| EP (MoE) | ALL_TO_ALL 等 | COLL_COMM | 诱导子图连通（v1 必要条件） |

### 7.3 内部编译流程（保留并改造）

1. 模型 builder + forward-only sharding CSV → 完整 TensorGraph
2. `GraphDistributer`：PP 分 temporal bucket + DP/TP/SP/EP 空间复制
3. 跨 PP bucket：Shadow 张量 + SEND/RECV
4. **新增**：Placement → Validation → 将通信目标重写为 physical_id
5. `BundledConvertChakra` → HybridGraph → readout

### 7.4 保留的模型能力

- Dense / LLaMA / GPT / MoE / VLM / VLM-MoE 等 **推理** forward 路径
- VLM vision/text 独立 PP 等已有扩展（去掉 backward 分支后保留 forward）
- 模型 config JSON（`models/model_configs/`）

### 7.5 环境变量（更名后）

| 变量 | 默认 | 作用 |
|------|------|------|
| `FLEXET_OPTIMIZED` | `1` | 分片 / Chakra 转换优化 |
| `FLEXET_MERGE_COMMS` | `0` | 合并相邻同类 collective |
| `FLEXET_LEGACY_ATTR` | `0` | Chakra 后端 legacy 属性 |

（训练相关 `STAGE_MICROBATCH_*` 等 **删除**，不保留 `FLEXET_*` 替代。）

---

## 八、实施分期建议

### Phase 1：重命名 + 推理-only 清理

- [ ] 目录 `flexet/` → `flexet/`，全面替换 `STAGE_*` → `FLEXET_*`
- [ ] 删除训练代码路径、`--include_backward`、GradUpdater、FSDP、训练 MicroBatch
- [ ] 清理 Sharding CSV：删除 `require_grads`、`grad_of` 及 backward 行
- [ ] 清理模型 builder / `main.py` 中所有 backward 分支
- [ ] 更新 README、schema title、项目内文档
- [ ] 回归：推理 ET 在**无物理拓扑**（identity / 全连接）下与 baseline forward 行为对齐

### Phase 2：物理拓扑输入 + 数据模型

- [ ] 定义 `physical_topology.json` schema（带宽矩阵）
- [ ] CLI `--physical-topology`
- [ ] 文档：矩阵语义、0=断链、physical_id 与下标一致
- [ ] 编写外部拓扑 → FlexET 矩阵的接口说明（转换由外部模块完成）

### Phase 3：Placement + Validation + ET 输出

- [ ] Placement 引擎：role → physical_id 双射搜索
- [ ] Validation：PP 直连 + comm_group 诱导子图连通
- [ ] Placement↔Validation 循环：`--max-placement-retries`（默认上限）+ 可选 `--placement-timeout-s`；耗尽/超时则 `feasibility_report.json`
- [ ] 输出 `placement.json`
- [ ] ET / comm_group 全部使用 physical_id
- [ ] 稀疏拓扑场景单元测试与 CLI 端到端示例

### Phase 4（后续，非 v1 必做）

- [ ] 大规模拓扑：矩阵二进制 / 稀疏格式
- [ ] 独立并行策略可行性分析工具（不自动改并行度，仅给用户建议）
- [ ] 推理 PP forward 流水：复用 `--micro_batch`，实现 forward-only 拆分器（非训练 MicroBatchReplicator）
- [ ] EP all-to-all 更强 validation 规则（若实践暴露不足）

---

## 九、验收标准（v1）

1. **命名**：项目内产品名均为 FlexET，无 `STAGE_*` 残留（pipeline stage 术语除外）
2. **范围**：无 backward/FSDP/训练 CLI；CSV 无 grad 列
3. **拓扑**：支持 N×N 带宽矩阵输入，0 表示断链
4. **编号**：ET、comm_group、文件名仅使用 physical_id；placement.json 记录 role
5. **PP**：仅直连边；多跳视为 Validation 失败
6. **Collective**：comm_group 不满足诱导子图连通则 Validation 失败
7. **Placement**：Validation 失败仅换 placement 重试，**不**自动改并行度；**重试次数或 wall-clock 达上限**即报错，禁止无限搜索
8. **输出**：`workload.{physical_id}.et` + `comm_group.json` + `placement.json`（+ 失败时 `feasibility_report.json`）
9. **CLI 端到端**：给定模型、并行策略、物理拓扑矩阵，FlexET 可独立完成 ET 生成或明确报错

---

## 十、风险与依赖

| 风险 | 说明 | 缓解 |
|------|------|------|
| Placement 搜索复杂度 | N 大时 permutation 空间爆炸 | 启发式 + 重试上限 + 明确超时 |
| 大矩阵 IO | N×N JSON 体积 | Phase 4 二进制格式；v1 文档说明推荐 N 上限 |
| VLM/MoE 推理路径回归 | 删 backward 时误伤 forward | Phase 1 专项回归用例 |
| EP validation 过松 | 诱导连通不足以保证 all-to-all | v1 先统一规则；EP 加强放 Phase 4 |
| 外部拓扑格式不一致 | 上游模块输出不符合 schema | 严格 schema 校验 + 清晰错误信息 |

**外部依赖（接口层）：** 物理拓扑由外部模块按 FlexET 矩阵 schema 提供；FlexET 不实现拓扑生成本身。
