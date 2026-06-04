# Stage 迭代：VLM 并行策略解耦 + MoE 文本 Backbone 支持

## 背景

当前 VLM 实现（`models/vlm.py` + `main.py`）将 ViT 和 Text Backbone 视为一个统一的 Dense 模型，使用相同的并行策略（DP/TP/PP/EP）进行分布。这限制了 VLM 的灵活性：
- ViT（视觉编码器）和 Text Backbone（文本主干网络）在模型结构、计算特性和内存需求上差异显著
- 实际场景中，ViT 通常使用较小的并行度（如 TP=1/2），而 Text Backbone 可能需要更大的并行度（如 TP=8）
- 需要支持 ViT 和 Text Backbone 采用不同的并行策略组合

此外，MoE（混合专家）作为文本 backbone 的支持尚未验证。当前 MoE 模型存在 pre-existing bug（`Micro(MicroBatch)` 畸形表达式），需要修复后才能支持 MoE 文本 backbone。

## 技术可行性评估

### VLM 并行策略解耦

**结论：可行。** 当前框架已支持子图级别的分布（`GraphDistributer.apply` 可以对任意 `TensorGraph` 进行分布），只需在 main.py 层面分别对 ViT 子图和 Text 子图进行分布，然后拼接。

关键步骤：
1. 分别构建 ViT 子图和 Text 子图（不拼接）
2. 分别对两个子图应用 MicroBatchReplicator 和各自的并行策略
3. 在拼接处（Projection → Concat → Text Backbone）处理跨子图的通信
4. 拼接后的图作为完整 VLM 进行后续处理（Chakra 转换、readout）

### MoE 文本 Backbone

**结论：需先修复 pre-existing bug。** 当前 MoE 在 Chakra 转换阶段因 `Micro(MicroBatch)` 表达式错误而失败。该 bug 根因与 VLM 中修复的 `KV(VisionHead)` 同源——`ReplicateGraph._update_symbols` 的 naive substring replacement。需要扩展修复以覆盖 MoE 场景。

## Phase 1: VLM 并行策略解耦架构设计

### Phase-1 AC

**AC-1**: `models/vlm.py` 中 `vlm()` 函数重构为返回**两个独立的子图**（ViT 子图 + Text 子图）以及拼接连接信息，而非一个拼接后的完整图：

```python
def vlm_subgraphs(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    regenerate=False,
    tpsp=True,
    include_backward=True,
):
    """Build VLM as two separate subgraphs for independent parallel strategies.
    
    Returns:
        (vision_graph, text_graph, links)
        - vision_graph: 包含 patch_embedding + vit.* + vision_projection + vlm_concat
        - text_graph: 包含 in_emb + transformer.* + out_emb + loss
        - links: vision_graph.vlm_concat.y -> text_graph.transformer.0.input_norm.x
    """
```

实现要求：
- `vision_graph`：包含 `vision_encoder.patch_embedding` + `vision_encoder.vit.*` + `vision_projection` + `vlm_concat`
- `text_graph`：包含 `in_emb` + `transformer.*` + `out_emb` + `loss`
- Projection + Concat 与 ViT 同属一个子图（共享 ViT 的并行策略）
- `links`：跨子图连接关系，如 `vlm_concat.y -> transformer.0.input_norm.x`

**AC-2**: `main.py` 中新增 `_build_and_distribute_vlm_model` 函数，支持对 ViT 和 Text 分别应用不同的并行策略：

```python
def _build_and_distribute_vlm_model(
    vision_graph,
    text_graph,
    links,
    vision_parallel_config,
    text_parallel_config,
    symbol_map_value,
    args,
    generated_filename,
):
```

处理流程：
1. 分别对 `vision_graph` 和 `text_graph` 应用 `MicroBatchReplicator`
2. 分别对 `vision_graph` 和 `text_graph` 应用 `GraphDistributer`，使用各自的并行配置
3. 使用 `ConnectGraph.apply` 拼接两个分布后的子图（处理跨子图 tensor 的 Shadow 节点）
4. 对拼接后的完整图进行 Chakra 转换和 readout

参数说明：
- `vision_parallel_config`: dict with keys `dp`, `tp`, `spp`, `ep`
- `text_parallel_config`: dict with keys `dp`, `tp`, `spp`, `ep`, `pp`

**AC-3**: CLI 参数扩展，新增 VLM 专用并行策略参数：
- `--vision_dp`, `--vision_tp`, `--vision_pp`, `--vision_ep`：ViT 并行策略
- `--text_dp`, `--text_tp`, `--text_pp`, `--text_ep`：Text Backbone 并行策略
- 若未显式指定 vision 参数，默认与 text 参数一致（保持向后兼容）

**AC-4**: 跨子图通信与 Pipeline Parallel 处理：
- 分别分布后的 `vision_graph` 和 `text_graph` 通过 `ConnectGraph.apply` 拼接
- 拼接处的跨子图 tensor（如 `vlm_concat.y` 到 `transformer.0.input_norm.x`）如果处于不同的并行布局，STAGE 框架的 `GraphDistributer._fix_cross_bucket_data_dependancies` 会自动创建 `Shadow` 节点表示跨设备通信
- Pipeline Parallel 跨子图时，`vision_pp` 和 `text_pp` 分别决定 ViT 和 Text 内部的 stage 划分。ViT 输出到 Text 输入的跨 stage 通信同样通过 `Shadow` 节点处理
- 具体实现：先分别对两个子图调用 `_create_pipeline_tensor_map`（各自使用自己的 `num_stacks` 和 `pp`），然后合并 pipeline_tensor_map，最后统一调用 `GraphDistributer`

---

## Phase 2: MoE 文本 Backbone 支持

### Phase-2 AC

**AC-5**: 修复 MoE pre-existing bug（`Micro(MicroBatch)`）：
- 根因：`moe_model.py` 中 `old_symbol_map_new_symbol={"Seq": "Seq*KExperts/(Experts*ep)"}` 导致 `Seq` 替换时，若 `op_attr` 中已有 `MicroBatch`（包含 `Batch` 子串），则会被错误替换
- 修复方式：在 `ReplicateGraph._update_symbols` 中，对替换目标字符串（`to_`）也进行 placeholder 保护，防止新插入的文本被后续替换

**AC-6**: 新增 `models/vlm_moe.py`（或扩展 `models/vlm.py`），支持 MoE 作为文本 backbone：

```python
def vlm_moe(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    experts=1,
    kexperts=1,
    regenerate=False,
    tpsp=True,
    include_backward=True,
):
```

**AC-7**: `main.py` 中 VLM 分支支持 `model_type="vlm_moe"`，使用 MoE 作为文本 backbone：
- 加载 MoE 专用 sharding spreadsheets（`sharding_spreadsheets/module/tpsp_moe/`）
- 支持 EP（Expert Parallelism）配置

---

## Phase 3: 虚拟 VLM 验证

### Phase-3 AC

**AC-8**: 构造虚拟 VLM 模型（ViT 2 blocks + Text 2 blocks），验证以下并行组合：

| 组合 | Vision 策略 | Text 策略 | 验证目标 |
|------|------------|-----------|---------|
| C1 | DP=2 | TP=2 | 基础解耦 |
| C2 | TP=2 | PP=2 | ViT TP + Text PP |
| C3 | DP=2 | PP=2 | ViT DP + Text PP |
| C4 | TP=2, DP=2 | TP=2, PP=2 | 混合策略 |

每个组合的验证标准：
- `main.py` 命令成功执行，无异常退出
- 生成的 `.et` 文件非空
- 使用高层 GraphML 可视化验证数据流正确（ViT → Projection → Concat → Text）

**AC-9**: 虚拟 VLM 模型参数：
- 使用 `qwen2_5_vl_3b` 配置但覆盖 `--num_stacks=2 --vision_num_hidden_layers=2`
- Batch=4, MicroBatch=2

---

## Phase 4: MoE 文本 Backbone 验证

### Phase-4 AC

**AC-10**: 验证 MoE 文本 backbone 在以下配置下正常运行：

| 配置 | EP | TP | DP | 预期结果 |
|------|-----|-----|-----|---------|
| M1 | 8 | 1 | 1 | EP=8 基础 |
| M2 | 8 | 2 | 2 | EP+TP+DP 混合 |

每个配置的验证标准：
- `main.py --model_type vlm_moe` 命令成功执行
- 生成的 `.et` 文件包含 MoE 相关节点（expert wrapper, moe frame 等）
- 无 `Micro(MicroBatch)` 或 `KV(VisionHead)` 类表达式错误

---

## Phase 5: 回归测试

### Phase-5 AC

**AC-11**: 向后兼容验证：
- `--model_type dense` 使用原有 CLI 方式，输出与上次迭代一致
- `--model_type gpt` 回归测试通过
- `--model_type vlm` 使用统一并行策略（不指定 vision/text 分离参数），行为与上次迭代一致

**AC-12**: 可视化输出：
- 为每个验证组合生成高层 GraphML 文件
- 文件保存到 `/tmp/vlm_parallel/` 目录
- 主人可通过 SCP 下载到本地 yEd 查看

---

## 风险与假设

1. **跨子图通信**：ViT 和 Text 使用不同并行策略时，Projection/Concat 边界处需要插入通信节点（如 AllGather/AllReduce）。当前 `GraphDistributer` 的 `_fix_cross_bucket_data_dependancies` 使用 Shadow 节点处理跨 bucket 通信，但不同并行策略的跨子图通信可能需要额外处理。
2. **MoE Bug 修复范围**：`Micro(MicroBatch)` 的修复可能需要改动 `ReplicateGraph._update_symbols` 的核心替换逻辑，影响所有模型类型。需要充分回归测试。
3. **PP 跨子图**：Pipeline Parallel 跨越 ViT 和 Text 边界时，需要确保 Concat 操作在正确的 stage 执行。
4. **向后兼容**：新增 CLI 参数必须保持默认值与旧行为一致。

---

## 文件变更清单

| 文件 | 动作 | 说明 |
|------|------|------|
| `models/vlm.py` | 修改 | 重构为返回子图 + 拼接信息 |
| `main.py` | 修改 | 新增 `_build_and_distribute_vlm_model`，VLM 并行解耦 CLI 参数 |
| `models/vlm_moe.py` | 新增 | MoE 作为文本 backbone 的 VLM |
| `symbolic_tensor_graph/graph/replicate_graph.py` | 修改 | 修复 MoE `Micro(MicroBatch)` bug |
| `models/moe_model.py` | 可能修改 | 配合修复 |

