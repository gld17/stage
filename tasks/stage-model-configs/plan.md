# Stage 迭代：模型配置文件化 + VLM（Qwen2.5-VL）支持

## 背景

当前 `main.py` 的模型结构参数（`--dvocal`, `--dmodel`, `--dff`, `--num_stacks`, `--head`, `--kvhead` 等）全部通过 argparse 手动传入，使用繁琐且容易出错。需要：

1. **配置文件化**：在 `models/` 下创建 `model_configs/` 子文件夹，以 JSON 文件存储各模型的结构参数，主入口通过 `--model_name` 直接调用。
2. **支持 VLM（Vision Language Model）**：以 Qwen2.5-VL 系列（3B/7B/32B/72B）为首批目标。Qwen2.5-VL 只是 VLM 中的一种具体实例，`model_type` 应泛化为 `"vlm"`。
3. **完整 VLM 建模**：VLM 包含 Vision Encoder（ViT）+ Projection + Text Backbone。需评估并实施用现有 STG 模块/算子组装 ViT 的可行性。

## 技术可行性评估：能否用现有模块构建 ViT？

**结论：可行。** STAGE 框架的底层算子完全通用：

- **Patch Embedding**：`Reshape`（image → patches）+ `Einsum`/`M`（线性投影）即可模拟 Patch Embedding。
- **ViT Block**：与 Transformer Decoder Block 结构相同（Self-Attn + FFN + LN + Residual）。现有 `group_query_attention`、`feed_forward_network`、`layer_norm`、`residual` 的**底层算子组合**（`Einsum`, `Add`, `Element` 等）可直接复用。ViT 使用 bidirectional self-attention（无 causal mask），但在 STG 层面 mask 模式不影响并行维度和通信模式，sharding 策略与文本 MHA 一致。
- **Vision-Text 拼接**：`Concat`/`C` 算子原生支持沿指定维度拼接 tensor，可将 ViT 输出的 visual tokens 与 text tokens 拼接后送入文本 backbone。
- **Projection 层**：单个 `Einsum`/`M`（线性映射）即可实现 vision hidden_size → text hidden_size 的投影。

**因此，本次迭代将完整实现：Vision Encoder 模型构建 + Projection + VLM 拼接。**

> 注意：ViT 的 tensor shapes（image tokens 而非 text tokens）与文本模型不同，需要**新增 ViT 专用的 sharding spreadsheets**（`sharding_spreadsheets/module/vit/`），不能直接复用文本模型的 CSV。

---

## Phase 1: 设计 model_configs JSON Schema

### Phase-1 AC

**AC-1**: 在 `models/model_configs/` 下创建 JSON Schema 文档 `schema.json`，规范每个配置文件的字段：
- `model_name`: 字符串，如 `"qwen2_5_vl_3b"`
- `model_type`: 枚举 `"dense" | "gpt" | "moe" | "vlm"`，决定调用哪个模型构建函数。`"vlm"` 为通用视觉语言模型类型，Qwen2.5-VL 只是其中一种实例。
- `display_name`: 字符串，人类可读名称
- `vocab_size` (`dvocal`): int — 文本词表大小
- `hidden_size` (`dmodel`): int — 文本 backbone hidden dimension
- `intermediate_size` (`dff`): int — 文本 backbone FFN 中间维度
- `num_hidden_layers` (`num_stacks`): int — 文本 backbone 层数
- `num_attention_heads` (`head`): int — 文本 backbone 注意力头数
- `num_key_value_heads` (`kvhead`): int — 文本 backbone GQA KV 头数
- `experts`: int（MoE 专用，dense/VLM 模型设为 1）
- `kexperts`: int（MoE 专用，dense/VLM 模型设为 1）
- `head_dim`: int（可选，如未提供则 `dmodel // head`）
- `vision_hidden_size`: int — ViT hidden dimension
- `vision_num_hidden_layers`: int — ViT 层数
- `vision_num_attention_heads`: int — ViT 注意力头数
- `vision_intermediate_size`: int — ViT FFN 中间维度
- `vision_image_size`: int — 输入图像尺寸（如 448）
- `vision_patch_size`: int — Patch 大小（如 14）
- `vision_in_channels`: int — 输入通道数（通常为 3）
- `vision_projection_size`: int — Vision → Text 投影层输出维度（通常等于 `hidden_size`）
- `notes`: 字符串数组，备注

**AC-2**: 创建 `models/model_configs/__init__.py`，提供两个公共 API：
- `load_model_config(model_name: str) -> dict`：按名称加载 JSON 配置，不存在则抛出 `ValueError`
- `list_available_models() -> list[str]`：返回所有可用的 `model_name`

---

## Phase 2: 创建模型配置文件（含 Vision Encoder 参数）

### Phase-2 AC

> **参数来源说明**：以下 Qwen2.5-VL 各规模参数基于公开资料估算。`vision_*` 系列参数（ViT 配置）可能跨规模相同（视觉编码器通常不与语言模型规模线性增长）。Builder 实现时须通过 HuggingFace 官方 `config.json` 精确核实，如有偏差在 Review 阶段提出修正。

**AC-3**: 创建 `models/model_configs/qwen3-4b.json`，参数如下（参考 Qwen3 官方配置）：
```json
{
  "model_name": "qwen3-4b",
  "model_type": "dense",
  "display_name": "Qwen3-4B",
  "vocab_size": 151936,
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "num_hidden_layers": 36,
  "num_attention_heads": 40,
  "num_key_value_heads": 8,
  "experts": 1,
  "kexperts": 1,
  "vision_hidden_size": 1,
  "vision_num_hidden_layers": 1,
  "vision_num_attention_heads": 1,
  "vision_intermediate_size": 1,
  "vision_image_size": 1,
  "vision_patch_size": 1,
  "vision_in_channels": 1,
  "vision_projection_size": 1,
  "notes": ["Qwen3-4B dense model, vision fields are dummy for dense models"]
}
```

**AC-4**: 创建 `models/model_configs/qwen2_5_vl_3b.json`（Qwen2.5-VL 3B）：
```json
{
  "model_name": "qwen2_5_vl_3b",
  "model_type": "vlm",
  "display_name": "Qwen2.5-VL-3B",
  "vocab_size": 151936,
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "experts": 1,
  "kexperts": 1,
  "vision_hidden_size": 1280,
  "vision_num_hidden_layers": 32,
  "vision_num_attention_heads": 16,
  "vision_intermediate_size": 5120,
  "vision_image_size": 448,
  "vision_patch_size": 14,
  "vision_in_channels": 3,
  "vision_projection_size": 2048,
  "notes": ["Qwen2.5-VL (3B). Vision encoder params are estimates; verify against HF config.json"]
}
```

**AC-5**: 创建 `models/model_configs/qwen2_5_vl_7b.json`：
```json
{
  "model_name": "qwen2_5_vl_7b",
  "model_type": "vlm",
  "display_name": "Qwen2.5-VL-7B",
  "vocab_size": 152064,
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "experts": 1,
  "kexperts": 1,
  "vision_hidden_size": 1664,
  "vision_num_hidden_layers": 32,
  "vision_num_attention_heads": 16,
  "vision_intermediate_size": 6656,
  "vision_image_size": 448,
  "vision_patch_size": 14,
  "vision_in_channels": 3,
  "vision_projection_size": 3584,
  "notes": ["Qwen2.5-VL (7B). Vision encoder params are estimates; verify against HF config.json"]
}
```

**AC-6**: 创建 `models/model_configs/qwen2_5_vl_32b.json`：
```json
{
  "model_name": "qwen2_5_vl_32b",
  "model_type": "vlm",
  "display_name": "Qwen2.5-VL-32B",
  "vocab_size": 152064,
  "hidden_size": 5120,
  "intermediate_size": 27648,
  "num_hidden_layers": 64,
  "num_attention_heads": 40,
  "num_key_value_heads": 8,
  "experts": 1,
  "kexperts": 1,
  "vision_hidden_size": 1664,
  "vision_num_hidden_layers": 32,
  "vision_num_attention_heads": 16,
  "vision_intermediate_size": 6656,
  "vision_image_size": 448,
  "vision_patch_size": 14,
  "vision_in_channels": 3,
  "vision_projection_size": 5120,
  "notes": ["Qwen2.5-VL (32B). Vision encoder params are estimates; verify against HF config.json"]
}
```

**AC-7**: 创建 `models/model_configs/qwen2_5_vl_72b.json`：
```json
{
  "model_name": "qwen2_5_vl_72b",
  "model_type": "vlm",
  "display_name": "Qwen2.5-VL-72B",
  "vocab_size": 152064,
  "hidden_size": 8192,
  "intermediate_size": 29696,
  "num_hidden_layers": 80,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "experts": 1,
  "kexperts": 1,
  "vision_hidden_size": 1664,
  "vision_num_hidden_layers": 32,
  "vision_num_attention_heads": 16,
  "vision_intermediate_size": 6656,
  "vision_image_size": 448,
  "vision_patch_size": 14,
  "vision_in_channels": 3,
  "vision_projection_size": 8192,
  "notes": ["Qwen2.5-VL (72B). Vision encoder params are estimates; verify against HF config.json"]
}
```

---

## Phase 3: main.py 重构 —— 支持 `--model_name` 与配置加载

### Phase-3 AC

**AC-8**: `main.py` 新增 `--model_name` 参数（`type=str`, `required=False`, `default=None`），并在 `argparse` 的 `description` 中说明：
> 可通过 `--model_name <name>` 从 `models/model_configs/` 自动加载模型结构参数；所有结构参数（`--dvocal`, `--dmodel`, `--dff`, `--num_stacks`, `--head`, `--kvhead`, `--experts`, `--kexperts`）变为可选，CLI 显式传入的值优先级高于配置文件。

**AC-9**: `main()` 中在 `parser.parse_args()` 之后、构建 `symbol_map_value` 之前，插入配置加载逻辑：
- 若 `args.model_name` 不为空，调用 `load_model_config(args.model_name)` 获取配置字典。
- 对配置中的每个结构参数字段（文本参数 + Vision 参数），仅在 `args` 中对应值为默认值（或 None）时，用配置值覆盖；CLI 显式传入的值始终优先。
- 若 `args.model_name` 为空，则保持现有行为（全部使用 CLI 默认值）。
- 若配置 `model_type == "vlm"` 但 CLI 未显式传入 `--model_type`，自动将 `args.model_type` 设为 `"vlm"`。

**AC-10**: `--model_type` 的 `choices` 更新为 `["dense", "gpt", "moe", "debug", "vlm"]`。

---

## Phase 4: Vision Encoder 模型构建

### Phase-4 AC

**AC-11**: 创建 `sharding_spreadsheets/module/vit/` 目录，复制 `sharding_spreadsheets/module/tpsp/` 中的以下文件作为 ViT 的初始 sharding 模板（ViT 的 MHA/FFN/LN/Residual 并行策略与文本模型相同）：
- `embedding.csv` → 重命名为 `patch_embedding.csv`（或保留 embedding.csv，作为 patch projection 的 sharding 模板）
- `group_query_attention_surrounding.csv` / `group_query_attention_kernel_fused.csv`
- `llama_feed_forward_network.csv`
- `layer_norm.csv`
- `residual.csv`

Builder 可根据需要调整这些 CSV 中的维度符号（如将 `Seq` 替换为 `NumPatches`，但保持 sharding 策略不变）。

**AC-12**: 创建 `models/vision_encoder.py`，提供 `vision_encoder(...)` 函数，使用现有 STG 算子（`Einsum`/`M`, `Reshape`, `Add`, `Element`）和 API（`TensorGraph`, `ConnectGraph`, `ReplicateGraph`）构建 ViT：

```python
def vision_encoder(
    num_layers,
    symbol_map_value,
    patch_embedding_path=None,
    regenerate=False,
    include_backward=True,
):
    """Build a Vision Transformer (ViT) encoder using existing STG primitives.

    Architecture:
      1. Patch Embedding: Reshape(image) -> Einsum(patch projection)
      2. N x ViT Block: Self-Attn (bidirectional) + FFN + LN + Residual
      3. Output: visual token embeddings
    """
```

实现要求：
- Patch Embedding 用 `Reshape` + `Einsum`（`"ab,bc->ac"` 风格的线性投影）模拟。
- ViT Block 内部复用与 `llama_model.py` 相似的连接逻辑（Self-Attn + FFN + LN + Residual），但**不引入 causal mask 相关的特殊处理**（ViT 是 bidirectional）。
- 使用 `sharding_spreadsheets/module/vit/` 下的 CSV 文件作为模块模板。
- 返回 `TensorGraph` 对象。

**AC-13**: 创建 `models/vlm_connector.py`（或直接在 `vision_encoder.py` 中），提供 `vision_projection(...)` 函数：
- 输入：ViT 输出的 visual tokens（shape: `[Batch, NumPatches, VisionHidden]`）
- 输出：投影后的 visual tokens（shape: `[Batch, NumPatches, TextHidden]`）
- 实现：单个 `Einsum`/`M` 算子（线性映射），权重符号维度为 `VisionHidden × TextHidden`。

---

## Phase 5: VLM 完整拼接与 main.py 集成

### Phase-5 AC

**AC-14**: 创建 `models/vlm.py`，提供 `vlm(...)` 函数，将 ViT + Projection + Text Backbone 拼接为完整 VLM：

```python
def vlm(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    text_backbone_fn,
    regenerate=False,
    tpsp=True,
    include_backward=True,
):
    """Build a full Vision-Language Model.

    Pipeline:
      image patches -> ViT -> Projection -> Concat(visual_tokens, text_tokens) -> Text Backbone
    """
```

实现要求：
1. 调用 `vision_encoder(...)` 构建 ViT。
2. 调用 `vision_projection(...)` 将 ViT 输出映射到文本 hidden_size。
3. 调用 `text_backbone_fn(...)`（即 `llama_model.llama` 或 `gpt_model.gpt`）构建文本 backbone。
4. 使用 `Concat` 算子将投影后的 visual tokens 与文本 backbone 的输入 token embeddings 沿 sequence 维度拼接。
5. 使用 `ConnectGraph.apply()` 建立完整的连接关系。
6. 返回完整的 VLM `TensorGraph`。

> **Concat 对后续执行图的关键影响**：
> - **Sequence 长度变化**：Concat 后 sequence 维度从 `TextSeq` 变为 `NumPatches + TextSeq`。`symbol_map_value` 中的 `Seq` 符号必须被更新为新的总长度（或引入新的符号 `TotalSeq`），否则下游 `GraphDistributer` 在解析 tensor shapes 时会因符号未定义而失败。
> - **Sharding 策略传播**：ViT 输出的 visual tokens 与文本 embedding 拼接后，整个 sequence 维度（`TotalSeq`）需要一致的 sharding 策略。`GraphDistributer` 必须将 `NumPatches` 和 `TextSeq` 视为同一 sequence 轴的不同组成部分，确保 TP/SP 切分在拼接边界正确对齐。
> - **Pipeline Parallel 影响**：ViT Encoder、Projection 层和 Text Backbone 在 PP 切分时可能分布在不同 stage。Concat 操作如果跨越 PP 边界，需要确保 `GraphDistributer` 正确处理跨 stage 的 tensor 传递。当前实现假设 ViT 和 Text Backbone 在同一 PP stage 内拼接（或 ViT 输出通过 pipeline 传递到 Text Backbone 的输入 stage）。
> - **MicroBatchReplicator**：在复制 micro-batch 时，拼接后的 `TotalSeq` 维度必须被正确传播到所有 replicated subgraphs，保证每个 micro-batch 的视觉+文本 token 比例一致。
> - **Backward 图**：Concat 的 backward 会产生 `dvisual` 和 `dtext` 两个梯度分支。`dvisual` 需要经过 Projection 的 backward 传回 ViT，`dtext` 则直接参与文本 backbone 的 backward。两个分支的符号映射必须独立且正确。

**AC-15**: `main.py` 在 `model_type` 判断分支中，新增 `"vlm"` 分支：

```python
elif args.model_type == "vlm":
    from models.vlm import vlm as build_vlm
    from models.llama_model import llama as build_text_backbone

    vlm_graph = build_vlm(
        text_num_layers=num_stacks,
        vision_num_layers=config["vision_num_hidden_layers"],
        symbol_map_value=symbol_map_value,
        text_backbone_fn=build_text_backbone,
        regenerate=True,
        tpsp=args.tpsp,
        include_backward=args.include_backward,
    )
    # ... 后续复用 MicroBatchReplicator, ReplicateGraph, GraphDistributer 等流程
```

VLM 图的后续处理（`MicroBatchReplicator`, `ReplicateGraph`, `GraphDistributer`, `BundledConvertChakra`, `readout`）复用现有 `_build_and_distribute_dense_model` 的逻辑，或提取为通用函数。

**AC-16**: 若 `--model_type vlm` 时 `--model_name` 未提供，使用合理的默认值构建一个最小 VLM（如 2-layer ViT + 2-layer text backbone），并打印警告提示用户应使用 `--model_name` 加载完整配置。

---

## Phase 6: 验证与回归测试

### Phase-6 AC

**AC-17**: `list_available_models()` 返回的列表包含全部 5 个配置名称，且每个 JSON 文件均通过 `json.load()` 验证无语法错误。

**AC-18**: 回归测试：使用原有的 CLI 方式（不指定 `--model_name`，`--model_type dense`）生成执行图，与上次迭代通过的命令对比，输出必须一致，确保向后兼容。

**AC-19**: 新增测试：使用 `--model_name qwen3-4b` 生成执行图，验证 `symbol_map_value` 中的 `Dmodel=2560`、`num_stacks=36`、`Head=40`、`KVHead=8` 等参数正确映射。

**AC-20**: 新增测试：使用 `--model_name qwen2_5_vl_7b --model_type vlm` 生成执行图，验证：
- `model_type` 正确路由到 VLM builder
- 文本参数：`Dmodel=3584`、`num_stacks=28`
- Vision 参数：`vision_hidden_size=1664`、`vision_num_hidden_layers=32`
- VLM 图包含 ViT + Projection + Text Backbone 三部分节点

**AC-21**: 新增测试：验证 CLI override 机制有效。例如 `--model_name qwen3-4b --dmodel 2048` 时，`Dmodel` 应取 CLI 值 `2048`，而非配置文件中的 `2560`。

**AC-22**: 新增测试：验证 ViT 子图的 tensor shapes 正确。例如 ViT 输入的 patch 数量应为 `(image_size / patch_size)^2`（如 448/14 = 32，32×32 = 1024 patches）。

**AC-23**: 新增全流程端到端测试：使用 `--model_name qwen2_5_vl_3b --model_type vlm` 执行完整的 ET（Execution Trace）生成流程，从模型构建、sharding 分布、Chakra 转换到最终的 `.et` 文件输出。验证：
- 命令成功执行，无异常退出
- 生成的 `.et` 文件包含完整的 VLM 执行图节点（包含 ViT encoder、projection、text backbone 各阶段）
- 输出文件路径和命名格式符合 `--output_dir` 和 `--output_name` 参数配置

---

## 风险与假设

1. **Qwen2.5-VL 参数准确性**：由于当前环境无法联网，AC-4~AC-7 中的 `vision_*` 参数基于公开资料估算。Builder 实现时须通过 HuggingFace `config.json` 核实。若官方值不同，Planner 将更新 AC 并重新进入 Review 循环。
2. **ViT sharding 策略假设**：假设 ViT 的 MHA/FFN/LN/Residual 的并行策略与文本 LLaMA 相同（TP 切分 attention heads，SP 切分 sequence）。ViT 处理的是 image patches 而非 text tokens，但并行维度的数学结构一致。如果实际 sharding 策略不同（如需要 spatial parallel），需新增专门的 sharding spreadsheets。
3. **Concat 符号映射**：VLM 拼接时，visual tokens 与 text tokens 的 Concat 会改变 `Seq` 符号的语义（从纯文本长度变为 `NumPatches + TextSeq`）。如果下游分布计算（`GraphDistributer`）对 `Seq` 有特殊假设，可能需要引入新的符号（如 `TotalSeq`）。
4. **Vision Encoder Frozen 假设**：当前实现假设 Vision Encoder 参与完整训练（含 backward）。如果实际场景中 ViT 是 frozen 的（如 LoRA 只训练文本部分），backward 图需要调整。后续迭代可根据需求扩展 `freeze_vision_encoder` 参数。
5. **向后兼容**：本次重构保留所有原有 argparse 参数，仅新增 `--model_name`。任何未指定 `--model_name` 的调用，行为与重构前完全一致。

---

## 文件变更清单

| 文件 | 动作 | 说明 |
|------|------|------|
| `models/model_configs/schema.json` | 新增 | JSON Schema 定义（含 vision_* 字段） |
| `models/model_configs/__init__.py` | 新增 | 配置加载 API |
| `models/model_configs/qwen3-4b.json` | 新增 | Qwen3-4B 配置 |
| `models/model_configs/qwen2_5_vl_3b.json` | 新增 | Qwen2.5-VL-3B 配置（含 ViT 参数） |
| `models/model_configs/qwen2_5_vl_7b.json` | 新增 | Qwen2.5-VL-7B 配置（含 ViT 参数） |
| `models/model_configs/qwen2_5_vl_32b.json` | 新增 | Qwen2.5-VL-32B 配置（含 ViT 参数） |
| `models/model_configs/qwen2_5_vl_72b.json` | 新增 | Qwen2.5-VL-72B 配置（含 ViT 参数） |
| `sharding_spreadsheets/module/vit/` | 新增 | ViT 专用 sharding spreadsheets（从 tpsp/ 复制初始模板） |
| `models/vision_encoder.py` | 新增 | ViT 模型构建（Patch Embedding + ViT Blocks） |
| `models/vlm_connector.py` | 新增 | Vision Projection 层 |
| `models/vlm.py` | 新增 | 完整 VLM 拼接（ViT + Projection + Text Backbone） |
| `main.py` | 修改 | 新增 `--model_name`，配置加载逻辑，`vlm` model_type 分支 |
