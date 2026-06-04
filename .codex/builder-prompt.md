# Builder Prompt: VLM 并行策略解耦 + MoE 文本 Backbone 支持

## 任务概述

根据 Plan 文档（`tasks/vlm-parallel-decoupling/plan.md`），对 Stage 项目的 VLM 模型支持进行迭代：

1. **VLM 并行策略解耦**：将 ViT（Vision Encoder）和 Text Backbone 分别构建为独立子图，支持配置不同的并行策略参数（PP/DP/TP/SP/EP）。Projection + Concat 跟 ViT 同属 vision 子图。
2. **MoE 文本 Backbone**：新增 VLM + MoE 支持，修复 MoE 的 `Micro(MicroBatch)` pre-existing bug。
3. **向后兼容**：不破坏现有 Dense/GPT/VLM 统一并行策略的行为。

## 工作目录

`/share/guolidong-nfs/SeeSpace/SCSP_v1/stage`

运行测试用：
```bash
export HTTPS_PROXY=http://127.0.0.1:7890
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py [args]
```

## 需要修改的文件

### 1. `models/vlm.py` — 重构为返回子图

**当前 `vlm()` 函数**（约 115-203 行）构建一个完整的拼接图并返回。需要：

**AC-1**: 新增 `vlm_subgraphs()` 函数，返回两个独立子图和连接信息：

```python
def vlm_subgraphs(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    text_backbone_fn=None,
    regenerate=False,
    tpsp=True,
    include_backward=True,
):
    """Build VLM as two separate subgraphs for independent parallel strategies.
    
    Returns:
        (vision_graph, text_graph, links)
        - vision_graph: 包含 vision_encoder.patch_embedding + vision_encoder.vit.* 
                        + vision_projection + vlm_concat
        - text_graph: 包含 in_emb + transformer.* + out_emb + loss
        - links: dict of cross-subgraph connections, e.g.:
            {"vlm_concat.y": "transformer.0.input_norm.x"}
    """
```

实现要求：
- 保留现有 `vlm()` 函数**不变**（向后兼容），内部调用 `vlm_subgraphs()` + `ConnectGraph.apply` + backward links
- `vlm_subgraphs()` 中：分别构建 vision 部分（patch_embed → vit blocks → projection → concat）和 text 部分（in_emb → transformers → out_emb → loss），但不拼接
- `links` 中只包含跨子图的连接关系（vision output → text input），不含子图内部的连接
- backward 的跨子图连接也放在 `links` 中（如 `"vlm_concat.dy": "transformer.0.input_norm.dx"`）

**关键提示**：当前 `vlm()` 中 `_set_identical_input` 将 `vlm_concat.y` 连接到 `transformer.0.input_norm.x`，这种连接关系应放入 `links` dict 返回。

### 2. `main.py` — VLM 并行解耦支持

**当前状态**：VLM 分支（约 400-422 行）调用 `_build_and_distribute_dense_model`，将 VLM 视为统一 Dense 模型。

**AC-3**: 新增 CLI 参数：
```python
parser.add_argument("--vision_dp", type=int, default=None, required=False)
parser.add_argument("--vision_tp", type=int, default=None, required=False)
parser.add_argument("--vision_pp", type=int, default=None, required=False)
parser.add_argument("--vision_ep", type=int, default=None, required=False)
parser.add_argument("--text_dp", type=int, default=None, required=False)
parser.add_argument("--text_tp", type=int, default=None, required=False)
parser.add_argument("--text_pp", type=int, default=None, required=False)
parser.add_argument("--text_ep", type=int, default=None, required=False)
```
- 默认值 `None` 表示"与全局参数一致"
- 在 `args` 解析后，如果 vision/text 参数为 `None`，用对应的全局参数填充：`args.vision_dp = args.vision_dp or args.dp`

**AC-2**: 新增 `_build_and_distribute_vlm_model()` 函数：

```python
def _build_and_distribute_vlm_model(
    vision_graph,
    text_graph,
    links,
    symbol_map_value,
    args,
    generated_filename,
    header="[VLM-Decoupled] ",
):
```

**实现方案**（分阶段分布）：

由于 ViT 和 Text 可能使用不同的空间并行策略（TP/DP/SP），采用以下流程：

```python
def _build_and_distribute_vlm_model(vision_graph, text_graph, links, symbol_map_value, args, generated_filename, header="[VLM-Decoupled] "):
    # Step 1: MicroBatchReplicator（统一）
    vision_graph = MicroBatchReplicator.apply(vision_graph, symbol_map_value)
    text_graph = MicroBatchReplicator.apply(text_graph, symbol_map_value)
    
    # Step 2: FSDP weight sharding（统一）
    if args.weight_sharded:
        vision_graph = ReplicateGraph.apply(vision_graph, inplace=True, old_symbol_map_new_symbol={"fsdp": "dp"})
        text_graph = ReplicateGraph.apply(text_graph, inplace=True, old_symbol_map_new_symbol={"fsdp": "dp"})
    else:
        vision_graph = ReplicateGraph.apply(vision_graph, inplace=True, old_symbol_map_new_symbol={"fsdp": 1})
        text_graph = ReplicateGraph.apply(text_graph, inplace=True, old_symbol_map_new_symbol={"fsdp": 1})
    
    # Step 3: Training mode
    vision_graph = _apply_training_mode(vision_graph, args.include_backward)
    text_graph = _apply_training_mode(text_graph, args.include_backward)
    
    # Step 4: 拼接
    full_graph = ConnectGraph.apply([vision_graph, text_graph], links)
    
    # Step 5: Pipeline mapping（支持 VLM 跨子图）
    # 需要扩展 _create_pipeline_tensor_map 或新增 _create_vlm_pipeline_tensor_map
    # 让 ViT 和 Text 的 tensor 可以分别映射到不同的 PP stage
    pipeline_tensor_map = _create_vlm_pipeline_tensor_map(
        full_graph.tensors,
        [sp.symbols("pp")],
        symbol_map_value,
        args.vision_num_hidden_layers,
        args.num_stacks,
        args.vision_pp or args.pp,
        args.text_pp or args.pp,
    )
    
    # Step 6: 统一 GraphDistributer
    # spatial_parallel_dims 使用全局值，但通过 sharding spreadsheets 的差异体现不同的并行策略
    spatial_parallel_dims = [sp.symbols("dp"), sp.symbols("tp"), sp.symbols("spp")]
    symbol_map_value[sp.symbols("tp")] *= symbol_map_value.get(sp.symbols("ep"), 1)
    
    distributed = GraphDistributer.apply(
        full_graph, symbol_map_value, spatial_parallel_dims, [sp.symbols("pp")], pipeline_tensor_map
    )
    
    # Step 7: Chakra 转换和 readout（复用现有逻辑）
    ...
```

**AC-4**: 新增 `_create_vlm_pipeline_tensor_map()` 函数：

```python
def _create_vlm_pipeline_tensor_map(
    _tensors, _temporal_parallel_dims, _symbol_map_value,
    vision_num_layers, text_num_layers,
    vision_pp, text_pp,
):
```

实现逻辑：
1. 计算全局总 stage 数 = vision_pp + text_pp（Projection+Concat 跟 vision 同 stage）
2. ViT tensor（`vision_encoder.*`, `vision_projection.*`, `vlm_concat.*`）分配到 stage 0 ~ vision_pp-1
3. Text tensor（`in_emb.*`, `transformer.*`, `out_emb.*`, `loss.*`）分配到 stage vision_pp ~ vision_pp+text_pp-1
4. 对于 ViT 内部：按 `vision_num_layers` 均匀分配到 `vision_pp` 个 stage
5. 对于 Text 内部：按 `text_num_layers` 均匀分配到 `text_pp` 个 stage

**向后兼容**：原有的 `_create_pipeline_tensor_map` **保持不变**。当 VLM 使用统一并行策略（不指定 `--vision_pp` 等）时，`_build_and_distribute_vlm_model` 退化为与旧行为一致（`vision_pp = text_pp = args.pp`）。

**修改 VLM 分支调用**（约 400-422 行）：
```python
elif args.model_type == "vlm":
    from models.vlm import vlm_subgraphs as build_vlm_subgraphs
    
    vision_graph, text_graph, links = build_vlm_subgraphs(
        text_num_layers=num_stacks,
        vision_num_layers=args.vision_num_hidden_layers,
        symbol_map_value=symbol_map_value,
        regenerate=True,
        tpsp=args.tpsp,
        include_backward=args.include_backward,
    )
    _build_and_distribute_vlm_model(
        vision_graph, text_graph, links,
        symbol_map_value, args, generated_filename,
        header="[VLM] ",
    )
```

### 3. `symbolic_tensor_graph/graph/replicate_graph.py` — 修复 MoE Bug

**当前状态**（上次迭代已修复 `KV(VisionHead)`）：
- `_update_symbols()` 中按 key 长度降序排序 + placeholder 两阶段替换

**AC-5**: 修复 `Micro(MicroBatch)` bug：

根因：`moe_model.py` 中 `old_symbol_map_new_symbol={"Seq": "Seq*KExperts/(Experts*ep)"}`。当 `op_attr` 中已有 `MicroBatch`（包含 `Batch` 子串），替换 `Batch` → `MicroBatch` 时，会把已经替换好的 `MicroBatch` 再次替换为 `MicroMicroBatch`。

当前代码对 `from_`（被替换的 key）做了排序和 placeholder 保护，但没有对 `to_`（替换后的值）做保护。

修复方式：在 placeholder 替换阶段，不仅替换 `from_`，也要确保 `to_` 中如果包含其他 `from_` 的子串，不会被后续替换影响。

实际上，当前的 placeholder 机制已经解决了大部分问题（先全部替换为 `__STAGE_SYMBOL_REPLACE_i__`，再统一替换为最终值）。`Micro(MicroBatch)` 的问题可能是因为 `to_` 中的 `Batch` 被后续轮次的 `Batch` → `MicroBatch` 替换影响。

具体修复：在 `_update_symbols` 中，**对所有 tensor 属性先做第一轮 placeholder 替换（全部 `from_` → placeholder），然后再做第二轮替换（placeholder → `to_`）**。当前代码已经这样做了，但只对 `op_attr` 做了，没有对 `x1_shape/x1_hidden/x2_shape/x2_hidden` 做。

检查当前代码：
```python
for from_, to_ in parsed_items:
    for tensor in graph.tensors:
        if not tensor.x1_shape is None:
            for i, dim in enumerate(tensor.x1_shape):
                tensor.x1_shape[i] = dim.replace(from_, to_)
        ...
```

这段代码是**逐轮替换**（每个 `from_`/`to_` 对直接替换），不是两阶段。而 `op_attr` 部分才是两阶段。

**修复**：对 `x1_shape`, `x1_hidden`, `x2_shape`, `x2_hidden` 也使用两阶段 placeholder 替换。

### 4. `models/vlm_moe.py` — 新增 MoE 文本 Backbone VLM

**AC-6/7**: 新增文件，结构与 `models/vlm.py` 类似，但 text backbone 使用 MoE：

```python
from models.moe_model import transformer as moe_transformer

def vlm_moe_subgraphs(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    regenerate=False,
    tpsp=True,
    include_backward=True,
):
    """Build VLM with MoE text backbone. Same subgraph structure as vlm_subgraphs."""
```

- Vision 部分与 `vlm_subgraphs` 完全相同
- Text 部分调用 `moe_transformer()` 而非 dense transformer
- 返回 `(vision_graph, text_graph, links)` 相同结构

**main.py 中新增 `vlm_moe` model_type**：
```python
elif args.model_type == "vlm_moe":
    from models.vlm_moe import vlm_moe_subgraphs
    # 类似 vlm 分支调用 _build_and_distribute_vlm_model
    # text_graph 中需要 experts/kexperts 参数
```

注意：`model_type` choices 中需添加 `"vlm_moe"`。

## 向后兼容要求

1. `models/vlm.py` 的 `vlm()` 函数**必须保留**，内部调用 `vlm_subgraphs()` + 拼接
2. `main.py` 中 `--model_type vlm` 不指定 `--vision_*` 参数时，行为与上次迭代完全一致
3. `main.py` 中 Dense/GPT/MoE/Debug 分支**不得修改**
4. `_create_pipeline_tensor_map` **不得修改**（或修改后保持原有行为不变）

## 测试验证命令

### VLM 虚拟模型验证（AC-8）
```bash
# C1: Vision DP=2, Text TP=2
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/vlm_parallel --output_name vlm_c1 \
  --model_type vlm --model_name qwen2_5_vl_3b \
  --num_stacks 2 --vision_num_hidden_layers 2 \
  --batch 4 --micro_batch 2 --dp 2 --tp 2 --pp 1

# C2: Vision TP=2, Text PP=2
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/vlm_parallel --output_name vlm_c2 \
  --model_type vlm --model_name qwen2_5_vl_3b \
  --num_stacks 2 --vision_num_hidden_layers 2 \
  --batch 4 --micro_batch 2 --tp 2 --pp 2

# C3: Vision DP=2, Text PP=2
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/vlm_parallel --output_name vlm_c3 \
  --model_type vlm --model_name qwen2_5_vl_3b \
  --num_stacks 2 --vision_num_hidden_layers 2 \
  --batch 4 --micro_batch 2 --dp 2 --pp 2

# C4: Vision TP+DP=2, Text TP+PP=2
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/vlm_parallel --output_name vlm_c4 \
  --model_type vlm --model_name qwen2_5_vl_3b \
  --num_stacks 2 --vision_num_hidden_layers 2 \
  --batch 4 --micro_batch 2 --dp 2 --tp 2 --pp 2
```

### MoE 验证（AC-10）
```bash
# M1: EP=8
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/vlm_moe --output_name vlm_moe_m1 \
  --model_type vlm_moe --model_name qwen2_5_vl_3b \
  --num_stacks 2 --vision_num_hidden_layers 2 \
  --batch 4 --micro_batch 2 --ep 8 --tp 1 --dp 1 --pp 1

# M2: EP=8, TP=2, DP=2
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/vlm_moe --output_name vlm_moe_m2 \
  --model_type vlm_moe --model_name qwen2_5_vl_3b \
  --num_stacks 2 --vision_num_hidden_layers 2 \
  --batch 4 --micro_batch 2 --ep 8 --tp 2 --dp 2 --pp 1
```

### 回归测试（AC-11）
```bash
# Dense 回归
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/regression --output_name dense_test \
  --model_type dense --num_stacks 1 --batch 2 --micro_batch 2 --dp 2 --tp 1 --pp 1

# VLM 统一策略回归（与上次迭代一致）
uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
  --output_dir /tmp/regression --output_name vlm_test \
  --model_type vlm --model_name qwen2_5_vl_3b \
  --batch 4 --micro_batch 2 --dp 2 --tp 1 --pp 1
```

## 关键约束

- **所有修改不得破坏现有测试**：Dense/GPT/VLM 统一策略的端到端测试必须仍然通过
- **Codex 必须使用 `patch` 或 `sed` 修改文件**，不要直接覆盖整个文件（以免丢失未读到的代码）
- **Python 语法必须正确**：修改后文件应能通过 `python3 -m py_compile file.py`
- **MoE Bug 修复**：修改 `replicate_graph.py` 后，运行 `python3 -c "from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph; print('OK')"` 确认无导入错误
