# Round 0 任务：FlexET Phase 1 — Rename + Inference-only Cleanup

## 强制执行规则
你必须遵守以下规则：
1. 你是唯一 Builder，只写代码，不做 review 判断。
2. 不做 review 判断。
3. 不执行任何 git 命令。
4. 完成后必须写入 summary 文件到：/share/guolidong-nfs/SeeSpace/SCSP_v1/flexet/tasks/flexet-phase1-rename-inference/impl/round-0-summary.md

## 重要限制
- **不要执行 git 命令**：如果你需要重命名文件，请直接创建新文件并写入内容，然后删除旧文件。不要尝试 `git mv`。Kimi 会在后续 commit 阶段处理 git 历史追溯性。
- 所有文件操作（创建、修改、删除）通过你自身的文件工具完成，不直接输出代码到对话中。
- 每次修改文件后，检查是否有 include 路径、import 路径或其他引用需要同步更新。
- 修改前先在相关文件中搜索所有引用，确认修改的安全性。

## 项目目录
项目根目录：`/share/guolidong-nfs/SeeSpace/SCSP_v1/stage`
当前目录名：`stage`
目标目录名：`flexet`

## 本轮需完成的工作（5个AC全部在本次完成）

### AC-1: 目录与产品名重命名
1. 将 `flexet/` 目录（本项目根目录）重命名为 `flexet/`。由于你不能执行 git 命令，请使用 copy+delete 方式：创建 `flexet/` 目录，将所有 `flexet/` 下的文件（除了 `.git/` 目录本身）复制到 `flexet/`，然后删除 `flexet/` 目录下的文件。
2. **重要**：`.git/` 目录不能动，保留在原位。
3. 将所有 `STAGE_` 前缀环境变量改为 `FLEXET_`：
   - `FLEXET_OPTIMIZED` → `FLEXET_OPTIMIZED`
   - `FLEXET_LEGACY_ATTR` → `FLEXET_LEGACY_ATTR`
   - `FLEXET_MERGE_COMMS` → `FLEXET_MERGE_COMMS`
   - `STAGE_MICROBATCH_OPTIMIZE` → 直接删除（仅训练使用）
4. 将 `__FLEXET_SYMBOL_REPLACE_*__` 占位符改为 `__FLEXET_SYMBOL_REPLACE_*__`
5. 将 `_FLEXET_ROOT` 等内部变量改为 `_FLEXET_ROOT`
6. JSON schema title 中的 "Stage" 改为 "FlexET"
7. CLI 描述中的 "Stage" 改为 "FlexET"
8. README、文档中的产品引用改为 FlexET
9. 代码中 `import flexet.` 或 `from flexet.` 路径更新为 `from flexet.` 或 `import flexet.`（因为目录已重命名）

**保留不变**：Pipeline Parallel 术语，如 `_stage_for_layer()`、`pipeline_tensor_map`、`num_stacks_each_stage`、`stage_offset`、`pipeline_parallel.py` 中 layer/stage 相关命名。

### AC-2: 删除训练相关代码路径
1. `main.py`：删除 `--include_backward` CLI 参数及所有 `include_backward` 分支
2. `main.py`：删除 `--weight_sharded` CLI 参数及所有 `weight_sharded` / FSDP 相关逻辑
3. 删除文件 `symbolic_tensor_graph/graph/grad_updater.py`
4. `main.py`：删除 `MicroBatchReplicator` 和 `MicroBatchReplicatorPostProcess` 的导入与调用
5. `models/graph_mode.py`：删除训练模式，内联为唯一 forward 路径
6. 模型 builder 文件：删除 `include_backward` 参数及 backward 子图、loss 构造：
   - `models/gpt_model.py`
   - `models/llama_model.py`
   - `models/moe_model.py`
   - `models/vlm.py`
   - `models/vlm_moe.py`
   - `models/vision_encoder.py`
   - `models/vlm_connector.py`
7. `symbolic_tensor_graph/vram_counting.py`：删除 grads 分量，仅保留 params + activations

**保留**：`--micro_batch` 参数（推理 PP 语义复用）

### AC-3: 清理 Sharding CSV 训练语义
1. Sharding CSV 中删除 `require_grads` 列
2. Sharding CSV 中删除 `grad_of` 列
3. CSV 中删除 backward 相关 tensor 行（`.dy`、`.dw`、`loss` 等）
4. 保留的列：id, op_type, op_attr, x1_shape, x2_shape, x1_hidden, x2_hidden, extra_attr 及 forward 并行符号

注意：需要同时修改生成和读取 CSV 的代码，确保删除列后不会导致列索引错位。

### AC-4: 更新文档与配置
1. `README.md`：更新产品名和目录引用为 FlexET / `flexet/`
2. `requirements.txt`：更新 stage-specific 说明
3. `environment.yml`：更新 stage-specific 环境变量说明
4. 项目内 `.codex/` 中的 builder prompt 等文档：更新产品名

### AC-5: 回归验证
1. 确保 `python -m flexet.main ...` 或等效命令可以成功运行（不实际执行，但确保语法正确、import 路径无误）
2. 检查无语法错误、ImportError、NameError
3. 确认推理 forward 路径未被破坏

## 实施顺序建议
1. 先 copy `flexet/` → `flexet/`
2. 在 `flexet/` 内做所有修改（重命名、删除、清理）
3. 删除 `flexet/` 目录（保留 `.git/`）
4. 验证 import 路径和语法

## 关键文件列表（必须在修改范围内）
- `main.py`
- `chakra_et_tools.py`
- `models/*.py` (gpt_model.py, llama_model.py, moe_model.py, vlm.py, vlm_moe.py, vision_encoder.py, vlm_connector.py, graph_mode.py, utils.py, __init__.py)
- `symbolic_tensor_graph/graph/*.py` (convert_chakra.py, grad_updater.py, graph.py, graph_distributer.py, replicate_graph.py)
- `symbolic_tensor_graph/chakra/backends/chakra_00_4_backend/chakra_00_4_backend.py`
- `symbolic_tensor_graph/vram_counting.py`
- `symbolic_tensor_graph/tensor.py`
- `sharding_spreadsheets/` 下的所有 CSV
- `README.md`
- `requirements.txt`
- `environment.yml`

## 完成后必须写入
将 summary 写入：
`/share/guolidong-nfs/SeeSpace/SCSP_v1/flexet/tasks/flexet-phase1-rename-inference/impl/round-0-summary.md`

summary 必须包含以下部分：
- 本轮实现内容
- AC 推进情况（每个 AC 的状态）
- 遗留问题
- Goal Tracker 更新请求
- Lesson Delta
