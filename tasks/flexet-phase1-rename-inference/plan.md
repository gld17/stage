# FlexET Phase 1: Rename + Inference-only Cleanup

## Goal Description
将 `stage` 项目重命名为 `FlexET`，删除全部训练相关能力，仅保留大模型推理执行图生成。具体包括：目录/环境变量/代码引用全面替换 `stage` → `flexet`（pipeline stage 术语除外）；删除 backward、grad、FSDP、训练 MicroBatch 等代码路径；清理 Sharding CSV 中的训练语义列与行；更新文档与 schema；确保推理 forward-only 输出与 baseline 对齐。

## Acceptance Criteria

### AC-1: 目录与产品名重命名
- Positive Tests：
  - `stage/` 目录已重命名为 `flexet/`
  - 项目中所有 `STAGE_` 前缀环境变量已改为 `FLEXET_`（`STAGE_OPTIMIZED` → `FLEXET_OPTIMIZED`，`STAGE_LEGACY_ATTR` → `FLEXET_LEGACY_ATTR`，`STAGE_MERGE_COMMS` → `FLEXET_MERGE_COMMS`，`STAGE_MICROBATCH_OPTIMIZE` → `FLEXET_MICROBATCH_OPTIMIZE`）
  - `__STAGE_SYMBOL_REPLACE_*__` 占位符已改为 `__FLEXET_SYMBOL_REPLACE_*__`
  - `_STAGE_ROOT` 等内部变量已改为 `_FLEXET_ROOT`
  - JSON schema title 中的 "Stage" 已改为 "FlexET"
  - CLI 描述中的 "Stage" 已改为 "FlexET"
  - README、文档、任务目录名中的产品引用已改为 FlexET
  - 代码中 `import stage.` 或 `from stage.` 路径已同步更新为 `flexet.`
- Negative Tests：
  - Pipeline Parallel 术语保留，如 `_stage_for_layer()`、`pipeline_tensor_map`、`num_stacks_each_stage`、`stage_offset`、`pipeline_parallel.py` 中 layer/stage 相关命名未被修改
  - 无 `STAGE_` 前缀残留（pipeline stage 语义除外）

### AC-2: 删除训练相关代码路径
- Positive Tests：
  - `main.py` 中 `--include_backward` CLI 参数及所有 `include_backward` 分支已删除
  - `main.py` 中 `--weight_sharded` CLI 参数及所有 `weight_sharded` / FSDP 相关逻辑已删除
  - `symbolic_tensor_graph/graph/grad_updater.py` 文件已删除
  - `main.py` 中 `MicroBatchReplicator` 和 `MicroBatchReplicatorPostProcess` 的导入与调用已删除
  - `models/graph_mode.py` 中训练模式已删除或内联为唯一 forward 路径
  - 模型 builder（`gpt_model.py`、`llama_model.py`、`moe_model.py`、`vlm.py`、`vlm_moe.py`、`vision_encoder.py`、`vlm_connector.py`）中 `include_backward` 参数及 backward 子图、loss 构造已删除
  - `vram_counting.py` 中 grads 分量已删除，仅保留 params + activations
- Negative Tests：
  - 推理 forward 路径未被破坏
  - `--micro_batch` 参数保留（推理 PP 语义复用）

### AC-3: 清理 Sharding CSV 训练语义
- Positive Tests：
  - Sharding CSV 中 `require_grads` 列已删除
  - Sharding CSV 中 `grad_of` 列已删除
  - CSV 中 backward 相关 tensor 行（`.dy`、`.dw`、`loss` 等）已删除
  - 保留的列仅包含：`id`、`op_type`、`op_attr`、`x1_shape`、`x2_shape`、`x1_hidden`、`x2_hidden`、`extra_attr` 及 forward 并行符号（`Batch/dp`、`Seq/cp`、`Dout/tp` 等）
- Negative Tests：
  - CSV 中不含任何 grad/backward/FSDP 语义
  - Forward 算子行未被误删

### AC-4: 更新文档与配置
- Positive Tests：
  - `README.md` 中所有产品名和目录引用已更新为 FlexET / `flexet/`
  - `requirements.txt` 中若含 stage-specific 依赖说明已更新
  - `environment.yml` 中若含 stage-specific 环境变量说明已更新
  - 项目内 `.codex/` 中的 builder prompt 等文档中产品名已更新（如需要）
- Negative Tests：
  - 文档中 pipeline stage 术语未被错误替换

### AC-5: 回归测试——推理 forward-only 行为对齐
- Positive Tests：
  - 运行推理命令（如 `python -m flexet.main ...` 或等效命令）可成功生成 Chakra ET
  - 生成的 `workload.*.et` 中不含 backward 算子（无 `.dy`、`.dw`、`loss` 等）
  - 在无物理拓扑（identity placement / 全连接假设）下，生成的推理 ET 结构与 baseline（重命名前 `stage` 的 forward-only 输出）一致
  - 环境变量 `FLEXET_OPTIMIZED=1` 时推理 ET 生成正常
- Negative Tests：
  - 不引入语法错误、ImportError、NameError
  - 不破坏现有推理模型支持（Dense / LLaMA / GPT / MoE / VLM / VLM-MoE 的 forward 路径）

## Implementation Notes
- 代码中禁止出现 AC-、Milestone、Step、Phase 等 plan 标记
- 重命名目录时，若使用 copy+delete 方式，需确保所有 import 路径同步更新
- 删除 backward 分支时，注意清理条件判断后的死代码和未使用变量
- 训练相关环境变量（如 `STAGE_MICROBATCH_OPTIMIZE`）若仅服务于训练，则直接删除，不保留 `FLEXET_*` 替代
- 保留的推理环境变量：`FLEXET_OPTIMIZED`、`FLEXET_MERGE_COMMS`、`FLEXET_LEGACY_ATTR`

## Path Boundaries
- 可接受的实现范围：`stage/` 目录重命名、`main.py`、模型 builder、symbolic_tensor_graph 子模块、Sharding CSV、文档、配置文件的修改与清理
- 不可接受的方向：引入物理拓扑输入（Phase 2）、实现 Placement + Validation（Phase 3）、修改 collective 路由逻辑
- 不得修改的文件：外层 `SCSP_v1/` 中 `astra-sim/`、`scsp/` 等非 stage 子目录的代码（除非 import 路径需要更新）
