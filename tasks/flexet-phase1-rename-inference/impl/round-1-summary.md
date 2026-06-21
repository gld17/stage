# Round 1 Summary

## 本轮实现内容

- **删除 `main.py` 中所有训练相关代码**：
  - 删除 `grad_updater.py` 的 import（GradUpdater、MicroBatchReplicator、MicroBatchReplicatorPostProcess）
  - 删除 `_apply_training_mode()` 函数
  - 删除 `--include_backward` CLI 参数
  - 删除 `--weight_sharded` CLI 参数
  - 删除所有 `args.weight_sharded` 条件分支（固定 fsdp=1）
  - 删除所有 `MicroBatchReplicator.apply()` 调用
  - 删除所有 `_apply_training_mode()` 调用
  - 删除所有 `include_backward=args.include_backward` 参数传递
  - 更新帮助文案为 "FlexET STG generator"

- **删除 `models/vlm.py` 中所有训练相关代码**：
  - 简化 `_tensor()` helper：删除 `require_grads` 和 `grad_of` 参数
  - `_concat_tokens()`：删除 `include_backward` 参数及 backward 分支
  - `vlm()`：删除 `include_backward` 参数
  - `vlm_subgraphs()`：删除 `include_backward` 参数
  - 删除 vision_encoder、projection、decoder 调用中的 `include_backward` 传递
  - 删除 backward text_links（`out_emb.dx`）
  - 删除 loss 构建及 backward links

- **删除 `models/vlm_moe.py` 中所有训练相关代码**：
  - `vlm_moe_subgraphs()`：删除 `include_backward` 参数
  - 删除所有 `include_backward` 传递
  - 删除 backward text_links
  - 删除 loss 构建及 backward links

- **回归验证**：
  - `python3 -m py_compile main.py`：通过
  - `python3 -m py_compile models/vlm.py`：通过
  - `python3 -m py_compile models/vlm_moe.py`：通过
  - `python3 main.py --help`：正常输出，无 `--include_backward`/`--weight_sharded`
  - `from models.vlm import vlm, vlm_subgraphs`：import 成功
  - `from models.vlm_moe import vlm_moe_subgraphs`：import 成功

## AC 推进情况

- AC-1: 目录与产品名重命名（stage → flexet，STAGE_* → FLEXET_*）：**已完成**
  - flexet/ 目录结构完整
  - 代码中无 stage. import、STAGE_* 环境变量残留
- AC-2: 删除训练相关代码路径：**已完成**
  - `--include_backward` 已删除
  - `--weight_sharded` 已删除
  - `grad_updater.py` 已删除
  - `MicroBatchReplicator` / `MicroBatchReplicatorPostProcess` 已删除
  - `_apply_training_mode()` 已删除
  - 模型 builder（vlm.py、vlm_moe.py）backward 分支已删除
- AC-3: 清理 Sharding CSV 训练语义：**已完成**
  - CSV 中无 `require_grads`、`grad_of` 列
  - 无 backward 行
  - tensor.py 解析逻辑已同步更新
- AC-4: 更新文档与配置：**部分完成**
  - 用户可见文档已更新（README、PHYSICAL_TOPOLOGY_REQUIREMENTS.md）
  - 历史任务文档（.codex/、.hermes/、tasks/ 下的历史 plan）保留为归档
- AC-5: 回归验证：**已完成**
  - 语法检查通过
  - import 验证通过
  - `--help` 正常输出

## 遗留问题

- 历史任务文档（tasks/ 下早期迭代 plan、.codex/、.hermes/）仍包含 "Stage" 描述，属于历史归档，不在产品代码清理范围
- 未执行端到端 ET 生成验证（需要完整运行 main.py 生成 ET 文件，耗时较长）

## Goal Tracker 更新请求

- AC-1 → **已完成**
- AC-2 → **已完成**
- AC-3 → **已完成**
- AC-4 → **部分完成**（历史文档保留）
- AC-5 → **已完成**（语法+import+--help 验证）

## Lesson Delta

- vlm.py 和 vlm_moe.py 中的 `_concat_tokens()`、loss 构建、backward links 与 `include_backward` 深度耦合，删除时需要一并清理，否则会产生悬空引用
- `main.py` 中 `if args.weight_sharded:` 分支删除后需确保 else 分支的缩进正确，避免 IndentationError
- `--help` 验证是检测 CLI 参数删除是否干净的快速手段
