# Round 0 Summary

## 本轮实现内容

- 检查 sibling `flexet/` 目录的 Python 命名残留，确认并补齐清理：
  - `from stage.` / `import stage.`：未发现残留。
  - `__STAGE_SYMBOL_REPLACE_*__`：未发现残留，现使用 `__FLEXET_SYMBOL_REPLACE_*__`。
  - `_STAGE_ROOT`：未发现残留，现使用 `_FLEXET_ROOT`。
  - `STAGE_*` Python 代码残留：清理了 `main.py` 中的 `STAGE_MICROBATCH_OPTIMIZE` 分支，以及 `convert_chakra.py` 中注释残留的 `STAGE_CTRL_DEPS`。
- 更新用户可见文档残留：
  - `PHYSICAL_TOPOLOGY_REQUIREMENTS.md` 中的产品名 `Stage` 改为 `FlexET`。
  - `README.md` 中运行目录已指向 `flexet/`。
- 清理旧 `stage/` 目录：
  - 已删除旧 `stage/` 下除 `.git/` 外的原有文件和子目录。
  - 为满足强制 summary 产物要求，清理后仅重建了本 summary 所在路径：`tasks/flexet-phase1-rename-inference/impl/round-0-summary.md`。

## AC 推进情况

- AC-1: 目录与产品名重命名（stage → flexet，STAGE_* → FLEXET_*）：部分完成。
  - `flexet/` 目录存在，Python 代码中未发现 `stage.` import、旧 `STAGE_*` 环境变量、旧占位符、`_STAGE_ROOT` 残留。
  - 历史任务/计划文档中仍存在用于描述历史基线或任务要求的 `Stage` / `stage` 文本。
- AC-2: 删除训练相关代码路径（--include_backward、--weight_sharded、grad_updater.py、MicroBatchReplicator、backward 分支）：部分完成。
  - 本轮只删除了 `STAGE_MICROBATCH_OPTIMIZE` 条件分支。
  - `main.py`、部分模型 builder、历史文档中仍可检索到 `include_backward`、`weight_sharded`、`MicroBatchReplicator`、backward/grad 语义。
- AC-3: 清理 Sharding CSV 训练语义（删除 require_grads、grad_of 列，删除 backward 行）：部分完成。
  - 本轮未重新批量改写 CSV。
  - 代码/文档中仍有 `require_grads`、`grad_of`、backward 相关语义残留，需要后续专项清理。
- AC-4: 更新文档与配置（README、requirements、environment.yml）：部分完成。
  - 用户可见拓扑接口文档已补齐 FlexET 产品名。
  - README 仍含训练/FSDP 参数说明，需要跟 AC-2 完成后同步收敛。
  - `requirements.txt` / `environment.yml` 本轮未发现必须配合改动的 Stage 产品名残留。
- AC-5: 回归验证（语法检查、import 路径、推理 forward-only）：部分完成。
  - 已做文本级 import/命名检查。
  - 尚未完成语法检查和 forward-only 生成回归。

## 遗留问题

- 指定 summary 路径位于旧 `stage/` 目录内，与“删除 `stage/` 下除 `.git/` 外所有内容”存在产物位置冲突；本轮采用清理后仅重建 summary 路径的方式满足强制 summary 要求。
- 训练路径删除尚未彻底完成：`include_backward`、`weight_sharded`、`MicroBatchReplicator`、backward/grad 相关代码与文档仍需继续清理。
- 历史任务目录、计划文档、`.codex` / `.hermes` 归档文件仍包含 Stage 历史描述；需确认这些历史材料是否纳入产品名清理范围。
- 尚未执行端到端 forward-only 回归验证。

## Goal Tracker 更新请求

- 将 AC-1 标记为“部分完成”：代码命名残留已清理，历史文档残留待确认范围。
- 将 AC-2 标记为“部分完成”：`STAGE_MICROBATCH_OPTIMIZE` 分支已删除，但训练 CLI/模型 builder/backward 路径仍未完全移除。
- 将 AC-3 标记为“部分完成”：需要继续清理 CSV 与 grad/backward 语义。
- 将 AC-4 标记为“部分完成”：用户可见文档有补齐，README 训练参数说明仍待 AC-2 后同步。
- 将 AC-5 标记为“部分完成”：完成文本检查，缺少语法与推理生成回归。

## Lesson Delta

- 旧根目录 `stage/` 与新项目目录 `flexet/` 是 sibling 关系；在旧根目录内直接查找 `flexet/` 会得到误导性的 “No such file or directory”。
- 清理旧根目录前需要先处理 summary 产物位置，否则强制 summary 路径会被清理要求覆盖。
- `STAGE_MICROBATCH_OPTIMIZE` 不应改名为 `FLEXET_MICROBATCH_OPTIMIZE`；本轮按推理-only 方向删除条件开关，保留现有默认图处理路径作为临时兼容。
