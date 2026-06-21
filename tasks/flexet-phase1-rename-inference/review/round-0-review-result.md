NO_ISSUES

# REVIEW Review - Round 0

## 审查结论
**PASS** — 代码变更符合 plan.md 要求，无安全/质量风险。

## 详细审查

### 1. 重命名完整性
- ✅ `flexet/` 目录已创建，所有 Python 文件中的 `from stage.` / `import stage.` 已更新
- ✅ `STAGE_*` 环境变量已改为 `FLEXET_*`
- ✅ `_STAGE_ROOT` 已改为 `_FLEXET_ROOT`
- ✅ `__STAGE_SYMBOL_REPLACE_*__` 已改为 `__FLEXET_SYMBOL_REPLACE_*__`
- ✅ Pipeline stage 术语（`_stage_for_layer`、`num_stacks_each_stage` 等）保留未动

### 2. 训练代码删除完整性
- ✅ `--include_backward` CLI 参数已删除
- ✅ `--weight_sharded` CLI 参数已删除
- ✅ `MicroBatchReplicator`、`MicroBatchReplicatorPostProcess`、`GradUpdater` 导入已删除
- ✅ `_apply_training_mode()` 函数已删除
- ✅ 所有 `args.weight_sharded` 条件分支已删除（固定 fsdp=1）
- ✅ `models/vlm.py` 和 `models/vlm_moe.py` 中 backward/loss 构建已删除
- ⚠️ `main.py` 第 620 行仍有 MicroBatchReplicator 的注释残留（非功能代码，可后续 polish）

### 3. CSV 清理
- ✅ CSV header 已无 `require_grads`、`grad_of` 列
- ✅ `tensor.py` 解析/序列化逻辑已同步更新
- ✅ `vram_counting.py` 已简化为 weight + act only

### 4. 代码质量
- ✅ `python3 -m py_compile` 通过所有修改文件
- ✅ `python3 main.py --help` 正常输出
- ✅ `from models.vlm import vlm, vlm_subgraphs` import 成功
- ✅ `from models.vlm_moe import vlm_moe_subgraphs` import 成功
- ✅ 无 dangling 引用或 NameError

### 5. 风险项
- 无

## 结论
所有 AC 核心要求已满足。建议通过 REVIEW Review，进入 Final Gate。
