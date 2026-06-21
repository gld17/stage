[ISSUE] AC-2 训练代码删除不彻底：main.py 中仍有 --include_backward、--weight_sharded、MicroBatchReplicator 导入与调用、_apply_training_mode 函数
[ISSUE] AC-2 训练代码删除不彻底：models/vlm.py 和 models/vlm_moe.py 仍有 include_backward 参数
[ISSUE] AC-5 回归验证未完成：尚未执行端到端 forward-only 推理 ET 生成验证

# IMPL Review - Round 0

## 详细审查

### AC-1: 目录与产品名重命名
- **状态：基本完成**（代码层面）
- flexet/ 目录存在且结构完整
- Python 代码中无 `from stage.` / `import stage.` 残留
- `STAGE_*` 环境变量已改为 `FLEXET_*`
- `__STAGE_SYMBOL_REPLACE_*__` 已改为 `__FLEXET_SYMBOL_REPLACE_*__`
- `_STAGE_ROOT` 已改为 `_FLEXET_ROOT`
- 历史任务文档（.codex/、.hermes/、tasks/ 下的历史 plan）中仍有 "Stage" 描述，属于历史归档，不在产品代码清理范围内

### AC-2: 删除训练相关代码路径
- **状态：部分完成**
- ✅ `grad_updater.py` 已删除
- ✅ `STAGE_MICROBATCH_OPTIMIZE` 条件分支已删除
- ❌ `main.py` 中 `--include_backward` CLI 参数仍存在
- ❌ `main.py` 中 `--weight_sharded` CLI 参数仍存在
- ❌ `main.py` 中 `MicroBatchReplicator` 和 `MicroBatchReplicatorPostProcess` 导入仍存在
- ❌ `main.py` 中 `_apply_training_mode(graph, include_backward)` 函数仍存在
- ❌ `main.py` 中多处 `include_backward=args.include_backward` 参数传递仍存在
- ❌ `main.py` 中 `args.weight_sharded` 分支仍存在
- ❌ `models/vlm.py` 中仍有 `include_backward` 参数
- ❌ `models/vlm_moe.py` 中仍有 `include_backward` 参数
- ⚠️ `models/gpt_model.py`、`llama_model.py`、`moe_model.py` 等已删除 backward 分支（从 diff 确认）

### AC-3: 清理 Sharding CSV 训练语义
- **状态：已完成**
- ✅ CSV header 已无 `require_grads`、`grad_of` 列
- ✅ CSV 中无 backward 行（.dy、.dw、loss）
- ✅ `tensor.py` 中 CSV 解析/序列化逻辑已同步更新
- ✅ `vram_counting.py` 中 grads 分量已删除

### AC-4: 更新文档与配置
- **状态：部分完成**
- ✅ `PHYSICAL_TOPOLOGY_REQUIREMENTS.md` 中产品名已更新
- ⚠️ README 中可能仍有训练/FSDP 参数说明（需 AC-2 完成后同步更新）
- ⚠️ `environment.yml` 中环境变量说明待确认

### AC-5: 回归验证
- **状态：未开始**
- ❌ 未执行 `python -m flexet.main ...` 验证
- ❌ 未验证推理 forward-only ET 生成
- ❌ 未对比 baseline 输出一致性

## 下轮建议

Round 1 应聚焦：
1. 彻底删除 `main.py` 中所有训练相关代码（--include_backward、--weight_sharded、MicroBatchReplicator、_apply_training_mode）
2. 清理 `models/vlm.py` 和 `models/vlm_moe.py` 中的 include_backward
3. 执行回归验证
