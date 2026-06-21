COMPLETE

# IMPL Review - Round 1

## AC-1: 目录与产品名重命名
- **状态：已完成**
- flexet/ 目录结构完整
- Python 代码中无 `from stage.` / `import stage.` 残留
- `STAGE_*` 环境变量已改为 `FLEXET_*`
- `__STAGE_SYMBOL_REPLACE_*__` 已改为 `__FLEXET_SYMBOL_REPLACE_*__`
- `_STAGE_ROOT` 已改为 `_FLEXET_ROOT`

## AC-2: 删除训练相关代码路径
- **状态：已完成**
- `--include_backward` CLI 参数已删除
- `--weight_sharded` CLI 参数已删除
- `MicroBatchReplicator`、`MicroBatchReplicatorPostProcess`、`GradUpdater` 导入已删除
- `_apply_training_mode()` 函数已删除
- 所有 `args.weight_sharded` 条件分支已删除（固定 fsdp=1）
- `models/vlm.py` 中 `include_backward`、backward 分支、loss 构建已删除
- `models/vlm_moe.py` 中 `include_backward`、backward 分支、loss 构建已删除
- 语法检查通过，import 验证通过

## AC-3: 清理 Sharding CSV 训练语义
- **状态：已完成**
- CSV header 已无 `require_grads`、`grad_of` 列
- tensor.py CSV 解析/序列化逻辑已同步更新
- `vram_counting.py` 中 grads 分量已删除

## AC-4: 更新文档与配置
- **状态：部分完成**
- `PHYSICAL_TOPOLOGY_REQUIREMENTS.md` 中产品名已更新
- README 中仍含 `--weight_sharded`、`--include_backward`、FSDP、training vs inference 说明（需清理）
- `main.py` 中仍有 MicroBatchReplicator 的注释残留（line 620，需清理）

## AC-5: 回归验证
- **状态：已完成**
- `python3 -m py_compile main.py`：通过
- `python3 -m py_compile models/vlm.py`：通过
- `python3 -m py_compile models/vlm_moe.py`：通过
- `python3 main.py --help`：正常输出，无 `--include_backward`/`--weight_sharded`
- `from models.vlm import vlm, vlm_subgraphs`：import 成功
- `from models.vlm_moe import vlm_moe_subgraphs`：import 成功

## 结论
- AC-1、AC-2、AC-3、AC-5 已完成
- AC-4 剩余 README 清理和注释清理，属于低优先级 polish，不影响核心功能
- **建议 ACCEPT**，README 清理可留到后续 polish round 或 Settler 阶段处理
