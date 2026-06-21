# Round 1 任务：补完训练代码删除 + 回归验证

## 强制执行规则
1. 你是唯一 Builder，只写代码，不做 review 判断。
2. 不执行任何 git 命令。
3. 完成后必须写入 summary 文件到：/share/guolidong-nfs/SeeSpace/SCSP_v1/flexet/tasks/flexet-phase1-rename-inference/impl/round-1-summary.md

## 重要限制
- 所有文件操作（创建、修改、删除）通过你自身的文件工具完成，不直接输出代码到对话中。
- 每次修改文件后，检查是否有 include 路径、import 路径或其他引用需要同步更新。
- 修改前先在相关文件中搜索所有引用，确认修改的安全性。

## 项目目录
项目根目录：`/share/guolidong-nfs/SeeSpace/SCSP_v1/flexet`

## 上轮发现的问题（必须在本轮解决）

### [ISSUE] AC-2 训练代码删除不彻底：main.py
`main.py` 中仍有以下训练相关代码，必须彻底删除：
1. `--include_backward` CLI 参数定义（argparse 中）
2. `--weight_sharded` CLI 参数定义（argparse 中）
3. `MicroBatchReplicator` 和 `MicroBatchReplicatorPostProcess` 的导入
4. `_apply_training_mode(graph, include_backward)` 函数及其所有调用点
5. 所有 `include_backward=args.include_backward` 参数传递
6. 所有 `args.weight_sharded` 条件分支
7. `main.py` 中所有 `include_backward` 局部变量/参数

### [ISSUE] AC-2 训练代码删除不彻底：模型 builder
- `models/vlm.py` 中仍有 `include_backward` 参数，删除
- `models/vlm_moe.py` 中仍有 `include_backward` 参数，删除
- 确保这些文件删除 backward 分支后，调用方（main.py）不需要再传递 include_backward

### [ISSUE] AC-5 回归验证未完成
1. 对修改后的所有 Python 文件执行语法检查：`python3 -m py_compile <file>`
2. 尝试运行推理命令验证（不实际执行完整生成，但至少验证 import 正确、参数解析正确）：
   ```bash
   cd /share/guolidong-nfs/SeeSpace/SCSP_v1/flexet
   python3 -m flexet.main --help
   ```
   如果 `--help` 能正常输出且不报 ImportError/NameError，说明基本语法和 import 路径正确。
3. 确认 `flexet/` 目录下无 `__pycache__` 残留旧 stage 模块缓存（必要时删除）

## 本轮实施范围
只修改以下文件（不要动其他已完成的部分）：
- `main.py`
- `models/vlm.py`
- `models/vlm_moe.py`

完成后验证语法和 import。

## 完成后必须写入
将 summary 写入：
`/share/guolidong-nfs/SeeSpace/SCSP_v1/flexet/tasks/flexet-phase1-rename-inference/impl/round-1-summary.md`

summary 必须包含以下部分：
- 本轮实现内容
- AC 推进情况（每个 AC 的状态：已完成/部分完成）
- 遗留问题
- Goal Tracker 更新请求
- Lesson Delta
