# VLM Parallel Decoupling — Summary

## AC 验证状态

| AC | 描述 | 状态 | 验证方式 |
|---|---|---|---|
| AC-1 | VLM 拆分为两个独立子图 | ✅ Verified | 代码审查：vlm.py `vlm_subgraphs()` 返回 vision_graph + text_graph |
| AC-2 | `vlm_moe.py` 复用 `vlm.py` 的 `_concat_tokens` | ✅ Verified | 代码审查：`vlm_moe.py` line 8 `from .vlm import _concat_tokens` |
| AC-3 | 新函数签名与调用点一致 | ✅ Verified | 代码审查：main.py 使用 `build_vlm_subgraphs(...)` 获取 vision_graph, text_graph, links |
| AC-4 | ConnectGraph links 字典键值规范 | ✅ Verified | 代码审查：links 使用原始 tensor 名称（无 @0/@1 后缀）|
| AC-5 | 重命名和 ReplicateGraph 在 ConnectGraph 之前 | ✅ Verified | 代码审查：vlm.py 中 vit → vision_encoder, projection, concat 先重命名再拼接 |
| AC-6 | 新增 `vlm_moe` model_type | ✅ Verified | 代码审查：main.py line 281 choices 包含 "vlm_moe"，line 501-520 处理 vlm_moe |
| AC-7 | VLM 回归测试通过 | ✅ Verified | 端到端运行：`--model_type vlm --model_name qwen2_5_vl_3b --dp 2 --tp 1 --pp 1` reading out 100% |
| AC-8 | 4 种并行组合 | ✅ Verified | C1(--dp2--tp2--pp1) C2(--tp2--pp2) C3(--dp2--pp2) C4(--dp2--tp2--pp2) 全部 reading out 100% |
| AC-9 | 拼接点检查 | ✅ Verified | 脚本验证：links = {in_emb.y→vlm_concat.text, vlm_concat.y→transformer.0.input_norm.x, vlm_concat.dtext→in_emb.dy, transformer.0.input_norm.dx→vlm_concat.dy} |
| AC-10 | MoE VLM 端到端 | ✅ Verified | M1(--ep8--tp1--dp1--pp1) M2(--ep8--tp2--dp2--pp1) 全部 reading out 100% |
| AC-11 | Dense 回归测试 | ✅ Verified | `--model_type dense --num_stacks 1 --dp 2` reading out 100% |
| AC-12 | MoE 回归测试 | ⚠️ Pre-existing Failure | HEAD 版本同样报错 `TypeError: Cannot convert expression to float`，非本次修改引入 |
| AC-13 | 文档与错误信息 | ✅ Verified | main.py 包含 vlm_moe 选项和提示信息 |

## 发现的问题与修复

1. **ConnectGraph 拼接顺序问题**：Codex Builder 在 `_build_and_distribute_vlm_model` 中先对 vision_graph 和 text_graph 分别应用 `MicroBatchReplicator`，再 `ConnectGraph.apply()`。但 `MicroBatchReplicator` 会给 tensor ID 添加 `mb0./mb1.` 前缀，导致 links 中的原始 ID 无法解析。**修复**：先 `ConnectGraph.apply()` 拼接，再统一应用 `MicroBatchReplicator`。

2. **replicate_graph.py subs 问题**：Codex Builder 将 `x1_shape` 等属性的逐轮 `replace` 改为 `subs(simultaneous=True)`，导致新插入表达式中的符号（如 `KExperts`、`ep`）不会被后续替换轮次处理，MoE 模型出现 `TypeError: Cannot convert expression to float`。**修复**：回退到原始逐轮 `replace` 逻辑。`Micro(MicroBatch)` bug 只影响字符串类型的 `op_attr`（已由上次迭代的 placeholder 修复），不影响 sympy 表达式类型的 shape 属性。

## 修改文件

- `main.py`: 新增 `_build_and_distribute_vlm_model`、`_create_vlm_pipeline_tensor_map`、`_stage_for_layer`；更新 `model_type` choices 和 VLM/vlm_moe 分支
- `models/vlm.py`: 新增 `vlm_subgraphs()`，导出 `_concat_tokens`
- `models/vlm_moe.py`: 新增（使用 MoE backbone）
