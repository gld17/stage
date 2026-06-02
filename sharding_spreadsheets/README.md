# Sharding Spreadsheets

This directory contains per-module sharding strategy definitions used by the model builders to construct symbolic tensor graphs.

## Directory Layout

| Directory | Description |
|-----------|-------------|
| `base/` | Common sharding definitions shared across multiple strategies. Loaded automatically as a fallback when a strategy-specific file is missing. |
| `tp/` | Tensor Parallel (TP) only — Dense LLaMA |
| `tpsp/` | Tensor Parallel + Sequence Parallel (TP+SP) — Dense LLaMA |
| `tp_gpt/` | TP only — GPT variant |
| `tpsp_gpt/` | TP+SP — GPT variant |
| `tp_gpt_moe/` | TP — GPT MoE |
| `tpsp_moe/` | TP+SP — MoE |
| `tpsp_fsdp/` | TP+SP with FSDP weight-sharding annotations |

## Files in `base/`

These files are identical across all (or most) strategy directories and are therefore centralized:

- `expert_wrapper.json` — Shared expert wrapper metadata (7 dirs)
- `mamba_mixer.csv` — Mamba mixer sharding (6 dirs)
- `vocab_parallel_cross_entropy.csv` — Vocab parallel cross-entropy sharding (7 dirs)

## Fallback Mechanism

The `TensorGraph.load_tensor_graph()` method automatically falls back to `base/` when a requested file is not found in the strategy-specific directory. This eliminates duplication without changing any hard-coded paths in the model files.
