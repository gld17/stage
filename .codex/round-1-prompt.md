# Round-1 Builder Prompt: Fix `_update_symbols` Substring Replacement Order

## Background
This is the Round-1 fix loop of the PBR build for VLM support in the STAGE project. Round-0 completed the core VLM model files (vision_encoder.py, vlm_connector.py, vlm.py, model_configs, main.py integration). Three framework-level bugs were already fixed in IMPL Review. The remaining blocker prevents VLM from completing end-to-end Execution Trace (ET) generation.

## Current Blocker
Running `main.py --model_type vlm --model_name qwen2_5_vl_3b` fails during Chakra conversion with:
```
TypeError: Cannot convert expression to float
expr = MicroBatch*NumPatches*VisionHidden*KV(VisionHead)/(VisionHead*cp*dp*tp)
```

## Root Cause (Confirmed)
The malformed expression `KV(VisionHead)` originates in `ReplicateGraph._update_symbols()` (`symbolic_tensor_graph/graph/replicate_graph.py`).

**Trigger path:**
1. `vision_encoder.py::_vit_block()` calls:
   ```python
   ReplicateGraph.apply(
       block,
       old_symbol_map_new_symbol={
           "Seq": "NumPatches",
           "Dmodel": "VisionHidden",
           "Dff": "VisionIntermediate",
           "Head": "VisionHead",
           "KVHead": "VisionHead",
       },
   )
   ```
2. Inside `_update_symbols()`, the dict is iterated in insertion order. `"Head"` (4 chars) is processed **before** `"KVHead"` (6 chars).
3. The replacement is a **naive substring replace** on `op_attr`:
   ```python
   tensor.op_attr = tensor.op_attr.replace(f"{str(from_)}", f"({str(to_)})")
   ```
4. When `from_="Head"`, `to_="VisionHead"`, on an `op_attr` containing `KVHead/tp`:
   - Before: `KVHead/tp`
   - After:  `KV(VisionHead)/tp`
5. Then when `from_="KVHead"` is processed, the original substring no longer exists.
6. `KV(VisionHead)` is later parsed by sympy as a function call `KV(VisionHead)`, causing the TypeError.

## Required Fix
Modify `ReplicateGraph._update_symbols()` in `symbolic_tensor_graph/graph/replicate_graph.py` to process replacements in **descending order of key string length** (longest first), preventing short keys from corrupting longer keys that contain them as substrings.

**Specific change:**
In `_update_symbols()`, before iterating `old_symbol_map_new_symbol.items()`, sort the items by the string length of `from_` (longest first) and process in that order.

For example:
```python
items = list(old_symbol_map_new_symbol.items())
items.sort(key=lambda item: len(str(item[0])), reverse=True)
for from_, to_ in items:
    ...
```

Also apply the same ordering logic to the shape/hidden replacements (x1_shape, x1_hidden, x2_shape, x2_hidden) for consistency, though the immediate bug is in `op_attr`.

## Constraints
- **Do NOT** change the existing replacement logic itself (`.replace()` call).
- **Do NOT** change `vision_encoder.py` or any caller — fix the framework method.
- The fix must be backward-compatible with all existing callers (GPT, LLaMA, MoE).
- If `from_` is already a sympy expression (not a string), use `str(from_)` for length comparison.

## Acceptance Criteria
- [ ] `symbolic_tensor_graph/graph/replicate_graph.py` is modified as described.
- [ ] Running the following command exits with code 0 (VLM end-to-end ET generation):
  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7890
  uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
    --output_dir /tmp --output_name vlm_test --model_type vlm --model_name qwen2_5_vl_3b \
    --batch 4 --micro_batch 2 --dp 2 --tp 1 --pp 1
  ```
- [ ] The generated `/tmp/vlm_test.0.et` exists and is non-empty.
- [ ] Running GPT baseline still works (backward compatibility):
  ```bash
  uv run --with numpy --with sympy --with pandas --with protobuf --with graphviz --with tqdm python3 main.py \
    --output_dir /tmp --output_name gpt_test --model_type gpt \
    --batch 4 --micro_batch 2 --dp 2 --tp 1 --pp 1
  ```
- [ ] After completing, write a summary of changes to `.codex/summary-2026-06-04.md` under a new "Round-1 Builder Output" section.

## Working Directory
`/share/guolidong-nfs/SeeSpace/SCSP_v1/stage`

## Notes
- Use `python3` (not `python`).
- The proxy `127.0.0.1:7890` is available on the server for Codex CLI if needed.
- Existing fixes in `grad_updater.py` and `reshape.py` must be preserved.
