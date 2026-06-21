
---

# Symbolic Tensor Graph (STG) Generator

## Overview

The Symbolic Tensor Graph is a generator for [Chakra Execution Trace (ET)](https://github.com/mlcommons/chakra) files. This tool is designed to generate synthetic workload traces for use in parallel strategy exploration without gathering data from a real system or implementing actual workload codes. It supports various parallelization strategies like Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP) and Sequence Parallelism (SP).

### Key Features
- Generate synthetic transformer workloads in Chakra ET format.
- Supports multiple parallelism strategies (DP, TP, PP, SP).
- Support customized model dimensions for Transformer Models (batch, seq, dmodel, dff, n_head)

## Installation

Install dependencies via pip (or uv):

```bash
# Using pip
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

For conda users, the same packages are available on conda-forge:
```bash
conda install numpy sympy python-graphviz protobuf pandas tqdm networkx -c conda-forge
```

## Usage

To generate symbolic workloads, use the following command:

```bash
python main.py –h
```

This will show all available options and their descriptions. Example of running the generator:

```bash
python main.py --output_dir generated/ \
               --output_name workload.%d.et \
               --comm_group_file comm_group.json \
               --dp 2 --tp 2 --pp 2
```

### Example Output:

```bash
$ ls generated/
comm_group.json  workload.0.et  workload.1.et  workload.2.et  workload.3.et
```

## Parameters

    | Argument               | Type    | Required | Default    | Description                                                                 |
    |------------------------|---------|----------|------------|-----------------------------------------------------------------------------|
    | --output_dir           | str     | Yes      | -          | Directory to store output traces.                                           |
    | --output_name          | str     | Yes      | -          | Name of the output traces.                                                  |
    | --dp                   | int     | No       | 1          | Data parallel degree.                                                       |
    | --tp                   | int     | No       | 1          | Tensor parallel degree.                                                     |
    | --sp                   | int     | No       | 1          | Sequence parallel degree.                                                   |
    | --ep                   | int     | No       | 1          | Expert parallel degree.                                                     |
    | --pp                   | int     | No       | 1          | Pipeline parallel degree.                                                   |
    | --activation_recompute | bool    | No       | False      | Whether to recompute activations.                                           |
    | --tpsp                 | bool    | No       | True       | Use tensor parallel + sequence parallel or tensor parallel only.            |
    | --dvocal               | int     | No       | 32000      | Vocabulary size.                                                            |
    | --dmodel               | int     | No       | 8192       | Model dimension.                                                            |
    | --dff                  | int     | No       | 28672      | Feed-forward dimension.                                                     |
    | --batch                | int     | No       | 64         | Batch size.                                                                 |
    | --micro_batch          | int     | No       | -1         | Micro-batch size. Default is -1 (same as batch size).                       |
    | --seq                  | int     | No       | 1024       | Sequence length.                                                            |
    | --head                 | int     | No       | 64         | Number of attention heads.                                                  |
    | --kvhead               | int     | No       | 8          | Number of key-value heads.                                                  |
    | --num_stacks           | int     | No       | 80         | Number of transformer layers.                                               |
    | --experts              | int     | No       | 8          | Number of experts in MoE.                                                   |
    | --kexperts             | int     | No       | 2          | Number of selected experts per token.                                       |
    | --chakra_schema_version| str     | No       | "v0.0.4"   | Chakra schema version.                                                      |
    | --model_type           | str     | No       | "dense"    | One of `dense`, `llama`, `gpt`, `moe`, `debug` (see [Supported Model Types](#supported-model-types)). |
    | --mixed_precision      | bool    | No       | False      | Whether to use mixed precision.                                             |
    | --print_gpu_vram       | bool    | No       | False      | Whether to print per-GPU VRAM footprint.                                    |
    
    \*: We do not specify number of total NPUs, which will be infered from the parallel degree as: ```num_NPUs=DP*TP*PP*SP```

## Supported Model Types

| Model Type | `--model_type` | Description | Supported Parallelism |
|-----------|----------------|-------------|----------------------|
| Dense (LLaMA) | `llama` / `dense` | Standard Transformer with TP+SP or TP-only | DP, TP, SP, PP |
| GPT | `gpt` | GPT variant with configurable TP/SP | DP, TP, SP, PP |
| MoE | `moe` | Mixture-of-Experts with expert parallelism | DP, TP, SP, PP, EP |
| Debug | `debug` | Minimal debug configuration | — |

**TP vs TP+SP:** Use `--tpsp true` (default) for Tensor Parallel + Sequence Parallel; `--tpsp false` for TP only. LLaMA (`dense`) uses `--tpsp` automatically; GPT allows explicit toggling.

## Example Commands

- **Generate with DP=8, TP=4, PP=4:**
  ```bash
  python main.py --output_dir generated/ --output_name workload_1.%d.et --comm_group_file comm_group_1.json --dp 8 --tp 4 --pp 4 --sp 1 --chakra_schema_version v0.0.4
  ```

- **Generate with DP=4, TP=4, PP=2, SP=2, output in JSON format:**
  ```bash
  python main.py --output_dir generated/ --output_name workload_2.%d.json --comm_group_file comm_group_2.json --dp 4 --tp 4 --pp 2 --sp 2 --chakra_schema_version json
  ```

## Tool workflow
Here is a breif workflow about how stg generate traces step by step.
![alt text](./docs/images/stg_workflow.png)

## Chakra Schema Version

The schema version used determines compatibility with different tools and repositories:
- **v0.0.4**: Current latest chakra version (by Oct.6 2024).
- **v0.0.1**: Supported for lagacy, not fully tested.

## ET analysis (`chakra_et_tools.py`)

After generating traces with `main.py`, use the bundled helper to inspect Chakra ET (v0.0.4 stream: `GlobalMetadata` + `Node` messages, same encoding as `Chakra004Backend`).

Run from this directory (`flexet/`) so `symbolic_tensor_graph` is importable. The script needs the same dependencies as `main.py` (see `requirements.txt`). If your interpreter does not have them yet:

```bash
# Using pip
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

Use `--jsonizer` and/or `--visualizer` together (omit both to run both). Outputs default to `results/<et_stem>.{json,graphml}` (same directory as `main.py --output_dir results`).

```bash
# JSON + GraphML together
python chakra_et_tools.py --et results/workload.0.et --jsonizer --visualizer

# same as above (default: both)
python chakra_et_tools.py --et results/workload.0.et

# JSON or GraphML only
python chakra_et_tools.py --et results/workload.0.et --jsonizer
python chakra_et_tools.py --et results/workload.0.et --visualizer --max-nodes 200
```

Open `results/workload.0.graphml` in Gephi, yEd, Cytoscape, or any GraphML-compatible viewer. Each node carries `name`, `node_type`, optional `op_type`, and a `label` attribute; each edge has a `dep_type` of `data` or `ctrl`.

Legacy subcommand names still work: `chakra_jsonizer` / `jsonizer` → `--jsonizer`; `chakra_visualizer` / `visualizer` → `--visualizer`. Aliases `--out-dot` / `--output` remain accepted for compatibility but now write GraphML.

## License

MIT

---
