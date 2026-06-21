from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.graph.convert_chakra import ConvertChakra

# ------------------------------------------------------------------
# Helper utilities for VRAM accounting
# ------------------------------------------------------------------
def _tensor_mem_class(tensor):
    """Classify tensor for persistent VRAM footprint.

    weight : parameter tensors stored persistently
    act    : forward activations
    None   : skip (temporary tensors not expected to occupy persistent VRAM)
    """

    name = tensor.name or ""

    if tensor.is_parameter:
        return "weight"

    return "act"


def _weight_size(tensor, symbol_map, mixed_precision=False):
    """Return bytes for a parameter tensor."""
    from symbolic_tensor_graph.tensor import Tensor as _Tensor

    elem_cnt = _Tensor.eval_expr(_Tensor.eval_size(tensor.y_shape), symbol_map)
    if mixed_precision:
        return int(elem_cnt * 1.5) * 4
    return elem_cnt * 4


def _tensor_size_bytes(tensor, symbol_map, mixed_precision=False):
    """Total size as computed by ConvertChakra (weight+opt if param)."""
    info = ConvertChakra._create_IOInfo(
        tensor, symbol_map, mixed_precision, fsdp_enabled=symbol_map.get("fsdp", 0) > 1
    )
    return info["size"]


def _print_gpu_vram(bundle_graph, symbol_map, mixed_precision=False, header=""):
    GiB = 1024**3
    for rank_key, tg in bundle_graph.graphs.items():
        stats = {"weight": 0, "act": 0}
        tensor_details = []  # Store details for sorting
        for tensor in tg.tensors:
            cls = _tensor_mem_class(tensor)
            if cls is None:
                continue
            if cls == "weight":
                w_b = _weight_size(tensor, symbol_map, mixed_precision)
                stats["weight"] += w_b
                tensor_details.append(
                    (cls, w_b, tensor.id, Tensor.stringfy_shape(tensor.y_shape))
                )
            else:
                size_b = _tensor_size_bytes(tensor, symbol_map, mixed_precision)
                stats[cls] += size_b
                tensor_details.append(
                    (cls, size_b, tensor.id, Tensor.stringfy_shape(tensor.y_shape))
                )
        total = sum(stats.values())
        rk_str = ",".join([f"{d[0]}={d[1]}" for d in rank_key])
        print(
            f"{header}[GPU {rk_str}] total={total / GiB:.3f} GiB | "
            f"weights={stats['weight'] / GiB:.3f} | "
            f"acts={stats['act'] / GiB:.3f}"
        )
        # Print top 5 largest tensors by size for this rank
        # tensor_details.sort(key=lambda x: x[1], reverse=True)
        # print(f"    Top 5 Tensors for GPU {rk_str}:")
        # for i in range(min(5, len(tensor_details))):
        #     cls, size_b, t_id, shape_str = tensor_details[i]
        #     print(f"      {i+1}. Type={cls}, Size={size_b / GiB:.4f} GiB, ID={t_id}, Shape={shape_str}")
