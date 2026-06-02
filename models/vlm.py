import sympy as sp

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.ops.concat import Concat
from symbolic_tensor_graph.tensor import Tensor

from .vision_encoder import vision_encoder
from .vlm_connector import vision_projection


def _new_concat(name, x1, x2):
    tensor = Tensor(create_empty=True)
    tensor.name = name
    tensor.revision = 0
    tensor.require_grads = False
    tensor.x1 = x1
    tensor.x2 = x2
    tensor.op_type = Concat.type_name
    tensor.op_attr = "1"
    tensor.x1_shape = list(x1.y_shape)
    tensor.x1_hidden = list(x1.y_hidden)
    tensor.x2_shape = list(x2.y_shape)
    tensor.x2_hidden = list(x2.y_hidden)
    tensor.grad_of = None
    return tensor


def _remap_text_embedding_to_text_seq(text_graph):
    text_seq = sp.parse_expr("TextSeq")
    total_seq = sp.parse_expr("TotalSeq")
    for tensor in text_graph.tensors:
        if not tensor.name.startswith("in_emb."):
            continue
        for attr in ("x1_shape", "x1_hidden", "x2_shape", "x2_hidden"):
            shape = getattr(tensor, attr)
            if shape is None:
                continue
            for i, dim in enumerate(shape):
                shape[i] = dim.replace(total_seq, text_seq)


def vlm(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    text_backbone_fn,
    regenerate=False,
    tpsp=True,
    include_backward=True,
):
    """Build a full Vision-Language Model."""
    vit_graph = ReplicateGraph.apply(
        vision_encoder(
            vision_num_layers,
            symbol_map_value,
            regenerate=regenerate,
            include_backward=include_backward,
        ),
        "vision_encoder.%s",
    )
    projection_graph = ReplicateGraph.apply(
        vision_projection(include_backward=include_backward),
        "vision_projection.%s",
    )
    text_graph = text_backbone_fn(
        text_num_layers,
        regenerate=regenerate,
        tpsp=tpsp,
        include_backward=include_backward,
    )
    text_graph = ReplicateGraph.apply(
        text_graph, old_symbol_map_new_symbol={"Seq": "TotalSeq"}
    )
    _remap_text_embedding_to_text_seq(text_graph)

    links = {
        f"vision_encoder.vit.{vision_num_layers - 1}.ffn_res.y": "vision_projection.x"
    }
    if include_backward:
        links["vision_projection.dx"] = (
            f"vision_encoder.vit.{vision_num_layers - 1}.ffn_res.dy"
        )
    graph = ConnectGraph.apply([vit_graph, projection_graph, text_graph], links)
    tensor_id_map = graph.get_tensor_id_map_tensor()

    visual_tokens = tensor_id_map["vision_projection.y@0"]
    text_tokens = tensor_id_map["in_emb.y@0"]
    concat = _new_concat("vision_text_concat.y", visual_tokens, text_tokens)
    graph.tensors.append(concat)

    first_text_input = tensor_id_map["transformer.0.input_norm.x@0"]
    first_text_input.x1 = concat
    first_text_input.x1_shape = list(concat.y_shape)
    first_text_input.x1_hidden = list(concat.y_hidden)
    if visual_tokens in graph.out_tensors:
        graph.out_tensors.remove(visual_tokens)

    return graph
