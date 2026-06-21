from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.ops import Identical, PlaceHolder
from symbolic_tensor_graph.tensor import Tensor

from .vision_encoder import vision_encoder
from .vlm_connector import vision_projection
from .llama_model import transformer_decoder_block, transformer_decoders


def _tensor(
    name,
    op_type,
    op_attr=None,
    x1=None,
    x2=None,
    x1_shape=None,
    x1_hidden="1",
    x2_shape=None,
    x2_hidden=None,
):
    tensor = Tensor(create_empty=True)
    tensor.name = name
    tensor.revision = 0
    tensor.op_type = op_type
    tensor.op_attr = op_attr
    tensor.x1 = x1
    tensor.x2 = x2
    tensor.x1_shape = Tensor.parse_shape(x1_shape)
    tensor.x1_hidden = Tensor.parse_shape(x1_hidden)
    tensor.x2_shape = Tensor.parse_shape(x2_shape)
    tensor.x2_hidden = Tensor.parse_shape(x2_hidden)
    return tensor


def _concat_tokens():
    visual = _tensor(
        "visual",
        PlaceHolder.type_name,
        x1_shape="Batch/dp, NumPatches, Dmodel/1",
    )
    text = _tensor(
        "text",
        PlaceHolder.type_name,
        x1_shape="Batch/dp, Seq, Dmodel/1",
    )
    y = _tensor(
        "y",
        "C",
        "1",
        x1=visual,
        x2=text,
        x1_shape="Batch/dp, NumPatches, Dmodel/1",
        x1_hidden="1",
        x2_shape="Batch/dp, Seq, Dmodel/1",
        x2_hidden="1",
    )
    return TensorGraph([visual, text, y], [visual, text], [y])


def _set_identical_input(graph, to_tensor_id, from_tensor_id):
    tensor_map = graph.get_tensor_id_map_tensor()
    to_tensor = tensor_map[to_tensor_id]
    from_tensor = tensor_map[from_tensor_id]
    to_tensor.op_type = Identical.type_name
    to_tensor.x1 = from_tensor
    if to_tensor in graph.in_tensors:
        graph.in_tensors.remove(to_tensor)
    if from_tensor in graph.out_tensors:
        graph.out_tensors.remove(from_tensor)


def vlm(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    text_backbone_fn=None,
    regenerate=False,
    tpsp=True,
):
    """Build a full Vision-Language Model.

    Pipeline:
      image patches -> ViT -> Projection -> Concat(visual_tokens, text_tokens) -> Text Backbone
    """
    vision_graph, text_graph, links = vlm_subgraphs(
        text_num_layers=text_num_layers,
        vision_num_layers=vision_num_layers,
        symbol_map_value=symbol_map_value,
        text_backbone_fn=text_backbone_fn,
        regenerate=regenerate,
        tpsp=tpsp,
    )
    return ConnectGraph.apply([vision_graph, text_graph], links)


def vlm_subgraphs(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    text_backbone_fn=None,
    regenerate=False,
    tpsp=True,
):
    """Build VLM as vision and text subgraphs for independent strategies.

    Returns:
        (vision_graph, text_graph, links)
    """
    # Build ViT and projection
    vit = ReplicateGraph.apply(
        vision_encoder(
            vision_num_layers,
            symbol_map_value,
            regenerate=regenerate,
        ),
        "vision_encoder.%s",
    )
    projection = ReplicateGraph.apply(
        vision_projection(), "vision_projection.%s"
    )

    # Build text decoders with TotalSeq (visual+text concatenated length)
    decoder_template = transformer_decoder_block()
    text_decoders = transformer_decoders(
        text_num_layers, decoder_template
    )
    text_decoders = ReplicateGraph.apply(
        text_decoders,
        inplace=True,
        old_symbol_map_new_symbol={"Seq": "TotalSeq"},
    )

    # Build in_emb (Seq, text tokens only)
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph("./sharding_spreadsheets/module/tpsp/embedding.csv"),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel"},
    )

    # Build out_emb (TotalSeq, concatenated visual+text)
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph("./sharding_spreadsheets/module/tpsp/embedding.csv"),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal", "Seq": "TotalSeq"},
    )

    # Build concat
    concat = ReplicateGraph.apply(
        _concat_tokens(), "vlm_concat.%s"
    )

    vision_links = {
        f"vision_encoder.vit.{vision_num_layers - 1}.ffn_res.y": "vision_projection.x",
        "vision_projection.y": "vlm_concat.visual",
    }
    vision_graph = ConnectGraph.apply([vit, projection, concat], vision_links)

    text_links = {f"transformer.{text_num_layers - 1}.ffn_res.y": "out_emb.x"}
    text_graph = ConnectGraph.apply([in_emb, text_decoders, out_emb], text_links)

    links = {
        "in_emb.y": "vlm_concat.text",
        "vlm_concat.y": "transformer.0.input_norm.x",
    }

    return vision_graph, text_graph, links
