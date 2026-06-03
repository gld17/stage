from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph

from .llama_model import transformer_decoders
from .moe_model import transformer_decoder_block as moe_transformer_decoder_block
from .vision_encoder import vision_encoder
from .vlm import _concat_tokens
from .vlm_connector import vision_projection


def vlm_moe_subgraphs(
    text_num_layers,
    vision_num_layers,
    symbol_map_value,
    regenerate=False,
    tpsp=True,
    include_backward=True,
):
    """Build VLM with a MoE text backbone as two independent subgraphs."""
    vit = ReplicateGraph.apply(
        vision_encoder(
            vision_num_layers,
            symbol_map_value,
            regenerate=regenerate,
            include_backward=include_backward,
        ),
        "vision_encoder.%s",
    )
    projection = ReplicateGraph.apply(
        vision_projection(include_backward=include_backward), "vision_projection.%s"
    )
    concat = ReplicateGraph.apply(
        _concat_tokens(include_backward=include_backward), "vlm_concat.%s"
    )
    vision_graph = ConnectGraph.apply(
        [vit, projection, concat],
        {
            f"vision_encoder.vit.{vision_num_layers - 1}.ffn_res.y": "vision_projection.x",
            "vision_projection.y": "vlm_concat.visual",
        },
    )

    decoder_template = moe_transformer_decoder_block(
        symbol_map_value, include_backward=include_backward
    )
    text_decoders = transformer_decoders(
        text_num_layers, decoder_template, include_backward=include_backward
    )
    text_decoders = ReplicateGraph.apply(
        text_decoders,
        inplace=True,
        old_symbol_map_new_symbol={"Seq": "TotalSeq"},
    )

    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph("./sharding_spreadsheets/module/tpsp_moe/embedding.csv"),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel", "tp": "tp"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph("./sharding_spreadsheets/module/tpsp_moe/embedding.csv"),
        "out_emb.%s",
        old_symbol_map_new_symbol={
            "Din": "Dmodel",
            "Dout": "Dvocal",
            "Seq": "TotalSeq",
            "tp": "tp",
        },
    )

    text_links = {f"transformer.{text_num_layers - 1}.ffn_res.y": "out_emb.x"}
    if include_backward:
        text_links["out_emb.dx"] = f"transformer.{text_num_layers - 1}.ffn_res.dy"
    text_graph = ConnectGraph.apply([in_emb, text_decoders, out_emb], text_links)

    links = {
        "in_emb.y": "vlm_concat.text",
        "vlm_concat.y": "transformer.0.input_norm.x",
    }

    if include_backward:
        loss = ReplicateGraph.apply(
            TensorGraph.load_tensor_graph("./sharding_spreadsheets/module/tpsp_moe/loss.csv"),
            "loss.%s",
            old_symbol_map_new_symbol={"Seq": "TotalSeq", "tp": "tp"},
        )
        text_graph = ConnectGraph.apply(
            [text_graph, loss],
            {"out_emb.y": "loss.y", "loss.dy": "out_emb.dy"},
        )
        links["vlm_concat.dtext"] = "in_emb.dy"
        links["transformer.0.input_norm.dx"] = "vlm_concat.dy"

    return vision_graph, text_graph, links
