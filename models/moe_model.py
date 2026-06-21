import sympy as sp

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from .llama_model import group_query_attention, transformer_decoders
from .utils import reduce_chain


def expert_branch(ffn_path=None, moe_wrapper_path=None):
    if ffn_path is None:
        ffn_path = "./sharding_spreadsheets/module/tpsp_moe/llama_feed_forward_network.csv"
    if moe_wrapper_path is None:
        moe_wrapper_path = "./sharding_spreadsheets/module/tpsp_moe/expert_wrapper.csv"

    ffn = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(ffn_path),
        "ffn.%s",
        old_symbol_map_new_symbol={"Seq": "Seq*KExperts/(Experts*ep)"},
    )
    moe_wrapper = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_wrapper_path),
        "ldis.%s",
    )
    return ConnectGraph.apply(
        [moe_wrapper, ffn],
        {
            "ldis.x_expert": "ffn.x0",
            "ffn.xdown": "ldis.y_expert",
        },
    )


def feed_forward_network(
    symbol_map_value,
    ffn_path=None,
    expert_wrapper_path=None,
    moe_frame_path=None,
):
    if moe_frame_path is None:
        moe_frame_path = "./sharding_spreadsheets/module/tpsp_moe/moe_frame.csv"
    experts_sym, kexperts_sym, ep_sym = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts_sym]
    ep = symbol_map_value[ep_sym]
    experts_each_group = experts / ep
    assert experts_each_group == int(experts_each_group)
    experts_each_group = int(experts_each_group)

    expert = expert_branch(ffn_path, expert_wrapper_path)
    moe_frame = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_frame_path), "moe.%s"
    )

    branches = [ReplicateGraph.apply(expert, f"moe.{i}.%s") for i in range(experts_each_group)]
    moe = ConnectGraph.apply([moe_frame] + branches, {})
    tensor_id_map_tensor = moe.get_tensor_id_map_tensor()
    moe_xrouted = tensor_id_map_tensor["moe.xrouted@0"]

    for i in range(experts_each_group):
        moe = ConnectGraph.apply(
            [moe], {"moe.xrouted": f"moe.{i}.ldis.x"}, inplace=True
        )
        moe.out_tensors.append(moe_xrouted)
    moe.out_tensors.remove(moe_xrouted)

    to_be_reduce_moe_yrouted = []
    for i in range(experts_each_group):
        branch_ldis_y = tensor_id_map_tensor[f"moe.{i}.ldis.y@0"]
        to_be_reduce_moe_yrouted.append(branch_ldis_y)
        moe.out_tensors.remove(branch_ldis_y)

    merged_yrouted = reduce_chain(to_be_reduce_moe_yrouted, "moe.yrouted_r%d", amp=0)
    moe.tensors.extend(merged_yrouted)
    if merged_yrouted:
        merged_yrouted[-1].op_attr = "1"
        merged_yrouted_last = merged_yrouted[-1]
    else:
        assert len(to_be_reduce_moe_yrouted) == 1
        merged_yrouted_last = to_be_reduce_moe_yrouted[0]
    moe.out_tensors.append(merged_yrouted_last)
    return ConnectGraph.apply([moe], {merged_yrouted_last.name: "moe.yrouted"})


def transformer_decoder_block(symbol_map_value, layernorm_path=None, residual_path=None):
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module/tpsp_moe/layer_norm.csv"
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module/tpsp_moe/residual.csv"

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "input_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    mha = ReplicateGraph.apply(
        group_query_attention(),
        "mha.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "mha_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "post_attn_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    ffn = feed_forward_network(symbol_map_value)
    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "ffn_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    return ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res],
        {
            "input_norm.y": "mha.x",
            "mha.o": "mha_res.x1",
            "input_norm.x": "mha_res.x2",
            "mha_res.y": "post_attn_norm.x",
            "post_attn_norm.y": "moe.x",
            "moe.y": "ffn_res.x1",
            "post_attn_norm.x": "ffn_res.x2",
        },
    )


def transformer(num_layers, symbol_map_value, embedding_path=None, regenerate=False):
    from . import CACHE_DIR
    import os

    experts_sym, ep_sym = sp.symbols("Experts ep")
    experts_each_group = symbol_map_value[experts_sym] / symbol_map_value[ep_sym]
    cache_filename = os.path.join(
        CACHE_DIR,
        f"moe_{num_layers}_{experts_each_group}_fwd.csv",
    )
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    if embedding_path is None:
        embedding_path = "./sharding_spreadsheets/module/tpsp_moe/embedding.csv"
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel", "tp": "tp"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal", "tp": "tp"},
    )

    decoders = transformer_decoders(
        num_layers, transformer_decoder_block(symbol_map_value)
    )
    transformer_graph = ConnectGraph.apply(
        [decoders, in_emb, out_emb],
        {
            "in_emb.y": "transformer.0.input_norm.x",
            f"transformer.{num_layers-1}.ffn_res.y": "out_emb.x",
        },
    )
    transformer_graph.save_tensor_graph(cache_filename)
    return transformer_graph
