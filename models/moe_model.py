import copy
import sympy as sp

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import FSDPWeightGradManager
from symbolic_tensor_graph.ops import Add, PlaceHolder
from .llama_model import group_query_attention, transformer_decoders
from .utils import reduce_chain


def expert_branch(ffn_path=None, moe_wrapper_path=None, include_backward=True):
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

    links = {
        "ldis.x_expert": "ffn.x0",
        "ffn.xdown": "ldis.y_expert",
    }
    if include_backward:
        links["ldis.dy_expert"] = "ffn.dxdown"
        links["ffn.dx0"] = "ldis.dx_expert"

    expert = ConnectGraph.apply([moe_wrapper, ffn], links)
    return expert


def feed_forward_network(
    symbol_map_value,
    ffn_path=None,
    expert_wrapper_path=None,
    moe_frame_path=None,
    include_backward=True,
):
    if moe_frame_path is None:
        moe_frame_path = "./sharding_spreadsheets/module/tpsp_moe/moe_frame.csv"
    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    kexperts = symbol_map_value[kexperts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    assert experts_each_group == int(experts_each_group)
    experts_each_group = int(experts_each_group)

    expert = expert_branch(ffn_path, expert_wrapper_path, include_backward=include_backward)
    moe_frame = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_frame_path), "moe.%s"
    )

    branches = list()
    for i in range(experts_each_group):
        branches.append(ReplicateGraph.apply(expert, f"moe.{i}.%s"))

    moe = ConnectGraph.apply([moe_frame] + branches, dict())
    tensor_id_map_tensor = moe.get_tensor_id_map_tensor()

    moe_xrouted = tensor_id_map_tensor["moe.xrouted@0"]
    moe_dyrouted = (
        tensor_id_map_tensor["moe.dyrouted@0"] if include_backward else None
    )
    for i in range(experts_each_group):
        route_links = {"moe.xrouted": f"moe.{i}.ldis.x"}
        if include_backward:
            route_links["moe.dyrouted"] = f"moe.{i}.ldis.dy"
        moe = ConnectGraph.apply([moe], route_links, inplace=True)
        moe.out_tensors.append(moe_xrouted)
        if include_backward:
            moe.out_tensors.append(moe_dyrouted)

    moe.out_tensors.remove(moe_xrouted)
    if include_backward:
        moe.out_tensors.remove(moe_dyrouted)

    to_be_reduce_moe_yrouted = list()
    for i in range(experts_each_group):
        branch_ldis_y = tensor_id_map_tensor[f"moe.{i}.ldis.y@0"]
        to_be_reduce_moe_yrouted.append(branch_ldis_y)
        moe.out_tensors.remove(branch_ldis_y)

    merged_yrouted = reduce_chain(to_be_reduce_moe_yrouted, "moe.yrouted_r%d", amp=0)
    moe.tensors.extend(merged_yrouted)
    if len(merged_yrouted) > 0:
        merged_yrouted[-1].op_attr = "1"
        merged_yrouted_last = merged_yrouted[-1]
    else:
        assert len(to_be_reduce_moe_yrouted) == 1
        merged_yrouted_last = to_be_reduce_moe_yrouted[0]
    moe.out_tensors.append(merged_yrouted_last)

    merge_links = {merged_yrouted_last.name: "moe.yrouted"}
    if include_backward:
        to_be_reduce_moe_dxrouted = list()
        for i in range(experts_each_group):
            branch_ldis_dx = tensor_id_map_tensor[f"moe.{i}.ldis.dx@0"]
            to_be_reduce_moe_dxrouted.append(branch_ldis_dx)
            moe.out_tensors.remove(branch_ldis_dx)

        merged_dxrouted = reduce_chain(
            to_be_reduce_moe_dxrouted, "moe.dxrouted_r%d", amp=0
        )
        moe.tensors.extend(merged_dxrouted)
        if len(merged_dxrouted) > 0:
            merged_dxrouted[-1].op_attr = "1"
            merged_dxrouted_last = merged_dxrouted[-1]
        else:
            assert len(to_be_reduce_moe_dxrouted) == 1
            merged_dxrouted_last = to_be_reduce_moe_dxrouted[0]
        moe.out_tensors.append(merged_dxrouted_last)
        merge_links[merged_dxrouted_last.name] = "moe.dxrouted"

    moe = ConnectGraph.apply([moe], merge_links)
    return moe


def transformer_decoder_block(
    symbol_map_value, layernorm_path=None, residual_path=None, include_backward=True
):
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
        group_query_attention(include_backward=include_backward),
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

    ffn = feed_forward_network(
        symbol_map_value, include_backward=include_backward
    )

    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "ffn_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    links = dict()
    links["input_norm.y"] = "mha.x"
    links["mha.o"] = "mha_res.x1"
    links["input_norm.x"] = "mha_res.x2"
    links["mha_res.y"] = "post_attn_norm.x"
    links["post_attn_norm.y"] = "moe.x"
    links["moe.y"] = "ffn_res.x1"
    links["post_attn_norm.x"] = "ffn_res.x2"
    if include_backward:
        links["mha_res.dx1"] = "mha.do"
        links["post_attn_norm.dx"] = "mha_res.dy"
        links["ffn_res.dx1"] = "moe.dy"

    decoder_block = ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res], links
    )

    if not include_backward:
        return decoder_block

    tensor_id_map_tensor = decoder_block.get_tensor_id_map_tensor()

    input_norm_dy = tensor_id_map_tensor["input_norm.dy@0"]
    assert input_norm_dy.op_type == PlaceHolder.type_name
    input_norm_dy.op_type = Add.type_name
    input_norm_dy.x1 = tensor_id_map_tensor["mha.dx@0"]
    input_norm_dy.x2 = tensor_id_map_tensor["mha_res.dx2@0"]
    input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
    input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(input_norm_dy)
    decoder_block.out_tensors.remove(input_norm_dy.x1)
    decoder_block.out_tensors.remove(input_norm_dy.x2)

    post_attn_norm_dy = tensor_id_map_tensor["post_attn_norm.dy@0"]
    assert post_attn_norm_dy.op_type == PlaceHolder.type_name
    post_attn_norm_dy.op_type = Add.type_name
    post_attn_norm_dy.x1 = tensor_id_map_tensor["moe.dx@0"]
    post_attn_norm_dy.x2 = tensor_id_map_tensor["ffn_res.dx2@0"]
    post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
    post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(post_attn_norm_dy)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x1)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x2)

    decoder_block = FSDPWeightGradManager.apply(decoder_block)

    return decoder_block


def transformer(
    num_layers, symbol_map_value, embedding_path=None, regenerate=False, include_backward=True
):
    from . import CACHE_DIR
    import os

    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    kexperts = symbol_map_value[kexperts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    cache_filename = os.path.join(
        CACHE_DIR,
        f"moe_{num_layers}_{experts_each_group}_bwd{int(include_backward)}.csv",
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

    decoder_template = transformer_decoder_block(
        symbol_map_value, include_backward=include_backward
    )
    decoders = transformer_decoders(
        num_layers, decoder_template, include_backward=include_backward
    )

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"
    if include_backward:
        links["transformer.0.input_norm.dx"] = "in_emb.dy"
        links["out_emb.dx"] = f"transformer.{num_layers-1}.ffn_res.dy"

    transformer = ConnectGraph.apply([decoders, in_emb, out_emb], links)

    if include_backward:
        loss = ReplicateGraph.apply(
            TensorGraph.load_tensor_graph(
                "./sharding_spreadsheets/module/tpsp_moe/loss.csv"
            ),
            "loss.%s",
            old_symbol_map_new_symbol={"tp": "tp"},
        )
        links = dict()
        links["out_emb.y"] = "loss.y"
        links["loss.dy"] = "out_emb.dy"
        transformer = ConnectGraph.apply([transformer, loss], links)

    transformer.save_tensor_graph(cache_filename)
    return transformer
