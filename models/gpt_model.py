from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph


def group_query_attention(GQA_surrounding_path=None, GQA_kernel_path=None, tpsp=True):
    if GQA_surrounding_path is None:
        GQA_surrounding_path = "./sharding_spreadsheets/module/tpsp_gpt/group_query_attention_surrounding.csv"
        if not tpsp:
            GQA_surrounding_path = (
                "./sharding_spreadsheets/module/tp_gpt/group_query_attention_surrounding.csv"
            )
    if GQA_kernel_path is None:
        GQA_kernel_path = "./sharding_spreadsheets/module/tpsp_gpt/group_query_attention_kernel_fused.csv"
        if not tpsp:
            GQA_kernel_path = (
                "./sharding_spreadsheets/module/tp_gpt/group_query_attention_kernel_fused.csv"
            )
    GQA_surrounding = TensorGraph.load_tensor_graph(GQA_surrounding_path)
    GQA_kernel = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(GQA_kernel_path), "attn_kernel.%s"
    )
    return ConnectGraph.apply(
        [GQA_surrounding, GQA_kernel],
        {
            "q": "attn_kernel.q",
            "k": "attn_kernel.k",
            "v": "attn_kernel.v",
            "attn_kernel.qkv": "attn",
        },
    )


def feed_forward_network(ffn_path=None, tpsp=True):
    if ffn_path is None:
        ffn_path = (
            "./sharding_spreadsheets/module/tpsp_gpt/llama_feed_forward_network.csv"
        )
        if not tpsp:
            ffn_path = (
                "./sharding_spreadsheets/module/tp_gpt/llama_feed_forward_network.csv"
            )
    return ReplicateGraph.apply(TensorGraph.load_tensor_graph(ffn_path), "ffn.%s")


def transformer_decoder_block(
    ffn_path=None,
    layernorm_path=None,
    residual_path=None,
    tpsp=True,
):
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module/tpsp_gpt/layer_norm.csv"
        if not tpsp:
            layernorm_path = "./sharding_spreadsheets/module/tp_gpt/layer_norm.csv"
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module/tpsp_gpt/residual.csv"
        if not tpsp:
            residual_path = "./sharding_spreadsheets/module/tp_gpt/residual.csv"

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "input_norm.%s"
    )
    mha = ReplicateGraph.apply(group_query_attention(tpsp=tpsp), "mha.%s")
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path), "mha_res.%s"
    )
    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "post_attn_norm.%s"
    )
    ffn = feed_forward_network(ffn_path, tpsp=tpsp)
    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path), "ffn_res.%s"
    )

    return ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res],
        {
            "input_norm.y": "mha.x",
            "mha.o": "mha_res.x1",
            "input_norm.x": "mha_res.x2",
            "mha_res.y": "post_attn_norm.x",
            "post_attn_norm.y": "ffn.x0",
            "ffn.xdown": "ffn_res.x1",
            "post_attn_norm.x": "ffn_res.x2",
        },
    )


def transformer_decoders(num_layers, decoder_template, tpsp=True):
    links = {}
    decoders = []
    for i in range(num_layers):
        decoder = ReplicateGraph.apply(decoder_template, f"transformer.{i}.%s")
        decoders.append(decoder)
        if i > 0:
            links[f"transformer.{i-1}.ffn_res.y"] = f"transformer.{i}.input_norm.x"
    return ConnectGraph.apply(decoders, links)


def gpt(num_layers, embedding_path=None, regenerate=False, tpsp=True):
    from . import CACHE_DIR
    import os

    cache_filename = os.path.join(CACHE_DIR, f"gpt_{num_layers}_{tpsp}_fwd.csv")
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    if embedding_path is None:
        embedding_path = "./sharding_spreadsheets/module/tpsp_gpt/embedding.csv"
        if not tpsp:
            embedding_path = "./sharding_spreadsheets/module/tp_gpt/embedding.csv"
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal"},
    )
    decoders = transformer_decoders(
        num_layers, transformer_decoder_block(tpsp=tpsp), tpsp=tpsp
    )
    transformer = ConnectGraph.apply(
        [decoders, in_emb, out_emb],
        {
            "in_emb.y": "transformer.0.input_norm.x",
            f"transformer.{num_layers-1}.ffn_res.y": "out_emb.x",
        },
    )
    transformer.save_tensor_graph(cache_filename)
    return transformer
