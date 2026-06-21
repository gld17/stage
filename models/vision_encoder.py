import os

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.ops import PlaceHolder
from symbolic_tensor_graph.tensor import Tensor


VIT_MODULE_DIR = "./sharding_spreadsheets/module/vit"


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


def _patch_embedding():
    image = _tensor(
        "image",
        PlaceHolder.type_name,
        x1_shape="Batch/dp, VisionImageSize, VisionImageSize, VisionInChannels",
    )
    patches = _tensor(
        "patches",
        "R",
        x1=image,
        x1_shape="Batch/dp, VisionImageSize, VisionImageSize, VisionInChannels",
        x2_shape="Batch/dp, NumPatches, VisionPatchDim",
        x2_hidden="1",
    )
    w = _tensor(
        "w_patch",
        PlaceHolder.type_name,
        x1_shape="VisionPatchDim, VisionHidden/1",
    )
    y = _tensor(
        "y",
        "M",
        "bsm,mn->bsn",
        x1=patches,
        x2=w,
        x1_shape="Batch/dp, NumPatches, VisionPatchDim",
        x1_hidden="1",
        x2_shape="VisionPatchDim, VisionHidden/1",
        x2_hidden="1",
    )
    return TensorGraph([image, patches, w, y], [image, w], [y])


def _vit_group_query_attention():
    surrounding = TensorGraph.load_tensor_graph(
        os.path.join(VIT_MODULE_DIR, "group_query_attention_surrounding.csv")
    )
    kernel = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(
            os.path.join(VIT_MODULE_DIR, "group_query_attention_kernel_fused.csv")
        ),
        "attn_kernel.%s",
    )
    return ConnectGraph.apply(
        [surrounding, kernel],
        {
            "q": "attn_kernel.q",
            "k": "attn_kernel.k",
            "v": "attn_kernel.v",
            "attn_kernel.qkv": "attn",
        },
    )


def _vit_block():
    layernorm_path = os.path.join(VIT_MODULE_DIR, "layer_norm.csv")
    residual_path = os.path.join(VIT_MODULE_DIR, "residual.csv")
    ffn_path = os.path.join(VIT_MODULE_DIR, "llama_feed_forward_network.csv")

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "input_norm.%s"
    )
    mha = ReplicateGraph.apply(_vit_group_query_attention(), "mha.%s")
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path), "mha_res.%s"
    )
    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "post_attn_norm.%s"
    )
    ffn = ReplicateGraph.apply(TensorGraph.load_tensor_graph(ffn_path), "ffn.%s")
    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path), "ffn_res.%s"
    )

    block = ConnectGraph.apply(
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

    return ReplicateGraph.apply(
        block,
        old_symbol_map_new_symbol={
            "Seq": "NumPatches",
            "Dmodel": "VisionHidden",
            "Dff": "VisionIntermediate",
            "Head": "VisionHead",
            "KVHead": "VisionHead",
        },
    )


def vision_encoder(
    num_layers,
    symbol_map_value,
    patch_embedding_path=None,
    regenerate=False,
):
    """Build a Vision Transformer encoder using existing STG primitives."""
    from . import CACHE_DIR

    cache_filename = os.path.join(CACHE_DIR, f"vit_{num_layers}_fwd.csv")
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    patch_embedding = (
        TensorGraph.load_tensor_graph(patch_embedding_path)
        if patch_embedding_path is not None
        else _patch_embedding()
    )
    patch_embedding = ReplicateGraph.apply(patch_embedding, "patch_embedding.%s")

    block_template = _vit_block()
    blocks = []
    links = {}
    for i in range(num_layers):
        block = ReplicateGraph.apply(block_template, f"vit.{i}.%s")
        blocks.append(block)
        if i > 0:
            links[f"vit.{i - 1}.ffn_res.y"] = f"vit.{i}.input_norm.x"

    blocks = ConnectGraph.apply(blocks, links)
    encoder = ConnectGraph.apply(
        [patch_embedding, blocks], {"patch_embedding.y": "vit.0.input_norm.x"}
    )
    encoder.save_tensor_graph(cache_filename)
    return encoder
