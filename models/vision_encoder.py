import copy
import os

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.grad_updater import FSDPWeightGradManager
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.ops import Add, Identical, PlaceHolder
from symbolic_tensor_graph.tensor import Tensor


VIT_MODULE_DIR = "./sharding_spreadsheets/module/vit"


def _tensor(
    name,
    require_grads,
    op_type,
    op_attr=None,
    x1=None,
    x2=None,
    x1_shape=None,
    x1_hidden="1",
    x2_shape=None,
    x2_hidden=None,
    grad_of=None,
):
    tensor = Tensor(create_empty=True)
    tensor.name = name
    tensor.revision = 0
    tensor.require_grads = require_grads
    tensor.op_type = op_type
    tensor.op_attr = op_attr
    tensor.x1 = x1
    tensor.x2 = x2
    tensor.x1_shape = Tensor.parse_shape(x1_shape)
    tensor.x1_hidden = Tensor.parse_shape(x1_hidden)
    tensor.x2_shape = Tensor.parse_shape(x2_shape)
    tensor.x2_hidden = Tensor.parse_shape(x2_hidden)
    tensor.grad_of = grad_of
    return tensor


def _patch_embedding(include_backward=True):
    image = _tensor(
        "image",
        False,
        PlaceHolder.type_name,
        x1_shape="Batch/dp, VisionImageSize, VisionImageSize, VisionInChannels",
    )
    patches = _tensor(
        "patches",
        False,
        "R",
        x1=image,
        x1_shape="Batch/dp, VisionImageSize, VisionImageSize, VisionInChannels",
        x2_shape="Batch/dp, NumPatches, VisionPatchDim",
        x2_hidden="1",
    )
    w = _tensor(
        "w_patch",
        True,
        PlaceHolder.type_name,
        x1_shape="VisionPatchDim, VisionHidden/1",
    )
    y = _tensor(
        "y",
        False,
        "M",
        "bsm,mn->bsn",
        x1=patches,
        x2=w,
        x1_shape="Batch/dp, NumPatches, VisionPatchDim",
        x1_hidden="1",
        x2_shape="VisionPatchDim, VisionHidden/1",
        x2_hidden="1",
    )
    tensors = [image, patches, w, y]
    in_tensors = [image, w]
    out_tensors = [y]

    if include_backward:
        dy = _tensor(
            "dy",
            False,
            PlaceHolder.type_name,
            x1_shape="Batch/dp, NumPatches, VisionHidden/1",
            grad_of=y,
        )
        dw = _tensor(
            "dw_patch",
            False,
            "M",
            "bsn,bsm->mn",
            x1=dy,
            x2=patches,
            x1_shape="Batch/dp, NumPatches, VisionHidden/1",
            x1_hidden="1",
            x2_shape="Batch/dp, NumPatches, VisionPatchDim",
            x2_hidden="1",
            grad_of=w,
        )
        dpatches = _tensor(
            "dpatches",
            False,
            "M",
            "bsn,mn->bsm",
            x1=dy,
            x2=w,
            x1_shape="Batch/dp, NumPatches, VisionHidden/1",
            x1_hidden="1",
            x2_shape="VisionPatchDim, VisionHidden/1",
            x2_hidden="1",
            grad_of=patches,
        )
        dimage = _tensor(
            "dimage",
            False,
            "R",
            x1=dpatches,
            x1_shape="Batch/dp, NumPatches, VisionPatchDim",
            x1_hidden="1",
            x2_shape="Batch/dp, VisionImageSize, VisionImageSize, VisionInChannels",
            x2_hidden="1",
            grad_of=image,
        )
        tensors.extend([dy, dw, dpatches, dimage])
        in_tensors.append(dy)
        out_tensors.extend([dw, dimage])

    return TensorGraph(tensors, in_tensors, out_tensors)


def _vit_group_query_attention(include_backward=True):
    surrounding = TensorGraph.load_tensor_graph(
        os.path.join(VIT_MODULE_DIR, "group_query_attention_surrounding.csv")
    )
    kernel = TensorGraph.load_tensor_graph(
        os.path.join(VIT_MODULE_DIR, "group_query_attention_kernel_fused.csv")
    )
    kernel = ReplicateGraph.apply(kernel, "attn_kernel.%s")
    links = {
        "q": "attn_kernel.q",
        "k": "attn_kernel.k",
        "v": "attn_kernel.v",
        "attn_kernel.qkv": "attn",
    }
    if include_backward:
        links.update(
            {
                "attn_kernel.dq": "dq",
                "attn_kernel.dk": "dk",
                "attn_kernel.dv": "dv",
                "dattn": "attn_kernel.dqkv",
            }
        )
    return ConnectGraph.apply([surrounding, kernel], links)


def _vit_block(include_backward=True):
    layernorm_path = os.path.join(VIT_MODULE_DIR, "layer_norm.csv")
    residual_path = os.path.join(VIT_MODULE_DIR, "residual.csv")
    ffn_path = os.path.join(VIT_MODULE_DIR, "llama_feed_forward_network.csv")

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "input_norm.%s"
    )
    mha = ReplicateGraph.apply(
        _vit_group_query_attention(include_backward=include_backward), "mha.%s"
    )
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

    links = {
        "input_norm.y": "mha.x",
        "mha.o": "mha_res.x1",
        "input_norm.x": "mha_res.x2",
        "mha_res.y": "post_attn_norm.x",
        "post_attn_norm.y": "ffn.x0",
        "ffn.xdown": "ffn_res.x1",
        "post_attn_norm.x": "ffn_res.x2",
    }
    if include_backward:
        links.update(
            {
                "mha_res.dx1": "mha.do",
                "post_attn_norm.dx": "mha_res.dy",
                "ffn_res.dx1": "ffn.dxdown",
            }
        )

    block = ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res], links
    )

    if include_backward:
        tensor_map = block.get_tensor_id_map_tensor()
        input_norm_dy = tensor_map["input_norm.dy@0"]
        input_norm_dy.op_type = Add.type_name
        input_norm_dy.x1 = tensor_map["mha.dx@0"]
        input_norm_dy.x2 = tensor_map["mha_res.dx2@0"]
        input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
        input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
        block.in_tensors.remove(input_norm_dy)
        block.out_tensors.remove(input_norm_dy.x1)
        block.out_tensors.remove(input_norm_dy.x2)

        post_attn_norm_dy = tensor_map["post_attn_norm.dy@0"]
        post_attn_norm_dy.op_type = Add.type_name
        post_attn_norm_dy.x1 = tensor_map["ffn.dx0@0"]
        post_attn_norm_dy.x2 = tensor_map["ffn_res.dx2@0"]
        post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
        post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
        block.in_tensors.remove(post_attn_norm_dy)
        block.out_tensors.remove(post_attn_norm_dy.x1)
        block.out_tensors.remove(post_attn_norm_dy.x2)
        block = FSDPWeightGradManager.apply(block)

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
    include_backward=True,
):
    """Build a Vision Transformer encoder using existing STG primitives."""
    from . import CACHE_DIR

    cache_filename = os.path.join(
        CACHE_DIR, f"vit_{num_layers}_bwd{int(include_backward)}.csv"
    )
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    patch_embedding = (
        TensorGraph.load_tensor_graph(patch_embedding_path)
        if patch_embedding_path is not None
        else _patch_embedding(include_backward=include_backward)
    )
    patch_embedding = ReplicateGraph.apply(patch_embedding, "patch_embedding.%s")

    block_template = _vit_block(include_backward=include_backward)
    blocks = []
    links = {}
    for i in range(num_layers):
        block = ReplicateGraph.apply(block_template, f"vit.{i}.%s")
        blocks.append(block)
        if i > 0:
            links[f"vit.{i - 1}.ffn_res.y"] = f"vit.{i}.input_norm.x"
            if include_backward:
                links[f"vit.{i}.input_norm.dx"] = f"vit.{i - 1}.ffn_res.dy"

    blocks = ConnectGraph.apply(blocks, links)
    links = {"patch_embedding.y": "vit.0.input_norm.x"}
    if include_backward:
        links["vit.0.input_norm.dx"] = "patch_embedding.dy"
    encoder = ConnectGraph.apply([patch_embedding, blocks], links)
    encoder.save_tensor_graph(cache_filename)
    return encoder
