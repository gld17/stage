import copy
import sympy as sp

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.ops import Add, PlaceHolder, Reshape
from symbolic_tensor_graph.tensor import Tensor


def _new_tensor(
    name,
    op_type,
    x1_shape,
    x1_hidden=None,
    x2_shape=None,
    x2_hidden=None,
    op_attr=None,
    x1=None,
    x2=None,
    require_grads=False,
    grad_of=None,
):
    tensor = Tensor(create_empty=True)
    tensor.name = name
    tensor.revision = 0
    tensor.require_grads = require_grads
    tensor.x1 = x1
    tensor.x2 = x2
    tensor.op_type = op_type
    tensor.op_attr = op_attr
    tensor.x1_shape = list(x1_shape)
    tensor.x1_hidden = list(x1_hidden or [1])
    tensor.x2_shape = list(x2_shape) if x2_shape is not None else None
    tensor.x2_hidden = list(x2_hidden) if x2_hidden is not None else None
    tensor.grad_of = grad_of
    return tensor


def patch_embedding(patch_embedding_path=None, include_backward=True):
    if patch_embedding_path is None:
        patch_embedding_path = "./sharding_spreadsheets/module/vit/patch_embedding.csv"

    batch, dp, cp, tp = sp.symbols("Batch dp cp tp")
    image_size, channels = sp.symbols("VisionImageSize VisionInChannels")
    num_patches, patch_dim, vision_hidden = sp.symbols(
        "NumPatches VisionPatchDim VisionHidden"
    )

    image = _new_tensor(
        "image",
        PlaceHolder.type_name,
        [batch / dp, image_size, image_size, channels],
    )
    patches = _new_tensor(
        "patches",
        Reshape.type_name,
        [batch / dp, image_size, image_size, channels],
        x2_shape=[batch / dp, (num_patches / cp) / tp, patch_dim],
        x2_hidden=[1],
        x1=image,
    )

    projection = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(patch_embedding_path),
        "patch_projection.%s",
        old_symbol_map_new_symbol={"Din": "VisionPatchDim", "Dout": "VisionHidden"},
    )
    graph = TensorGraph([image, patches], [image], [patches])
    links = {"patches": "patch_projection.x"}
    if include_backward:
        links["patch_projection.dx"] = "patches"
    return ConnectGraph.apply([graph, projection], links, force_connect=True)


def vit_block(
    ffn_path=None,
    layernorm_path=None,
    residual_path=None,
    gqa_surrounding_path=None,
    gqa_kernel_path=None,
    include_backward=True,
):
    from models.llama_model import group_query_attention

    if ffn_path is None:
        ffn_path = "./sharding_spreadsheets/module/vit/llama_feed_forward_network.csv"
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module/vit/layer_norm.csv"
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module/vit/residual.csv"
    if gqa_surrounding_path is None:
        gqa_surrounding_path = (
            "./sharding_spreadsheets/module/vit/group_query_attention_surrounding.csv"
        )
    if gqa_kernel_path is None:
        gqa_kernel_path = (
            "./sharding_spreadsheets/module/vit/group_query_attention_kernel_fused.csv"
        )

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "input_norm.%s"
    )
    mha = ReplicateGraph.apply(
        group_query_attention(
            GQA_surrounding_path=gqa_surrounding_path,
            GQA_kernel_path=gqa_kernel_path,
            include_backward=include_backward,
        ),
        "mha.%s",
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
    block = ReplicateGraph.apply(
        block,
        old_symbol_map_new_symbol={
            "Seq": "NumPatches",
            "Dmodel": "VisionHidden",
            "Dff": "VisionIntermediate",
            "Head": "VisionHead",
            "KVHead": "VisionHead",
        },
    )

    if include_backward:
        tensor_id_map_tensor = block.get_tensor_id_map_tensor()
        input_norm_dy = tensor_id_map_tensor["input_norm.dy@0"]
        if input_norm_dy.op_type == PlaceHolder.type_name:
            input_norm_dy.op_type = Add.type_name
            input_norm_dy.x1 = tensor_id_map_tensor["mha.dx@0"]
            input_norm_dy.x2 = tensor_id_map_tensor["mha_res.dx2@0"]
            input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
            input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
            block.in_tensors.remove(input_norm_dy)
            block.out_tensors.remove(input_norm_dy.x1)
            block.out_tensors.remove(input_norm_dy.x2)

        post_attn_norm_dy = tensor_id_map_tensor["post_attn_norm.dy@0"]
        if post_attn_norm_dy.op_type == PlaceHolder.type_name:
            post_attn_norm_dy.op_type = Add.type_name
            post_attn_norm_dy.x1 = tensor_id_map_tensor["ffn.dx0@0"]
            post_attn_norm_dy.x2 = tensor_id_map_tensor["ffn_res.dx2@0"]
            post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
            post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
            block.in_tensors.remove(post_attn_norm_dy)
            block.out_tensors.remove(post_attn_norm_dy.x1)
            block.out_tensors.remove(post_attn_norm_dy.x2)

    return block


def vision_encoder(
    num_layers,
    symbol_map_value,
    patch_embedding_path=None,
    regenerate=False,
    include_backward=True,
):
    """Build a Vision Transformer encoder using existing STG primitives."""
    del symbol_map_value, regenerate

    patch = patch_embedding(
        patch_embedding_path=patch_embedding_path, include_backward=include_backward
    )
    block_template = vit_block(include_backward=include_backward)

    blocks = []
    links = {"patch_projection.y": "vit.0.input_norm.x"}
    if include_backward:
        links["vit.0.input_norm.dx"] = "patch_projection.dy"
    for i in range(num_layers):
        blocks.append(ReplicateGraph.apply(block_template, f"vit.{i}.%s"))
        if i > 0:
            links[f"vit.{i - 1}.ffn_res.y"] = f"vit.{i}.input_norm.x"
            if include_backward:
                links[f"vit.{i}.input_norm.dx"] = f"vit.{i - 1}.ffn_res.dy"

    return ConnectGraph.apply([patch] + blocks, links)
