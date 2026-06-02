import sympy as sp

from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.ops import Einsum, PlaceHolder
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


def vision_projection(include_backward=True):
    """Project ViT visual tokens from VisionHidden to Dmodel."""
    batch, dp, cp, tp = sp.symbols("Batch dp cp tp")
    num_patches, vision_hidden, dmodel = sp.symbols(
        "NumPatches VisionHidden Dmodel"
    )

    x = _new_tensor(
        "x",
        PlaceHolder.type_name,
        [batch / dp, (num_patches / cp) / tp, vision_hidden],
    )
    w = _new_tensor(
        "w",
        PlaceHolder.type_name,
        [vision_hidden, dmodel],
        require_grads=True,
    )
    y = _new_tensor(
        "y",
        Einsum.type_name,
        [batch / dp, (num_patches / cp) / tp, vision_hidden],
        x2_shape=[vision_hidden, dmodel],
        x2_hidden=[1],
        op_attr="bsv,vd->bsd",
        x1=x,
        x2=w,
    )
    tensors = [x, w, y]
    in_tensors = [x, w]
    out_tensors = [y]

    if include_backward:
        dy = _new_tensor(
            "dy",
            PlaceHolder.type_name,
            [batch / dp, (num_patches / cp) / tp, dmodel],
            grad_of=y,
        )
        dw = _new_tensor(
            "dw",
            Einsum.type_name,
            [batch / dp, (num_patches / cp) / tp, dmodel],
            x2_shape=[batch / dp, (num_patches / cp) / tp, vision_hidden],
            x2_hidden=[1],
            op_attr="bsd,bsv->vd",
            x1=dy,
            x2=x,
            grad_of=w,
        )
        dx = _new_tensor(
            "dx",
            Einsum.type_name,
            [batch / dp, (num_patches / cp) / tp, dmodel],
            x2_shape=[vision_hidden, dmodel],
            x2_hidden=[1],
            op_attr="bsd,vd->bsv",
            x1=dy,
            x2=w,
            grad_of=x,
        )
        tensors.extend([dy, dw, dx])
        in_tensors.append(dy)
        out_tensors.extend([dw, dx])

    return TensorGraph(tensors, in_tensors, out_tensors)
