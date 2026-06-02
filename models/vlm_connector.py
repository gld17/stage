from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.ops import PlaceHolder
from symbolic_tensor_graph.tensor import Tensor


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


def vision_projection(include_backward=True):
    """Project ViT tokens from VisionHidden to Dmodel."""
    x = _tensor(
        "x",
        False,
        PlaceHolder.type_name,
        x1_shape="Batch/dp, NumPatches, VisionHidden/1",
    )
    w = _tensor(
        "w",
        True,
        PlaceHolder.type_name,
        x1_shape="VisionHidden, Dmodel/1",
    )
    y = _tensor(
        "y",
        False,
        "M",
        "bsm,mn->bsn",
        x1=x,
        x2=w,
        x1_shape="Batch/dp, NumPatches, VisionHidden",
        x1_hidden="1",
        x2_shape="VisionHidden, Dmodel/1",
        x2_hidden="1",
    )
    tensors = [x, w, y]
    in_tensors = [x, w]
    out_tensors = [y]

    if include_backward:
        dy = _tensor(
            "dy",
            False,
            PlaceHolder.type_name,
            x1_shape="Batch/dp, NumPatches, Dmodel/1",
            grad_of=y,
        )
        dw = _tensor(
            "dw",
            False,
            "M",
            "bsn,bsm->mn",
            x1=dy,
            x2=x,
            x1_shape="Batch/dp, NumPatches, Dmodel/1",
            x1_hidden="1",
            x2_shape="Batch/dp, NumPatches, VisionHidden",
            x2_hidden="1",
            grad_of=w,
        )
        dx = _tensor(
            "dx",
            False,
            "M",
            "bsn,mn->bsm",
            x1=dy,
            x2=w,
            x1_shape="Batch/dp, NumPatches, Dmodel/1",
            x1_hidden="1",
            x2_shape="VisionHidden, Dmodel/1",
            x2_hidden="1",
            grad_of=x,
        )
        tensors.extend([dy, dw, dx])
        in_tensors.append(dy)
        out_tensors.extend([dw, dx])

    return TensorGraph(tensors, in_tensors, out_tensors)
