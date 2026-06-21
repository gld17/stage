from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.ops import PlaceHolder
from symbolic_tensor_graph.tensor import Tensor


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


def vision_projection():
    """Project ViT tokens from VisionHidden to Dmodel."""
    x = _tensor(
        "x",
        PlaceHolder.type_name,
        x1_shape="Batch/dp, NumPatches, VisionHidden/1",
    )
    w = _tensor(
        "w",
        PlaceHolder.type_name,
        x1_shape="VisionHidden, Dmodel/1",
    )
    y = _tensor(
        "y",
        "M",
        "bsm,mn->bsn",
        x1=x,
        x2=w,
        x1_shape="Batch/dp, NumPatches, VisionHidden",
        x1_hidden="1",
        x2_shape="VisionHidden, Dmodel/1",
        x2_hidden="1",
    )
    return TensorGraph([x, w, y], [x, w], [y])
