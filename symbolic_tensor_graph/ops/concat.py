import sympy as sp

from .op_base import OPBase


class Concat(OPBase):
    type_name = "C"

    @staticmethod
    def _parse_dim(op_attr):
        dim = float(op_attr)
        assert dim.is_integer()
        return int(dim)

    @staticmethod
    def _same_expr(left, right):
        return sp.simplify(left - right) == 0

    @classmethod
    def _same_shape(cls, left_shape, right_shape):
        if len(left_shape) != len(right_shape):
            return False
        return all(
            cls._same_expr(left, right)
            for left, right in zip(left_shape, right_shape)
        )

    @classmethod
    def _sanity_check(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr
        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden
        assert op_attr is not None
        
        dim = cls._parse_dim(op_attr)
        assert len(x1_shape) == len(x2_shape)
        assert cls._same_shape(x1_hidden, x2_hidden)
        
        if dim < 0:
            dim += len(x1_shape)
        for i in range(len(x1_shape)):
            if i != dim:
                assert cls._same_expr(x1_shape[i], x2_shape[i])
            else:
                pass

    @classmethod
    def _eval_impl(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr
        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden
        
        dim = cls._parse_dim(op_attr)
        if dim < 0:
            dim += len(x1_shape)
            
        y_shape = list(tensor.x1_shape)
        y_shape[dim] += x2_shape[dim]
        y_shape = tuple(y_shape)
        
        return y_shape, x1_hidden, Tensor.eval_size(y_shape)

    @classmethod
    def _shardable_options_impl(cls, tensor):
        ret = list()
        cat_dim = cls._parse_dim(tensor.op_attr)
        for i, shape in enumerate(tensor.x1_shape):
            if i == cat_dim:
                continue
            ret.append(i)
        return ret
