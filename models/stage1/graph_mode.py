"""Forward-only vs training (forward + backward + weight update) graph modes."""

from __future__ import annotations

import copy
import re
from collections import defaultdict, deque
from typing import Set

from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.tensor import Tensor

# Matches backward tensor id stems (revision suffix @N stripped).
_BACKWARD_STEM_RE = re.compile(
    r"(?:"
    r"\.(?:dy|dx|dw|dq|dk|dv|do|dattn|dwo|dqkv|dwqkv|dx1|dx2|dx0|dxdown|dxgate|dx000"
    r"|dx_expert|dy_expert|dxrouted|dyrouted)"
    r"|(?:_sharded_grad|_assembled_grad|_assembled_weight_backward|_backward)"
    r"|(?:^loss\.(?:dy|dx))"
    r"|(?:^|\.)(?:dattn|dqkv|dwqkv)(?:\.|@|$)"
    r")"
)


def tensor_stem(tensor: Tensor) -> str:
    return tensor.id.split("@")[0]


def is_backward_tensor(tensor: Tensor) -> bool:
    if getattr(tensor, "grad_of", None):
        return True
    stem = tensor_stem(tensor)
    if _BACKWARD_STEM_RE.search(stem):
        return True
    last = stem.split(".")[-1]
    if last.startswith(("dw", "dq", "dk", "dv", "do")) and len(last) > 1:
        return True
    return False


def strip_to_forward_only(graph: TensorGraph, inplace: bool = False) -> TensorGraph:
    """
    Keep only tensors on forward paths from graph inputs.

    Removes backward activations/grads, loss backward, FSDP grad collectives' tensors,
    and GradUpdater weight-update chains when applied before this pass.
    """
    if not inplace:
        graph = copy.deepcopy(graph)

    backward_ids: Set[str] = {t.id for t in graph.tensors if is_backward_tensor(t)}

    children: dict = defaultdict(list)
    for t in graph.tensors:
        for parent in (t.x1, t.x2):
            if parent is not None:
                children[parent.id].append(t)

    reachable: Set[str] = set()
    queue: deque = deque()

    for t in graph.in_tensors:
        if t.id not in backward_ids:
            reachable.add(t.id)
            queue.append(t.id)

    if not queue:
        for t in graph.tensors:
            if t.id in backward_ids:
                continue
            stem = tensor_stem(t)
            if stem.endswith(".x") or stem in ("x", "w"):
                reachable.add(t.id)
                queue.append(t.id)

    while queue:
        tid = queue.popleft()
        for child in children.get(tid, []):
            if child.id in backward_ids or child.id in reachable:
                continue
            reachable.add(child.id)
            queue.append(child.id)

    graph.tensors = [t for t in graph.tensors if t.id in reachable]
    keep_ids = {t.id for t in graph.tensors}
    graph.in_tensors = [t for t in graph.in_tensors if t.id in keep_ids]
    graph.out_tensors = [t for t in graph.out_tensors if t.id in keep_ids]
    return graph
