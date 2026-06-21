from typing import List, Optional, Set


def validate_physical_topology(data: dict) -> None:
    """Validate physical_topology.json schema. Raises ValueError on failure."""
    if not isinstance(data, dict):
        raise ValueError("physical topology must be a JSON object")

    required_fields = ("schema_version", "num_npus", "bandwidth_gbps")
    for field in required_fields:
        if field not in data:
            raise ValueError(f"missing required field: {field}")

    num_npus = data["num_npus"]
    if not isinstance(num_npus, int) or isinstance(num_npus, bool) or num_npus < 0:
        raise ValueError("num_npus must be a non-negative integer")

    bandwidth_gbps = data["bandwidth_gbps"]
    if not isinstance(bandwidth_gbps, list) or len(bandwidth_gbps) != num_npus:
        raise ValueError("num_npus must match bandwidth_gbps dimension")

    for i, row in enumerate(bandwidth_gbps):
        if not isinstance(row, list) or len(row) != num_npus:
            raise ValueError("bandwidth_gbps must be N×N")
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("bandwidth_gbps elements must be numeric")
            if value < 0:
                raise ValueError("bandwidth_gbps elements must be >= 0")
            if i == j and value != 0:
                raise ValueError("bandwidth_gbps diagonal must be 0")

    if data.get("symmetric", True):
        for i in range(num_npus):
            for j in range(i + 1, num_npus):
                if bandwidth_gbps[i][j] != bandwidth_gbps[j][i]:
                    raise ValueError("bandwidth_gbps must be symmetric")


class PhysicalTopology:
    """Physical topology data model for FlexET."""

    def __init__(
        self,
        bandwidth_gbps: List[List[float]],
        latency_ns: Optional[List[List[float]]] = None,
        symmetric: bool = True,
        schema_version: str = "1.0",
        num_npus: Optional[int] = None,
        **kwargs,
    ):
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_ns = latency_ns
        self.symmetric = symmetric
        self.schema_version = schema_version
        self._num_npus = num_npus or len(bandwidth_gbps)
        if len(bandwidth_gbps) != self._num_npus:
            raise ValueError("bandwidth_gbps dimension mismatch with num_npus")
        for row in bandwidth_gbps:
            if len(row) != self._num_npus:
                raise ValueError("bandwidth_gbps must be N×N")

    @property
    def num_npus(self) -> int:
        return self._num_npus

    def has_direct_link(self, src: int, dst: int) -> bool:
        return self.bandwidth_gbps[src][dst] > 0

    def link_bandwidth_gbps(self, src: int, dst: int) -> float:
        return self.bandwidth_gbps[src][dst]

    def is_induced_subgraph_connected(self, node_set: Set[int]) -> bool:
        """Check if the induced subgraph of node_set is connected."""
        if not node_set:
            return True
        node_list = list(node_set)
        start = node_list[0]
        visited = {start}
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in node_list:
                if v not in visited and self.has_direct_link(u, v):
                    visited.add(v)
                    queue.append(v)
        return len(visited) == len(node_set)
