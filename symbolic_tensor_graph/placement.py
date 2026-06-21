import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from symbolic_tensor_graph.graph.logical_to_physicall_rank_mapper import (
    LogicalToPhysicalRankMapper,
)
from symbolic_tensor_graph.topology import PhysicalTopology


Role = Tuple[int, int, int, int, int]
ReadableRank = Tuple[Tuple[str, int], ...]


def generate_readable_ranks(dp, tp, pp, sp, ep, dim_names=None) -> List[ReadableRank]:
    if dim_names is None:
        dim_names = ("dp", "tp", "pp", "sp", "ep")
    ranks = []
    for d in range(dp):
        for t in range(tp):
            for p in range(pp):
                for s in range(sp):
                    for e in range(ep):
                        ranks.append(
                            ((dim_names[0], d), (dim_names[1], t), (dim_names[2], p),
                             (dim_names[3], s), (dim_names[4], e))
                        )
    return ranks


def role_from_readable(readable_rank, dim_names=None) -> Role:
    d = {str(k): v for k, v in readable_rank}
    if dim_names is None:
        dim_names = ("dp", "tp", "pp", "sp", "ep")
    return tuple(d[str(name)] for name in dim_names)


def readable_rank_to_role(readable_rank, dim_names=None) -> Role:
    """Convert any readable rank (with string or sympy Symbol dims) to a role tuple."""
    return role_from_readable(readable_rank, dim_names)


def generate_comm_groups(dp, tp, pp, sp, ep, dim_names=None) -> Dict[str, List[Set[Role]]]:
    if dim_names is None:
        dim_names = ("dp", "tp", "pp", "sp", "ep")
    comm_groups = {
        str(dim_names[0]): [],
        str(dim_names[1]): [],
        str(dim_names[2]): [],
        str(dim_names[3]): [],
        str(dim_names[4]): [],
    }
    dp_key, tp_key, pp_key, sp_key, ep_key = str(dim_names[0]), str(dim_names[1]), str(dim_names[2]), str(dim_names[3]), str(dim_names[4])

    for t in range(tp):
        for p in range(pp):
            for s in range(sp):
                for e in range(ep):
                    comm_groups[dp_key].append({(d, t, p, s, e) for d in range(dp)})

    for d in range(dp):
        for p in range(pp):
            for s in range(sp):
                for e in range(ep):
                    comm_groups[tp_key].append({(d, t, p, s, e) for t in range(tp)})

    for d in range(dp):
        for t in range(tp):
            for p in range(pp - 1):
                for s in range(sp):
                    for e in range(ep):
                        comm_groups[pp_key].append(
                            {(d, t, p, s, e), (d, t, p + 1, s, e)}
                        )

    for d in range(dp):
        for t in range(tp):
            for p in range(pp):
                for e in range(ep):
                    comm_groups[sp_key].append({(d, t, p, s, e) for s in range(sp)})

    for d in range(dp):
        for t in range(tp):
            for p in range(pp):
                for s in range(sp):
                    comm_groups[ep_key].append({(d, t, p, s, e) for e in range(ep)})

    return comm_groups


@dataclass
class ValidationResult:
    passed: bool = False
    failed_pp_edges: List[dict] = field(default_factory=list)
    failed_comm_groups: List[dict] = field(default_factory=list)
    error_message: str = ""


class ValidationEngine:
    def __init__(self, physical_topology: PhysicalTopology):
        self.physical_topology = physical_topology

    def validate(
        self,
        placement: Dict[Role, int],
        dp,
        tp,
        pp,
        sp,
        ep,
    ) -> ValidationResult:
        """Validate a placement.

        placement: {role_tuple -> physical_id}
        Returns ValidationResult with passed=True if all checks pass.
        """
        result = ValidationResult()

        expected_npus = math.prod((dp, tp, pp, sp, ep))
        if len(placement) != expected_npus:
            result.error_message = (
                f"Placement has {len(placement)} entries, expected {expected_npus}"
            )
            return result

        physical_ids = set(placement.values())
        if len(physical_ids) != expected_npus:
            result.error_message = (
                f"Placement has {len(physical_ids)} unique physical_ids, "
                f"expected {expected_npus}"
            )
            return result

        for d in range(dp):
            for t in range(tp):
                for p in range(pp - 1):
                    for s in range(sp):
                        for e in range(ep):
                            src_role = (d, t, p, s, e)
                            dst_role = (d, t, p + 1, s, e)
                            src_phy = placement[src_role]
                            dst_phy = placement[dst_role]
                            if not self.physical_topology.has_direct_link(
                                src_phy, dst_phy
                            ):
                                result.failed_pp_edges.append(
                                    {
                                        "src_role": src_role,
                                        "dst_role": dst_role,
                                        "physical_src": src_phy,
                                        "physical_dst": dst_phy,
                                    }
                                )

        comm_groups = generate_comm_groups(dp, tp, pp, sp, ep)
        for dim, groups in comm_groups.items():
            if dim == "pp":
                continue
            for idx, group in enumerate(groups):
                physical_ids_in_group = {placement[role] for role in group}
                if not self.physical_topology.is_induced_subgraph_connected(
                    physical_ids_in_group
                ):
                    result.failed_comm_groups.append(
                        {
                            "dimension": dim,
                            "group_index": idx,
                            "members": list(group),
                            "physical_ids": sorted(list(physical_ids_in_group)),
                        }
                    )

        result.passed = (
            len(result.failed_pp_edges) == 0 and len(result.failed_comm_groups) == 0
        )
        if not result.passed:
            parts = []
            if result.failed_pp_edges:
                parts.append(
                    f"{len(result.failed_pp_edges)} PP edge(s) not directly connected"
                )
            if result.failed_comm_groups:
                parts.append(
                    f"{len(result.failed_comm_groups)} comm group(s) not connected"
                )
            result.error_message = "; ".join(parts)

        return result


class PlacementEngine:
    def __init__(
        self,
        dp,
        tp,
        pp,
        sp,
        ep,
        physical_topology: Optional[PhysicalTopology] = None,
        dim_names=None,
    ):
        self.dp = dp
        self.tp = tp
        self.pp = pp
        self.sp = sp
        self.ep = ep
        self.physical_topology = physical_topology
        self.num_npus = math.prod((dp, tp, pp, sp, ep))
        self.validator = ValidationEngine(physical_topology) if physical_topology else None
        self.dim_names = dim_names

    def generate_all_placements(self) -> List[Dict[Role, int]]:
        """Generate all valid placements.

        If physical_topology is provided, only return placements that pass validation.
        If not provided, return the identity-like placement(s).
        """
        readable_ranks = generate_readable_ranks(
            self.dp, self.tp, self.pp, self.sp, self.ep, dim_names=self.dim_names
        )
        physical = [self.dp, self.tp, self.pp, self.sp, self.ep]

        expanded_mappings, _ = LogicalToPhysicalRankMapper.generate_all_readable_mappings(
            readable_ranks, physical
        )

        placements = []
        for mapping in expanded_mappings:
            placement = {}
            for readable_rank, physical_rank in mapping.items():
                role = role_from_readable(readable_rank, self.dim_names)
                placement[role] = physical_rank

            if self.validator:
                result = self.validator.validate(
                    placement, self.dp, self.tp, self.pp, self.sp, self.ep
                )
                if result.passed:
                    placements.append(placement)
            else:
                placements.append(placement)

        return placements

    def find_placement(self, max_retries: int = 1000) -> Optional[Dict[Role, int]]:
        """Find a valid placement, trying up to max_retries candidates."""
        readable_ranks = generate_readable_ranks(
            self.dp, self.tp, self.pp, self.sp, self.ep, dim_names=self.dim_names
        )
        physical = [self.dp, self.tp, self.pp, self.sp, self.ep]

        expanded_mappings, _ = LogicalToPhysicalRankMapper.generate_all_readable_mappings(
            readable_ranks, physical
        )

        for i, mapping in enumerate(expanded_mappings):
            if i >= max_retries:
                break
            placement = {}
            for readable_rank, physical_rank in mapping.items():
                role = role_from_readable(readable_rank, self.dim_names)
                placement[role] = physical_rank

            if self.validator:
                result = self.validator.validate(
                    placement, self.dp, self.tp, self.pp, self.sp, self.ep
                )
                if result.passed:
                    return placement
            else:
                return placement

        return None


def write_placement_json(placement, dp, tp, pp, sp, ep, output_path):
    """Write placement.json in the specified schema."""
    data = {
        "schema_version": "1.0",
        "num_npus": math.prod((dp, tp, pp, sp, ep)),
        "parallelism": {
            "dp": dp,
            "tp": tp,
            "pp": pp,
            "sp": sp,
            "ep": ep,
        },
        "placement": [
            {
                "physical_id": physical_id,
                "dp": role[0],
                "tp": role[1],
                "pp": role[2],
                "sp": role[3],
                "ep": role[4],
            }
            for role, physical_id in sorted(placement.items())
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_feasibility_report(
    failed_pp_edges, failed_comm_groups, retries, output_path
):
    data = {
        "schema_version": "1.0",
        "failure_reason": "placement_validation_failed",
        "retries": retries,
        "failed_pp_edges": failed_pp_edges,
        "failed_comm_groups": failed_comm_groups,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
