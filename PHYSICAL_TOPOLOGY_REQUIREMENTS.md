# FlexET-to-SCSP Physical Topology Interface Requirements

## Overview

This document defines the interface contract for FlexET to supply physical constellation topology data to SCSP in a future iteration.

**Current iteration:** SCSP does **not** depend on FlexET for physical topology. The fields `distance_km`, `num_sats`, `npus_per_sat`, and `feasible` are provided directly by the user in the SCSP JSON configuration. `normalize_raw_config` in `scsp/config.py` already accepts and processes these fields.

**Future iteration:** FlexET shall output a `physical_topology.json` file that SCSP can read and pass directly to `isl_sim.py`.

## Schema

```json
{
  "num_sats": 16,
  "npus_per_sat": 4,
  "distance_km": [[0, 100, null], [100, 0, 150], [null, 150, 0]],
  "feasible": [[true, true, false], [true, true, true], [false, true, true]]
}
```

## Field Semantics and Types

| Field | Type | Description |
|---|---|---|
| `num_sats` | `int` | Total number of satellites in the constellation. |
| `npus_per_sat` | `int` | Number of NPUs per satellite. |
| `distance_km` | `list[list[float \| null]]` | Square matrix of pairwise satellite distances in **km**. `distance_km[i][j]` is the distance between satellite `i` and satellite `j`. A value of `null` indicates that no direct ISL edge exists between the two satellites. |
| `feasible` | `list[list[bool]]` | Square matrix indicating whether the ISL between two satellites is feasible. A value of `false` means the link is blocked (e.g., by Earth occlusion, antenna pointing limits, or other geometric constraints), even if a distance is present. |

## Units

- **Distance:** kilometers (`km`)
- **Feasibility:** boolean (`true` / `false`)

## Null Semantics

- `distance_km[i][j] == null` explicitly denotes the absence of a direct ISL edge between satellite `i` and satellite `j`.
- If `distance_km[i][j]` is `null` or `0`, or if `feasible[i][j]` is `false`, SCSP `isl_sim.compute_isl_edges` will **not** generate an edge for that pair.

## Integration with SCSP `isl_sim.py`

SCSP provides `scsp/isl_sim.py::compute_isl_edges` which consumes exactly this schema:

```python
compute_isl_edges(
    distance_km: list[list[float | None]],
    isl_link_Gbps: float,
    feasible: list[list[bool]] | None = None,
) -> list[dict]
```

The SCSP bridge can read `physical_topology.json`, extract its fields, and pass them directly to `compute_isl_edges` along with the user-supplied `isl_link_Gbps`. The returned edge list is then fed into `build_network_yaml` to produce the ASTRA-sim `network.yml`.

## Current Status

- SCSP `normalize_raw_config` already accepts `distance_km`, `num_sats`, `npus_per_sat`, and `feasible` from the user-provided JSON config.
- Integration with a FlexET-generated `physical_topology.json` is a **future work item**. No changes to FlexET code are required in the current iteration.
