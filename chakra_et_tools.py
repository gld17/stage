#!/usr/bin/env python3
"""
Chakra ET (v0.0.4 stream format) utilities: JSON dump and GraphML graph.

Run from the stage/ directory (same as main.py) so symbolic_tensor_graph is importable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape as _xml_escape

# -----------------------------------------------------------------------------
# Package path (stage/ contains symbolic_tensor_graph)
# -----------------------------------------------------------------------------
_STAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = _STAGE_ROOT / "results"
if str(_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_ROOT))

from google.protobuf.json_format import MessageToDict  # noqa: E402

from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.et_def.et_def_pb2 import (  # noqa: E402
    GlobalMetadata,
    Node,
    NodeType,
)
from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.protolib import (  # noqa: E402
    decodeMessage,
    openFileRd,
)

_LEGACY_COMMANDS = frozenset(
    ("chakra_jsonizer", "jsonizer", "chakra_visualizer", "visualizer")
)


@dataclass
class ParsedET:
    path: Path
    metadata: GlobalMetadata
    nodes: List[Node]


def parse_et_file(path: Path) -> ParsedET:
    """Decode varint-length-prefixed stream: GlobalMetadata then Node messages."""
    fh = openFileRd(str(path))
    try:
        meta = GlobalMetadata()
        if not decodeMessage(fh, meta):
            raise ValueError(f"Empty or unreadable ET file: {path}")
        nodes: List[Node] = []
        while True:
            n = Node()
            if not decodeMessage(fh, n):
                break
            nodes.append(n)
    finally:
        fh.close()
    return ParsedET(path=path, metadata=meta, nodes=nodes)


def _node_type_name(t: int) -> str:
    try:
        return NodeType.Name(t)
    except ValueError:
        return f"UNKNOWN({t})"


def _xml_attr(value: str) -> str:
    return _xml_escape(str(value), {'"': "&quot;"})


def _resolve_output_path(
    et_path: Path, out: Optional[str], suffix: str
) -> Optional[Path]:
    """None means write to stdout (--out-* -)."""
    if out is not None:
        if out == "-":
            return None
        return Path(out)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{et_path.stem}{suffix}"


def _write_output(path: Optional[Path], text: str, tool: str) -> None:
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    sys.stderr.write(f"[{tool}] Wrote {path}\n")


def run_jsonizer(parsed: ParsedET, out: Optional[str]) -> None:
    payload = {
        "source_et": str(parsed.path),
        "metadata": MessageToDict(
            parsed.metadata, preserving_proto_field_name=True
        ),
        "nodes": [
            MessageToDict(n, preserving_proto_field_name=True) for n in parsed.nodes
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    out_path = _resolve_output_path(parsed.path, out, ".json")
    _write_output(out_path, text, "chakra_jsonizer")


def _op_type_of(node: Node) -> str:
    for a in node.attr:
        if a.name == "op_type":
            which = a.WhichOneof("value")
            if which is None:
                return ""
            v = getattr(a, which)
            return str(v)
    return ""


# yEd 节点配色：先按 NodeType 大类区分；COMP_NODE 再按 op_type 细分
_NODE_TYPE_FILL = {
    "COMP_NODE": "#A6C8FF",
    "COMM_COLL_NODE": "#FFC069",
    "COMM_SEND_NODE": "#95DE64",
    "COMM_RECV_NODE": "#FF85C0",
    "MEM_LOAD_NODE": "#FFE58F",
    "MEM_STORE_NODE": "#D3ADF7",
    "METADATA_NODE": "#D9D9D9",
    "INVALID_NODE": "#BFBFBF",
}
_COMP_OP_FILL = {
    "M": "#4F81BD",       # MatMul
    "A": "#9BBB59",       # Add
    "SLICE": "#C0504D",
    "E": "#8064A2",       # Elementwise
    "CUSTOM": "#F79646",
}
_DEFAULT_FILL = "#F2F2F2"


def _node_fill_color(node_type_name: str, op_type: str) -> str:
    if node_type_name == "COMP_NODE" and op_type in _COMP_OP_FILL:
        return _COMP_OP_FILL[op_type]
    return _NODE_TYPE_FILL.get(node_type_name, _DEFAULT_FILL)


def _yed_node_block(
    nid: int, node_type_name: str, op_type: str, label_text: str
) -> List[str]:
    fill = _node_fill_color(node_type_name, op_type)
    label_lines = [_xml_escape(line) for line in label_text.split("\n")]
    label_inner = "&#10;".join(label_lines)
    return [
        '      <data key="d_yfiles">',
        '        <y:ShapeNode>',
        '          <y:Geometry height="48.0" width="200.0" x="0.0" y="0.0"/>',
        f'          <y:Fill color="{fill}" transparent="false"/>',
        '          <y:BorderStyle color="#000000" type="line" width="1.0"/>',
        '          <y:NodeLabel alignment="center" autoSizePolicy="content"'
        ' fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false"'
        ' hasLineColor="false" horizontalTextPosition="center"'
        ' verticalTextPosition="bottom" iconTextGap="4" modelName="internal"'
        ' modelPosition="c" textColor="#000000" visible="true">'
        f'{label_inner}</y:NodeLabel>',
        '          <y:Shape type="roundrectangle"/>',
        '        </y:ShapeNode>',
        '      </data>',
    ]


def _yed_edge_block(dep_type: str) -> List[str]:
    if dep_type == "ctrl":
        line_type = "dashed"
        color = "#FF7F0E"
    else:
        line_type = "line"
        color = "#1F77B4"
    return [
        '      <data key="d_yedges">',
        '        <y:PolyLineEdge>',
        '          <y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>',
        f'          <y:LineStyle color="{color}" type="{line_type}" width="1.5"/>',
        '          <y:Arrows source="none" target="standard"/>',
        f'          <y:EdgeLabel modelName="centered">{dep_type}</y:EdgeLabel>',
        '          <y:BendStyle smoothed="false"/>',
        '        </y:PolyLineEdge>',
        '      </data>',
    ]


def run_visualizer(
    parsed: ParsedET, out: Optional[str], max_nodes: Optional[int]
) -> None:
    # 节点数量不宜过多：边数随依赖大致线性增长，过大时 GraphML 文件会很大，可视化工具难以布局/阅读。
    # TODO: 单层或分层折叠可视化（例如按 pipeline stage / 子图聚合），便于大图浏览；
    #       visualizer 目前尚未充分测试。
    nodes = parsed.nodes
    if max_nodes is not None and max_nodes > 0 and len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        sys.stderr.write(
            f"[chakra_visualizer] Truncated to first {max_nodes} nodes "
            f"(ET has {len(parsed.nodes)} total).\n"
        )

    id_set = {int(n.id) for n in nodes}
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xmlns:y="http://www.yworks.com/xml/graphml"',
        '         xmlns:yed="http://www.yworks.com/xml/yed/3"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">',
        '  <key id="d_name"    for="node" attr.name="name"      attr.type="string"/>',
        '  <key id="d_type"    for="node" attr.name="node_type" attr.type="string"/>',
        '  <key id="d_op_type" for="node" attr.name="op_type"   attr.type="string"/>',
        '  <key id="d_label"   for="node" attr.name="label"     attr.type="string"/>',
        '  <key id="d_dep"     for="edge" attr.name="dep_type"  attr.type="string"/>',
        '  <key id="d_yfiles"  for="node" yfiles.type="nodegraphics"/>',
        '  <key id="d_yedges"  for="edge" yfiles.type="edgegraphics"/>',
        '  <graph id="ChakraET" edgedefault="directed">',
    ]

    for n in nodes:
        nid = int(n.id)
        ntype = _node_type_name(n.type)
        op_type = _op_type_of(n)
        label_text = f"{nid}: {ntype}"
        if op_type:
            label_text += f" ({op_type})"
        if n.name:
            label_text += f"\n{n.name}"
        lines.append(f'    <node id="n{nid}">')
        lines.append(f'      <data key="d_name">{_xml_attr(n.name or "")}</data>')
        lines.append(f'      <data key="d_type">{_xml_attr(ntype)}</data>')
        if op_type:
            lines.append(f'      <data key="d_op_type">{_xml_attr(op_type)}</data>')
        lines.append(f'      <data key="d_label">{_xml_attr(label_text)}</data>')
        lines.extend(_yed_node_block(nid, ntype, op_type, label_text))
        lines.append('    </node>')

    edge_idx = 0
    for n in nodes:
        tid = int(n.id)
        for dep in n.data_deps:
            d = int(dep)
            if d in id_set and tid in id_set:
                lines.append(
                    f'    <edge id="e{edge_idx}" source="n{d}" target="n{tid}">'
                )
                lines.append('      <data key="d_dep">data</data>')
                lines.extend(_yed_edge_block("data"))
                lines.append('    </edge>')
                edge_idx += 1
        for dep in n.ctrl_deps:
            d = int(dep)
            if d in id_set and tid in id_set:
                lines.append(
                    f'    <edge id="e{edge_idx}" source="n{d}" target="n{tid}">'
                )
                lines.append('      <data key="d_dep">ctrl</data>')
                lines.extend(_yed_edge_block("ctrl"))
                lines.append('    </edge>')
                edge_idx += 1

    lines.append('  </graph>')
    lines.append('</graphml>')
    text = "\n".join(lines) + "\n"
    out_path = _resolve_output_path(parsed.path, out, ".graphml")
    _write_output(out_path, text, "chakra_visualizer")


def _normalize_argv(argv: Sequence[str]) -> List[str]:
    """Map legacy subcommands to --jsonizer / --visualizer flags."""
    out = list(argv)
    if not out or out[0] not in _LEGACY_COMMANDS:
        return out

    legacy = out.pop(0)
    has_mode = any(
        a in ("--jsonizer", "--visualizer") or a.startswith("--no-")
        for a in out
    )
    if has_mode:
        return out

    if legacy in ("chakra_jsonizer", "jsonizer"):
        return ["--jsonizer", *out]
    return ["--visualizer", *out]


def _build_parser() -> argparse.ArgumentParser:
    results_name = DEFAULT_OUTPUT_DIR.name
    p = argparse.ArgumentParser(
        description=(
            "Chakra ET tools: JSON dump and/or GraphML graph. "
            "Use --jsonizer and/or --visualizer together; "
            "omit both to run both (default)."
        )
    )
    p.add_argument("--et", required=True, help="Path to .et file")
    p.add_argument(
        "--jsonizer",
        action="store_true",
        help=f"Emit readable JSON to {results_name}/<et_stem>.json",
    )
    p.add_argument(
        "--visualizer",
        action="store_true",
        help=f"Emit GraphML to {results_name}/<et_stem>.graphml",
    )
    p.add_argument(
        "--out-json",
        "--out",
        dest="out_json",
        metavar="PATH",
        help=(
            f"JSON output path (default: {results_name}/<et_stem>.json; "
            "use '-' for stdout)"
        ),
    )
    p.add_argument(
        "--out-graphml",
        "--output",
        "--out-dot",
        dest="out_graphml",
        metavar="PATH",
        help=(
            f"GraphML output path (default: {results_name}/<et_stem>.graphml; "
            "use '-' for stdout)"
        ),
    )
    p.add_argument(
        "--format",
        choices=["graphml"],
        default="graphml",
        help="Output format for visualizer (only graphml supported)",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        metavar="N",
        help="Visualizer: only include the first N nodes",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw = list(argv) if argv is not None else sys.argv[1:]
    normalized = _normalize_argv(raw)
    args = _build_parser().parse_args(normalized)

    do_json = args.jsonizer
    do_vis = args.visualizer
    if not do_json and not do_vis:
        do_json = do_vis = True

    et_path = Path(args.et)
    parsed: Optional[ParsedET] = None

    if do_json:
        parsed = parse_et_file(et_path)
        run_jsonizer(parsed, args.out_json)
    if do_vis:
        if parsed is None:
            parsed = parse_et_file(et_path)
        run_visualizer(parsed, args.out_graphml, args.max_nodes)


if __name__ == "__main__":
    main()
