"""Generate Device Gateway topology metadata from Qubex calibration files."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_QUBIT_LABEL_PATTERN = re.compile(r"^Q(\d+)$")
_COUPLING_LABEL_PATTERN = re.compile(r"^Q(\d+)-Q(\d+)$")


def qid_to_label(qid: int, num_qubits: int) -> str:
    """Return the Qubex label for a physical qubit id."""
    return f"Q{qid:0{max(2, len(str(num_qubits)))}d}"


def label_to_qid(label: str) -> int:
    """Parse a Qubex qubit label such as ``Q00``."""
    match = _QUBIT_LABEL_PATTERN.match(label)
    if match is None:
        raise ValueError(f"Invalid Qubex qubit label {label!r}.")
    return int(match.group(1))


def build_device_topology(
    *,
    calib_note_path: str | Path,
    request: Mapping[str, Any] | None = None,
    params_dir: str | Path | None = None,
    topology: Mapping[str, Any] | None = None,
    topology_json: str | Path | None = None,
    name: str = "anemone",
    device_id: str = "anemone",
    qubits: Iterable[int] | None = None,
    exclude_couplings: Iterable[tuple[int, int]] | None = None,
    qubit_fidelity_metric: str = "x90_gate_fidelity",
    coupling_fidelity_metric: str = "zx90_gate_fidelity",
    readout_fidelity_metric: str = "average_readout_fidelity",
    qubit_fidelity_range: tuple[float, float] = (0.0, 1.0),
    coupling_fidelity_range: tuple[float, float] = (0.0, 1.0),
    readout_fidelity_range: tuple[float, float] = (0.0, 1.0),
    only_maximum_connected: bool = True,
) -> dict[str, Any]:
    """Build a Device Gateway ``device-topology.json`` compatible mapping.

    The generator accepts the same core Qubex inputs used by Device Gateway:
    ``calib_note.json`` supplies pulse durations and CR calibration entries,
    while the optional params directory supplies metric YAML files whose
    ``data:`` section maps Qubex labels such as ``Q00`` and ``Q00-Q01`` to
    numeric values.
    """
    if request is not None:
        request_options = _device_topology_request_options(request)
        name = request_options.get("name", name)
        device_id = request_options.get("device_id", device_id)
        qubits = request_options.get("qubits", qubits)
        exclude_couplings = request_options.get("exclude_couplings", exclude_couplings)
        qubit_fidelity_metric = request_options.get(
            "qubit_fidelity_metric",
            qubit_fidelity_metric,
        )
        coupling_fidelity_metric = request_options.get(
            "coupling_fidelity_metric",
            coupling_fidelity_metric,
        )
        readout_fidelity_metric = request_options.get(
            "readout_fidelity_metric",
            readout_fidelity_metric,
        )
        qubit_fidelity_range = request_options.get(
            "qubit_fidelity_range",
            qubit_fidelity_range,
        )
        coupling_fidelity_range = request_options.get(
            "coupling_fidelity_range",
            coupling_fidelity_range,
        )
        readout_fidelity_range = request_options.get(
            "readout_fidelity_range",
            readout_fidelity_range,
        )
        only_maximum_connected = request_options.get(
            "only_maximum_connected",
            only_maximum_connected,
        )
    calib_note = json.loads(Path(calib_note_path).read_text(encoding="utf-8"))
    metrics = _load_metrics(
        params_dir,
        {
            qubit_fidelity_metric,
            coupling_fidelity_metric,
            "t1",
            "t1_average",
            "t2_echo",
            "t2_echo_average",
            readout_fidelity_metric,
            "readout_fidelity_0",
            "readout_fidelity_1",
            "average_readout_fidelity",
        },
    )
    physical_qids = _infer_physical_qids(calib_note, metrics, qubits)
    topology_data = _load_topology_data(
        topology=topology,
        topology_json=topology_json,
        physical_qids=physical_qids,
        calib_note=calib_note,
    )
    excluded = set(exclude_couplings or ())

    selected_qubits = [
        qid
        for qid in physical_qids
        if _qubit_passes_filters(
            qid,
            len(physical_qids),
            metrics,
            qubit_fidelity_metric=qubit_fidelity_metric,
            qubit_fidelity_range=qubit_fidelity_range,
            readout_fidelity_metric=readout_fidelity_metric,
            readout_fidelity_range=readout_fidelity_range,
        )
    ]
    selected_set = set(selected_qubits)
    couplings = _infer_couplings(
        calib_note,
        topology_data=topology_data,
        selected_qids=selected_set,
        num_qubits=len(physical_qids),
        metrics=metrics,
        coupling_fidelity_metric=coupling_fidelity_metric,
        coupling_fidelity_range=coupling_fidelity_range,
        excluded=excluded,
    )
    if only_maximum_connected:
        selected_set = _largest_connected_component(selected_set, couplings)
        selected_qubits = [qid for qid in selected_qubits if qid in selected_set]
        couplings = [
            coupling
            for coupling in couplings
            if coupling[0] in selected_set and coupling[1] in selected_set
        ]

    id_by_qid = {qid: index for index, qid in enumerate(selected_qubits)}
    return {
        "name": name,
        "device_id": device_id,
        "qubits": [
            _build_qubit_entry(
                qid,
                logical_id=id_by_qid[qid],
                num_qubits=len(physical_qids),
                calib_note=calib_note,
                metrics=metrics,
                topology_data=topology_data,
                qubit_fidelity_metric=qubit_fidelity_metric,
            )
            for qid in selected_qubits
        ],
        "couplings": [
            _build_coupling_entry(
                control,
                target,
                id_by_qid=id_by_qid,
                num_qubits=len(physical_qids),
                calib_note=calib_note,
                metrics=metrics,
                coupling_fidelity_metric=coupling_fidelity_metric,
            )
            for control, target in couplings
        ],
        "calibrated_at": _calibrated_at(calib_note),
    }


def write_device_topology(
    output_json: str | Path,
    output_image: str | Path | bool | None = None,
    **build_options: Any,
) -> dict[str, Any]:
    """Build and write Device Gateway topology metadata."""
    topology = build_device_topology(**build_options)
    output_json_path = Path(output_json)
    output_json_path.write_text(
        json.dumps(topology, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if output_image is not False:
        image_path = (
            output_json_path.with_suffix(".svg")
            if output_image is None or output_image is True
            else Path(output_image)
        )
        write_device_topology_image(topology, image_path)
    return topology


def write_device_topology_image(
    topology: Mapping[str, Any],
    output_image: str | Path,
) -> None:
    """Write a topology visualization for Device Gateway topology metadata.

    Plotly is used when available. Static image formats such as PNG and SVG
    require Plotly's Kaleido backend. If Plotly is not installed and the output
    path ends in ``.svg``, a dependency-free SVG fallback is written.
    """
    output_path = Path(output_image)
    suffix = output_path.suffix.lower()
    try:
        figure = build_device_topology_figure(topology)
        if suffix == ".html":
            figure.write_html(str(output_path), include_plotlyjs="cdn")
        else:
            figure.write_image(str(output_path), scale=2)
    except (ImportError, ValueError) as exc:
        if suffix == ".html":
            raise RuntimeError(
                "Writing topology plots to HTML requires Plotly. Install the "
                "'plot' extra: pip install 'qiskit-qubex-provider[plot]'."
            ) from exc
        if suffix != ".svg":
            raise RuntimeError(
                "Writing topology plots to static image formats requires Plotly "
                "and a working Kaleido backend. Install the 'plot' extra and use "
                "--output-image device-topology.html if Kaleido is unavailable."
            ) from exc
        output_path.write_text(build_device_topology_svg(topology), encoding="utf-8")


def build_device_topology_figure(topology: Mapping[str, Any]):
    """Return a Plotly figure for Device Gateway topology metadata."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Topology plotting requires the optional 'plot' extra: "
            "pip install 'qiskit-qubex-provider[plot]'."
        ) from exc

    qubits = [
        qubit
        for qubit in topology.get("qubits", [])
        if isinstance(qubit, Mapping) and "id" in qubit
    ]
    couplings = [
        coupling
        for coupling in topology.get("couplings", [])
        if isinstance(coupling, Mapping)
        and "control" in coupling
        and "target" in coupling
    ]
    positions = _plot_positions(qubits)
    figure = go.Figure()

    for coupling in couplings:
        control = int(coupling["control"])
        target = int(coupling["target"])
        if control not in positions or target not in positions:
            continue
        x1, y1 = positions[control]
        x2, y2 = positions[target]
        fidelity = _safe_float(coupling.get("fidelity"), default=0.0)
        duration = _safe_float(
            _mapping(coupling.get("gate_duration")).get("rzx90"),
            default=0.0,
        )
        figure.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line={"width": 2 + 5 * max(0.0, min(1.0, fidelity)), "color": "#64748b"},
                hovertemplate=(
                    f"direction: q{control} -> q{target}<br>"
                    f"coupling fidelity: {fidelity:.4f}<br>"
                    f"rzx90 duration: {duration:.0f} ns<extra></extra>"
                ),
                showlegend=False,
            )
        )
        _add_direction_annotation(figure, x1, y1, x2, y2, fidelity)

    if qubits:
        physical_ids = [int(qubit.get("physical_id", qubit["id"])) for qubit in qubits]
        label_base = _label_width_base(qubits)
        x_values = [positions[int(qubit["id"])][0] for qubit in qubits]
        y_values = [positions[int(qubit["id"])][1] for qubit in qubits]
        fidelities = [_safe_float(qubit.get("fidelity"), default=0.0) for qubit in qubits]
        labels = [qid_to_label(physical_id, label_base) for physical_id in physical_ids]
        hover_text = [
            _qubit_hover_text(qubit, label, fidelity)
            for qubit, label, fidelity in zip(qubits, labels, fidelities, strict=True)
        ]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                text=labels,
                textposition="middle center",
                textfont={"color": "white", "size": 12},
                marker={
                    "size": 42,
                    "color": fidelities,
                    "colorscale": "Viridis",
                    "cmin": min(fidelities),
                    "cmax": max(fidelities),
                    "line": {"color": "white", "width": 2},
                    "colorbar": {"title": "Qubit fidelity"},
                },
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>",
                name="qubits",
            )
        )

    name = str(topology.get("name") or topology.get("device_id") or "Qubex")
    figure.update_layout(
        title={
            "text": (
                f"{name} topology: {len(qubits)} qubits / "
                f"{len(couplings)} directed couplings"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        width=1200,
        height=900,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
        margin={"l": 40, "r": 40, "t": 90, "b": 40},
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        showlegend=False,
    )
    return figure


def build_device_topology_svg(topology: Mapping[str, Any]) -> str:
    """Return an SVG visualization for Device Gateway topology metadata."""
    qubits = [
        qubit
        for qubit in topology.get("qubits", [])
        if isinstance(qubit, Mapping) and "id" in qubit
    ]
    couplings = [
        coupling
        for coupling in topology.get("couplings", [])
        if isinstance(coupling, Mapping)
        and "control" in coupling
        and "target" in coupling
    ]
    width = 960
    height = 720
    margin = 88
    positions = _svg_positions(qubits, width=width, height=height, margin=margin)
    name = html.escape(str(topology.get("name") or topology.get("device_id") or "Qubex"))
    calibrated_at = html.escape(str(topology.get("calibrated_at") or ""))

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{name} device topology">'
        ),
        "<defs>",
        (
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" '
            'refY="3" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L0,6 L9,3 z" fill="#64748b" /></marker>'
        ),
        (
            '<filter id="shadow" x="-25%" y="-25%" width="150%" height="150%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" '
            'flood-opacity="0.18" /></filter>'
        ),
        "</defs>",
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="48" y="52" fill="#0f172a" font-family="Inter, Arial, sans-serif" '
        f'font-size="28" font-weight="700">{name}</text>',
        f'<text x="48" y="80" fill="#64748b" font-family="Inter, Arial, sans-serif" '
        f'font-size="14">{len(qubits)} qubits / {len(couplings)} directed couplings'
        f'{(" / calibrated " + calibrated_at) if calibrated_at else ""}</text>',
    ]
    for coupling in couplings:
        control = int(coupling["control"])
        target = int(coupling["target"])
        if control not in positions or target not in positions:
            continue
        x1, y1 = positions[control]
        x2, y2 = positions[target]
        fidelity = _safe_float(coupling.get("fidelity"), default=0.0)
        color = _fidelity_color(fidelity)
        line_start, line_end = _shortened_line(x1, y1, x2, y2, radius=28)
        parts.append(
            f'<line x1="{line_start[0]:.2f}" y1="{line_start[1]:.2f}" '
            f'x2="{line_end[0]:.2f}" y2="{line_end[1]:.2f}" '
            f'stroke="{color}" stroke-width="5" stroke-linecap="round" '
            f'marker-end="url(#arrow)" opacity="0.82">'
            f'<title>q{control} -> q{target}, fidelity {fidelity:.4f}</title></line>'
        )
    for qubit in qubits:
        qubit_id = int(qubit["id"])
        x, y = positions[qubit_id]
        physical_id = int(qubit.get("physical_id", qubit_id))
        fidelity = _safe_float(qubit.get("fidelity"), default=0.0)
        color = _fidelity_color(fidelity)
        label = html.escape(qid_to_label(physical_id, _label_width_base(qubits)))
        parts.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="30" fill="{color}" '
                f'stroke="#ffffff" stroke-width="4" filter="url(#shadow)">'
                f'<title>{label}, logical q{qubit_id}, fidelity {fidelity:.4f}</title>'
                "</circle>",
                f'<text x="{x:.2f}" y="{y + 5:.2f}" text-anchor="middle" '
                f'fill="#ffffff" font-family="Inter, Arial, sans-serif" '
                f'font-size="15" font-weight="700">{label}</text>',
                f'<text x="{x:.2f}" y="{y + 48:.2f}" text-anchor="middle" '
                f'fill="#334155" font-family="Inter, Arial, sans-serif" '
                f'font-size="12">q{qubit_id} / {fidelity:.3f}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate device-topology.json from Qubex calibration files.",
    )
    parser.add_argument("--calib-note", required=True, help="Path to calib_note.json.")
    parser.add_argument(
        "--request-json",
        help="Optional QDash-style DeviceTopologyRequest JSON file.",
    )
    parser.add_argument("--params-dir", help="Path to the Qubex params directory.")
    parser.add_argument("--topology-json", help="Optional physical topology JSON.")
    parser.add_argument("--output-json", default="device-topology.json")
    parser.add_argument(
        "--output-image",
        help="Path to write an SVG topology image. Defaults to output-json with .svg.",
    )
    parser.add_argument(
        "--no-output-image",
        action="store_true",
        help="Do not write the topology SVG image.",
    )
    parser.add_argument("--name", default="anemone")
    parser.add_argument("--device-id", default="anemone")
    parser.add_argument("--qubits", help="Comma-separated physical qubit ids.")
    parser.add_argument(
        "--exclude-couplings",
        help="Comma-separated physical couplings like 0-1,2-3.",
    )
    parser.add_argument("--qubit-fidelity-metric", default="x90_gate_fidelity")
    parser.add_argument("--coupling-fidelity-metric", default="zx90_gate_fidelity")
    parser.add_argument("--readout-fidelity-metric", default="average_readout_fidelity")
    parser.add_argument("--qubit-fidelity-range", default="0:1")
    parser.add_argument("--coupling-fidelity-range", default="0:1")
    parser.add_argument("--readout-fidelity-range", default="0:1")
    parser.add_argument(
        "--keep-disconnected",
        action="store_true",
        help="Keep all filtered qubits instead of extracting the largest component.",
    )
    args = parser.parse_args(argv)

    write_device_topology(
        args.output_json,
        output_image=False if args.no_output_image else args.output_image,
        calib_note_path=args.calib_note,
        request=_load_request_json(args.request_json),
        params_dir=args.params_dir,
        topology_json=args.topology_json,
        name=args.name,
        device_id=args.device_id,
        qubits=_parse_int_list(args.qubits),
        exclude_couplings=_parse_coupling_list(args.exclude_couplings),
        qubit_fidelity_metric=args.qubit_fidelity_metric,
        coupling_fidelity_metric=args.coupling_fidelity_metric,
        readout_fidelity_metric=args.readout_fidelity_metric,
        qubit_fidelity_range=_parse_range(args.qubit_fidelity_range),
        coupling_fidelity_range=_parse_range(args.coupling_fidelity_range),
        readout_fidelity_range=_parse_range(args.readout_fidelity_range),
        only_maximum_connected=not args.keep_disconnected,
    )
    return 0


def _load_metrics(
    params_dir: str | Path | None,
    metric_names: Iterable[str],
) -> dict[str, dict[str, Any]]:
    if params_dir is None:
        return {}
    base = Path(params_dir)
    metrics: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        path = base / f"{name}.yaml"
        if path.exists():
            metrics[name] = _load_metric_yaml(path)
    return metrics


def _load_metric_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    in_data = False
    data_indent: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip())
        stripped = without_comment.strip()
        if stripped == "data:":
            in_data = True
            data_indent = None
            continue
        if not in_data:
            continue
        if data_indent is None:
            data_indent = indent
        elif indent < data_indent:
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().strip("'\"")
        if key:
            data[key] = _parse_yaml_scalar(value.strip())
    return data


def _parse_yaml_scalar(value: str) -> Any:
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _infer_physical_qids(
    calib_note: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    explicit_qubits: Iterable[int] | None,
) -> list[int]:
    if explicit_qubits is not None:
        return sorted({int(qid) for qid in explicit_qubits})
    qids: set[int] = set()
    for section_name in ("drag_hpi_params", "drag_pi_params"):
        for label in _mapping(calib_note.get(section_name)):
            qid = _maybe_label_to_qid(label)
            if qid is not None:
                qids.add(qid)
    for label in _mapping(calib_note.get("cr_params")):
        pair = _parse_coupling_label(label)
        if pair is not None:
            qids.update(pair)
    for metric in metrics.values():
        for label in metric:
            qid = _maybe_label_to_qid(label)
            if qid is not None:
                qids.add(qid)
    if not qids:
        raise ValueError("Could not infer qubits; pass explicit qubits.")
    return sorted(qids)


def _load_topology_data(
    *,
    topology: Mapping[str, Any] | None,
    topology_json: str | Path | None,
    physical_qids: Sequence[int],
    calib_note: Mapping[str, Any],
) -> dict[str, Any]:
    if topology is not None and topology_json is not None:
        raise ValueError("Pass either topology or topology_json, not both.")
    if topology_json is not None:
        topology = json.loads(Path(topology_json).read_text(encoding="utf-8"))
    if topology is None:
        generated = _generate_square_lattice_topology(physical_qids)
        cr_couplings = _cr_couplings(calib_note)
        if cr_couplings:
            generated["couplings"] = cr_couplings
        return generated
    return _normalize_topology(topology, physical_qids)


def _generate_square_lattice_topology(physical_qids: Sequence[int]) -> dict[str, Any]:
    if not physical_qids:
        return {"positions": {}, "couplings": []}
    side = math.ceil(math.sqrt(len(physical_qids)))
    positions = {
        qid: {"row": index // side, "col": index % side}
        for index, qid in enumerate(physical_qids)
    }
    position_to_qid = {(pos["row"], pos["col"]): qid for qid, pos in positions.items()}
    couplings: list[tuple[int, int]] = []
    for qid, pos in positions.items():
        for delta_row, delta_col in ((1, 0), (0, 1)):
            neighbor = position_to_qid.get((pos["row"] + delta_row, pos["col"] + delta_col))
            if neighbor is not None:
                couplings.append((qid, neighbor))
                couplings.append((neighbor, qid))
    return {"positions": positions, "couplings": couplings}


def _normalize_topology(
    topology: Mapping[str, Any],
    physical_qids: Sequence[int],
) -> dict[str, Any]:
    positions: dict[int, dict[str, float]] = {}
    qubits = topology.get("qubits")
    if isinstance(qubits, Mapping):
        for key, value in qubits.items():
            qid = int(key)
            positions[qid] = _normalize_position(value)
    elif isinstance(qubits, list):
        for entry in qubits:
            if isinstance(entry, Mapping):
                qid = int(entry.get("physical_id", entry.get("id")))
                positions[qid] = _normalize_position(entry.get("position", entry))
    if not positions:
        positions = _generate_square_lattice_topology(physical_qids)["positions"]

    couplings: list[tuple[int, int]] = []
    for entry in topology.get("couplings", []):
        pair = _normalize_coupling(entry)
        if pair is not None:
            couplings.append(pair)
    if not couplings:
        couplings = _generate_square_lattice_topology(physical_qids)["couplings"]
    return {"positions": positions, "couplings": couplings}


def _normalize_position(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        if "row" in value or "col" in value:
            return {"row": float(value.get("row", 0)), "col": float(value.get("col", 0))}
        if "x" in value or "y" in value:
            return {"row": float(value.get("y", 0)), "col": float(value.get("x", 0))}
    return {"row": 0.0, "col": 0.0}


def _normalize_coupling(entry: Any) -> tuple[int, int] | None:
    if isinstance(entry, Mapping):
        if "control" in entry and "target" in entry:
            return int(entry["control"]), int(entry["target"])
        nodes = entry.get("nodes") or entry.get("qubits")
        if isinstance(nodes, Sequence) and len(nodes) == 2:
            return int(nodes[0]), int(nodes[1])
    if isinstance(entry, Sequence) and not isinstance(entry, str) and len(entry) == 2:
        return int(entry[0]), int(entry[1])
    return None


def _qubit_passes_filters(
    qid: int,
    num_qubits: int,
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    qubit_fidelity_metric: str,
    qubit_fidelity_range: tuple[float, float],
    readout_fidelity_metric: str,
    readout_fidelity_range: tuple[float, float],
) -> bool:
    fidelity = _metric_value(metrics, qubit_fidelity_metric, qid, num_qubits, default=0.25)
    readout = _metric_value(
        metrics,
        readout_fidelity_metric,
        qid,
        num_qubits,
        default=1.0,
    )
    return _in_range(fidelity, qubit_fidelity_range) and _in_range(
        readout,
        readout_fidelity_range,
    )


def _infer_couplings(
    calib_note: Mapping[str, Any],
    *,
    topology_data: Mapping[str, Any],
    selected_qids: set[int],
    num_qubits: int,
    metrics: Mapping[str, Mapping[str, Any]],
    coupling_fidelity_metric: str,
    coupling_fidelity_range: tuple[float, float],
    excluded: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cr_params = _mapping(calib_note.get("cr_params"))
    for control, target in topology_data.get("couplings", []):
        if control not in selected_qids or target not in selected_qids:
            continue
        if (control, target) in excluded:
            continue
        cr_entry = _cr_entry(cr_params, control, target, num_qubits)
        if cr_entry is None:
            continue
        if float(cr_entry.get("duration", 0) or 0) <= 0:
            continue
        fidelity = _coupling_metric_value(
            metrics,
            coupling_fidelity_metric,
            control,
            target,
            num_qubits,
            default=None,
        )
        if fidelity is not None and _in_range(float(fidelity), coupling_fidelity_range):
            result.append((control, target))
    return sorted(dict.fromkeys(result))


def _build_qubit_entry(
    qid: int,
    *,
    logical_id: int,
    num_qubits: int,
    calib_note: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    topology_data: Mapping[str, Any],
    qubit_fidelity_metric: str,
) -> dict[str, Any]:
    label = qid_to_label(qid, num_qubits)
    position = _scaled_position(qid, topology_data)
    return {
        "id": logical_id,
        "physical_id": qid,
        "position": position,
        "fidelity": _metric_value(metrics, qubit_fidelity_metric, qid, num_qubits, 0.25),
        "meas_error": {
            "prob_meas1_prep0": 1.0
            - _metric_value(metrics, "readout_fidelity_0", qid, num_qubits, 1.0),
            "prob_meas0_prep1": 1.0
            - _metric_value(metrics, "readout_fidelity_1", qid, num_qubits, 1.0),
            "readout_assignment_error": 1.0
            - _metric_value(metrics, "average_readout_fidelity", qid, num_qubits, 1.0),
        },
        "qubit_lifetime": {
            "t1": _metric_value(metrics, "t1", qid, num_qubits, None)
            or _metric_value(metrics, "t1_average", qid, num_qubits, 100.0),
            "t2": _metric_value(metrics, "t2_echo", qid, num_qubits, None)
            or _metric_value(metrics, "t2_echo_average", qid, num_qubits, 100.0),
        },
        "gate_duration": {
            "rz": 0,
            "sx": _duration(calib_note, "drag_hpi_params", label, default=20),
            "x": _duration(calib_note, "drag_pi_params", label, default=20),
        },
    }


def _build_coupling_entry(
    control: int,
    target: int,
    *,
    id_by_qid: Mapping[int, int],
    num_qubits: int,
    calib_note: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    coupling_fidelity_metric: str,
) -> dict[str, Any]:
    fidelity = _coupling_metric_value(
        metrics,
        coupling_fidelity_metric,
        control,
        target,
        num_qubits,
        default=None,
    )
    if fidelity is None:
        raise ValueError(
            f"Missing {coupling_fidelity_metric!r} for coupling "
            f"{qid_to_label(control, num_qubits)}-{qid_to_label(target, num_qubits)}."
        )
    return {
        "control": id_by_qid[control],
        "target": id_by_qid[target],
        "fidelity": float(fidelity),
        "gate_duration": {
            "rzx90": _coupling_duration(calib_note, control, target, num_qubits),
        },
    }


def _metric_value(
    metrics: Mapping[str, Mapping[str, Any]],
    metric_name: str,
    qid: int,
    num_qubits: int,
    default: Any,
) -> Any:
    value = metrics.get(metric_name, {}).get(qid_to_label(qid, num_qubits), default)
    return default if value is None else value


def _coupling_metric_value(
    metrics: Mapping[str, Mapping[str, Any]],
    metric_name: str,
    control: int,
    target: int,
    num_qubits: int,
    default: Any,
) -> Any:
    metric = metrics.get(metric_name, {})
    forward = f"{qid_to_label(control, num_qubits)}-{qid_to_label(target, num_qubits)}"
    reverse = f"{qid_to_label(target, num_qubits)}-{qid_to_label(control, num_qubits)}"
    value = metric.get(forward, metric.get(reverse, default))
    return default if value is None else value


def _duration(
    calib_note: Mapping[str, Any],
    section_name: str,
    label: str,
    *,
    default: int,
) -> int:
    entry = _mapping(calib_note.get(section_name)).get(label)
    if not isinstance(entry, Mapping):
        return default
    return int(entry.get("duration", default))


def _coupling_duration(
    calib_note: Mapping[str, Any],
    control: int,
    target: int,
    num_qubits: int,
) -> int:
    cr_params = _mapping(calib_note.get("cr_params"))
    entry = _cr_entry(cr_params, control, target, num_qubits)
    if entry is None:
        return 0
    return int(entry.get("duration", 0))


def _cr_entry(
    cr_params: Mapping[str, Any],
    control: int,
    target: int,
    num_qubits: int,
) -> Mapping[str, Any] | None:
    forward = f"{qid_to_label(control, num_qubits)}-{qid_to_label(target, num_qubits)}"
    reverse = f"{qid_to_label(target, num_qubits)}-{qid_to_label(control, num_qubits)}"
    entry = cr_params.get(forward, cr_params.get(reverse))
    return entry if isinstance(entry, Mapping) else None


def _scaled_position(
    qid: int,
    topology_data: Mapping[str, Any],
) -> dict[str, float]:
    positions = topology_data.get("positions", {})
    position = positions.get(qid, {"row": 0.0, "col": 0.0})
    rows = [float(pos["row"]) for pos in positions.values()] or [0.0]
    cols = [float(pos["col"]) for pos in positions.values()] or [0.0]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    return {
        "x": 0.0 if col_span == 0 else (float(position["col"]) - min(cols)) / col_span,
        "y": 0.0 if row_span == 0 else (float(position["row"]) - min(rows)) / row_span,
    }


def _largest_connected_component(
    qubits: set[int],
    couplings: Sequence[tuple[int, int]],
) -> set[int]:
    if not qubits:
        return set()
    neighbors = {qid: set() for qid in qubits}
    for control, target in couplings:
        if control in neighbors and target in neighbors:
            neighbors[control].add(target)
            neighbors[target].add(control)
    components: list[set[int]] = []
    unseen = set(qubits)
    while unseen:
        root = unseen.pop()
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current] - component:
                component.add(neighbor)
                unseen.discard(neighbor)
                stack.append(neighbor)
        components.append(component)
    return max(components, key=lambda component: (len(component), -min(component)))


def _calibrated_at(calib_note: Mapping[str, Any]) -> str:
    for key in ("calibrated_at", "updated_at", "created_at", "timestamp"):
        value = calib_note.get(key)
        if isinstance(value, str):
            return value
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_request_json(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _device_topology_request_options(request: Mapping[str, Any]) -> dict[str, Any]:
    condition = _mapping(request.get("condition"))
    options: dict[str, Any] = {}
    if "name" in request:
        options["name"] = str(request["name"])
    if "device_id" in request:
        options["device_id"] = str(request["device_id"])
    if "qubits" in request and request["qubits"] is not None:
        options["qubits"] = [_parse_request_qid(qid) for qid in request["qubits"]]
    if "exclude_couplings" in request and request["exclude_couplings"] is not None:
        options["exclude_couplings"] = _parse_request_couplings(
            request["exclude_couplings"],
        )
    qubit_fidelity = _mapping(condition.get("qubit_fidelity"))
    if qubit_fidelity:
        options["qubit_fidelity_range"] = _request_range(qubit_fidelity)
        metric = qubit_fidelity.get("metric")
        if metric:
            options["qubit_fidelity_metric"] = str(metric)
    coupling_fidelity = _mapping(condition.get("coupling_fidelity"))
    if coupling_fidelity:
        options["coupling_fidelity_range"] = _request_range(coupling_fidelity)
        metric = coupling_fidelity.get("metric")
        if metric:
            options["coupling_fidelity_metric"] = str(metric)
    readout_fidelity = _mapping(condition.get("readout_fidelity"))
    if readout_fidelity:
        options["readout_fidelity_range"] = _request_range(readout_fidelity)
        metric = readout_fidelity.get("metric")
        if metric:
            options["readout_fidelity_metric"] = str(metric)
    if "only_maximum_connected" in condition:
        options["only_maximum_connected"] = bool(condition["only_maximum_connected"])
    return options


def _request_range(condition: Mapping[str, Any]) -> tuple[float, float]:
    return (
        _safe_float(condition.get("min"), default=0.0),
        _safe_float(condition.get("max"), default=1.0),
    )


def _parse_request_qid(value: Any) -> int:
    if isinstance(value, str):
        qid = _maybe_label_to_qid(value)
        if qid is not None:
            return qid
    return int(value)


def _parse_request_couplings(values: Iterable[Any]) -> list[tuple[int, int]]:
    couplings: list[tuple[int, int]] = []
    for value in values:
        if isinstance(value, str):
            left, right = value.strip().split("-", 1)
            couplings.append((_parse_request_qid(left), _parse_request_qid(right)))
        elif isinstance(value, Sequence) and len(value) == 2:
            couplings.append((_parse_request_qid(value[0]), _parse_request_qid(value[1])))
        else:
            raise ValueError(f"Invalid coupling request entry {value!r}.")
    return couplings


def _plot_positions(qubits: Sequence[Mapping[str, Any]]) -> dict[int, tuple[float, float]]:
    raw = {
        int(qubit["id"]): (
            _safe_float(_mapping(qubit.get("position")).get("x"), default=0.0),
            _safe_float(_mapping(qubit.get("position")).get("y"), default=0.0),
        )
        for qubit in qubits
    }
    if not raw:
        return {}
    xs = [value[0] for value in raw.values()]
    ys = [value[1] for value in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = max(span_x, span_y, 1.0)
    return {
        qubit_id: (
            100 * (0.5 if span_x == 0 else (x - min_x) / scale),
            -100 * (0.5 if span_y == 0 else (y - min_y) / scale),
        )
        for qubit_id, (x, y) in raw.items()
    }


def _add_direction_annotation(
    figure: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    fidelity: float,
) -> None:
    start, end = _shortened_line(x1, y1, x2, y2, radius=2.8)
    figure.add_annotation(
        x=end[0],
        y=end[1],
        ax=start[0],
        ay=start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2 + 4 * max(0.0, min(1.0, fidelity)),
        arrowcolor=_fidelity_color(fidelity),
        opacity=0.9,
    )


def _qubit_hover_text(
    qubit: Mapping[str, Any],
    label: str,
    fidelity: float,
) -> str:
    lifetime = _mapping(qubit.get("qubit_lifetime"))
    meas_error = _mapping(qubit.get("meas_error"))
    gate_duration = _mapping(qubit.get("gate_duration"))
    return (
        f"{html.escape(label)}<br>"
        f"logical q{int(qubit['id'])}<br>"
        f"qubit fidelity: {fidelity:.4f}<br>"
        f"readout fidelity: "
        f"{1.0 - _safe_float(meas_error.get('readout_assignment_error'), default=1.0):.4f}"
        f"<br>t1: {_safe_float(lifetime.get('t1'), default=0.0):.2f} us"
        f"<br>t2: {_safe_float(lifetime.get('t2'), default=0.0):.2f} us"
        f"<br>sx: {_safe_float(gate_duration.get('sx'), default=0.0):.0f} ns"
        f"<br>x: {_safe_float(gate_duration.get('x'), default=0.0):.0f} ns"
    )


def _svg_positions(
    qubits: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    margin: int,
) -> dict[int, tuple[float, float]]:
    if not qubits:
        return {}
    raw_positions: dict[int, tuple[float, float]] = {}
    for index, qubit in enumerate(qubits):
        qubit_id = int(qubit["id"])
        position = qubit.get("position")
        if isinstance(position, Mapping):
            raw_positions[qubit_id] = (
                _safe_float(position.get("x"), default=0.0),
                _safe_float(position.get("y"), default=0.0),
            )
        else:
            angle = 2 * math.pi * index / len(qubits)
            raw_positions[qubit_id] = (0.5 + 0.45 * math.cos(angle), 0.5 + 0.45 * math.sin(angle))
    xs = [position[0] for position in raw_positions.values()]
    ys = [position[1] for position in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    drawable_width = width - 2 * margin
    drawable_height = height - 2 * margin - 48
    y_offset = margin + 48
    return {
        qubit_id: (
            margin + (0.5 if span_x == 0 else (x - min_x) / span_x) * drawable_width,
            y_offset + (0.5 if span_y == 0 else (y - min_y) / span_y) * drawable_height,
        )
        for qubit_id, (x, y) in raw_positions.items()
    }


def _shortened_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    if distance == 0:
        return (x1, y1), (x2, y2)
    ux = dx / distance
    uy = dy / distance
    return (x1 + ux * radius, y1 + uy * radius), (x2 - ux * radius, y2 - uy * radius)


def _safe_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fidelity_color(fidelity: float) -> str:
    if fidelity >= 0.98:
        return "#16a34a"
    if fidelity >= 0.95:
        return "#65a30d"
    if fidelity >= 0.90:
        return "#d97706"
    return "#dc2626"


def _label_width_base(qubits: Sequence[Mapping[str, Any]]) -> int:
    physical_ids = [
        int(qubit.get("physical_id", qubit.get("id", 0)))
        for qubit in qubits
        if isinstance(qubit, Mapping)
    ]
    return (max(physical_ids) + 1) if physical_ids else len(qubits)


def _maybe_label_to_qid(label: str) -> int | None:
    match = _QUBIT_LABEL_PATTERN.match(str(label))
    return int(match.group(1)) if match else None


def _parse_coupling_label(label: str) -> tuple[int, int] | None:
    match = _COUPLING_LABEL_PATTERN.match(str(label))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _cr_couplings(calib_note: Mapping[str, Any]) -> list[tuple[int, int]]:
    couplings = []
    for label in _mapping(calib_note.get("cr_params")):
        pair = _parse_coupling_label(label)
        if pair is not None:
            couplings.append(pair)
    return sorted(dict.fromkeys(couplings))


def _in_range(value: Any, bounds: tuple[float, float]) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bounds[0] <= number <= bounds[1]


def _parse_range(value: str) -> tuple[float, float]:
    parts = value.replace(",", ":").split(":")
    if len(parts) != 2:
        raise ValueError(f"Range must be MIN:MAX, got {value!r}.")
    return float(parts[0]), float(parts[1])


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_coupling_list(value: str | None) -> list[tuple[int, int]]:
    if value is None or not value.strip():
        return []
    couplings = []
    for part in value.split(","):
        left, right = part.strip().split("-", 1)
        couplings.append((int(left), int(right)))
    return couplings


if __name__ == "__main__":
    raise SystemExit(main())
