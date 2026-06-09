"""Generate Device Gateway topology metadata from Qubex calibration files."""

from __future__ import annotations

import argparse
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
    params_dir: str | Path | None = None,
    topology: Mapping[str, Any] | None = None,
    topology_json: str | Path | None = None,
    name: str = "anemone",
    device_id: str = "anemone",
    qubits: Iterable[int] | None = None,
    exclude_couplings: Iterable[tuple[int, int]] | None = None,
    qubit_fidelity_metric: str = "x90_gate_fidelity",
    coupling_fidelity_metric: str = "zx90_gate_fidelity",
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
    **build_options: Any,
) -> dict[str, Any]:
    """Build and write Device Gateway topology metadata."""
    topology = build_device_topology(**build_options)
    Path(output_json).write_text(
        json.dumps(topology, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return topology


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate device-topology.json from Qubex calibration files.",
    )
    parser.add_argument("--calib-note", required=True, help="Path to calib_note.json.")
    parser.add_argument("--params-dir", help="Path to the Qubex params directory.")
    parser.add_argument("--topology-json", help="Optional physical topology JSON.")
    parser.add_argument("--output-json", default="device-topology.json")
    parser.add_argument("--name", default="anemone")
    parser.add_argument("--device-id", default="anemone")
    parser.add_argument("--qubits", help="Comma-separated physical qubit ids.")
    parser.add_argument(
        "--exclude-couplings",
        help="Comma-separated physical couplings like 0-1,2-3.",
    )
    parser.add_argument("--qubit-fidelity-metric", default="x90_gate_fidelity")
    parser.add_argument("--coupling-fidelity-metric", default="zx90_gate_fidelity")
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
        calib_note_path=args.calib_note,
        params_dir=args.params_dir,
        topology_json=args.topology_json,
        name=args.name,
        device_id=args.device_id,
        qubits=_parse_int_list(args.qubits),
        exclude_couplings=_parse_coupling_list(args.exclude_couplings),
        qubit_fidelity_metric=args.qubit_fidelity_metric,
        coupling_fidelity_metric=args.coupling_fidelity_metric,
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
    readout_fidelity_range: tuple[float, float],
) -> bool:
    fidelity = _metric_value(metrics, qubit_fidelity_metric, qid, num_qubits, default=0.25)
    readout = _metric_value(
        metrics,
        "average_readout_fidelity",
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
            default=0.25,
        )
        if _in_range(fidelity, coupling_fidelity_range):
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
    return {
        "control": id_by_qid[control],
        "target": id_by_qid[target],
        "fidelity": _coupling_metric_value(
            metrics,
            coupling_fidelity_metric,
            control,
            target,
            num_qubits,
            default=0.25,
        ),
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
