#!/usr/bin/env python3
"""Benchmark readout stagger timing on Qubex hardware.

The benchmark runs two calibration-style circuits for each requested stagger:

* ground: prepare |00...0> and measure all selected qubits
* excited: apply an X/pi pulse to every selected qubit, then measure all

It reports all-state accuracy and per-qubit assignment accuracy for both
prepared states.  Use this to sweep small readout offsets within a shared
readout multiplexing group.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit, transpile

from qiskit_qubex_provider import (
    QubexProvider,
    build_device_topology,
    build_device_topology_svg,
    label_to_qid,
)

from bell_state import (
    DEFAULT_DEVICE_ID,
    DEFAULT_QUBIT_LABELS,
    _apply_native_gate_durations,
    make_experiment,
    topology_json_path,
)


HERE = Path(__file__).resolve().parent
DEFAULT_STAGGERS_NS = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DEFAULT_SHOTS = 2000


def generate_readout_topology(
    *,
    config_root: Path,
    device_id: str,
    qubit_labels: tuple[str, ...],
    output_path: Path,
    pulse_source=None,
) -> Path:
    """Generate and write a topology subset without requiring a coupling edge."""
    device_config = config_root / device_id
    full_topology = build_device_topology(
        name=f"{device_id}-{'-'.join(qubit_labels)}-readout",
        device_id=device_id,
        calib_note_path=device_config / "calibration" / "calib_note.json",
        params_dir=device_config / "params",
        qubit_fidelity_range=(0.0, 100.0),
        coupling_fidelity_range=(0.0, 100.0),
        readout_fidelity_range=(0.0, 100.0),
        only_maximum_connected=False,
    )

    qids = tuple(label_to_qid(label) for label in qubit_labels)
    labels_by_id = dict(zip(qids, qubit_labels, strict=True))
    logical_id_by_physical = {
        physical_id: index for index, physical_id in enumerate(qids)
    }

    qubits = []
    for qubit in full_topology["qubits"]:
        physical_id = int(qubit["physical_id"])
        if physical_id not in logical_id_by_physical:
            continue
        qubit = dict(qubit)
        qubit["id"] = logical_id_by_physical[physical_id]
        qubit["label"] = labels_by_id[physical_id]
        qubits.append(qubit)

    couplings = []
    for coupling in full_topology["couplings"]:
        control = int(coupling["control"])
        target = int(coupling["target"])
        if control not in logical_id_by_physical or target not in logical_id_by_physical:
            continue
        coupling = dict(coupling)
        coupling["control"] = logical_id_by_physical[control]
        coupling["target"] = logical_id_by_physical[target]
        couplings.append(coupling)

    if len(qubits) != len(qubit_labels):
        found = {str(qubit["label"]) for qubit in qubits}
        missing = sorted(set(qubit_labels) - found)
        raise ValueError(f"Selected qubits are missing from topology: {missing}")

    topology = {
        **full_topology,
        "qubits": qubits,
        "couplings": couplings,
    }
    _apply_native_gate_durations(topology, pulse_source=pulse_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(topology, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".svg").write_text(
        build_device_topology_svg(topology),
        encoding="utf-8",
    )
    return output_path


def make_readout_circuit(num_qubits: int, *, prepared_state: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits, num_qubits)
    if prepared_state == 1:
        circuit.x(range(num_qubits))
    elif prepared_state != 0:
        raise ValueError("prepared_state must be 0 or 1.")
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit


def summarize_counts(
    counts: Mapping[str, int],
    *,
    prepared_state: int,
    qubit_labels: Sequence[str],
    shots: int,
) -> dict[str, Any]:
    num_qubits = len(qubit_labels)
    normalized = {
        _normalize_count_key(key, num_qubits): int(count)
        for key, count in counts.items()
    }
    expected_key = str(prepared_state) * num_qubits
    per_qubit_correct = {label: 0 for label in qubit_labels}
    for bitstring, count in normalized.items():
        for index, label in enumerate(qubit_labels):
            # Qiskit displays classical bit 0 as the rightmost bit.
            if bitstring[-1 - index] == str(prepared_state):
                per_qubit_correct[label] += count
    return {
        "prepared_state": prepared_state,
        "expected_key": expected_key,
        "all_state_accuracy": normalized.get(expected_key, 0) / shots,
        "per_qubit_accuracy": {
            label: correct / shots for label, correct in per_qubit_correct.items()
        },
        "counts": dict(sorted(normalized.items())),
    }


def benchmark_stagger(
    *,
    experiment: Any,
    topology_path: Path,
    qubit_labels: tuple[str, ...],
    device_id: str,
    stagger_ns: float,
    readout_stagger_mode: str,
    readout_multiplex_groups: dict[str, str] | None,
    shots: int,
    scheduling_method: str,
) -> dict[str, Any]:
    provider = QubexProvider.from_experiment(
        experiment,
        name=f"{device_id}-readout-stagger-{stagger_ns:g}ns",
        device_topology=topology_path,
        qubit_labels=qubit_labels,
        native=True,
        readout_stagger_ns=stagger_ns,
        readout_stagger_mode=readout_stagger_mode,
        readout_multiplex_groups=readout_multiplex_groups,
        execute_options={
            "state_classification": False,
            "time_integration": True,
            "plot": False,
        },
    )
    backend = provider.get_backend()
    circuits = [
        transpile(
            make_readout_circuit(len(qubit_labels), prepared_state=prepared_state),
            backend,
            scheduling_method=scheduling_method,
            optimization_level=1,
        )
        for prepared_state in (0, 1)
    ]
    backend.validate(circuits)
    job = backend.run(circuits, shots=shots)
    result = job.result()
    summaries = [
        summarize_counts(
            result.get_counts(index),
            prepared_state=prepared_state,
            qubit_labels=qubit_labels,
            shots=shots,
        )
        for index, prepared_state in enumerate((0, 1))
    ]
    return {
        "stagger_ns": stagger_ns,
        "job_id": job.job_id(),
        "conditions": summaries,
    }


def flatten_rows(results: Sequence[Mapping[str, Any]], qubit_labels: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for condition in result["conditions"]:
            row = {
                "stagger_ns": result["stagger_ns"],
                "job_id": result["job_id"],
                "prepared_state": condition["prepared_state"],
                "all_state_accuracy": condition["all_state_accuracy"],
                "expected_key": condition["expected_key"],
            }
            for label in qubit_labels:
                row[f"{label}_accuracy"] = condition["per_qubit_accuracy"][label]
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    *,
    output_dir: Path,
    device_id: str,
    result_tag: str,
    results: Sequence[Mapping[str, Any]],
    qubit_labels: Sequence[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not results:
        return written

    prefix = output_dir / f"readout-stagger-{device_id.lower()}-{result_tag}"
    overview = _make_overview_figure(results)
    written.extend(_write_figure(overview, prefix.with_name(f"{prefix.name}-overview")))

    per_qubit = _make_per_qubit_figure(results, qubit_labels)
    written.extend(_write_figure(per_qubit, prefix.with_name(f"{prefix.name}-per-qubit")))
    return written


def _make_overview_figure(results: Sequence[Mapping[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for prepared_state in (0, 1):
        points = [
            (
                float(result["stagger_ns"]),
                float(condition["all_state_accuracy"]),
            )
            for result in results
            for condition in result["conditions"]
            if int(condition["prepared_state"]) == prepared_state
        ]
        points.sort(key=lambda item: item[0])
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in points],
                y=[point[1] for point in points],
                mode="lines+markers",
                name=f"prepared {prepared_state}: all-state",
            )
        )
    _style_accuracy_figure(
        fig,
        title="Readout stagger overview",
        yaxis_title="all-state accuracy",
    )
    return fig


def _make_per_qubit_figure(
    results: Sequence[Mapping[str, Any]],
    qubit_labels: Sequence[str],
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("prepared 0", "prepared 1"),
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    for column, prepared_state in enumerate((0, 1), start=1):
        for label in qubit_labels:
            points = [
                (
                    float(result["stagger_ns"]),
                    float(condition["per_qubit_accuracy"][label]),
                )
                for result in results
                for condition in result["conditions"]
                if int(condition["prepared_state"]) == prepared_state
            ]
            points.sort(key=lambda item: item[0])
            fig.add_trace(
                go.Scatter(
                    x=[point[0] for point in points],
                    y=[point[1] for point in points],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
    _style_accuracy_figure(
        fig,
        title="Readout stagger per-qubit assignment accuracy",
        yaxis_title="per-qubit accuracy",
    )
    fig.update_xaxes(title_text="readout stagger step [ns]", row=1, col=1)
    fig.update_xaxes(title_text="readout stagger step [ns]", row=1, col=2)
    return fig


def _style_accuracy_figure(fig: go.Figure, *, title: str, yaxis_title: str) -> None:
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=520,
        legend_title_text="condition",
    )
    fig.update_xaxes(title_text="readout stagger step [ns]")
    fig.update_yaxes(title_text=yaxis_title, range=[0.0, 1.02])


def _write_figure(fig: go.Figure, base_path: Path) -> list[Path]:
    html_path = base_path.with_suffix(".html")
    png_path = base_path.with_suffix(".png")
    fig.write_html(html_path, include_plotlyjs="cdn")
    written = [html_path]
    try:
        fig.write_image(png_path, scale=2)
    except Exception as exc:  # pragma: no cover - depends on local kaleido setup.
        print(f"could not write {png_path}: {exc}")
    else:
        written.append(png_path)
    return written


def _normalize_count_key(key: str, num_qubits: int) -> str:
    compact = str(key).replace(" ", "")
    if compact.startswith("0x"):
        return format(int(compact, 16), f"0{num_qubits}b")[-num_qubits:]
    return compact.zfill(num_qubits)[-num_qubits:]


def _labels(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        raise argparse.ArgumentTypeError("expected at least one comma-separated label")
    return labels


def _float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("stagger values must be non-negative")
    return values


def _multiplex_groups(value: str | None) -> dict[str, str] | None:
    if value is None or not value.strip():
        return None
    groups: dict[str, str] = {}
    for group_index, group_text in enumerate(value.split(";")):
        labels = [label.strip() for label in group_text.split(",") if label.strip()]
        if not labels:
            continue
        group_id = f"mux{group_index}"
        for label in labels:
            groups[label] = group_id
    return groups or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep readout stagger timing and report assignment accuracy.",
    )
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path(os.environ.get("QUBEX_CONFIG_ROOT", HERE / "qubex-config")),
        help="Directory containing DEVICE_ID/config, params, and calibration folders. "
        "Can also be set with QUBEX_CONFIG_ROOT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "generated",
        help="Directory for generated topology and benchmark outputs.",
    )
    parser.add_argument("--qubits", type=_labels, default=DEFAULT_QUBIT_LABELS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--classifier-shots", type=int, default=None)
    parser.add_argument("--staggers-ns", type=_float_list, default=DEFAULT_STAGGERS_NS)
    parser.add_argument(
        "--readout-multiplex-groups",
        type=_multiplex_groups,
        default=None,
        help=(
            "Optional semicolon-separated mux groups, for example "
            "'Q00,Q01,Q02,Q03;Q04,Q05,Q06,Q07'. If omitted, Qubex readout "
            "resource metadata is used when available."
        ),
    )
    parser.add_argument(
        "--readout-stagger-mode",
        choices=("start", "sequential"),
        default="start",
        help=(
            "'start' offsets readout starts by N ns. 'sequential' starts each "
            "readout after the previous readout in the same mux group ends, "
            "plus N ns."
        ),
    )
    parser.add_argument(
        "--scheduling-method",
        choices=("alap", "asap"),
        default="alap",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to Qubex hardware and execute. Without this, only validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be positive.")
    classifier_shots = args.classifier_shots or args.shots
    if classifier_shots <= 0:
        raise ValueError("--classifier-shots must be positive.")

    config_root = args.config_root.resolve()
    output_dir = args.output_dir.resolve()
    topology_path = topology_json_path(
        output_dir=output_dir,
        device_id=args.device_id,
        qubit_labels=args.qubits,
    )

    experiment = make_experiment(
        config_root=config_root,
        device_id=args.device_id,
        qubit_labels=args.qubits,
    )
    generate_readout_topology(
        config_root=config_root,
        device_id=args.device_id,
        qubit_labels=args.qubits,
        output_path=topology_path,
        pulse_source=experiment,
    )

    print(f"topology: {topology_path}")
    print(f"qubits: {', '.join(args.qubits)}")
    print(f"readout stagger mode: {args.readout_stagger_mode}")
    print(f"staggers ns: {', '.join(f'{value:g}' for value in args.staggers_ns)}")
    print(f"shots: {args.shots}")
    print(f"classifier shots: {classifier_shots}")
    if args.readout_multiplex_groups:
        print(f"readout multiplex groups: {json.dumps(args.readout_multiplex_groups, sort_keys=True)}")

    if not args.execute:
        provider = QubexProvider.from_experiment(
            experiment,
            name=f"{args.device_id}-readout-stagger-dry-run",
            device_topology=topology_path,
            qubit_labels=args.qubits,
            native=True,
            readout_stagger_ns=args.staggers_ns[0],
            readout_stagger_mode=args.readout_stagger_mode,
            readout_multiplex_groups=args.readout_multiplex_groups,
        )
        backend = provider.get_backend()
        circuits = [
            transpile(
                make_readout_circuit(len(args.qubits), prepared_state=prepared_state),
                backend,
                scheduling_method=args.scheduling_method,
                optimization_level=1,
            )
            for prepared_state in (0, 1)
        ]
        backend.validate(circuits)
        print("dry run validated ground/excited circuits; pass --execute to run hardware sweep")
        return

    results = []
    print("connecting to Qubex hardware...")
    try:
        experiment.connect()
        classifier_provider = QubexProvider.from_experiment(
            experiment,
            name=f"{args.device_id}-readout-stagger-classifier",
            device_topology=topology_path,
            qubit_labels=args.qubits,
            native=True,
        )
        print("building classifiers...")
        classifier_provider.build_classifier(targets=list(args.qubits), shots=classifier_shots)
        for stagger_ns in args.staggers_ns:
            print(f"running stagger_ns={stagger_ns:g}...")
            result = benchmark_stagger(
                experiment=experiment,
                topology_path=topology_path,
                qubit_labels=args.qubits,
                device_id=args.device_id,
                stagger_ns=stagger_ns,
                readout_stagger_mode=args.readout_stagger_mode,
                readout_multiplex_groups=args.readout_multiplex_groups,
                shots=args.shots,
                scheduling_method=args.scheduling_method,
            )
            results.append(result)
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        disconnect = getattr(experiment, "disconnect", None)
        if callable(disconnect):
            disconnect()

    output_dir.mkdir(parents=True, exist_ok=True)
    result_tag = args.readout_stagger_mode
    json_path = output_dir / f"readout-stagger-{args.device_id.lower()}-{result_tag}-results.json"
    csv_path = output_dir / f"readout-stagger-{args.device_id.lower()}-{result_tag}-summary.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = flatten_rows(results, args.qubits)
    write_csv(csv_path, rows)
    plot_paths = write_plots(
        output_dir=output_dir,
        device_id=args.device_id,
        result_tag=result_tag,
        results=results,
        qubit_labels=args.qubits,
    )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    for plot_path in plot_paths:
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
