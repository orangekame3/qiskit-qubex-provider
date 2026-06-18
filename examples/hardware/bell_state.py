#!/usr/bin/env python3
"""Build and optionally run a Bell circuit on a Qubex hardware device."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qiskit import QuantumCircuit, transpile

from qiskit_qubex_provider import (
    QubexProvider,
    build_device_topology,
    build_device_topology_svg,
    label_to_qid,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DEVICE_ID = "64Qv3"
DEFAULT_QUBIT_LABELS = (
    "Q24",
    "Q25",
    "Q26",
    "Q27",
    "Q28",
    "Q29",
    "Q30",
    "Q31",
    "Q40",
    "Q41",
    "Q42",
)
DEFAULT_BELL_PAIR = ("Q28", "Q25")
DEFAULT_4Q_WORKLOAD_LABELS = ("Q28", "Q25", "Q30", "Q31")
DEFAULT_4Q_CHAIN_LABELS = ("Q25", "Q28", "Q30", "Q31")
DEFAULT_SHOTS = 1000


def topology_json_path(
    *,
    output_dir: Path,
    device_id: str,
    qubit_labels: tuple[str, ...],
) -> Path:
    labels = "-".join(label.lower() for label in qubit_labels)
    return output_dir / f"device-topology-{device_id.lower()}-{labels}.json"


def generate_device_topology(
    *,
    config_root: Path,
    device_id: str,
    qubit_labels: tuple[str, ...],
    bell_pair: tuple[str, str],
    output_path: Path,
    pulse_source=None,
    calibration_valid_days: int | None = None,
) -> Path:
    """Generate and write a Device Gateway topology for the selected qubits."""
    device_config = config_root / device_id
    full_topology = build_device_topology(
        name=f"{device_id}-{'-'.join(qubit_labels)}",
        device_id=device_id,
        calib_note_path=device_config / "calibration" / "calib_note.json",
        params_dir=device_config / "params",
        qubit_fidelity_range=(0.0, 100.0),
        coupling_fidelity_range=(0.0, 100.0),
        readout_fidelity_range=(0.0, 100.0),
        only_maximum_connected=False,
        calibration_valid_days=calibration_valid_days,
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
    qubits.sort(key=lambda qubit: int(qubit["id"]))

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

    topology = {
        **full_topology,
        "qubits": qubits,
        "couplings": couplings,
    }
    _apply_native_gate_durations(topology, pulse_source=pulse_source)
    _validate_topology_subset(topology, qubit_labels, bell_pair=bell_pair)
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


def _apply_native_gate_durations(topology: dict, *, pulse_source=None) -> None:
    pulse = _resolve_pulse_source(pulse_source)
    labels_by_logical_id = {
        int(qubit["id"]): str(qubit["label"])
        for qubit in topology["qubits"]
    }

    for qubit in topology["qubits"]:
        label = str(qubit["label"])
        original = dict(qubit.get("gate_duration") or {})
        gate_duration = {"rz": 0}
        sx_duration = _pulse_duration_ns(pulse, "x90", label) if pulse is not None else None
        if sx_duration is None:
            sx_duration = original.get("sx")
        if sx_duration is not None:
            gate_duration["sx"] = int(sx_duration)
        measure_duration = (
            _pulse_duration_ns(pulse, "readout", label) if pulse is not None else None
        )
        if measure_duration is not None:
            gate_duration["measure"] = int(measure_duration)
        qubit["gate_duration"] = gate_duration

    for coupling in topology["couplings"]:
        control = int(coupling["control"])
        target = int(coupling["target"])
        original = dict(coupling.get("gate_duration") or {})
        duration = None
        if pulse is not None:
            duration = _pulse_duration_ns(
                pulse,
                "zx90",
                labels_by_logical_id[control],
                labels_by_logical_id[target],
                echo=True,
            )
        gate_duration = {"rzx90": int(duration or 0)}
        cx_duration = None
        if pulse is not None:
            cx_duration = _pulse_duration_ns(
                pulse,
                "cx",
                labels_by_logical_id[control],
                labels_by_logical_id[target],
            )
        if cx_duration is None:
            cx_duration = original.get("cx")
        if cx_duration is not None:
            gate_duration["cx"] = int(cx_duration)
        coupling["gate_duration"] = gate_duration


def _resolve_pulse_source(source):
    if source is None:
        return None
    for candidate in (
        getattr(source, "pulse", None),
        getattr(source, "pulse_service", None),
        source,
    ):
        if candidate is not None:
            return candidate
    return None


def _pulse_duration_ns(pulse, method_name: str, *args, **kwargs) -> int | None:
    method = getattr(pulse, method_name, None)
    if method is None:
        return None
    try:
        obj = method(*args, **kwargs)
    except TypeError:
        if "echo" not in kwargs:
            return None
        kwargs = dict(kwargs)
        kwargs.pop("echo", None)
        obj = method(*args, **kwargs)
    duration = getattr(obj, "cached_duration", None)
    if duration is None:
        duration = getattr(obj, "duration", None)
    if duration is None:
        return None
    return int(duration)


def make_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def make_experiment(*, config_root: Path, device_id: str, qubit_labels: tuple[str, ...]):
    try:
        from qubex import Experiment
    except ImportError as exc:
        raise ImportError("This hardware example requires qubex to be installed.") from exc

    device_config = config_root / device_id
    return Experiment(
        chip_id=device_id,
        qubits=qubit_labels,
        config_dir=device_config / "config",
        params_dir=device_config / "params",
        calib_note_path=device_config / "calibration" / "calib_note.json",
    )


def make_provider(
    *,
    experiment,
    topology_path: Path,
    qubit_labels: tuple[str, ...],
    device_id: str,
) -> QubexProvider:
    return QubexProvider.from_experiment(
        experiment,
        name=f"{device_id}-real",
        device_topology=topology_path,
        qubit_labels=qubit_labels,
        execute_options={
            "state_classification": False,
            "time_integration": True,
            "plot": False,
        },
    )


def prepare_provider(
    *,
    config_root: Path,
    device_id: str,
    qubit_labels: tuple[str, ...],
    bell_pair: tuple[str, str],
    topology_path: Path,
) -> QubexProvider:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    experiment = make_experiment(
        config_root=config_root,
        device_id=device_id,
        qubit_labels=qubit_labels,
    )
    generate_device_topology(
        config_root=config_root,
        device_id=device_id,
        qubit_labels=qubit_labels,
        bell_pair=bell_pair,
        output_path=topology_path,
        pulse_source=experiment,
    )
    return make_provider(
        experiment=experiment,
        topology_path=topology_path,
        qubit_labels=qubit_labels,
        device_id=device_id,
    )


def bell_initial_layout(
    *,
    qubit_labels: tuple[str, ...],
    bell_pair: tuple[str, str],
) -> list[int]:
    return labels_initial_layout(qubit_labels=qubit_labels, circuit_labels=bell_pair)


def labels_initial_layout(
    *,
    qubit_labels: tuple[str, ...],
    circuit_labels: tuple[str, ...],
) -> list[int]:
    index_by_label = {label: index for index, label in enumerate(qubit_labels)}
    try:
        return [index_by_label[label] for label in circuit_labels]
    except KeyError as exc:
        raise ValueError(
            f"circuit_labels={circuit_labels!r} must be contained in "
            f"qubit_labels={qubit_labels!r}."
        ) from exc


def _validate_topology_subset(
    topology: dict,
    qubit_labels: tuple[str, ...],
    *,
    bell_pair: tuple[str, str],
) -> None:
    if len(topology["qubits"]) != len(qubit_labels):
        found = tuple(qubit.get("label") for qubit in topology["qubits"])
        raise ValueError(
            f"Could not build topology for all requested qubits: requested "
            f"{qubit_labels!r}, found {found!r}."
        )

    initial_layout = bell_initial_layout(
        qubit_labels=qubit_labels,
        bell_pair=bell_pair,
    )
    requested_edge = tuple(initial_layout)
    available_edges = {
        (int(coupling["control"]), int(coupling["target"]))
        for coupling in topology["couplings"]
    }
    if requested_edge not in available_edges:
        available = ", ".join(
            f"{qubit_labels[control]}->{qubit_labels[target]}"
            for control, target in sorted(available_edges)
        )
        raise ValueError(
            f"bell_pair={bell_pair!r} is not a calibrated directed coupling. "
            f"Available couplings in qubit_labels: {available or 'none'}."
        )


def _labels(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        raise argparse.ArgumentTypeError("expected at least one comma-separated label")
    return labels


def _pair(value: str) -> tuple[str, str]:
    labels = _labels(value)
    if len(labels) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated labels")
    return labels[0], labels[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qubex hardware Bell sample.")
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
        help="Directory for generated topology JSON files.",
    )
    parser.add_argument("--qubits", type=_labels, default=DEFAULT_QUBIT_LABELS)
    parser.add_argument("--bell-pair", type=_pair, default=DEFAULT_BELL_PAIR)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to Qubex hardware and execute. Without this, only validate.",
    )
    parser.add_argument(
        "--show-circuit",
        action="store_true",
        help="Print the transpiled Qiskit circuit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_root = args.config_root.resolve()
    output_dir = args.output_dir.resolve()
    topology_path = topology_json_path(
        output_dir=output_dir,
        device_id=args.device_id,
        qubit_labels=args.qubits,
    )

    provider = prepare_provider(
        config_root=config_root,
        device_id=args.device_id,
        qubit_labels=args.qubits,
        bell_pair=args.bell_pair,
        topology_path=topology_path,
    )
    backend = provider.get_backend()

    circuit = make_circuit()
    transpiled = transpile(
        circuit,
        backend,
        initial_layout=bell_initial_layout(
            qubit_labels=args.qubits,
            bell_pair=args.bell_pair,
        ),
        scheduling_method="alap",
        optimization_level=1,
    )

    if args.show_circuit:
        print(transpiled)

    schedules = backend.validate(transpiled)
    print(f"validated {len(schedules)} schedule(s)")
    print(f"backend: {backend.name}")
    print(f"config root: {config_root}")
    print(f"topology: {topology_path}")
    print(f"topology svg: {topology_path.with_suffix('.svg')}")
    print(f"qubits: {', '.join(args.qubits)}")
    print(f"bell pair: {args.bell_pair[0]} -> {args.bell_pair[1]}")
    print(f"shots: {args.shots}")

    if not args.execute:
        print("dry run only; pass --execute to connect and run on hardware")
        return

    print("connecting to Qubex hardware...")
    experiment = make_experiment(
        config_root=config_root,
        device_id=args.device_id,
        qubit_labels=args.qubits,
    )
    try:
        experiment.connect()
        execute_provider = make_provider(
            experiment=experiment,
            topology_path=topology_path,
            qubit_labels=args.qubits,
            device_id=args.device_id,
        )
        print("building classifiers...")
        execute_provider.build_classifier(targets=list(args.bell_pair), shots=args.shots)
        execute_backend = execute_provider.get_backend()
        job = execute_backend.run(transpiled, shots=args.shots)
        print(f"job id: {job.job_id()}")
        result = job.result()
        print("counts:")
        print(json.dumps(result.get_counts(), indent=2, sort_keys=True))
    finally:
        experiment.disconnect()


if __name__ == "__main__":
    main()
