"""Conversion from Qubex system metadata to Qiskit Target objects."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

from qiskit.circuit import Delay, Measure, Parameter, Reset
from qiskit.circuit.library import (
    CXGate,
    HGate,
    IGate,
    RXGate,
    RYGate,
    RZGate,
    SXGate,
    XGate,
)
from qiskit.providers import QubitProperties
from qiskit.transpiler import InstructionProperties, Target

QubexTargetSource: TypeAlias = Any

_DEFAULT_BASIS_GATES = (
    "id",
    "rz",
    "sx",
    "x",
    "rx",
    "ry",
    "h",
    "cx",
    "measure",
    "reset",
    "delay",
)


def build_qubex_target(
    qubex: QubexTargetSource | None = None,
    *,
    num_qubits: int | None = None,
    coupling_map: Iterable[tuple[int, int]] | None = None,
    basis_gates: Iterable[str] | None = None,
    dt: float | None = 1e-9,
    description: str = "Qiskit target for Qubex",
) -> Target:
    """Build a Qiskit Target from Qubex system metadata.

    Args:
        qubex: Qubex ``ExperimentSystem``, ``QuantumSystem``, target registry,
            integer qubit count, or any object with compatible ``qubits`` /
            ``ge_targets`` / ``cr_targets`` attributes.
        num_qubits: Explicit qubit count. Required when it cannot be inferred.
        coupling_map: Optional directed two-qubit connectivity.
        basis_gates: Optional operation names to expose. Defaults to common
            single-qubit gates, ``cx``, measurement, reset, and delay.
        dt: Backend sampling period in seconds.
        description: Target description string.
    """
    qubit_labels = _infer_qubit_labels(qubex, num_qubits)
    qubit_count = len(qubit_labels)
    label_to_index = {label: index for index, label in enumerate(qubit_labels)}
    properties = _infer_qubit_properties(qubex, qubit_labels)
    edges = _infer_coupling_map(qubex, label_to_index, coupling_map)

    target = Target(
        description=description,
        num_qubits=qubit_count,
        dt=dt,
        qubit_properties=properties,
    )
    _add_operations(target, basis_gates or _DEFAULT_BASIS_GATES, qubit_count, edges)
    return target


def _infer_qubit_labels(
    qubex: QubexTargetSource | None,
    num_qubits: int | None,
) -> list[str]:
    if isinstance(qubex, int):
        if num_qubits is not None and num_qubits != qubex:
            raise ValueError("num_qubits conflicts with integer qubex source.")
        return [f"Q{i}" for i in range(qubex)]
    if num_qubits is not None:
        return [f"Q{i}" for i in range(num_qubits)]
    if qubex is None:
        raise ValueError("num_qubits is required when no Qubex source is provided.")

    qubits = _get_attr_chain(
        qubex,
        ("qubits",),
        ("quantum_system", "qubits"),
        ("chip", "qubits"),
    )
    if qubits is None:
        raise ValueError("Could not infer qubits from Qubex source; pass num_qubits.")
    return [str(getattr(qubit, "label", f"Q{index}")) for index, qubit in enumerate(qubits)]


def _infer_qubit_properties(
    qubex: QubexTargetSource | None,
    qubit_labels: Sequence[str],
) -> list[QubitProperties] | None:
    if qubex is None or isinstance(qubex, int):
        return None

    qubits = _get_attr_chain(
        qubex,
        ("qubits",),
        ("quantum_system", "qubits"),
        ("chip", "qubits"),
    )
    if qubits is None:
        return None

    by_label = {str(getattr(qubit, "label", index)): qubit for index, qubit in enumerate(qubits)}
    props: list[QubitProperties] = []
    for label in qubit_labels:
        qubit = by_label.get(label)
        if qubit is None:
            props.append(QubitProperties())
            continue
        frequency = _finite_or_none(getattr(qubit, "frequency", None))
        if frequency is not None:
            frequency *= 1e9
        props.append(QubitProperties(frequency=frequency))
    return props


def _infer_coupling_map(
    qubex: QubexTargetSource | None,
    label_to_index: dict[str, int],
    coupling_map: Iterable[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    if coupling_map is not None:
        return list(coupling_map)

    edges: set[tuple[int, int]] = set()
    cr_targets = _get_attr_chain(qubex, ("cr_targets",), ("target_registry", "cr_targets"))
    if cr_targets:
        values = cr_targets.values() if isinstance(cr_targets, dict) else cr_targets
        for target in values:
            pair = _parse_cr_target(target)
            if pair is None:
                continue
            control, target_qubit = pair
            if control in label_to_index and target_qubit in label_to_index:
                edges.add((label_to_index[control], label_to_index[target_qubit]))

    graph = _get_attr_chain(qubex, ("chip_graph",), ("quantum_system", "chip_graph"))
    undirected = getattr(graph, "qubit_undirected_edges", None)
    if undirected:
        for edge in undirected.values() if isinstance(undirected, dict) else undirected:
            labels = _edge_labels(edge)
            if labels and labels[0] in label_to_index and labels[1] in label_to_index:
                a, b = label_to_index[labels[0]], label_to_index[labels[1]]
                edges.add((a, b))
                edges.add((b, a))
    return sorted(edges)


def _add_operations(
    target: Target,
    basis_gates: Iterable[str],
    num_qubits: int,
    coupling_map: Sequence[tuple[int, int]],
) -> None:
    one_qubit_props = {(qubit,): InstructionProperties() for qubit in range(num_qubits)}
    two_qubit_props = {edge: InstructionProperties() for edge in coupling_map}
    angle = Parameter("theta")
    duration = Parameter("duration")

    factories = {
        "id": (lambda: IGate(), one_qubit_props),
        "rz": (lambda: RZGate(angle), one_qubit_props),
        "sx": (lambda: SXGate(), one_qubit_props),
        "x": (lambda: XGate(), one_qubit_props),
        "rx": (lambda: RXGate(angle), one_qubit_props),
        "ry": (lambda: RYGate(angle), one_qubit_props),
        "h": (lambda: HGate(), one_qubit_props),
        "cx": (lambda: CXGate(), two_qubit_props),
        "measure": (lambda: Measure(), one_qubit_props),
        "reset": (lambda: Reset(), one_qubit_props),
        "delay": (lambda: Delay(duration), one_qubit_props),
    }
    for gate_name in basis_gates:
        if gate_name not in factories:
            raise ValueError(f"Unsupported basis gate {gate_name!r}.")
        factory, props = factories[gate_name]
        if gate_name == "cx" and not props:
            continue
        target.add_instruction(factory(), props, name=gate_name)


def _parse_cr_target(target: Any) -> tuple[str, str] | None:
    label = getattr(target, "label", None)
    if label and "-" in label:
        left, right = label.split("-", 1)
        if left.startswith("Q") and right.startswith("Q"):
            return left, right

    obj = getattr(target, "object", None)
    control = getattr(obj, "label", None)
    target_qubit = getattr(target, "target_qubit", None)
    if control and target_qubit:
        return str(control), str(getattr(target_qubit, "label", target_qubit))
    return None


def _edge_labels(edge: Any) -> tuple[str, str] | None:
    if isinstance(edge, dict):
        label = edge.get("label")
        if isinstance(label, str) and "-" in label:
            left, right = label.split("-", 1)
            return left, right
        nodes = edge.get("nodes") or edge.get("qubits")
        if nodes and len(nodes) == 2:
            return str(nodes[0]), str(nodes[1])
    if isinstance(edge, tuple) and len(edge) >= 2:
        return f"Q{edge[0]}", f"Q{edge[1]}"
    return None


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _get_attr_chain(obj: Any, *chains: tuple[str, ...]) -> Any | None:
    for chain in chains:
        current = obj
        for attr in chain:
            if current is None or not hasattr(current, attr):
                current = None
                break
            current = getattr(current, attr)
        if current is not None:
            return current
    return None
