"""Dynamical decoupling helpers for Qubex-backed Qiskit circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import PassManager, Target
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    ContextAwareDynamicalDecoupling,
    PadDynamicalDecoupling,
)


def build_dynamical_decoupling_pass_manager(
    backend: Any | None = None,
    *,
    target: Target | None = None,
    sequence: str | Sequence[Gate] = "xy4",
    scheduling_method: str = "alap",
    qubits: Iterable[int] | None = None,
    spacing: Sequence[float] | None = None,
    skip_reset_qubits: bool = True,
    pulse_alignment: int = 1,
    extra_slack_distribution: str = "middle",
    context_aware: bool = False,
    min_duration: int | None = None,
    skip_dd_threshold: float = 1.0,
) -> PassManager:
    """Return a pass manager that schedules and inserts dynamical decoupling.

    Args:
        backend: Backend with a Qiskit ``target``. Optional when ``target`` is supplied.
        target: Explicit Qiskit target. Takes precedence over ``backend.target``.
        sequence: DD gate sequence. Built-in names are ``"xx"``, ``"xy4"``,
            and ``"x"``. A concrete gate sequence may also be supplied.
        scheduling_method: ``"alap"`` or ``"asap"``.
        qubits: Optional physical qubit indices to receive DD.
        spacing: Optional relative spacing for ``PadDynamicalDecoupling``.
        skip_reset_qubits: Whether to skip initial/reset idle windows.
        pulse_alignment: Delay alignment for ``PadDynamicalDecoupling`` when
            ``context_aware`` is false. Ignored when target supplies alignment.
        extra_slack_distribution: Slack placement for ``PadDynamicalDecoupling``.
        context_aware: Use Qiskit's context-aware X-sequence DD pass. This pass
            accounts for coupling-map context around CX/ECR-like interactions.
        min_duration: Minimum delay duration in dt for context-aware DD.
        skip_dd_threshold: Context-aware DD occupancy threshold.
    """
    resolved_target = target or getattr(backend, "target", None)
    if resolved_target is None:
        raise ValueError("backend or target is required for dynamical decoupling.")

    passes = [_schedule_analysis_pass(scheduling_method, resolved_target)]
    if context_aware:
        passes.append(
            ContextAwareDynamicalDecoupling(
                resolved_target,
                min_duration=min_duration,
                skip_reset_qubits=skip_reset_qubits,
                skip_dd_threshold=skip_dd_threshold,
            )
        )
    else:
        passes.append(
            PadDynamicalDecoupling(
                target=resolved_target,
                dd_sequence=_dd_sequence(sequence),
                qubits=list(qubits) if qubits is not None else None,
                spacing=list(spacing) if spacing is not None else None,
                skip_reset_qubits=skip_reset_qubits,
                pulse_alignment=pulse_alignment,
                extra_slack_distribution=extra_slack_distribution,
            )
        )
    return PassManager(passes)


def build_topology_aware_dynamical_decoupling_pass_manager(
    backend: Any | None = None,
    *,
    target: Target | None = None,
    scheduling_method: str = "alap",
    skip_reset_qubits: bool = True,
    min_duration: int | None = None,
    skip_dd_threshold: float = 1.0,
) -> PassManager:
    """Return a topology-aware dynamical decoupling pass manager.

    This is a convenience wrapper around Qiskit's
    ``ContextAwareDynamicalDecoupling``. It uses the backend/target coupling map
    to choose mutually orthogonal X-sequence DD on adjacent qubits and around
    CX/ECR-like interactions. It is topology-aware, but not a global optimizer
    over all possible DD sequences.
    """
    return build_dynamical_decoupling_pass_manager(
        backend,
        target=target,
        scheduling_method=scheduling_method,
        skip_reset_qubits=skip_reset_qubits,
        context_aware=True,
        min_duration=min_duration,
        skip_dd_threshold=skip_dd_threshold,
    )


def _schedule_analysis_pass(method: str, target: Target) -> Any:
    normalized = method.lower()
    if normalized == "alap":
        return ALAPScheduleAnalysis(target=target)
    if normalized == "asap":
        return ASAPScheduleAnalysis(target=target)
    raise ValueError("scheduling_method must be 'alap' or 'asap'.")


def _dd_sequence(sequence: str | Sequence[Gate]) -> list[Gate]:
    if isinstance(sequence, str):
        normalized = sequence.lower()
        if normalized == "xx":
            return [XGate(), XGate()]
        if normalized == "xy4":
            return [XGate(), YGate(), XGate(), YGate()]
        if normalized in {"x", "hahn"}:
            return [XGate()]
        raise ValueError("sequence must be 'xx', 'xy4', 'x', or a gate sequence.")
    return list(sequence)
