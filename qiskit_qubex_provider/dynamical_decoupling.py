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
    pulse_interval: float | None = None,
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
            Mutually exclusive with ``pulse_interval``.
        pulse_interval: Target pulse interval in seconds. When set, each idle
            window is padded with the ``sequence`` repeated as many times as
            needed to keep roughly one pulse every ``pulse_interval`` —
            experiment-style fixed-interval DD — instead of stretching a
            single sequence block over the window. Windows too short for one
            repetition fall back to a plain delay.
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
        if pulse_interval is not None:
            raise ValueError("pulse_interval is not supported with context_aware DD.")
        passes.append(
            ContextAwareDynamicalDecoupling(
                resolved_target,
                min_duration=min_duration,
                skip_reset_qubits=skip_reset_qubits,
                skip_dd_threshold=skip_dd_threshold,
            )
        )
    elif pulse_interval is not None:
        if spacing is not None:
            raise ValueError(
                "spacing and pulse_interval are mutually exclusive; fixed-interval "
                "DD computes its own per-window spacing."
            )
        if pulse_interval <= 0:
            raise ValueError("pulse_interval must be a positive duration in seconds.")
        dt = getattr(resolved_target, "dt", None) or getattr(backend, "dt", None)
        if dt is None:
            raise ValueError(
                "pulse_interval requires the target to define dt (the sampling "
                "period in seconds)."
            )
        passes.append(
            _FixedIntervalPadDynamicalDecoupling(
                pulse_interval_dt=pulse_interval / dt,
                target=resolved_target,
                dd_sequence=_dd_sequence(sequence),
                qubits=list(qubits) if qubits is not None else None,
                skip_reset_qubits=skip_reset_qubits,
                pulse_alignment=pulse_alignment,
                extra_slack_distribution=extra_slack_distribution,
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


class _FixedIntervalPadDynamicalDecoupling(PadDynamicalDecoupling):
    """Pad each idle window by repeating the base DD sequence.

    Qiskit's ``PadDynamicalDecoupling`` stretches a single sequence block over
    every idle window, so the pulse interval grows with the window length.
    Experiments instead keep the pulse interval fixed and repeat the sequence
    (e.g. Pokharel et al., PRL 121, 220502 (2018)). This subclass picks, per
    window, the repetition count that best matches ``pulse_interval_dt`` and
    delegates the actual padding to the parent implementation with the
    repeated sequence swapped in.
    """

    def __init__(self, *, pulse_interval_dt: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pulse_interval_dt = float(pulse_interval_dt)

    def _pre_runhook(self, dag: Any) -> None:
        super()._pre_runhook(dag)
        # Snapshot the normalized base sequence; _pad swaps repeated versions
        # of these in and out per idle window.
        self._base_sequence = list(self._dd_sequence)
        self._base_lengths = {
            qubit: list(lengths) for qubit, lengths in self._dd_sequence_lengths.items()
        }
        self._base_phase = self._sequence_phase

    def _repetitions(self, qubit: Any, window: int) -> int:
        num_base = len(self._base_sequence)
        reps = max(1, round(window / (self._pulse_interval_dt * num_base)))
        if num_base % 2 == 1:
            # An odd base (e.g. a bare X) only composes to identity for an
            # even number of repetitions.
            reps += reps % 2
        lengths = self._base_lengths.get(qubit)
        if lengths is not None:
            step = 2 if num_base % 2 == 1 else 1
            floor = 2 if num_base % 2 == 1 else 1
            while reps > floor and window - reps * sum(lengths) <= 0:
                reps -= step
        return reps

    def _pad(
        self,
        dag: Any,
        qubit: Any,
        t_start: int,
        t_end: int,
        next_node: Any,
        prev_node: Any,
    ) -> None:
        reps = self._repetitions(qubit, t_end - t_start)
        saved = (
            self._dd_sequence,
            self._dd_sequence_lengths,
            self._spacing,
            self._sequence_phase,
        )
        num_pulses = len(self._base_sequence) * reps
        mid = 1 / num_pulses
        end = mid / 2
        self._dd_sequence = self._base_sequence * reps
        self._dd_sequence_lengths = {
            q: lengths * reps for q, lengths in self._base_lengths.items()
        }
        self._spacing = [end] + [mid] * (num_pulses - 1) + [end]
        self._sequence_phase = self._base_phase * reps
        try:
            super()._pad(dag, qubit, t_start, t_end, next_node, prev_node)
        finally:
            (
                self._dd_sequence,
                self._dd_sequence_lengths,
                self._spacing,
                self._sequence_phase,
            ) = saved


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
