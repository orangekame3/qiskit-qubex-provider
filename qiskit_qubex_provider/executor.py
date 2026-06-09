"""Qubex hardware execution adapter for Qiskit circuits."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from typing import Any

from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit
from qiskit.circuit import Delay as QiskitDelay
from qiskit.result import Result

from .job import QubexJob


@dataclass(frozen=True)
class QubexCircuitExecution:
    """Qubex execution artifacts for one Qiskit circuit."""

    circuit: QuantumCircuit
    schedule: Any
    raw_result: Any
    measured_targets: tuple[str, ...]
    target_to_clbit: Mapping[str, int]


class QubexPulseExecutor:
    """Execute Qiskit circuits through Qubex pulse and measurement APIs.

    The executor converts a supported, hardware-oriented subset of Qiskit
    circuits into Qubex ``PulseSchedule`` objects. It expects a Qubex
    ``Experiment``-like object or service object that exposes calibrated pulse
    methods such as ``x90``, ``x180``, ``y90``, ``z90``, ``cx`` and an
    ``execute(schedule=..., ...)`` method either directly or through
    ``measurement_service``.
    """

    def __init__(
        self,
        qubex: Any,
        *,
        qubit_labels: Sequence[str] | None = None,
        execute_options: Mapping[str, Any] | None = None,
    ) -> None:
        if qubex is None:
            raise ValueError(
                "QubexPulseExecutor requires a Qubex Experiment-like object. "
                "Create qubex.Experiment(...) and pass it as QubexProvider(qubex=exp, use_qubex_executor=True), "
                "or pass a custom executor."
            )
        self._qubex = qubex
        self._qubit_labels = tuple(qubit_labels or self._infer_qubit_labels(qubex))
        self._execute_options = dict(execute_options or {})
        if not self._qubit_labels:
            raise ValueError(
                "qubit_labels must be supplied or inferable from the Qubex object. "
                "For production use, pass a configured qubex.Experiment with selected qubits."
            )
        self._pulse_source()

    @property
    def qubit_labels(self) -> tuple[str, ...]:
        """Return Qubex qubit labels in Qiskit physical qubit order."""
        return self._qubit_labels

    def run(self, run_input: Any, **options: Any) -> QubexJob:
        """Execute one or more Qiskit circuits on Qubex."""
        circuits = _normalize_circuits(run_input)
        shots = int(options.pop("shots", 1024))
        memory = bool(options.pop("memory", False))
        job_id = str(uuid.uuid4())
        executions = [
            self._execute_circuit(circuit, shots=shots, options=options)
            for circuit in circuits
        ]
        result = self._to_qiskit_result(
            executions,
            job_id=job_id,
            shots=shots,
            memory=memory,
        )
        return QubexJob(backend=None, job_id=job_id, result=result)

    def build_schedule(self, circuit: QuantumCircuit) -> Any:
        """Convert a supported Qiskit circuit into a Qubex PulseSchedule."""
        pulse = self._pulse_source()
        pulse_schedule_cls = _import_pulse_schedule()
        blank_cls = _import_blank()
        op_start_times = _op_start_times(circuit)
        channel_offsets: dict[str, float] = {
            label: 0.0 for label in self._qubit_labels
        }
        with pulse_schedule_cls(list(self._qubit_labels)) as schedule:
            for index, instruction in enumerate(circuit.data):
                operation = instruction.operation
                name = operation.name
                qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
                labels = [self._qubit_labels[index] for index in qubit_indices]
                start_ns = (
                    _time_to_ns(op_start_times[index], _circuit_time_unit(circuit), self._dt_seconds())
                    if op_start_times is not None
                    else None
                )

                if start_ns is not None and labels:
                    self._align_channels(
                        schedule,
                        blank_cls,
                        channel_offsets,
                        labels,
                        start_ns,
                    )

                if name == "barrier":
                    schedule.barrier(labels or None)
                    self._sync_offsets_after_barrier(schedule, channel_offsets, labels)
                elif name == "delay":
                    delay_ns = _delay_duration_ns(operation, self._dt_seconds())
                    for label in labels:
                        if delay_ns > 0:
                            schedule.add(label, blank_cls(delay_ns))
                        channel_offsets[label] = channel_offsets.get(label, 0.0) + delay_ns
                elif name == "measure":
                    continue
                elif name in {"id", "reset"}:
                    continue
                elif name == "x":
                    waveform = pulse.x180(labels[0])
                    schedule.add(labels[0], waveform)
                    self._advance_offsets(channel_offsets, labels, _duration_ns(waveform))
                elif name == "sx":
                    waveform = pulse.x90(labels[0])
                    schedule.add(labels[0], waveform)
                    self._advance_offsets(channel_offsets, labels, _duration_ns(waveform))
                elif name == "sxdg":
                    waveform = pulse.x90m(labels[0])
                    schedule.add(labels[0], waveform)
                    self._advance_offsets(channel_offsets, labels, _duration_ns(waveform))
                elif name == "y":
                    waveform = pulse.y180(labels[0])
                    schedule.add(labels[0], waveform)
                    self._advance_offsets(channel_offsets, labels, _duration_ns(waveform))
                elif name == "h":
                    waveform = pulse.hadamard(labels[0])
                    schedule.add(labels[0], waveform)
                    self._advance_offsets(channel_offsets, labels, _duration_ns(waveform))
                elif name == "s":
                    schedule.add(labels[0], pulse.z90())
                elif name == "sdg":
                    schedule.add(labels[0], self._virtual_z(-pi / 2))
                elif name == "z":
                    schedule.add(labels[0], pulse.z180())
                elif name == "rz":
                    schedule.add(labels[0], self._virtual_z(float(operation.params[0])))
                elif name == "rx":
                    duration_ns = self._add_rx(schedule, pulse, labels[0], float(operation.params[0]))
                    self._advance_offsets(channel_offsets, labels, duration_ns)
                elif name == "ry":
                    duration_ns = self._add_ry(schedule, pulse, labels[0], float(operation.params[0]))
                    self._advance_offsets(channel_offsets, labels, duration_ns)
                elif name == "cx":
                    sub_schedule = pulse.cx(labels[0], labels[1])
                    schedule.call(sub_schedule)
                    self._advance_offsets_for_schedule(channel_offsets, sub_schedule)
                elif name == "cz":
                    sub_schedule = pulse.cz(labels[0], labels[1])
                    schedule.call(sub_schedule)
                    self._advance_offsets_for_schedule(channel_offsets, sub_schedule)
                else:
                    raise ValueError(
                        f"Unsupported Qiskit instruction {name!r} for Qubex pulse execution. "
                        "Transpile to the backend target or provide a custom executor."
                    )
        return schedule

    def instruction_durations_seconds(self) -> dict[str, dict[tuple[int, ...], float]]:
        """Infer Qiskit Target instruction durations from calibrated Qubex pulses."""
        pulse = self._pulse_source()
        durations: dict[str, dict[tuple[int, ...], float]] = {}
        for index, label in enumerate(self._qubit_labels):
            qarg = (index,)
            self._set_duration(durations, "x", qarg, _duration_seconds_safe(lambda: pulse.x180(label)))
            self._set_duration(durations, "sx", qarg, _duration_seconds_safe(lambda: pulse.x90(label)))
            self._set_duration(durations, "sxdg", qarg, _duration_seconds_safe(lambda: pulse.x90m(label)))
            self._set_duration(durations, "y", qarg, _duration_seconds_safe(lambda: pulse.y180(label)))
            self._set_duration(durations, "h", qarg, _duration_seconds_safe(lambda: pulse.hadamard(label)))
            for virtual_gate in ("id", "rz", "s", "sdg", "z", "reset", "measure"):
                self._set_duration(durations, virtual_gate, qarg, 0.0)
        for control, control_label in enumerate(self._qubit_labels):
            for target, target_label in enumerate(self._qubit_labels):
                if control == target:
                    continue
                qarg = (control, target)
                self._set_duration(durations, "cx", qarg, _duration_seconds_safe(lambda c=control_label, t=target_label: pulse.cx(c, t)))
                self._set_duration(durations, "cz", qarg, _duration_seconds_safe(lambda c=control_label, t=target_label: pulse.cz(c, t)))
        return durations

    def _execute_circuit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int,
        options: Mapping[str, Any],
    ) -> QubexCircuitExecution:
        measured_targets, target_to_clbit = self._measurement_mapping(circuit)
        schedule = self.build_schedule(circuit)
        execute_options = dict(self._execute_options)
        execute_options.update(options)
        execute_options.setdefault("state_classification", True)
        execute_options.setdefault("final_measurement", True)
        execute_options.setdefault("plot", False)
        execute_options["n_shots"] = shots
        raw_result = self._execute_source().execute(schedule=schedule, **execute_options)
        return QubexCircuitExecution(
            circuit=circuit,
            schedule=schedule,
            raw_result=raw_result,
            measured_targets=tuple(measured_targets),
            target_to_clbit=target_to_clbit,
        )

    def _measurement_mapping(
        self,
        circuit: QuantumCircuit,
    ) -> tuple[list[str], dict[str, int]]:
        measured_targets: list[str] = []
        target_to_clbit: dict[str, int] = {}
        seen_non_measure_after_measure = False
        measurement_started = False
        for instruction in circuit.data:
            name = instruction.operation.name
            if name == "measure":
                measurement_started = True
                if len(instruction.qubits) != 1 or len(instruction.clbits) != 1:
                    raise ValueError("Only one-qubit Qiskit measurements are supported.")
                qubit_index = circuit.find_bit(instruction.qubits[0]).index
                clbit_index = circuit.find_bit(instruction.clbits[0]).index
                target = self._qubit_labels[qubit_index]
                measured_targets.append(target)
                target_to_clbit[target] = clbit_index
            elif measurement_started and name not in {"barrier", "delay"}:
                seen_non_measure_after_measure = True
        if seen_non_measure_after_measure:
            raise ValueError("Mid-circuit measurement is not supported by QubexPulseExecutor.")
        if not measured_targets:
            measured_targets = list(self._qubit_labels[: circuit.num_qubits])
            target_to_clbit = {
                target: index for index, target in enumerate(measured_targets)
            }
        return measured_targets, target_to_clbit

    def _to_qiskit_result(
        self,
        executions: Sequence[QubexCircuitExecution],
        *,
        job_id: str,
        shots: int,
        memory: bool,
    ) -> Result:
        results = []
        for execution in executions:
            counts, memory_values = self._qiskit_counts_and_memory(execution)
            data: dict[str, Any] = {"counts": dict(counts)}
            if memory:
                data["memory"] = memory_values
            circuit = execution.circuit
            results.append(
                {
                    "shots": shots,
                    "success": True,
                    "status": "DONE",
                    "name": circuit.name,
                    "header": _circuit_header(circuit),
                    "data": data,
                }
            )
        return Result.from_dict(
            {
                "backend_name": "qubex",
                "backend_version": "0.1.0",
                "qobj_id": job_id,
                "job_id": job_id,
                "success": True,
                "status": "COMPLETED",
                "results": results,
            }
        )

    def _qiskit_counts_and_memory(
        self,
        execution: QubexCircuitExecution,
    ) -> tuple[Counter[str], list[str]]:
        raw_counts = execution.raw_result.get_counts(execution.measured_targets)
        counts: Counter[str] = Counter()
        memory: list[str] = []
        for qubex_bitstring, count in raw_counts.items():
            hex_value = self._qubex_bitstring_to_hex(str(qubex_bitstring), execution)
            counts[hex_value] += int(count)
            memory.extend([hex_value] * int(count))
        return counts, memory

    def _qubex_bitstring_to_hex(
        self,
        bitstring: str,
        execution: QubexCircuitExecution,
    ) -> str:
        if len(bitstring) != len(execution.measured_targets):
            raise ValueError(
                "Qubex result bitstring length does not match measured target count."
            )
        value = 0
        for target, bit in zip(execution.measured_targets, bitstring):
            if bit not in {"0", "1"}:
                raise ValueError(f"Unsupported classified state {bit!r}; only 0/1 results can become Qiskit counts.")
            if bit == "1":
                value |= 1 << execution.target_to_clbit[target]
        return hex(value)

    def _pulse_source(self) -> Any:
        for candidate in (
            self._qubex,
            getattr(self._qubex, "pulse", None),
            getattr(self._qubex, "pulse_service", None),
        ):
            if candidate is not None and all(
                hasattr(candidate, name)
                for name in ("x90", "x180", "y90", "y180", "z90", "z180")
            ):
                return candidate
        raise TypeError(
            "Qubex object does not expose the calibrated pulse API required for circuit execution. "
            "Use a qubex.Experiment instance, or an object exposing x90/x180/y90/y180/z90/z180/cx/cz "
            "and execute(schedule=..., ...). A bare qubex.Measurement session is not sufficient "
            "for gate-level Qiskit circuit execution."
        )

    def _execute_source(self) -> Any:
        measurement_service = getattr(self._qubex, "measurement_service", None)
        if measurement_service is not None and hasattr(measurement_service, "execute"):
            return measurement_service
        if hasattr(self._qubex, "execute"):
            return self._qubex
        raise TypeError(
            "Qubex object does not expose execute(schedule=..., ...) or measurement_service.execute(...). "
            "Pass a configured qubex.Experiment instance or a custom executor."
        )

    def _virtual_z(self, theta: float) -> Any:
        return self._pulse_source().z90().__class__(theta)

    @staticmethod
    def _set_duration(
        durations: dict[str, dict[tuple[int, ...], float]],
        name: str,
        qarg: tuple[int, ...],
        duration: float | None,
    ) -> None:
        if duration is not None:
            durations.setdefault(name, {})[qarg] = duration

    @staticmethod
    def _add_rx(schedule: Any, pulse: Any, target: str, theta: float) -> float:
        if _is_close(theta, pi):
            waveform = pulse.x180(target)
            schedule.add(target, waveform)
            return _duration_ns(waveform)
        elif _is_close(theta, pi / 2):
            waveform = pulse.x90(target)
            schedule.add(target, waveform)
            return _duration_ns(waveform)
        elif _is_close(theta, -pi / 2):
            waveform = pulse.x90m(target)
            schedule.add(target, waveform)
            return _duration_ns(waveform)
        elif _is_close(theta, 0):
            return 0.0
        else:
            raise ValueError("QubexPulseExecutor supports rx angles of 0, +/-pi/2, and pi.")

    @staticmethod
    def _add_ry(schedule: Any, pulse: Any, target: str, theta: float) -> float:
        if _is_close(theta, pi):
            waveform = pulse.y180(target)
            schedule.add(target, waveform)
            return _duration_ns(waveform)
        elif _is_close(theta, pi / 2):
            waveform = pulse.y90(target)
            schedule.add(target, waveform)
            return _duration_ns(waveform)
        elif _is_close(theta, -pi / 2):
            waveform = pulse.y90m(target)
            schedule.add(target, waveform)
            return _duration_ns(waveform)
        elif _is_close(theta, 0):
            return 0.0
        else:
            raise ValueError("QubexPulseExecutor supports ry angles of 0, +/-pi/2, and pi.")

    @staticmethod
    def _align_channels(
        schedule: Any,
        blank_cls: type,
        channel_offsets: dict[str, float],
        labels: Sequence[str],
        start_ns: float,
    ) -> None:
        for label in labels:
            offset = channel_offsets.get(label, 0.0)
            delta = start_ns - offset
            if delta < -1e-9:
                raise ValueError(
                    f"Scheduled operation starts at {start_ns} ns on {label}, "
                    f"but the Qubex channel is already at {offset} ns."
                )
            if delta > 1e-9:
                schedule.add(label, blank_cls(delta))
                channel_offsets[label] = start_ns

    @staticmethod
    def _advance_offsets(
        channel_offsets: dict[str, float],
        labels: Sequence[str],
        duration_ns: float,
    ) -> None:
        for label in labels:
            channel_offsets[label] = channel_offsets.get(label, 0.0) + duration_ns

    @staticmethod
    def _advance_offsets_for_schedule(
        channel_offsets: dict[str, float],
        schedule: Any,
    ) -> None:
        labels = getattr(schedule, "labels", [])
        start = max((channel_offsets.get(label, 0.0) for label in labels), default=0.0)
        duration = _duration_ns(schedule)
        for label in labels:
            channel_offsets[label] = start + duration

    @staticmethod
    def _sync_offsets_after_barrier(
        schedule: Any,
        channel_offsets: dict[str, float],
        labels: Sequence[str],
    ) -> None:
        selected = labels or list(getattr(schedule, "labels", []))
        barrier_time = max(
            (channel_offsets.get(label, 0.0) for label in selected),
            default=0.0,
        )
        for label in selected:
            channel_offsets[label] = barrier_time

    @staticmethod
    def _infer_qubit_labels(qubex: Any) -> tuple[str, ...]:
        for source in (
            qubex,
            getattr(qubex, "ctx", None),
            getattr(qubex, "context", None),
            getattr(qubex, "qubex_system", None),
            getattr(qubex, "quantum_system", None),
        ):
            if source is None:
                continue
            labels = getattr(source, "qubit_labels", None)
            if labels is not None:
                return tuple(str(label) for label in labels)
            qubits = getattr(source, "qubits", None)
            if isinstance(qubits, Mapping):
                return tuple(str(label) for label in qubits)
            if qubits is not None:
                return tuple(
                    str(getattr(qubit, "label", index))
                    for index, qubit in enumerate(qubits)
                )
        return ()

    def _dt_seconds(self) -> float:
        return self.dt_seconds()

    def dt_seconds(self) -> float:
        """Return the Qubex sampling period in seconds for Qiskit scheduling."""
        dt = getattr(self._qubex, "dt", None)
        if dt is not None:
            return float(dt)
        measurement = getattr(self._qubex, "measurement", None)
        sampling_period = getattr(measurement, "sampling_period", None)
        if sampling_period is not None:
            return float(sampling_period) * 1e-9
        ctx = getattr(self._qubex, "ctx", None)
        ctx_measurement = getattr(ctx, "measurement", None)
        sampling_period = getattr(ctx_measurement, "sampling_period", None)
        if sampling_period is not None:
            return float(sampling_period) * 1e-9
        qxpulse_sampling_period = _qxpulse_default_sampling_period_ns()
        if qxpulse_sampling_period is not None:
            return qxpulse_sampling_period * 1e-9
        return 1e-9


def _normalize_circuits(run_input: Any) -> list[QuantumCircuit]:
    if isinstance(run_input, QuantumCircuit):
        return [run_input]
    if isinstance(run_input, Iterable):
        circuits = list(run_input)
        if all(isinstance(circuit, QuantumCircuit) for circuit in circuits):
            return circuits
    raise TypeError("QubexPulseExecutor.run expects a QuantumCircuit or iterable of QuantumCircuit objects.")


def _import_pulse_schedule() -> type:
    try:
        from qxpulse import PulseSchedule
    except ImportError as exc:
        raise ImportError(
            "QubexPulseExecutor requires qxpulse/qubex to be installed."
        ) from exc
    return PulseSchedule


def _import_blank() -> type:
    try:
        from qxpulse import Blank
    except ImportError as exc:
        raise ImportError(
            "QubexPulseExecutor requires qxpulse/qubex to be installed."
        ) from exc
    return Blank


def _op_start_times(circuit: QuantumCircuit) -> list[float] | None:
    try:
        start_times = circuit.op_start_times
    except AttributeError:
        return None
    if start_times is None:
        return None
    return list(start_times)


def _circuit_time_unit(circuit: QuantumCircuit) -> str:
    return getattr(circuit, "_unit", None) or getattr(circuit, "unit", "dt")


def _time_to_ns(value: float, unit: str, dt_seconds: float) -> float:
    if unit == "dt":
        return float(value) * dt_seconds * 1e9
    if unit == "s":
        return float(value) * 1e9
    if unit == "ms":
        return float(value) * 1e6
    if unit == "us":
        return float(value) * 1e3
    if unit == "ns":
        return float(value)
    raise ValueError(f"Unsupported Qiskit time unit {unit!r}.")


def _delay_duration_ns(operation: Any, dt_seconds: float) -> float:
    if not isinstance(operation, QiskitDelay) and operation.name != "delay":
        return 0.0
    return _time_to_ns(operation.duration, operation.unit, dt_seconds)


def _duration_ns(obj: Any) -> float:
    duration = getattr(obj, "cached_duration", None)
    if duration is None:
        duration = getattr(obj, "duration", None)
    if duration is None:
        return 0.0
    return float(duration)


def _duration_seconds_safe(factory: Any) -> float | None:
    try:
        obj = factory()
    except Exception:
        return None
    return _duration_ns(obj) * 1e-9


def _qxpulse_default_sampling_period_ns() -> float | None:
    try:
        from qxpulse.waveform import DEFAULT_SAMPLING_PERIOD
    except ImportError:
        return None
    return float(DEFAULT_SAMPLING_PERIOD)


def _circuit_header(circuit: QuantumCircuit) -> dict[str, Any]:
    return {
        "name": circuit.name,
        "n_qubits": circuit.num_qubits,
        "memory_slots": circuit.num_clbits,
        "qreg_sizes": [[reg.name, reg.size] for reg in circuit.qregs],
        "creg_sizes": [
            [reg.name, reg.size]
            for reg in circuit.cregs
            if isinstance(reg, ClassicalRegister)
        ],
        "qubit_labels": _bit_labels(circuit.qubits, circuit),
        "clbit_labels": _bit_labels(circuit.clbits, circuit),
        "global_phase": float(circuit.global_phase),
        "metadata": circuit.metadata or {},
    }


def _bit_labels(bits: Sequence[Any], circuit: QuantumCircuit) -> list[list[Any]]:
    labels = []
    for bit in bits:
        registers = getattr(bit, "_registers", [])
        if registers:
            register, index = registers[0]
            labels.append([register.name, index])
        elif isinstance(bit, Qubit):
            labels.append(["q", circuit.find_bit(bit).index])
        else:
            labels.append(["c", circuit.find_bit(bit).index])
    return labels


def _is_close(value: float, expected: float, *, atol: float = 1e-9) -> bool:
    return abs(value - expected) <= atol
