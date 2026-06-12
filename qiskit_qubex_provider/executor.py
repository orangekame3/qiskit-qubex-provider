"""Qubex hardware execution adapter for Qiskit circuits."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from numbers import Integral
from typing import Any, Literal

from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit
from qiskit.circuit import Delay as QiskitDelay
from qiskit.result import Result

from .job import QubexJob

TimingPolicy = Literal["qiskit", "legacy_device_gateway"]


@dataclass(frozen=True)
class QubexCircuitExecution:
    """Qubex execution artifacts for one Qiskit circuit."""

    circuit: QuantumCircuit
    schedule: Any
    raw_result: Any
    measured_targets: tuple[str | tuple[str, int], ...]
    target_to_clbit: Mapping[str | tuple[str, int], int]


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
        timing_policy: TimingPolicy | str = "qiskit",
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
        self._timing_policy = _timing_policy(timing_policy)
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
        shots = _shot_count(options.pop("shots", 1024))
        memory = _bool_option("memory", options.pop("memory", False))
        options.pop("seed_simulator", None)
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

    def validate(self, run_input: Any) -> list[Any]:
        """Build Qubex pulse schedules and run preflight checks without execution."""
        return [
            self.build_schedule(circuit)
            for circuit in _normalize_circuits(run_input)
        ]

    def build_schedule(self, circuit: QuantumCircuit) -> Any:
        """Convert a supported Qiskit circuit into a Qubex PulseSchedule."""
        self._validate_circuit_qubits(circuit)
        self._validate_static_circuit(circuit)
        if self._timing_policy == "legacy_device_gateway":
            schedule = self._build_legacy_device_gateway_schedule(circuit)
        else:
            schedule = self._build_qiskit_timed_schedule(circuit)
        self._validate_native_schedule(schedule)
        self._validate_resource_constraints(schedule)
        return schedule

    def _build_qiskit_timed_schedule(self, circuit: QuantumCircuit) -> Any:
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
                    readout_label = self._readout_label(labels[0])
                    if start_ns is not None:
                        self._align_channels(
                            schedule,
                            blank_cls,
                            channel_offsets,
                            [readout_label],
                            start_ns,
                        )
                    waveform = pulse.readout(labels[0])
                    schedule.add(readout_label, waveform)
                    duration_ns = _duration_ns(waveform)
                    self._advance_offsets(channel_offsets, [readout_label], duration_ns)
                    self._advance_offsets(channel_offsets, labels, duration_ns)
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
                elif name == "ecr":
                    sub_schedule = pulse.zx90(labels[0], labels[1], echo=True)
                    self._sync_cr_channel_frames(schedule, sub_schedule)
                    schedule.call(sub_schedule)
                    self._advance_offsets_for_schedule(channel_offsets, sub_schedule)
                elif name == "cx":
                    sub_schedule = pulse.cx(labels[0], labels[1])
                    self._sync_cr_channel_frames(schedule, sub_schedule)
                    schedule.call(sub_schedule)
                    self._advance_offsets_for_schedule(channel_offsets, sub_schedule)
                elif name == "cz":
                    sub_schedule = pulse.cz(labels[0], labels[1])
                    self._sync_cr_channel_frames(schedule, sub_schedule)
                    schedule.call(sub_schedule)
                    self._advance_offsets_for_schedule(channel_offsets, sub_schedule)
                else:
                    raise ValueError(
                        f"Unsupported Qiskit instruction {name!r} for Qubex pulse execution. "
                        "Transpile to the backend target or provide a custom executor."
                    )
        return schedule

    def _build_legacy_device_gateway_schedule(self, circuit: QuantumCircuit) -> Any:
        """Build the deprecated device-gateway-compatible sequential schedule."""
        pulse = self._pulse_source()
        pulse_schedule_cls = _import_pulse_schedule()
        blank_cls = _import_blank()
        with pulse_schedule_cls(self._used_legacy_labels(circuit)) as schedule:
            for instruction in circuit.data:
                operation = instruction.operation
                name = operation.name
                qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
                labels = [self._qubit_labels[index] for index in qubit_indices]
                if name == "barrier":
                    schedule.barrier()
                elif name == "delay":
                    delay_ns = _delay_duration_ns(operation, self._dt_seconds())
                    if delay_ns > 0:
                        schedule.add(labels[0], blank_cls(delay_ns))
                elif name == "measure":
                    continue
                elif name in {"id", "reset"}:
                    continue
                elif name == "x":
                    schedule.add(labels[0], pulse.x180(labels[0]))
                elif name == "sx":
                    schedule.add(labels[0], pulse.x90(labels[0]))
                elif name == "sxdg":
                    schedule.add(labels[0], pulse.x90m(labels[0]))
                elif name == "y":
                    schedule.add(labels[0], pulse.y180(labels[0]))
                elif name == "h":
                    schedule.add(labels[0], pulse.hadamard(labels[0]))
                elif name == "s":
                    schedule.add(labels[0], pulse.z90())
                elif name == "sdg":
                    schedule.add(labels[0], self._virtual_z(-pi / 2))
                elif name == "z":
                    schedule.add(labels[0], pulse.z180())
                elif name == "rz":
                    schedule.barrier()
                    schedule.add(labels[0], self._virtual_z(float(operation.params[0])))
                    schedule.barrier()
                elif name == "rx":
                    self._add_rx(schedule, pulse, labels[0], float(operation.params[0]))
                elif name == "ry":
                    self._add_ry(schedule, pulse, labels[0], float(operation.params[0]))
                elif name == "ecr":
                    sub_schedule = pulse.zx90(labels[0], labels[1], echo=True)
                    self._sync_cr_channel_frames(schedule, sub_schedule)
                    schedule.call(sub_schedule)
                elif name == "cx":
                    sub_schedule = pulse.cx(labels[0], labels[1])
                    self._sync_cr_channel_frames(schedule, sub_schedule)
                    schedule.call(sub_schedule)
                elif name == "cz":
                    sub_schedule = pulse.cz(labels[0], labels[1])
                    self._sync_cr_channel_frames(schedule, sub_schedule)
                    schedule.call(sub_schedule)
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
            self._set_duration(durations, "measure", qarg, _duration_seconds_safe(lambda: pulse.readout(label)))
            for virtual_gate in ("id", "rz", "s", "sdg", "z", "reset"):
                self._set_duration(durations, virtual_gate, qarg, 0.0)
        for control, control_label in enumerate(self._qubit_labels):
            for target, target_label in enumerate(self._qubit_labels):
                if control == target:
                    continue
                qarg = (control, target)
                self._set_duration(durations, "ecr", qarg, _duration_seconds_safe(lambda c=control_label, t=target_label: pulse.zx90(c, t, echo=True)))
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
        execute_options = self._execution_options(
            circuit,
            options=options,
            shots=shots,
        )
        measured_targets, target_to_clbit = self._measurement_mapping(circuit)
        schedule = self.build_schedule(circuit)
        raw_result = self._execute_source().execute(schedule=schedule, **execute_options)
        return QubexCircuitExecution(
            circuit=circuit,
            schedule=schedule,
            raw_result=raw_result,
            measured_targets=tuple(measured_targets),
            target_to_clbit=target_to_clbit,
        )

    def _execution_options(
        self,
        circuit: QuantumCircuit,
        *,
        options: Mapping[str, Any],
        shots: int,
    ) -> dict[str, Any]:
        execute_options = dict(self._execute_options)
        execute_options.update(options)
        execute_options.setdefault("state_classification", True)
        execute_options["state_classification"] = _bool_option(
            "state_classification",
            execute_options["state_classification"],
        )
        if not execute_options["state_classification"]:
            raise ValueError(
                "QubexPulseExecutor requires state_classification=True to "
                "produce Qiskit counts."
            )
        execute_options.setdefault("final_measurement", not _has_explicit_measurements(circuit))
        execute_options["final_measurement"] = _bool_option(
            "final_measurement",
            execute_options["final_measurement"],
        )
        if (
            not _has_explicit_measurements(circuit)
            and not execute_options["final_measurement"]
        ):
            raise ValueError(
                "QubexPulseExecutor cannot produce Qiskit counts for a circuit "
                "without explicit measurements when final_measurement=False."
            )
        execute_options.setdefault("plot", False)
        execute_options["plot"] = _bool_option("plot", execute_options["plot"])
        execute_options["n_shots"] = shots
        return execute_options

    def _measurement_mapping(
        self,
        circuit: QuantumCircuit,
    ) -> tuple[list[str | tuple[str, int]], dict[str | tuple[str, int], int]]:
        self._validate_circuit_qubits(circuit)
        self._validate_static_circuit(circuit)
        measured_targets: list[str | tuple[str, int]] = []
        target_to_clbit: dict[str | tuple[str, int], int] = {}
        capture_counts: dict[str, int] = {}
        for instruction in circuit.data:
            name = instruction.operation.name
            if name == "measure":
                qubit_index = circuit.find_bit(instruction.qubits[0]).index
                clbit_index = circuit.find_bit(instruction.clbits[0]).index
                target = self._qubit_labels[qubit_index]
                capture_index = capture_counts.get(target, 0)
                capture_counts[target] = capture_index + 1
                measured_target = (target, capture_index)
                measured_targets.append(measured_target)
                target_to_clbit[measured_target] = clbit_index
        if not measured_targets:
            measured_targets = list(self._qubit_labels[: circuit.num_qubits])
            target_to_clbit = {
                target: index for index, target in enumerate(measured_targets)
            }
        return measured_targets, target_to_clbit

    def _validate_circuit_qubits(self, circuit: QuantumCircuit) -> None:
        if circuit.num_qubits > len(self._qubit_labels):
            raise ValueError(
                "Qiskit circuit uses more qubits than the Qubex executor has "
                f"labels: circuit has {circuit.num_qubits}, executor has "
                f"{len(self._qubit_labels)}."
            )

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
            try:
                counts, memory_values = self._qiskit_counts_and_memory(
                    execution,
                    include_memory=memory,
                )
                self._validate_result_shots(
                    counts=counts,
                    memory_values=memory_values if memory else None,
                    shots=shots,
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Failed to convert Qubex result for circuit "
                    f"{execution.circuit.name!r}: {exc}"
                ) from exc
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
                    "header": _circuit_header(
                        circuit,
                        memory_slots=_memory_slots(execution),
                    ),
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
        *,
        include_memory: bool,
    ) -> tuple[Counter[str], list[str]]:
        raw_counts = self._raw_counts(execution)
        raw_memory = self._raw_memory(execution) if include_memory else None
        counts: Counter[str] = Counter()
        memory: list[str] = []
        for qubex_bitstring, count in raw_counts.items():
            count_value = _classified_count(count)
            hex_value = self._qubex_bitstring_to_hex(
                _classified_bitstring(qubex_bitstring),
                execution,
            )
            counts[hex_value] += count_value
            if include_memory and raw_memory is None:
                memory.extend([hex_value] * count_value)
        return counts, memory if raw_memory is None else raw_memory

    @staticmethod
    def _raw_counts(execution: QubexCircuitExecution) -> Mapping[Any, Any]:
        raw_result = execution.raw_result
        get_counts = getattr(raw_result, "get_counts", None)
        if callable(get_counts):
            try:
                return get_counts(execution.measured_targets)
            except TypeError:
                return get_counts()
        if isinstance(raw_result, Mapping):
            if "counts" in raw_result:
                nested_counts = raw_result["counts"]
                if isinstance(nested_counts, Mapping):
                    return nested_counts
                raise TypeError("Qubex execution result 'counts' must be a mapping.")
            if "memory" in raw_result:
                raise TypeError(
                    "Qubex execution result must include a counts mapping."
                )
            return raw_result
        counts = getattr(raw_result, "counts", None)
        if callable(counts):
            counts = counts()
        if isinstance(counts, Mapping):
            return counts
        raise TypeError(
            "Qubex execution result must expose get_counts(...), a counts "
            "mapping, or a {'counts': ...} mapping."
        )

    def _raw_memory(self, execution: QubexCircuitExecution) -> list[str] | None:
        raw_memory = self._raw_memory_values(execution)
        if raw_memory is None:
            return None
        return [
            self._qubex_bitstring_to_hex(
                _classified_bitstring(value),
                execution,
            )
            for value in raw_memory
        ]

    @staticmethod
    def _raw_memory_values(execution: QubexCircuitExecution) -> Sequence[Any] | None:
        raw_result = execution.raw_result
        get_memory = getattr(raw_result, "get_memory", None)
        if callable(get_memory):
            try:
                memory = get_memory(execution.measured_targets)
            except TypeError:
                memory = get_memory()
            if isinstance(memory, Sequence) and not isinstance(memory, str):
                return memory
        if isinstance(raw_result, Mapping):
            memory = raw_result.get("memory")
            if isinstance(memory, Sequence) and not isinstance(memory, str):
                return memory
        memory = getattr(raw_result, "memory", None)
        if callable(memory):
            memory = memory()
        if isinstance(memory, Sequence) and not isinstance(memory, str):
            return memory
        return None

    @staticmethod
    def _validate_result_shots(
        *,
        counts: Mapping[str, int],
        memory_values: Sequence[str] | None,
        shots: int,
    ) -> None:
        count_total = sum(int(count) for count in counts.values())
        if count_total != shots:
            raise ValueError(
                "Qubex result count total does not match requested shots: "
                f"got {count_total}, expected {shots}."
            )
        if memory_values is not None and len(memory_values) != shots:
            raise ValueError(
                "Qubex result memory length does not match requested shots: "
                f"got {len(memory_values)}, expected {shots}."
            )

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

    @staticmethod
    def _validate_static_circuit(circuit: QuantumCircuit) -> None:
        active_operations_started = False
        measurement_started = False
        measured_clbits: set[int] = set()
        for instruction in circuit.data:
            operation = instruction.operation
            name = operation.name
            if getattr(operation, "condition", None) is not None or name in {
                "if_else",
                "while_loop",
                "for_loop",
                "switch_case",
            }:
                raise ValueError(
                    "Classically controlled or dynamic Qiskit circuits are not "
                    "supported by QubexPulseExecutor."
                )
            if name == "measure":
                measurement_started = True
                if len(instruction.qubits) != 1 or len(instruction.clbits) != 1:
                    raise ValueError("Only one-qubit Qiskit measurements are supported.")
                clbit = circuit.find_bit(instruction.clbits[0]).index
                if clbit in measured_clbits:
                    raise ValueError("Multiple measurements into the same clbit are not supported.")
                measured_clbits.add(clbit)
            elif name == "reset":
                if active_operations_started or measurement_started:
                    raise ValueError("Mid-circuit reset is not supported by QubexPulseExecutor.")
            elif name not in {"barrier", "delay"}:
                active_operations_started = True

    def _virtual_z(self, theta: float) -> Any:
        return self._pulse_source().z90().__class__(theta)

    def _sync_cr_channel_frames(self, schedule: Any, sub_schedule: Any) -> None:
        """Align CR channel frames with their frequency-target qubit frames.

        A cross-resonance channel ``Qc-Qt`` drives ``Qc`` in the frame of
        ``Qt``, so virtual-Z rotations accumulated on ``Qt`` must also rotate
        the frame of subsequent CR pulses on that channel. Production Qubex
        mirrors ``VirtualZ`` onto the CR channel the same way inside its
        ``cnot``/``cz`` constructions.
        """
        for label in getattr(sub_schedule, "labels", []):
            if "-" not in label:
                continue
            frequency_target = label.split("-", 1)[1]
            if frequency_target not in self._qubit_labels:
                continue
            delta = self._channel_frame_shift(schedule, frequency_target) - self._channel_frame_shift(schedule, label)
            if abs(delta) > 1e-12:
                # VirtualZ(theta) stores PhaseShift(-theta), so negate to add a raw frame shift.
                schedule.add(label, self._virtual_z(-delta))

    @staticmethod
    def _channel_frame_shift(schedule: Any, label: str) -> float:
        get_final_frame_shift = getattr(schedule, "get_final_frame_shift", None)
        if get_final_frame_shift is None or label not in getattr(schedule, "labels", []):
            return 0.0
        return float(get_final_frame_shift(label))

    def _used_legacy_labels(self, circuit: QuantumCircuit) -> list[str]:
        labels: list[str] = []
        for instruction in circuit.data:
            for qubit in instruction.qubits:
                label = self._qubit_labels[circuit.find_bit(qubit).index]
                if label not in labels:
                    labels.append(label)
            if instruction.operation.name in {"cx", "cz", "ecr"} and len(instruction.qubits) == 2:
                left, right = [
                    self._qubit_labels[circuit.find_bit(qubit).index]
                    for qubit in instruction.qubits
                ]
                coupling_label = f"{left}-{right}"
                if coupling_label not in labels:
                    labels.append(coupling_label)
        return labels or list(self._qubit_labels[: circuit.num_qubits])

    def _readout_label(self, target: str) -> str:
        for source in (
            self._qubex,
            getattr(self._qubex, "experiment_system", None),
            getattr(self._qubex, "ctx", None),
            getattr(self._qubex, "context", None),
            getattr(self._qubex, "target_registry", None),
        ):
            resolver = getattr(source, "resolve_read_label", None)
            if resolver is None:
                continue
            try:
                return str(resolver(target, allow_legacy=True))
            except TypeError:
                return str(resolver(target))
            except ValueError:
                continue
        return f"R{target}"

    def _validate_resource_constraints(self, schedule: Any) -> None:
        get_pulse_ranges = getattr(schedule, "get_pulse_ranges", None)
        if get_pulse_ranges is None:
            return
        resource_windows: dict[str, list[tuple[int, int, str]]] = {}
        for label, ranges in get_pulse_ranges().items():
            if self._is_readout_label(label):
                continue
            resource_key = self._resource_key(label)
            if resource_key is None:
                continue
            for pulse_range in ranges:
                if pulse_range.start == pulse_range.stop:
                    continue
                resource_windows.setdefault(resource_key, []).append(
                    (pulse_range.start, pulse_range.stop, label)
                )
        for resource_key, windows in resource_windows.items():
            windows.sort()
            previous: tuple[int, int, str] | None = None
            for current in windows:
                if previous is not None and current[0] < previous[1]:
                    raise ValueError(
                        "Qubex resource conflict: "
                        f"channels {previous[2]!r} and {current[2]!r} overlap "
                        f"on hardware resource {resource_key!r}."
                    )
                previous = current

    @staticmethod
    def _validate_native_schedule(schedule: Any) -> None:
        is_valid = getattr(schedule, "is_valid", None)
        if is_valid is None:
            return
        if not is_valid():
            raise ValueError("Invalid Qubex pulse schedule.")

    def _resource_key(self, label: str) -> str | None:
        target = self._target_metadata(label)
        channel = getattr(target, "channel", None)
        if channel is None:
            return None
        channel_id = getattr(channel, "id", None)
        if channel_id is not None:
            return str(channel_id)
        port = getattr(channel, "port", None)
        port_id = getattr(port, "id", None)
        number = getattr(channel, "number", None)
        if port_id is not None and number is not None:
            return f"{port_id}:{number}"
        return None

    def _is_readout_label(self, label: str) -> bool:
        if label in {self._readout_label(qubit_label) for qubit_label in self._qubit_labels}:
            return True
        return label.startswith("R") and label[1:] in self._qubit_labels

    def _target_metadata(self, label: str) -> Any | None:
        for source in (
            self._qubex,
            getattr(self._qubex, "experiment_system", None),
            getattr(self._qubex, "ctx", None),
            getattr(self._qubex, "context", None),
            getattr(self._qubex, "target_registry", None),
        ):
            for method_name in ("get_target", "get_read_out_target", "get_cap_target"):
                resolver = getattr(source, method_name, None)
                if resolver is None:
                    continue
                try:
                    return resolver(label)
                except (KeyError, ValueError):
                    continue
        return None

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


def _timing_policy(value: str) -> TimingPolicy:
    if value in {"qiskit", "legacy_device_gateway"}:
        return value  # type: ignore[return-value]
    raise ValueError("timing_policy must be 'qiskit' or 'legacy_device_gateway'.")


def _normalize_circuits(run_input: Any) -> list[QuantumCircuit]:
    if isinstance(run_input, QuantumCircuit):
        return [run_input]
    if isinstance(run_input, Iterable):
        circuits = list(run_input)
        if circuits and all(isinstance(circuit, QuantumCircuit) for circuit in circuits):
            return circuits
    raise TypeError(
        "QubexPulseExecutor.run expects a QuantumCircuit or non-empty "
        "iterable of QuantumCircuit objects."
    )


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


def _has_explicit_measurements(circuit: QuantumCircuit) -> bool:
    return any(instruction.operation.name == "measure" for instruction in circuit.data)


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


def _classified_bitstring(value: Any) -> str:
    if isinstance(value, str):
        return value.replace(" ", "")
    if isinstance(value, Sequence):
        return "".join(str(bit) for bit in value)
    return str(value)


def _classified_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Qubex classified counts must be non-negative integers.")
    if isinstance(value, Integral):
        count = int(value)
    elif isinstance(value, str) and value.isdecimal():
        count = int(value)
    else:
        raise ValueError("Qubex classified counts must be non-negative integers.")
    if count < 0:
        raise ValueError("Qubex classified counts must be non-negative integers.")
    return count


def _shot_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Qubex shots must be a positive integer.")
    if isinstance(value, Integral):
        shots = int(value)
    elif isinstance(value, str) and value.isdecimal():
        shots = int(value)
    else:
        raise ValueError("Qubex shots must be a positive integer.")
    if shots <= 0:
        raise ValueError("Qubex shots must be a positive integer.")
    return shots


def _bool_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Qubex option {name!r} must be a boolean.")
    return value


def _qxpulse_default_sampling_period_ns() -> float | None:
    try:
        from qxpulse.waveform import DEFAULT_SAMPLING_PERIOD
    except ImportError:
        return None
    return float(DEFAULT_SAMPLING_PERIOD)


def _memory_slots(execution: QubexCircuitExecution) -> int:
    return max(
        [execution.circuit.num_clbits]
        + [index + 1 for index in execution.target_to_clbit.values()]
    )


def _circuit_header(
    circuit: QuantumCircuit,
    *,
    memory_slots: int | None = None,
) -> dict[str, Any]:
    return {
        "name": circuit.name,
        "n_qubits": circuit.num_qubits,
        "memory_slots": circuit.num_clbits if memory_slots is None else memory_slots,
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
