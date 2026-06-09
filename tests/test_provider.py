from __future__ import annotations

from math import pi
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.quantum_info import SparsePauliOp

from qiskit_qubex_provider import (
    QubexBackend,
    QubexEstimatorV2,
    QubexPulseExecutor,
    QubexProvider,
    QubexSamplerV2,
    build_qubex_target,
)
import qiskit_qubex_provider.executor as executor_module


class DurationObject:
    def __init__(self, name: str, duration: float = 0.0):
        self.name = name
        self.cached_duration = duration
        self.duration = duration


class DurationPulse:
    def x90(self, target):
        return DurationObject(f"x90-{target}", 4)

    def x90m(self, target):
        return DurationObject(f"x90m-{target}", 4)

    def x180(self, target):
        return DurationObject(f"x180-{target}", 8)

    def y90(self, target):
        return DurationObject(f"y90-{target}", 4)

    def y90m(self, target):
        return DurationObject(f"y90m-{target}", 4)

    def y180(self, target):
        return DurationObject(f"y180-{target}", 8)

    def z90(self):
        return DurationVirtualZ(pi / 2)

    def z180(self):
        return DurationVirtualZ(pi)

    def hadamard(self, target):
        return DurationObject(f"h-{target}", 12)

    def readout(self, target):
        return DurationObject(f"readout-{target}", 20)

    def cx(self, control, target):
        return DurationSchedule(
            [control, target, f"{control}-{target}"],
            duration=24,
            ops=[("cx", control, target)],
        )

    def cz(self, control, target):
        return DurationSchedule(
            [control, target, f"{control}-{target}"],
            duration=28,
            ops=[("cz", control, target)],
        )


class DurationSchedule:
    def __init__(self, channels=None, *, duration: float = 0.0, ops=None):
        self.labels = list(channels or [])
        self.ops = list(ops or [])
        self.duration = duration
        self.cached_duration = duration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.barrier()

    def add(self, label, obj):
        self.ops.append(("add", label, obj))
        if label not in self.labels:
            self.labels.append(label)
        self.duration += getattr(obj, "cached_duration", 0.0)
        self.cached_duration = self.duration

    def call(self, schedule):
        self.ops.append(("call", schedule))
        for label in schedule.labels:
            if label not in self.labels:
                self.labels.append(label)
        self.duration += schedule.duration
        self.cached_duration = self.duration

    def barrier(self, labels=None):
        self.ops.append(("barrier", labels))


class DurationBlank(DurationObject):
    def __init__(self, duration: float):
        super().__init__("blank", duration)


class DurationVirtualZ(DurationObject):
    def __init__(self, theta: float):
        super().__init__("virtual_z", 0)
        self.theta = theta


def test_provider_returns_backend_for_integer_qubit_count() -> None:
    provider = QubexProvider(num_qubits=2, coupling_map=[(0, 1)])

    backend = provider.get_backend()

    assert isinstance(backend, QubexBackend)
    assert backend.num_qubits == 2
    assert "cx" in backend.target.operation_names


def test_unknown_backend_name_raises() -> None:
    provider = QubexProvider(num_qubits=1)

    with pytest.raises(QiskitBackendNotFoundError):
        provider.get_backend("missing")


def test_target_uses_qubex_like_system_metadata() -> None:
    qubits = [
        SimpleNamespace(label="Q0", frequency=5.0),
        SimpleNamespace(label="Q1", frequency=5.1),
    ]
    source = SimpleNamespace(
        qubits=qubits,
        cr_targets=[SimpleNamespace(label="Q0-Q1")],
    )

    target = build_qubex_target(source)

    assert target.num_qubits == 2
    assert target.qubit_properties[0].frequency == 5.0e9
    assert (0, 1) in target["cx"]


def test_backend_runs_qiskit_circuit_locally() -> None:
    backend = QubexProvider(num_qubits=2, coupling_map=[(0, 1)]).get_backend()
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    transpiled = transpile(circuit, backend)
    result = backend.run(transpiled, shots=128).result()

    counts = result.get_counts()
    assert sum(counts.values()) == 128


def test_provider_primitives_are_executable() -> None:
    provider = QubexProvider(num_qubits=2, coupling_map=[(0, 1)])
    backend = provider.get_backend()
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    measured = circuit.copy()
    measured.measure_all()

    sampler = provider.get_sampler()
    estimator = provider.get_estimator()

    assert isinstance(sampler, QubexSamplerV2)
    assert isinstance(estimator, QubexEstimatorV2)
    assert len(sampler.run([measured], shots=32).result()) == 1
    assert len(estimator.run([(circuit, SparsePauliOp("ZZ"))]).result()) == 1


def test_backend_can_delegate_to_executor() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = []

        def run(self, run_input, **options):
            self.calls.append((run_input, options))
            return "job"

    executor = RecordingExecutor()
    backend = QubexProvider(num_qubits=1, executor=executor).get_backend()
    circuit = QuantumCircuit(1)

    job = backend.run(circuit, shots=7)

    assert job == "job"
    assert executor.calls[0][0] is circuit
    assert executor.calls[0][1]["shots"] == 7


def test_qubex_pulse_executor_converts_and_runs_circuit(monkeypatch) -> None:
    class FakePulseSchedule:
        def __init__(self, channels=None):
            self.labels = list(channels or [])
            self.ops = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.barrier()

        def add(self, label, obj):
            self.ops.append(("add", label, obj))
            if label not in self.labels:
                self.labels.append(label)

        def call(self, schedule):
            self.ops.append(("call", schedule))

        def barrier(self, labels=None):
            self.ops.append(("barrier", labels))

    class FakeVirtualZ:
        def __init__(self, theta):
            self.theta = theta

    class FakePulse:
        def x90(self, target):
            return ("x90", target)

        def x90m(self, target):
            return ("x90m", target)

        def x180(self, target):
            return ("x180", target)

        def y90(self, target):
            return ("y90", target)

        def y90m(self, target):
            return ("y90m", target)

        def y180(self, target):
            return ("y180", target)

        def z90(self):
            return FakeVirtualZ(1.5708)

        def z180(self):
            return FakeVirtualZ(3.14159)

        def hadamard(self, target):
            return ("h", target)

        def readout(self, target):
            return ("readout", target)

        def cx(self, control, target):
            schedule = FakePulseSchedule([control, target])
            schedule.add(f"{control}-{target}", ("cx", control, target))
            return schedule

        def cz(self, control, target):
            schedule = FakePulseSchedule([control, target])
            schedule.add(f"{control}-{target}", ("cz", control, target))
            return schedule

    class FakeMeasureResult:
        def get_counts(self, targets):
            assert tuple(targets) == (("Q0", 0), ("Q1", 0))
            return {"10": 3, "01": 2}

    class FakeMeasurementService:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return FakeMeasureResult()

    class FakeQubex:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = FakePulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: FakePulseSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    qubex = FakeQubex()
    provider = QubexProvider(
        num_qubits=2,
        coupling_map=[(0, 1)],
        qubex=qubex,
        use_qubex_executor=True,
    )
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    result = provider.get_backend().run(circuit, shots=5).result()

    assert isinstance(provider.get_backend()._executor, QubexPulseExecutor)
    assert result.get_counts() == {"01": 3, "10": 2}
    execute_call = qubex.measurement_service.calls[0]
    assert execute_call["n_shots"] == 5
    assert execute_call["state_classification"] is True
    assert execute_call["final_measurement"] is False


def test_provider_from_experiment_uses_qubex_executor() -> None:
    class FakePulse:
        def x90(self, target):
            return ("x90", target)

        def x180(self, target):
            return ("x180", target)

        def y90(self, target):
            return ("y90", target)

        def y180(self, target):
            return ("y180", target)

        def z90(self):
            return "z90"

        def z180(self):
            return "z180"

    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = FakePulse()
            self.measurement_service = FakeMeasurementService()

    provider = QubexProvider.from_experiment(FakeExperiment())
    backend = provider.get_backend()

    assert backend.num_qubits == 2
    assert backend.target.num_qubits == 2
    assert isinstance(backend._executor, QubexPulseExecutor)


def test_provider_from_experiment_populates_target_durations() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 2e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()

    assert backend.target.dt == pytest.approx(2e-9)
    assert backend.target["x"][(0,)].duration == pytest.approx(8e-9)
    assert backend.target["sx"][(0,)].duration == pytest.approx(4e-9)
    assert backend.target["h"][(0,)].duration == pytest.approx(12e-9)
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(24e-9)
    assert backend.target["measure"][(0,)].duration == pytest.approx(20e-9)
    assert backend.target["rz"][(0,)].duration == 0.0


def test_scheduled_circuit_start_times_become_qubex_blanks(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.x(1)
    circuit.rz(pi / 2, 0)
    circuit.delay(5, 1, unit="dt")
    circuit.measure([0, 1], [0, 1])
    circuit._op_start_times = [0, 10, 18, 20, 25, 25]
    circuit._duration = 25
    circuit._unit = "dt"

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    q0_ops = [op for op in schedule.ops if op[0] == "add" and op[1] == "Q0"]
    q1_ops = [op for op in schedule.ops if op[0] == "add" and op[1] == "Q1"]
    rq0_ops = [op for op in schedule.ops if op[0] == "add" and op[1] == "RQ0"]
    rq1_ops = [op for op in schedule.ops if op[0] == "add" and op[1] == "RQ1"]
    assert q0_ops[0][2].name == "x180-Q0"
    assert q0_ops[1][2].name == "blank"
    assert q0_ops[1][2].duration == pytest.approx(10)
    assert q0_ops[2][2].name == "virtual_z"
    assert q0_ops[2][2].theta == pytest.approx(pi / 2)
    assert q0_ops[2][2].duration == 0
    assert q1_ops[0][2].name == "blank"
    assert q1_ops[0][2].duration == pytest.approx(10)
    assert q1_ops[1][2].name == "x180-Q1"
    assert q1_ops[2][2].name == "blank"
    assert q1_ops[2][2].duration == 2
    assert q1_ops[3][2].name == "blank"
    assert q1_ops[3][2].duration == 5
    assert rq0_ops[0][2].name == "blank"
    assert rq0_ops[0][2].duration == pytest.approx(25)
    assert rq0_ops[1][2].name == "readout-Q0"
    assert rq1_ops[0][2].name == "blank"
    assert rq1_ops[0][2].duration == pytest.approx(25)
    assert rq1_ops[1][2].name == "readout-Q1"


def test_qubex_executor_supports_mid_circuit_measurement_without_feedback(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

        def resolve_read_label(self, target, allow_legacy=False):
            return f"R{target}"

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(1, 2)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.x(0)
    circuit.measure(0, 1)

    executor = QubexPulseExecutor(FakeExperiment())
    schedule = executor.build_schedule(circuit)
    measured_targets, target_to_clbit = executor._measurement_mapping(circuit)

    readout_ops = [op for op in schedule.ops if op[0] == "add" and op[1] == "RQ0"]
    assert [op[2].name for op in readout_ops] == ["readout-Q0", "readout-Q0"]
    assert measured_targets == [("Q0", 0), ("Q0", 1)]
    assert target_to_clbit == {("Q0", 0): 0, ("Q0", 1): 1}


def test_qubex_executor_rejects_dynamic_circuit_control() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)

    with pytest.raises(ValueError, match="dynamic Qiskit circuits"):
        QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)


def test_qubex_executor_rejects_mid_circuit_reset() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.reset(0)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="Mid-circuit reset"):
        QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)


def test_qubex_executor_rejects_repeated_clbit_measurement() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    circuit = QuantumCircuit(1, 2)
    circuit.measure(0, 0)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="same clbit"):
        QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)


def test_qubex_executor_allows_initial_reset(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(1, 1)
    circuit.reset(0)
    circuit.x(0)
    circuit.measure(0, 0)

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    assert any(op[0] == "add" and op[1] == "Q0" for op in schedule.ops)


def test_qubex_executor_rejects_overlapping_hardware_resource_windows() -> None:
    class FakeSchedule:
        def get_pulse_ranges(self):
            return {
                "Q0": [range(0, 10)],
                "Q1": [range(5, 15)],
            }

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = DurationPulse()

        def get_target(self, label):
            channel = SimpleNamespace(id="shared-control")
            return SimpleNamespace(channel=channel)

    executor = QubexPulseExecutor(FakeExperiment())

    with pytest.raises(ValueError, match="resource conflict"):
        executor._validate_resource_constraints(FakeSchedule())


def test_qubex_executor_allows_non_overlapping_hardware_resource_windows() -> None:
    class FakeSchedule:
        def get_pulse_ranges(self):
            return {
                "Q0": [range(0, 10)],
                "Q1": [range(10, 20)],
            }

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = DurationPulse()

        def get_target(self, label):
            channel = SimpleNamespace(id="shared-control")
            return SimpleNamespace(channel=channel)

    QubexPulseExecutor(FakeExperiment())._validate_resource_constraints(FakeSchedule())


def test_transpile_scheduling_uses_qubex_target_durations() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        coupling_map=[(0, 1)],
    ).get_backend()
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.x(1)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    scheduled = transpile(circuit, backend, scheduling_method="asap")

    assert scheduled.op_start_times is not None


def test_independent_cx_operations_are_scheduled_in_parallel(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1", "Q2", "Q3")
        dt = 2e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        coupling_map=[(0, 1), (2, 3)],
    ).get_backend()
    circuit = QuantumCircuit(4, 4)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.measure([0, 1, 2, 3], [0, 1, 2, 3])

    scheduled = transpile(circuit, backend, scheduling_method="asap")

    cx_starts = [
        start
        for instruction, start in zip(scheduled.data, scheduled.op_start_times)
        if instruction.operation.name == "cx"
    ]
    assert cx_starts == [0, 0]

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(scheduled)
    calls = [op for op in schedule.ops if op[0] == "call"]

    assert len(calls) == 2
    assert calls[0][1].labels == ["Q0", "Q1", "Q0-Q1"]
    assert calls[1][1].labels == ["Q2", "Q3", "Q2-Q3"]
    assert not [
        op for op in schedule.ops
        if op[0] == "add" and op[1] in {"Q2", "Q3"} and op[2].name == "blank"
    ]


def test_transpile_scheduling_decomposes_parameterized_rotations() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.rx(pi / 2, 0)
    circuit.measure(0, 0)

    scheduled = transpile(circuit, backend, scheduling_method="asap")

    assert "rx" not in scheduled.count_ops()
    assert scheduled.op_start_times is not None


def test_qubex_executor_requires_experiment_like_object() -> None:
    with pytest.raises(ValueError, match="requires a Qubex Experiment-like object"):
        QubexPulseExecutor(None)

    class BareMeasurement:
        qubit_labels = ("Q0",)

        def execute(self, **kwargs):
            return None

    with pytest.raises(TypeError, match="bare qubex.Measurement"):
        QubexPulseExecutor(BareMeasurement())
