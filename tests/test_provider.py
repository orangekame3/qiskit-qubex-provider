from __future__ import annotations

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
            assert tuple(targets) == ("Q0", "Q1")
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
    assert execute_call["final_measurement"] is True
