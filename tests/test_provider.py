from __future__ import annotations

from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.quantum_info import SparsePauliOp

from qiskit_qubex_provider import (
    QubexBackend,
    QubexEstimatorV2,
    QubexProvider,
    QubexSamplerV2,
    build_qubex_target,
)


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
