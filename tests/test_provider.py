from __future__ import annotations

import json
import sys
from math import pi
from pathlib import Path
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
    QUBEX_NATIVE_BASIS_GATES,
    build_device_topology,
    build_device_topology_svg,
    build_dynamical_decoupling_pass_manager,
    build_topology_aware_dynamical_decoupling_pass_manager,
    build_qubex_target,
    qid_to_label,
    write_device_topology,
)
from qiskit_qubex_provider.device_topology import main as device_topology_main
import qiskit_qubex_provider.executor as executor_module


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    def zx90(self, control, target, *, echo=True):
        suffix = "echo" if echo else "direct"
        return DurationSchedule(
            [control, target, f"{control}-{target}"],
            duration=24,
            ops=[("zx90", control, target, suffix)],
        )

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

    def is_valid(self):
        return True


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


def test_target_uses_device_topology_metadata() -> None:
    topology = {
        "name": "anemone",
        "qubits": [
            {
                "id": 0,
                "physical_id": 5,
                "qubit_lifetime": {"t1": 25.0, "t2": 30.0},
                "gate_duration": {"rz": 0, "sx": 16, "x": 24, "measure": 120},
            },
            {
                "id": 1,
                "physical_id": 7,
                "qubit_lifetime": {"t1": 20.0, "t2": 22.0},
                "gate_duration": {"rz": 0, "sx": 18, "x": 26, "measure": 120},
            },
        ],
        "couplings": [
            {
                "control": 0,
                "target": 1,
                "gate_duration": {"rzx90": 272},
            }
        ],
    }

    target = build_qubex_target(topology)

    assert target.num_qubits == 2
    assert target.qubit_properties[0].t1 == pytest.approx(25e-6)
    assert target["sx"][(0,)].duration == pytest.approx(16e-9)
    assert target["measure"][(0,)].duration == pytest.approx(120e-9)
    assert target["ecr"][(0, 1)].duration == pytest.approx(272e-9)
    assert target["cx"][(0, 1)].duration == pytest.approx(272e-9)


def test_provider_from_device_topology_file(tmp_path) -> None:
    topology = {
        "name": "anemone",
        "qubits": [
            {"id": 0, "physical_id": 5, "gate_duration": {"sx": 16, "x": 24}},
            {"id": 1, "physical_id": 7, "gate_duration": {"sx": 16, "x": 24}},
        ],
        "couplings": [
            {"control": 0, "target": 1, "gate_duration": {"rzx90": 272}}
        ],
    }
    topology_path = tmp_path / "device-topology.json"
    topology_path.write_text(json.dumps(topology), encoding="utf-8")

    backend = QubexProvider.from_device_topology(topology_path).get_backend()

    assert backend.name == "anemone"
    assert backend.target.num_qubits == 2
    assert (0, 1) in backend.target["cx"]


def test_device_topology_label_width_matches_device_gateway() -> None:
    assert qid_to_label(7, 64) == "Q07"
    assert qid_to_label(7, 100) == "Q007"


def test_build_device_topology_from_qubex_calibration_files(tmp_path) -> None:
    calib_note_path = tmp_path / "calib_note.json"
    calib_note_path.write_text(
        json.dumps(
            {
                "drag_hpi_params": {
                    "Q00": {"duration": 16},
                    "Q01": {"duration": 18},
                },
                "drag_pi_params": {
                    "Q00": {"duration": 24},
                    "Q01": {"duration": 26},
                },
                "cr_params": {
                    "Q00-Q01": {"duration": 272},
                },
                "calibrated_at": "2026-01-02T03:04:05Z",
            }
        ),
        encoding="utf-8",
    )
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "x90_gate_fidelity.yaml").write_text(
        "meta:\n  description: X90\n\ndata:\n  Q00: 0.99\n  Q01: 0.98\n",
        encoding="utf-8",
    )
    (params_dir / "zx90_gate_fidelity.yaml").write_text(
        "data:\n  Q00-Q01: 0.97\n",
        encoding="utf-8",
    )
    (params_dir / "t1.yaml").write_text("data:\n  Q00: 25.0\n  Q01: 20.0\n", encoding="utf-8")
    (params_dir / "t2_echo.yaml").write_text(
        "data:\n  Q00: 30.0\n  Q01: 22.0\n",
        encoding="utf-8",
    )
    (params_dir / "readout_fidelity_0.yaml").write_text(
        "data:\n  Q00: 0.91\n  Q01: 0.92\n",
        encoding="utf-8",
    )
    (params_dir / "readout_fidelity_1.yaml").write_text(
        "data:\n  Q00: 0.93\n  Q01: 0.94\n",
        encoding="utf-8",
    )
    (params_dir / "average_readout_fidelity.yaml").write_text(
        "data:\n  Q00: 0.92\n  Q01: 0.93\n",
        encoding="utf-8",
    )

    topology = build_device_topology(
        calib_note_path=calib_note_path,
        params_dir=params_dir,
        topology={
            "qubits": {
                "0": {"row": 0, "col": 0},
                "1": {"row": 0, "col": 1},
            },
            "couplings": [[0, 1]],
        },
        name="test-device",
        device_id="test-device",
    )

    assert topology["name"] == "test-device"
    assert topology["qubits"][0]["id"] == 0
    assert topology["qubits"][0]["physical_id"] == 0
    assert topology["qubits"][0]["gate_duration"] == {"rz": 0, "sx": 16, "x": 24}
    assert topology["qubits"][0]["qubit_lifetime"] == {"t1": 25.0, "t2": 30.0}
    assert topology["qubits"][0]["meas_error"]["prob_meas1_prep0"] == pytest.approx(0.09)
    assert topology["couplings"] == [
        {
            "control": 0,
            "target": 1,
            "fidelity": 0.97,
            "gate_duration": {"rzx90": 272},
        }
    ]

    backend = QubexProvider.from_device_topology(topology).get_backend()
    assert backend.target["sx"][(0,)].duration == pytest.approx(16e-9)
    assert backend.target["ecr"][(0, 1)].duration == pytest.approx(272e-9)
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(272e-9)


def test_build_device_topology_accepts_qdash_style_request(tmp_path) -> None:
    calib_note_path = tmp_path / "calib_note.json"
    calib_note_path.write_text(
        json.dumps(
            {
                "drag_hpi_params": {
                    "Q00": {"duration": 16},
                    "Q01": {"duration": 18},
                    "Q02": {"duration": 20},
                },
                "drag_pi_params": {
                    "Q00": {"duration": 24},
                    "Q01": {"duration": 26},
                    "Q02": {"duration": 28},
                },
                "cr_params": {
                    "Q00-Q01": {"duration": 272},
                    "Q01-Q02": {"duration": 288},
                },
            }
        ),
        encoding="utf-8",
    )
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "x90_gate_fidelity.yaml").write_text(
        "data:\n  Q00: 0.99\n  Q01: 0.98\n  Q02: 0.91\n",
        encoding="utf-8",
    )
    (params_dir / "zx90_gate_fidelity.yaml").write_text(
        "data:\n  Q00-Q01: 0.97\n  Q01-Q02: 0.96\n",
        encoding="utf-8",
    )
    (params_dir / "average_readout_fidelity.yaml").write_text(
        "data:\n  Q00: 0.99\n  Q01: 0.98\n  Q02: 0.97\n",
        encoding="utf-8",
    )

    topology = build_device_topology(
        calib_note_path=calib_note_path,
        params_dir=params_dir,
        request={
            "name": "request-device",
            "device_id": "request-device",
            "qubits": ["Q00", "1", "2"],
            "exclude_couplings": ["Q01-Q02"],
            "condition": {
                "qubit_fidelity": {"min": 0.95, "max": 1.0},
                "coupling_fidelity": {"min": 0.95, "max": 1.0},
                "readout_fidelity": {"min": 0.95, "max": 1.0},
                "only_maximum_connected": False,
            },
        },
        topology={"couplings": [[0, 1], [1, 2]]},
    )

    assert topology["name"] == "request-device"
    assert [qubit["physical_id"] for qubit in topology["qubits"]] == [0, 1]
    assert topology["couplings"] == [
        {
            "control": 0,
            "target": 1,
            "fidelity": 0.97,
            "gate_duration": {"rzx90": 272},
        }
    ]


def test_build_device_topology_request_filters_by_selected_metrics(tmp_path) -> None:
    calib_note_path = tmp_path / "calib_note.json"
    calib_note_path.write_text(
        json.dumps(
            {
                "drag_hpi_params": {
                    "Q00": {"duration": 16},
                    "Q01": {"duration": 18},
                    "Q02": {"duration": 20},
                },
                "drag_pi_params": {
                    "Q00": {"duration": 24},
                    "Q01": {"duration": 26},
                    "Q02": {"duration": 28},
                },
                "cr_params": {
                    "Q00-Q01": {"duration": 272},
                    "Q01-Q02": {"duration": 288},
                },
            }
        ),
        encoding="utf-8",
    )
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "x90_gate_fidelity.yaml").write_text(
        "data:\n  Q00: 0.99\n  Q01: 0.99\n  Q02: 0.99\n",
        encoding="utf-8",
    )
    (params_dir / "custom_qubit_score.yaml").write_text(
        "data:\n  Q00: 0.92\n  Q01: 0.96\n  Q02: 0.98\n",
        encoding="utf-8",
    )
    (params_dir / "custom_coupling_score.yaml").write_text(
        "data:\n  Q00-Q01: 0.94\n  Q01-Q02: 0.99\n",
        encoding="utf-8",
    )
    (params_dir / "custom_readout_score.yaml").write_text(
        "data:\n  Q00: 0.99\n  Q01: 0.97\n  Q02: 0.93\n",
        encoding="utf-8",
    )

    topology = build_device_topology(
        calib_note_path=calib_note_path,
        params_dir=params_dir,
        request={
            "qubits": ["0", "1", "2"],
            "condition": {
                "qubit_fidelity": {
                    "metric": "custom_qubit_score",
                    "min": 0.95,
                    "max": 1.0,
                },
                "coupling_fidelity": {
                    "metric": "custom_coupling_score",
                    "min": 0.95,
                    "max": 1.0,
                },
                "readout_fidelity": {
                    "metric": "custom_readout_score",
                    "min": 0.95,
                    "max": 1.0,
                    "is_within_24h": True,
                },
                "only_maximum_connected": False,
            },
        },
        topology={"couplings": [[0, 1], [1, 2]]},
    )

    assert [qubit["physical_id"] for qubit in topology["qubits"]] == [1]
    assert topology["couplings"] == []


def test_build_device_topology_skips_couplings_without_fidelity_metric(tmp_path) -> None:
    calib_note_path = tmp_path / "calib_note.json"
    calib_note_path.write_text(
        json.dumps(
            {
                "drag_hpi_params": {"Q00": {"duration": 16}, "Q01": {"duration": 18}},
                "drag_pi_params": {"Q00": {"duration": 24}, "Q01": {"duration": 26}},
                "cr_params": {"Q00-Q01": {"duration": 272}},
            }
        ),
        encoding="utf-8",
    )
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "x90_gate_fidelity.yaml").write_text(
        "data:\n  Q00: 0.99\n  Q01: 0.98\n",
        encoding="utf-8",
    )
    (params_dir / "average_readout_fidelity.yaml").write_text(
        "data:\n  Q00: 0.99\n  Q01: 0.98\n",
        encoding="utf-8",
    )

    topology = build_device_topology(
        calib_note_path=calib_note_path,
        params_dir=params_dir,
        topology={"couplings": [[0, 1]]},
        qubits=[0, 1],
        only_maximum_connected=False,
    )

    assert [qubit["physical_id"] for qubit in topology["qubits"]] == [0, 1]
    assert topology["couplings"] == []


def test_write_device_topology_cli(tmp_path) -> None:
    calib_note_path = tmp_path / "calib_note.json"
    calib_note_path.write_text(
        json.dumps(
            {
                "drag_hpi_params": {"Q00": {"duration": 16}, "Q01": {"duration": 16}},
                "drag_pi_params": {"Q00": {"duration": 24}, "Q01": {"duration": 24}},
                "cr_params": {"Q00-Q01": {"duration": 272}},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "device-topology.json"

    topology = write_device_topology(
        output_path,
        calib_note_path=calib_note_path,
        qubits=[0, 1],
        topology={"couplings": [[0, 1]]},
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["couplings"] == topology[
        "couplings"
    ]
    default_image_path = tmp_path / "device-topology.svg"
    assert default_image_path.exists()
    svg = default_image_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Q00" in svg

    explicit_image_path = tmp_path / "explicit-topology.svg"
    write_device_topology(
        tmp_path / "explicit-device-topology.json",
        output_image=explicit_image_path,
        calib_note_path=calib_note_path,
        qubits=[0, 1],
        topology={"couplings": [[0, 1]]},
    )
    assert explicit_image_path.exists()

    cli_output_path = tmp_path / "cli-device-topology.json"
    assert (
        device_topology_main(
            [
                "--calib-note",
                str(calib_note_path),
                "--qubits",
                "0,1",
                "--output-json",
                str(cli_output_path),
            ]
        )
        == 0
    )
    assert json.loads(cli_output_path.read_text(encoding="utf-8"))["qubits"][0][
        "physical_id"
    ] == 0
    assert (tmp_path / "cli-device-topology.svg").exists()

    no_image_path = tmp_path / "no-image-device-topology.json"
    assert (
        device_topology_main(
            [
                "--calib-note",
                str(calib_note_path),
                "--qubits",
                "0,1",
                "--output-json",
                str(no_image_path),
                "--no-output-image",
            ]
        )
        == 0
    )
    assert not (tmp_path / "no-image-device-topology.svg").exists()

    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "name": "cli-request-device",
                "device_id": "cli-request-device",
                "qubits": ["1"],
                "exclude_couplings": [],
                "condition": {
                    "qubit_fidelity": {"min": 0.0, "max": 1.0},
                    "coupling_fidelity": {"min": 0.0, "max": 1.0},
                    "readout_fidelity": {"min": 0.0, "max": 1.0},
                    "only_maximum_connected": False,
                },
            }
        ),
        encoding="utf-8",
    )
    request_output_path = tmp_path / "request-device-topology.json"
    assert (
        device_topology_main(
            [
                "--calib-note",
                str(calib_note_path),
                "--request-json",
                str(request_path),
                "--output-json",
                str(request_output_path),
                "--no-output-image",
            ]
        )
        == 0
    )
    request_topology = json.loads(request_output_path.read_text(encoding="utf-8"))
    assert request_topology["name"] == "cli-request-device"
    assert [qubit["physical_id"] for qubit in request_topology["qubits"]] == [1]


def test_build_device_topology_svg_renders_topology() -> None:
    svg = build_device_topology_svg(
        {
            "name": "svg-device",
            "qubits": [
                {
                    "id": 0,
                    "physical_id": 0,
                    "position": {"x": 0, "y": 0},
                    "fidelity": 0.99,
                },
                {
                    "id": 1,
                    "physical_id": 1,
                    "position": {"x": 1, "y": 0},
                    "fidelity": 0.94,
                },
            ],
            "couplings": [{"control": 0, "target": 1, "fidelity": 0.97}],
            "calibrated_at": "2026-01-02T03:04:05Z",
        }
    )

    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "svg-device" in svg
    assert "2 qubits / 1 directed couplings" in svg
    assert "Q00" in svg
    assert "Q01" in svg
    assert "q0 -> q1" in svg


def test_device_topology_examples_are_loadable() -> None:
    topology = json.loads(
        (REPO_ROOT / "examples" / "device-topology.json").read_text(encoding="utf-8")
    )
    svg = (REPO_ROOT / "examples" / "device-topology.svg").read_text(encoding="utf-8")

    assert topology["name"] == "4Q-DEMO"
    assert len(topology["qubits"]) == 4
    assert len(topology["couplings"]) == 3
    assert "<svg" in svg
    assert "q0 -> q1" in svg


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


def test_sampler_constructor_options_set_delegate_defaults() -> None:
    provider = QubexProvider(num_qubits=1)
    circuit = QuantumCircuit(1)
    circuit.measure_all()

    sampler = provider.get_sampler(default_shots=32)
    result = sampler.run([circuit]).result()

    assert result[0].data.meas.num_shots == 32


def test_estimator_uses_backend_estimator_only_with_executor() -> None:
    from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator

    class RecordingExecutor:
        def run(self, run_input, **options):
            raise NotImplementedError

    hardware_backend = QubexProvider(
        num_qubits=1, executor=RecordingExecutor()
    ).get_backend()
    simulator_backend = QubexProvider(num_qubits=1).get_backend()

    assert isinstance(
        QubexEstimatorV2(hardware_backend)._delegate, BackendEstimatorV2
    )
    assert isinstance(
        QubexEstimatorV2(simulator_backend)._delegate, StatevectorEstimator
    )


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


def test_qubex_executor_rejects_empty_circuit_iterable() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    executor = QubexPulseExecutor(FakeExperiment())

    with pytest.raises(TypeError, match="non-empty"):
        executor.run([])
    with pytest.raises(TypeError, match="non-empty"):
        executor.validate([])


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

    result = provider.get_backend().run(
        circuit,
        shots=5,
        seed_simulator=1234,
        acquisition_timeout=30.0,
    ).result()

    assert isinstance(provider.get_backend()._executor, QubexPulseExecutor)
    assert result.get_counts() == {"01": 3, "10": 2}
    execute_call = qubex.measurement_service.calls[0]
    assert execute_call["n_shots"] == 5
    assert execute_call["state_classification"] is True
    assert execute_call["final_measurement"] is False
    assert execute_call["acquisition_timeout"] == 30.0
    assert "seed_simulator" not in execute_call


def test_qubex_executor_implicit_final_measurement_sets_memory_slots(monkeypatch) -> None:
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

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()
    circuit = QuantumCircuit(2)
    circuit.x(0)

    result = backend.run(circuit, shots=5).result()

    assert result.results[0].header["memory_slots"] == 2
    assert result.get_counts() == {"01": 3, "10": 2}
    assert experiment.measurement_service.calls[0]["final_measurement"] is True


def test_qubex_executor_rejects_disabled_implicit_final_measurement(monkeypatch) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return {"counts": {"1": 1}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()
    circuit = QuantumCircuit(1)
    circuit.x(0)

    with pytest.raises(ValueError, match="final_measurement=False"):
        backend.run(circuit, shots=1, final_measurement=False).result()
    assert experiment.measurement_service.calls == []


def test_qubex_executor_rejects_disabled_state_classification(monkeypatch) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return {"counts": {"1": 1}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="state_classification=True"):
        backend.run(circuit, shots=1, state_classification=False).result()
    assert experiment.measurement_service.calls == []


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("memory", "False"),
        ("state_classification", "False"),
        ("final_measurement", "False"),
        ("plot", "False"),
    ],
)
def test_qubex_executor_rejects_non_bool_options(
    option_name,
    option_value,
    monkeypatch,
) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return {"counts": {"1": 1}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match=option_name):
        backend.run(circuit, shots=1, **{option_name: option_value}).result()
    assert experiment.measurement_service.calls == []


def test_qubex_executor_validates_options_before_schedule_build(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return {"counts": {"1": 1}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    def fail_import():
        raise AssertionError("schedule construction should not start")

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", fail_import)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="state_classification=True"):
        backend.run(circuit, shots=1, state_classification=False).result()


@pytest.mark.parametrize(
    "raw_result",
    [
        {"counts": {"1": 3, "0": 2}},
        {"1": 3, "0": 2},
        SimpleNamespace(counts={"1": 3, "0": 2}),
    ],
)
def test_qubex_executor_accepts_mapping_counts(raw_result, monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return raw_result

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    result = backend.run(circuit, shots=5).result()

    assert result.get_counts() == {"1": 3, "0": 2}


def test_qubex_executor_accepts_no_argument_get_counts(monkeypatch) -> None:
    class FakeRawResult:
        def get_counts(self):
            return {(1,): 3, (0,): 2}

    class FakeMeasurementService:
        def execute(self, **kwargs):
            return FakeRawResult()

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    result = backend.run(circuit, shots=5).result()

    assert result.get_counts() == {"1": 3, "0": 2}


@pytest.mark.parametrize(
    "raw_result",
    [
        {"counts": {"1": 2, "0": 1}, "memory": ["1", "0", "1"]},
        SimpleNamespace(
            counts={"1": 2, "0": 1},
            memory=[(1,), (0,), (1,)],
        ),
    ],
)
def test_qubex_executor_uses_raw_memory_when_available(raw_result, monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return raw_result

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    result = backend.run(circuit, shots=3, memory=True).result()

    assert result.get_counts() == {"1": 2, "0": 1}
    assert result.get_memory() == ["1", "0", "1"]


def test_qubex_executor_skips_raw_memory_when_memory_false(monkeypatch) -> None:
    class FakeRawResult:
        def get_counts(self):
            return {"1": 3, "0": 2}

        def get_memory(self):
            raise AssertionError("raw memory should not be read")

    class FakeMeasurementService:
        def execute(self, **kwargs):
            return FakeRawResult()

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    result = backend.run(circuit, shots=5, memory=False).result()

    assert result.get_counts() == {"1": 3, "0": 2}


def test_qubex_executor_accepts_no_argument_get_memory(monkeypatch) -> None:
    class FakeRawResult:
        def get_counts(self):
            return {"1": 2, "0": 1}

        def get_memory(self):
            return ["1", "0", "1"]

    class FakeMeasurementService:
        def execute(self, **kwargs):
            return FakeRawResult()

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    result = backend.run(circuit, shots=3, memory=True).result()

    assert result.get_memory() == ["1", "0", "1"]


def test_qubex_executor_rejects_count_total_mismatch(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return {"counts": {"1": 2, "0": 1}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="count total does not match"):
        backend.run(circuit, shots=5).result()


def test_qubex_executor_accepts_integer_string_shots(monkeypatch) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return {"counts": {"1": 3, "0": 2}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    result = backend.run(circuit, shots="5").result()

    assert result.get_counts() == {"1": 3, "0": 2}
    assert experiment.measurement_service.calls[0]["n_shots"] == 5


@pytest.mark.parametrize("shots", [0, -1, 2.5, True, "2.5"])
def test_qubex_executor_rejects_invalid_shots(shots, monkeypatch) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return {"counts": {"1": 1}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="positive integer"):
        backend.run(circuit, shots=shots)
    assert experiment.measurement_service.calls == []


def test_qubex_executor_accepts_integer_string_counts(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return {"counts": {"1": "3", "0": "2"}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    result = backend.run(circuit, shots=5).result()

    assert result.get_counts() == {"1": 3, "0": 2}


@pytest.mark.parametrize("count", [2.5, -1, True, "2.5"])
def test_qubex_executor_rejects_invalid_count_values(count, monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return {"counts": {"1": count, "0": 2}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="non-negative integers"):
        backend.run(circuit, shots=5).result()


@pytest.mark.parametrize(
    ("raw_result", "message"),
    [
        ({"counts": ["1", "0"]}, "counts"),
        ({"memory": ["1", "0"]}, "counts mapping"),
    ],
)
def test_qubex_executor_rejects_invalid_count_result_shape(
    raw_result,
    message,
    monkeypatch,
) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return raw_result

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1, name="shape_error")
    circuit.measure(0, 0)

    with pytest.raises(TypeError, match=message):
        backend.run(circuit, shots=1).result()


def test_qubex_executor_rejects_memory_length_mismatch(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return {
                "counts": {"1": 2, "0": 1},
                "memory": ["1", "0"],
            }

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="memory length does not match"):
        backend.run(circuit, shots=3, memory=True).result()


def test_qubex_executor_result_error_includes_circuit_name(monkeypatch) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.calls = 0

        def execute(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"counts": {"1": 1}}
            return {"counts": {"1": 2}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    first = QuantumCircuit(1, 1, name="ok_circuit")
    first.measure(0, 0)
    second = QuantumCircuit(1, 1, name="bad_circuit")
    second.measure(0, 0)

    with pytest.raises(ValueError, match="bad_circuit"):
        backend.run([first, second], shots=1).result()


def test_qubex_executor_rejects_empty_raw_memory(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return {
                "counts": {"1": 2, "0": 1},
                "memory": [],
            }

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match="memory length does not match"):
        backend.run(circuit, shots=3, memory=True).result()


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
    assert backend.target["ecr"][(0, 1)].duration == pytest.approx(24e-9)
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(24e-9)
    assert backend.target["measure"][(0,)].duration == pytest.approx(20e-9)
    assert backend.target["rz"][(0,)].duration == 0.0


def test_native_basis_target_exposes_ecr_without_cx_cz() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        coupling_map=[(0, 1)],
        basis_gates=QUBEX_NATIVE_BASIS_GATES,
    ).get_backend()

    assert "ecr" in backend.target.operation_names
    assert "cx" not in backend.target.operation_names
    assert "cz" not in backend.target.operation_names
    assert backend.target["ecr"][(0, 1)].duration == pytest.approx(24e-9)


def test_native_flag_target_exposes_ecr_without_cx_cz() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        coupling_map=[(0, 1)],
        native=True,
    ).get_backend()

    assert "ecr" in backend.target.operation_names
    assert "cx" not in backend.target.operation_names
    assert "cz" not in backend.target.operation_names


def test_native_basis_transpiles_cx_to_ecr() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        coupling_map=[(0, 1)],
        basis_gates=QUBEX_NATIVE_BASIS_GATES,
    ).get_backend()
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)

    transpiled = transpile(circuit, backend, optimization_level=1)

    assert "cx" not in transpiled.count_ops()
    assert "ecr" in transpiled.count_ops()


def test_provider_from_experiment_can_use_device_topology_target() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 2e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    topology = {
        "name": "topology-device",
        "qubits": [
            {
                "id": 0,
                "physical_id": 5,
                "qubit_lifetime": {"t1": 25.0, "t2": 30.0},
                "gate_duration": {"sx": 16, "x": 24},
            },
            {
                "id": 1,
                "physical_id": 7,
                "qubit_lifetime": {"t1": 20.0, "t2": 22.0},
                "gate_duration": {"sx": 18, "x": 26},
            },
        ],
        "couplings": [
            {"control": 0, "target": 1, "gate_duration": {"rzx90": 272}},
        ],
    }

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        device_topology=topology,
    ).get_backend()

    assert backend.qubex is topology
    assert backend.target.num_qubits == 2
    assert backend.target.qubit_properties[0].t1 == pytest.approx(25e-6)
    assert (0, 1) in backend.target["cx"]
    assert (1, 0) not in backend.target["cx"]
    assert backend.target["sx"][(0,)].duration == pytest.approx(4e-9)
    assert backend._executor.qubit_labels == ("Q05", "Q07")


def test_provider_from_device_topology_native_flag_exposes_ecr_without_cx_cz() -> None:
    topology = {
        "name": "topology-device",
        "qubits": [
            {"id": 0, "physical_id": 5},
            {"id": 1, "physical_id": 7},
        ],
        "couplings": [
            {"control": 0, "target": 1, "gate_duration": {"rzx90": 272}},
        ],
    }

    backend = QubexProvider.from_device_topology(
        topology,
        native=True,
    ).get_backend()

    assert "ecr" in backend.target.operation_names
    assert "cx" not in backend.target.operation_names
    assert "cz" not in backend.target.operation_names


def test_provider_from_experiment_allows_explicit_topology_qubit_labels() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    topology = {
        "qubits": [
            {"id": 0, "physical_id": 7},
            {"id": 1, "physical_id": 42},
        ],
        "couplings": [
            {"control": 0, "target": 1, "gate_duration": {"rzx90": 272}},
        ],
    }

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        device_topology=topology,
        qubit_labels=("Q007", "Q042"),
    ).get_backend()

    assert backend._executor.qubit_labels == ("Q007", "Q042")
    assert backend.target.num_qubits == 2


def test_provider_from_experiment_config_infers_qubits_from_topology(monkeypatch) -> None:
    created = {}

    class FakeExperiment:
        dt = 1e-9

        def __init__(self, *, system_id, chip_id, qubits, **options):
            created["system_id"] = system_id
            created["chip_id"] = chip_id
            created["qubits"] = qubits
            created["options"] = options
            self.qubit_labels = tuple(qubits)
            self.pulse = DurationPulse()
            self.measurement_service = SimpleNamespace(execute=lambda **kwargs: None)

    monkeypatch.setitem(
        sys.modules,
        "qubex",
        SimpleNamespace(Experiment=FakeExperiment),
    )
    topology = {
        "qubits": [
            {"id": 0, "physical_id": 5},
            {"id": 1, "physical_id": 7},
        ],
        "couplings": [
            {"control": 0, "target": 1, "gate_duration": {"rzx90": 272}},
        ],
    }

    backend = QubexProvider.from_experiment_config(
        system_id="system",
        chip_id="chip",
        device_topology=topology,
        config_dir="config",
    ).get_backend()

    assert created["qubits"] == ["Q05", "Q07"]
    assert created["options"] == {"config_dir": "config"}
    assert backend.qubex is topology
    assert backend._executor.qubit_labels == ("Q05", "Q07")


def test_provider_from_experiment_config_keeps_experiment_label_inference(monkeypatch) -> None:
    class FakeExperiment:
        dt = 1e-9

        def __init__(self, *, system_id, chip_id, qubits, **options):
            self.qubit_labels = ("Q00", "Q01")
            self.pulse = DurationPulse()
            self.measurement_service = SimpleNamespace(execute=lambda **kwargs: None)

    monkeypatch.setitem(
        sys.modules,
        "qubex",
        SimpleNamespace(Experiment=FakeExperiment),
    )

    backend = QubexProvider.from_experiment_config(
        system_id="system",
        qubits=[0, 1],
    ).get_backend()

    assert backend._executor.qubit_labels == ("Q00", "Q01")


def test_provider_from_experiment_config_forwards_backend_basis_gates(monkeypatch) -> None:
    class FakeExperiment:
        dt = 1e-9

        def __init__(self, *, system_id, chip_id, qubits, **options):
            assert "basis_gates" not in options
            self.qubit_labels = tuple(f"Q0{qubit}" for qubit in qubits)
            self.pulse = DurationPulse()
            self.measurement_service = SimpleNamespace(execute=lambda **kwargs: None)

    monkeypatch.setitem(
        sys.modules,
        "qubex",
        SimpleNamespace(Experiment=FakeExperiment),
    )

    backend = QubexProvider.from_experiment_config(
        system_id="system",
        qubits=[0, 1],
        coupling_map=[(0, 1)],
        basis_gates=QUBEX_NATIVE_BASIS_GATES,
    ).get_backend()

    assert "ecr" in backend.target.operation_names
    assert "cx" not in backend.target.operation_names


def test_provider_from_experiment_config_native_flag_is_backend_only(monkeypatch) -> None:
    class FakeExperiment:
        dt = 1e-9

        def __init__(self, *, system_id, chip_id, qubits, **options):
            assert "native" not in options
            assert "basis_gates" not in options
            self.qubit_labels = tuple(f"Q0{qubit}" for qubit in qubits)
            self.pulse = DurationPulse()
            self.measurement_service = SimpleNamespace(execute=lambda **kwargs: None)

    monkeypatch.setitem(
        sys.modules,
        "qubex",
        SimpleNamespace(Experiment=FakeExperiment),
    )

    backend = QubexProvider.from_experiment_config(
        system_id="system",
        qubits=[0, 1],
        coupling_map=[(0, 1)],
        native=True,
    ).get_backend()

    assert "ecr" in backend.target.operation_names
    assert "cx" not in backend.target.operation_names


def test_backend_validate_builds_schedule_without_executing(monkeypatch) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return None

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)

    schedules = backend.validate(circuit)

    assert len(schedules) == 1
    assert any(op[0] == "add" and op[1] == "Q0" for op in schedules[0].ops)
    assert experiment.measurement_service.calls == []


def test_provider_validate_delegates_to_backend(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(
        executor_module,
        "_import_pulse_schedule",
        lambda: DurationSchedule,
    )
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    provider = QubexProvider.from_experiment(FakeExperiment())
    circuit = QuantumCircuit(1)
    circuit.x(0)

    schedules = provider.validate(circuit)

    assert len(schedules) == 1


def test_validate_without_qubex_executor_raises() -> None:
    backend = QubexProvider(num_qubits=1).get_backend()
    circuit = QuantumCircuit(1)

    with pytest.raises(ValueError, match="requires a Qubex executor"):
        backend.validate(circuit)


def test_qubex_executor_rejects_circuit_with_too_many_qubits(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return {"counts": {"1": 1}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(2, 2)
    circuit.measure([0, 1], [0, 1])

    with pytest.raises(ValueError, match="more qubits"):
        backend.validate(circuit)
    with pytest.raises(ValueError, match="more qubits"):
        backend.run(circuit, shots=1).result()


def test_scheduled_circuit_start_times_become_qubex_blanks(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        # Pin the sampling period the timing arithmetic below assumes,
        # regardless of whether qxpulse (default 2 ns) is installed.
        dt = 1e-9

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


def test_legacy_device_gateway_timing_ignores_qiskit_start_times(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        # Pin the sampling period the timing arithmetic below assumes,
        # regardless of whether qxpulse (default 2 ns) is installed.
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.x(1)
    circuit.rz(pi / 2, 0)
    circuit.cx(0, 1)
    circuit.delay(5, 1, unit="dt")
    circuit.measure([0, 1], [0, 1])
    circuit._op_start_times = [0, 20, 30, 40, 70, 75, 75]
    circuit._duration = 75
    circuit._unit = "dt"

    schedule = QubexPulseExecutor(
        FakeExperiment(),
        timing_policy="legacy_device_gateway",
    ).build_schedule(circuit)

    add_ops = [op for op in schedule.ops if op[0] == "add"]
    call_ops = [op for op in schedule.ops if op[0] == "call"]
    barrier_indices = [index for index, op in enumerate(schedule.ops) if op[0] == "barrier"]
    virtual_z_indices = [
        index
        for index, op in enumerate(schedule.ops)
        if op[0] == "add" and op[2].name == "virtual_z"
    ]

    assert schedule.labels == ["Q0", "Q1", "Q0-Q1"]
    assert [(op[1], op[2].name) for op in add_ops] == [
        ("Q0", "x180-Q0"),
        ("Q1", "x180-Q1"),
        ("Q0", "virtual_z"),
        ("Q1", "blank"),
    ]
    assert add_ops[-1][2].duration == 5
    assert call_ops[0][1].ops == [("cx", "Q0", "Q1")]
    assert not any(op[2].name.startswith("readout") for op in add_ops)
    assert barrier_indices[0] < virtual_z_indices[0] < barrier_indices[1]


def test_provider_from_experiment_forwards_timing_policy() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        timing_policy="legacy_device_gateway",
    ).get_backend()

    assert backend._executor._timing_policy == "legacy_device_gateway"


def test_qubex_executor_rejects_unknown_timing_policy() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    with pytest.raises(ValueError, match="timing_policy"):
        QubexPulseExecutor(FakeExperiment(), timing_policy="serial")


def test_qubex_executor_converts_ecr_to_echoed_zx90(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        # Pin the sampling period the timing arithmetic below assumes,
        # regardless of whether qxpulse (default 2 ns) is installed.
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(2)
    circuit.ecr(0, 1)

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    calls = [op for op in schedule.ops if op[0] == "call"]
    assert len(calls) == 1
    assert calls[0][1].ops == [("zx90", "Q0", "Q1", "echo")]


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


def test_unscheduled_measurement_barriers_readout_to_qubit_channel(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    readout_index = next(
        index
        for index, op in enumerate(schedule.ops)
        if op[0] == "add" and op[2].name == "readout-Q0"
    )
    barriers_before_readout = [
        op for op in schedule.ops[:readout_index] if op[0] == "barrier"
    ]
    assert ("barrier", ["Q0", "RQ0"]) in barriers_before_readout


def test_mid_circuit_measurement_blocks_drive_channel_during_readout(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)
        # Pin the sampling period the timing arithmetic below assumes,
        # regardless of whether qxpulse (default 2 ns) is installed.
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    # x180 is 8 ns and readout is 20 ns in the duration fixtures.
    circuit = QuantumCircuit(1, 2)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.x(0)
    circuit.measure(0, 1)
    circuit._op_start_times = [0, 8, 28, 36]
    circuit._duration = 56
    circuit._unit = "dt"

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    q0_ops = [op for op in schedule.ops if op[0] == "add" and op[1] == "Q0"]
    assert [(op[2].name, op[2].duration) for op in q0_ops] == [
        ("x180-Q0", 8),
        ("blank", 20),
        ("x180-Q0", 8),
        ("blank", 20),
    ]


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


def test_qubex_executor_allows_multiplexed_readout_resource_windows() -> None:
    class FakeSchedule:
        def get_pulse_ranges(self):
            return {
                "RQ0": [range(0, 10)],
                "RQ1": [range(0, 10)],
            }

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = DurationPulse()

        def resolve_read_label(self, target, allow_legacy=False):
            return f"R{target}"

        def get_read_out_target(self, label):
            channel = SimpleNamespace(id="shared-readout")
            return SimpleNamespace(channel=channel)

    QubexPulseExecutor(FakeExperiment())._validate_resource_constraints(FakeSchedule())


def test_qubex_executor_rejects_invalid_native_pulse_schedule(monkeypatch) -> None:
    class InvalidSchedule(DurationSchedule):
        def is_valid(self):
            return False

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: InvalidSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(1)
    circuit.x(0)

    with pytest.raises(ValueError, match="Invalid Qubex pulse schedule"):
        QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)


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
    first_readout_index = min(
        index
        for index, op in enumerate(schedule.ops)
        if op[0] == "add" and op[2].name.startswith("readout")
    )
    assert not [
        op for op in schedule.ops[:first_readout_index]
        if op[0] == "add" and op[1] in {"Q2", "Q3"} and op[2].name == "blank"
    ]
    # Readout windows occupy the drive channels so later gates cannot overlap.
    readout_window_blanks = [
        op for op in schedule.ops[first_readout_index:]
        if op[0] == "add" and op[1] in {"Q2", "Q3"} and op[2].name == "blank"
    ]
    assert [blank[2].duration for blank in readout_window_blanks] == [20, 20]


def test_dynamical_decoupling_pass_manager_inserts_dd_sequence(monkeypatch) -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0",)
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(FakeExperiment()).get_backend()
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.delay(100, 0, unit="ns")
    circuit.x(0)

    dd_circuit = build_dynamical_decoupling_pass_manager(
        backend,
        sequence="xy4",
    ).run(circuit)

    assert dd_circuit.count_ops()["x"] >= 4
    assert dd_circuit.count_ops()["y"] >= 2
    assert dd_circuit.count_ops()["delay"] >= 2

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    schedule = backend._executor.build_schedule(dd_circuit)
    added_names = [
        op[2].name
        for op in schedule.ops
        if op[0] == "add" and op[1] == "Q0" and hasattr(op[2], "name")
    ]

    assert "x180-Q0" in added_names
    assert "y180-Q0" in added_names
    assert "blank" in added_names


def test_context_aware_dynamical_decoupling_pass_manager_runs() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        coupling_map=[(0, 1)],
    ).get_backend()
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.delay(100, 0, unit="ns")
    circuit.cx(0, 1)

    dd_circuit = build_dynamical_decoupling_pass_manager(
        backend,
        context_aware=True,
    ).run(circuit)

    assert dd_circuit.num_qubits == 2
    assert "delay" in dd_circuit.count_ops()


def test_topology_aware_dynamical_decoupling_helper_runs() -> None:
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        coupling_map=[(0, 1)],
    ).get_backend()
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.delay(100, 0, unit="ns")
    circuit.cx(0, 1)

    dd_circuit = build_topology_aware_dynamical_decoupling_pass_manager(
        backend,
    ).run(circuit)

    assert dd_circuit.num_qubits == 2
    assert "delay" in dd_circuit.count_ops()


def _fixed_interval_dd_backend(qubit_labels=("Q0", "Q1")):
    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        dt = 1e-9

        def __init__(self):
            self.qubit_labels = qubit_labels
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

    return QubexProvider.from_experiment(FakeExperiment()).get_backend()


def test_fixed_interval_dd_repeats_sequence_per_window() -> None:
    backend = _fixed_interval_dd_backend()
    # x180 is 8 ns in the duration fixtures; dt is 1 ns.
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.delay(400, 0, unit="ns")
    circuit.x(0)
    circuit.x(1)
    circuit.delay(100, 1, unit="ns")
    circuit.x(1)

    dd_circuit = build_dynamical_decoupling_pass_manager(
        backend,
        sequence="xx",
        pulse_interval=50e-9,
        scheduling_method="asap",
    ).run(circuit)

    # Windows: q0 400 ns -> 4 reps (8 X); q1 100 ns -> 1 rep (2 X) plus the
    # trailing 300 ns idle until the circuit end -> 3 reps (6 X). Original
    # circuit contributes 4 X.
    assert dd_circuit.count_ops()["x"] == 4 + 8 + 2 + 6


def test_fixed_interval_dd_falls_back_for_short_windows() -> None:
    backend = _fixed_interval_dd_backend(qubit_labels=("Q0",))
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.delay(12, 0, unit="ns")  # too short for even one XX repetition
    circuit.x(0)

    dd_circuit = build_dynamical_decoupling_pass_manager(
        backend,
        sequence="xx",
        pulse_interval=50e-9,
        scheduling_method="asap",
    ).run(circuit)

    assert dd_circuit.count_ops()["x"] == 2


def test_fixed_interval_dd_keeps_odd_base_sequences_identity() -> None:
    backend = _fixed_interval_dd_backend(qubit_labels=("Q0",))
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.delay(300, 0, unit="ns")  # round(300 / 100) = 3 -> bumped to 4
    circuit.x(0)

    dd_circuit = build_dynamical_decoupling_pass_manager(
        backend,
        sequence="hahn",
        pulse_interval=100e-9,
        scheduling_method="asap",
    ).run(circuit)

    inserted = dd_circuit.count_ops()["x"] - 2
    assert inserted == 4


def test_fixed_interval_dd_rejects_conflicting_options() -> None:
    backend = _fixed_interval_dd_backend(qubit_labels=("Q0",))

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_dynamical_decoupling_pass_manager(
            backend, pulse_interval=50e-9, spacing=[0.5, 0.5]
        )
    with pytest.raises(ValueError, match="context_aware"):
        build_dynamical_decoupling_pass_manager(
            backend, pulse_interval=50e-9, context_aware=True
        )
    with pytest.raises(ValueError, match="positive"):
        build_dynamical_decoupling_pass_manager(backend, pulse_interval=0.0)


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
