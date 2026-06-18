from __future__ import annotations

import importlib.util
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
    FakeQubexExperiment,
    build_device_topology,
    build_device_topology_svg,
    build_dynamical_decoupling_pass_manager,
    build_qxsimulator_system,
    build_topology_aware_dynamical_decoupling_pass_manager,
    build_qubex_target,
    filter_pulse_schedule_for_simulation,
    materialize_pulse_schedule_for_simulation,
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

def _load_bell_state_module():
    path = REPO_ROOT / "examples" / "hardware" / "bell_state.py"
    spec = importlib.util.spec_from_file_location("bell_state_example", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DurationSchedule:
    def __init__(self, channels=None, *, duration: float = 0.0, ops=None):
        self.labels = list(channels or [])
        self.ops = list(ops or [])
        self.duration = duration
        self.cached_duration = duration
        self.frequencies = {}
        self.targets = {}
        self.frame_shifts = {label: 0.0 for label in self.labels}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.barrier()

    def add(self, label, obj):
        self.ops.append(("add", label, obj))
        if label not in self.labels:
            self.labels.append(label)
        self.frame_shifts.setdefault(label, 0.0)
        if isinstance(obj, DurationVirtualZ):
            self.frame_shifts[label] -= obj.theta
        self.duration += getattr(obj, "cached_duration", 0.0)
        self.cached_duration = self.duration

    def call(self, schedule):
        self.ops.append(("call", schedule))
        for label in schedule.labels:
            if label not in self.labels:
                self.labels.append(label)
            self.frame_shifts[label] = self.frame_shifts.get(label, 0.0) + schedule.get_final_frame_shift(label)
        self.duration += schedule.duration
        self.cached_duration = self.duration

    def barrier(self, labels=None):
        self.ops.append(("barrier", labels))

    def is_valid(self):
        return True

    def set_frequency(self, label, frequency):
        self.frequencies[label] = frequency

    def get_frequency(self, label):
        return self.frequencies.get(label)

    def set_target(self, label, target):
        self.targets[label] = target

    def get_target(self, label):
        return self.targets.get(label)

    def get_final_frame_shift(self, label):
        return self.frame_shifts.get(label, 0.0)


class DurationBlank(DurationObject):
    def __init__(self, duration: float):
        super().__init__("blank", duration)


class DurationVirtualZ(DurationObject):
    def __init__(self, theta: float):
        super().__init__("virtual_z", 0)
        self.theta = theta


def test_bell_state_topology_helper_outputs_native_durations_only() -> None:
    bell_state = _load_bell_state_module()

    class Pulse(DurationPulse):
        def x90(self, target):
            return DurationObject(f"x90-{target}", 11 if target == "Q00" else 13)

        def readout(self, target):
            return DurationObject(f"readout-{target}", 120 if target == "Q00" else 128)

        def zx90(self, control, target, *, echo=True):
            return DurationSchedule([control, target], duration=251)

        def cx(self, control, target):
            return DurationSchedule([control, target], duration=301)

    topology = {
        "qubits": [
            {
                "id": 0,
                "label": "Q00",
                "gate_duration": {"rz": 0, "sx": 16, "sxdg": 16, "x": 24, "y": 24},
            },
            {
                "id": 1,
                "label": "Q01",
                "gate_duration": {"rz": 0, "sx": 18, "sxdg": 18, "x": 26, "y": 26},
            },
        ],
        "couplings": [
            {
                "control": 0,
                "target": 1,
                "gate_duration": {"rzx90": 272, "ecr": 272, "cx": 301},
            }
        ],
    }

    bell_state._apply_native_gate_durations(
        topology,
        pulse_source=SimpleNamespace(pulse=Pulse()),
    )

    assert topology["qubits"][0]["gate_duration"] == {
        "rz": 0,
        "sx": 11,
        "measure": 120,
    }
    assert topology["qubits"][1]["gate_duration"] == {
        "rz": 0,
        "sx": 13,
        "measure": 128,
    }
    assert topology["couplings"][0]["gate_duration"] == {"rzx90": 251, "cx": 301}


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
                "frequency": 5.125,
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
    assert target.qubit_properties[0].frequency == pytest.approx(5.125e9)
    assert target.qubit_properties[0].t1 == pytest.approx(25e-6)
    assert target["sx"][(0,)].duration == pytest.approx(16e-9)
    assert target["sxdg"][(0,)].duration == pytest.approx(16e-9)
    assert target["y"][(0,)].duration == pytest.approx(24e-9)
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


def test_device_topology_target_decomposes_h_for_scheduling() -> None:
    topology = {
        "name": "anemone",
        "qubits": [
            {
                "id": 0,
                "physical_id": 0,
                "gate_duration": {"rz": 0, "sx": 16, "x": 24},
            },
        ],
        "couplings": [],
    }
    backend = QubexProvider.from_device_topology(topology).get_backend()
    circuit = QuantumCircuit(1)
    circuit.h(0)

    scheduled = transpile(
        circuit,
        backend,
        scheduling_method="alap",
        optimization_level=1,
    )

    assert "h" not in backend.target.operation_names
    assert "h" not in scheduled.count_ops()


def test_device_topology_target_omits_durationless_compatibility_gates() -> None:
    topology = {
        "name": "anemone",
        "qubits": [
            {"id": 0, "physical_id": 0, "gate_duration": {"rz": 0, "sx": 16}},
        ],
        "couplings": [],
    }

    backend = QubexProvider.from_device_topology(topology).get_backend()

    assert "sx" in backend.target.operation_names
    assert "sxdg" in backend.target.operation_names
    assert "x" not in backend.target.operation_names
    assert "y" not in backend.target.operation_names
    assert "h" not in backend.target.operation_names


def test_device_topology_target_schedules_sxdg_from_sx_duration() -> None:
    topology = {
        "name": "anemone",
        "qubits": [
            {
                "id": 0,
                "physical_id": 0,
                "gate_duration": {"rz": 0, "sx": 16, "x": 24},
            },
        ],
        "couplings": [],
    }
    backend = QubexProvider.from_device_topology(topology).get_backend()
    circuit = QuantumCircuit(1)
    circuit.sxdg(0)

    transpile(
        circuit,
        backend,
        scheduling_method="alap",
        optimization_level=1,
    )

    assert backend.target["sxdg"][(0,)].duration == pytest.approx(16e-9)


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
    (params_dir / "control_frequency.yaml").write_text(
        "data:\n  Q00: 5.1\n  Q01: 5.2\n",
        encoding="utf-8",
    )
    (params_dir / "qubit_anharmonicity.yaml").write_text(
        "data:\n  Q00: -0.31\n  Q01: -0.32\n",
        encoding="utf-8",
    )
    (params_dir / "readout_frequency.yaml").write_text(
        "data:\n  Q00: 7.1\n  Q01: 7.2\n",
        encoding="utf-8",
    )
    (params_dir / "resonator_frequency.yaml").write_text(
        "data:\n  Q00: 6.9\n  Q01: 7.0\n",
        encoding="utf-8",
    )
    (params_dir / "qubit_qubit_coupling_strength.yaml").write_text(
        "data:\n  Q00-Q01: 4.5\n",
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
    assert topology["qubits"][0]["gate_duration"] == {"rz": 0, "sx": 16}
    assert "sxdg" not in topology["qubits"][0]["gate_duration"]
    assert "y" not in topology["qubits"][0]["gate_duration"]
    assert "duration_probe_failures" not in topology
    assert topology["qubits"][0]["qubit_lifetime"] == {"t1": 25.0, "t2": 30.0}
    assert topology["qubits"][0]["frequency"] == pytest.approx(5.1)
    assert topology["qubits"][0]["anharmonicity"] == pytest.approx(-0.31)
    assert topology["qubits"][0]["readout_frequency"] == pytest.approx(7.1)
    assert topology["qubits"][0]["resonator_frequency"] == pytest.approx(6.9)
    assert topology["qubits"][0]["meas_error"]["prob_meas1_prep0"] == pytest.approx(0.09)
    assert topology["couplings"] == [
        {
            "control": 0,
            "target": 1,
            "fidelity": 0.97,
            "gate_duration": {"rzx90": 272},
            "coupling_strength_mhz": 4.5,
        }
    ]

    backend = QubexProvider.from_device_topology(topology).get_backend()
    assert backend.target.qubit_properties[0].frequency == pytest.approx(5.1e9)
    assert backend.target["sx"][(0,)].duration == pytest.approx(16e-9)
    assert backend.target["sxdg"][(0,)].duration == pytest.approx(16e-9)
    assert backend.target["ecr"][(0, 1)].duration == pytest.approx(272e-9)
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(272e-9)


def test_build_qxsimulator_system_uses_device_topology_metadata() -> None:
    pytest.importorskip("qxsimulator")
    topology = {
        "qubits": [
            {
                "id": 0,
                "physical_id": 28,
                "label": "Q28",
                "frequency": 7.9,
                "anharmonicity": -0.31,
                "qubit_lifetime": {"t1": 25.0, "t2": 30.0},
            },
            {
                "id": 1,
                "physical_id": 25,
                "label": "Q25",
                "frequency": 8.7,
                "anharmonicity": -0.32,
                "qubit_lifetime": {"t1": 20.0, "t2": 22.0},
            },
        ],
        "couplings": [
            {"control": 0, "target": 1, "coupling_strength_mhz": 4.5}
        ],
    }

    system = build_qxsimulator_system(topology)

    assert system.object_labels == ["Q28", "Q25"]
    assert system.get_object("Q28").frequency == pytest.approx(7.9)
    assert system.get_object("Q28").anharmonicity == pytest.approx(-0.31)
    assert system.get_object("Q28").relaxation_rate == pytest.approx(1.0 / 25000.0)
    assert system.get_object("Q28").dephasing_rate == pytest.approx(
        1.0 / 30000.0 - 0.5 / 25000.0
    )
    assert system.get_coupling(("Q28", "Q25")).strength == pytest.approx(0.0045)


def test_fake_qubex_experiment_generates_device_topology(tmp_path) -> None:
    fake = FakeQubexExperiment.two_qubit_cr_demo(
        qubit_lifetimes=((4.0, 3.0), (5.0, 3.5)),
    )

    topology = fake.device_topology()

    assert topology["name"] == "fake-qubex-cr-demo"
    assert topology["qubits"][0]["label"] == "Q00"
    assert topology["qubits"][0]["physical_id"] == 0
    assert topology["qubits"][0]["frequency"] == pytest.approx(7.157231)
    assert topology["qubits"][0]["anharmonicity"] == pytest.approx(-0.393715)
    assert topology["qubits"][0]["qubit_lifetime"] == {"t1": 4.0, "t2": 3.0}
    assert topology["qubits"][0]["gate_duration"]["sx"] == 24.0
    assert "fidelity" not in topology["qubits"][0]
    assert "meas_error" not in topology["qubits"][0]
    assert topology["qubits"][1]["label"] == "Q01"
    assert topology["qubits"][1]["qubit_lifetime"] == {"t1": 5.0, "t2": 3.5}
    assert topology["couplings"] == [
        {
            "control": 0,
            "target": 1,
            "coupling_strength_mhz": 5.0,
        }
    ]

    calibrated_fake = FakeQubexExperiment.two_qubit_cr_demo(
        rzx90_duration=112.0,
        cx_duration=112.0,
    )
    topology_path = calibrated_fake.write_device_topology(
        tmp_path / "device-topology.json"
    )
    loaded = json.loads(topology_path.read_text(encoding="utf-8"))
    backend = QubexProvider.from_device_topology(loaded).get_backend()

    assert backend.target.qubit_properties[0].frequency == pytest.approx(7.157231e9)
    assert backend.target["sx"][(0,)].duration == pytest.approx(24e-9)
    assert backend.target["measure"][(0,)].duration == pytest.approx(1000e-9)
    assert backend.target["ecr"][(0, 1)].duration == pytest.approx(112e-9)
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(112e-9)


def test_fake_qubex_experiment_topology_feeds_qxsimulator() -> None:
    pytest.importorskip("qxsimulator")
    topology = FakeQubexExperiment.two_qubit_cr_demo().device_topology()

    system = build_qxsimulator_system(topology)

    assert system.object_labels == ["Q00", "Q01"]
    assert system.get_object("Q00").frequency == pytest.approx(7.157231)
    assert system.get_object("Q01").anharmonicity == pytest.approx(-0.487412)
    assert system.get_coupling(("Q00", "Q01")).strength == pytest.approx(0.005)


def test_fake_qubex_experiment_exposes_calibration_api() -> None:
    pytest.importorskip("qubex")
    pytest.importorskip("qxsimulator")
    fake = FakeQubexExperiment.two_qubit_cr_demo()

    drag_result = fake.calibrate_drag_hpi_pulse(repetitions=2, plot=False)
    classifier_result = fake.build_classifier(
        fake.qubit_labels,
        n_states=2,
        n_shots=100,
        plot=False,
    )
    assert "gate_duration" not in fake.device_topology()["couplings"][0]
    cr_result = fake.obtain_cr_params(
        fake.qubit_labels[0],
        fake.qubit_labels[1],
        tomography_duration=160,
        tomography_samples=32,
        plot=False,
    )
    zx90 = fake.zx90(fake.qubit_labels[0], fake.qubit_labels[1])
    pulse_tomography = fake.pulse_tomography(
        zx90,
        initial_state={fake.qubit_labels[0]: "0"},
        n_samples=4,
        plot=False,
    )
    bell_result = fake.measure_bell_state(
        fake.qubit_labels[0],
        fake.qubit_labels[1],
        plot=False,
    )

    for qubit in fake.qubit_labels:
        assert drag_result[qubit]["duration"] == pytest.approx(24.0)
        assert drag_result[qubit]["repeat"].iloc[-1]["1"] > 0.99
        assert qubit in fake.drag_hpi_pulses
        assert classifier_result["average_readout_fidelity"][qubit] > 0.9
        assert qubit in fake.classifiers
        qubit_topology = next(
            item for item in fake.device_topology()["qubits"] if item["label"] == qubit
        )
        assert qubit_topology["meas_error"]["readout_assignment_error"] < 0.1
    assert fake.device_topology()["couplings"][0]["gate_duration"] == {
        "rzx90": fake.rzx90_duration,
        "cx": fake.cx_duration,
    }
    assert fake.rzx90_duration > cr_result["cr_param"]["duration"]
    assert set(pulse_tomography.keys()) == set(fake.qubit_labels)
    assert bell_result["raw"].shape == (4,)
    assert float(sum(bell_result["raw"])) <= 1.0


def test_fake_qubex_experiment_exposes_pulse_schedule_rb_api() -> None:
    pytest.importorskip("qubex")
    pytest.importorskip("qxsimulator")
    fake = FakeQubexExperiment.two_qubit_cr_demo()
    for qubit in fake.qubit_labels:
        fake.calibrate_drag_hpi_pulse(qubit, repetitions=1, plot=False)
        fake.calibrate_drag_pi_pulse(qubit, repetitions=1, plot=False)
    fake.obtain_cr_params(
        fake.qubit_labels[0],
        fake.qubit_labels[1],
        tomography_duration=160,
        tomography_samples=32,
        plot=False,
    )

    cr_label = f"{fake.qubit_labels[0]}-{fake.qubit_labels[1]}"
    sequence = fake.rb_sequence(cr_label, n=1, seed=1)
    result = fake.randomized_benchmarking(
        cr_label,
        n_cliffords_range=[0, 1],
        n_trials=1,
        seeds=[1],
        plot=False,
    )

    assert list(sequence.labels) == [fake.qubit_labels[0], cr_label, fake.qubit_labels[1]]
    assert result[cr_label]["n_cliffords"].tolist() == [0, 1]
    assert result[cr_label]["mean"][0] == pytest.approx(1.0)
    assert 0.0 <= result[cr_label]["mean"][1] <= 1.0


def test_filter_pulse_schedule_for_simulation_keeps_active_channels() -> None:
    qxpulse = pytest.importorskip("qxpulse")
    with qxpulse.PulseSchedule(["Q0", "Q1"]) as schedule:
        schedule.add("Q0", qxpulse.Rect(duration=4, amplitude=0.1))
        schedule.add("Q1", qxpulse.Blank(4))
    schedule.set_frequency("Q0", 5.0)
    schedule._channels["Q0"].target = "Q0"

    filtered = filter_pulse_schedule_for_simulation(schedule)

    assert filtered.labels == ["Q0"]
    assert filtered.get_frequency("Q0") == pytest.approx(5.0)
    assert filtered.get_target("Q0") == "Q0"


def test_build_device_topology_can_use_pulse_source_durations(tmp_path) -> None:
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
    def write_metric(path, values):
        path.write_text(
            "data:\n"
            + "".join(f"  {key}: {value}\n" for key, value in values.items()),
            encoding="utf-8",
        )

    write_metric(params_dir / "x90_gate_fidelity.yaml", {"Q00": 0.99, "Q01": 0.98})
    write_metric(params_dir / "zx90_gate_fidelity.yaml", {"Q00-Q01": 0.97})
    write_metric(params_dir / "average_readout_fidelity.yaml", {"Q00": 0.96, "Q01": 0.95})

    class Pulse(DurationPulse):
        def readout(self, target):
            return DurationObject(f"readout-{target}", 120 if target == "Q00" else 128)

        def x90(self, target, *, valid_days=None):
            assert valid_days == 5
            return DurationObject(f"x90-{target}", 11 if target == "Q00" else 13)

        def x180(self, target, *, valid_days=None):
            assert valid_days == 5
            return DurationObject(f"x180-{target}", 21 if target == "Q00" else 23)

        def zx90(self, control, target, *, echo=True):
            return DurationSchedule([control, target, f"{control}-{target}"], duration=251)

        def cx(self, control, target):
            return DurationSchedule([control, target, f"{control}-{target}"], duration=301)

    topology = build_device_topology(
        calib_note_path=calib_note_path,
        params_dir=params_dir,
        pulse_source=Pulse(),
        calibration_valid_days=5,
    )

    assert topology["qubits"][0]["gate_duration"] == {
        "rz": 0,
        "sx": 11,
        "measure": 120,
    }
    assert "sxdg" not in topology["qubits"][0]["gate_duration"]
    assert "y" not in topology["qubits"][0]["gate_duration"]
    assert topology["couplings"][0]["gate_duration"] == {"rzx90": 251, "cx": 301}
    assert "duration_probe_failures" not in topology

    backend = QubexProvider.from_device_topology(topology).get_backend()
    assert backend.target["measure"][(0,)].duration == pytest.approx(120e-9)
    assert backend.target["ecr"][(0, 1)].duration == pytest.approx(251e-9)
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(301e-9)


def test_provider_from_experiment_uses_topology_durations_without_refresh() -> None:
    class FailingPulse(DurationPulse):
        def x90(self, target):
            raise AssertionError("duration probing should be skipped")

    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0",)
        dt = 1e-9

        def __init__(self):
            self.pulse = FailingPulse()
            self.measurement_service = FakeMeasurementService()

    topology = {
        "qubits": [
            {"id": 0, "physical_id": 0, "gate_duration": {"sx": 17, "x": 29}},
        ],
        "couplings": [],
    }

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        device_topology=topology,
    ).get_backend()

    assert backend.target["sx"][(0,)].duration == pytest.approx(17e-9)
    assert backend.target["sxdg"][(0,)].duration == pytest.approx(17e-9)
    assert backend.target["x"][(0,)].duration == pytest.approx(29e-9)
    assert backend.target["y"][(0,)].duration == pytest.approx(29e-9)


def test_build_device_topology_infers_one_current_cr_direction(tmp_path) -> None:
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
                    "Q00-Q01": {"duration": 272, "timestamp": "2026-01-01 00:00:00"},
                    "Q01-Q00": {"duration": 288, "timestamp": "2026-01-02 00:00:00"},
                },
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
    (params_dir / "zx90_gate_fidelity.yaml").write_text(
        "data:\n  Q00-Q01: 0.97\n",
        encoding="utf-8",
    )
    (params_dir / "average_readout_fidelity.yaml").write_text(
        "data:\n  Q00: 0.99\n  Q01: 0.98\n",
        encoding="utf-8",
    )

    topology = build_device_topology(
        calib_note_path=calib_note_path,
        params_dir=params_dir,
        qubits=[0, 1],
        only_maximum_connected=False,
    )

    assert topology["couplings"] == [
        {
            "control": 1,
            "target": 0,
            "fidelity": 0.97,
            "gate_duration": {"rzx90": 288},
        }
    ]

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
    assert '<path d="M ' in svg


def test_device_topology_examples_are_loadable() -> None:
    topology = json.loads(
        (REPO_ROOT / "examples" / "simulation" / "device-topology.json").read_text(encoding="utf-8")
    )
    svg = (REPO_ROOT / "examples" / "simulation" / "device-topology.svg").read_text(encoding="utf-8")

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
    assert execute_call["mode"] == "single"
    assert execute_call["state_classification"] is False
    assert execute_call["time_integration"] is True
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


def test_qubex_executor_allows_software_classification_path(monkeypatch) -> None:
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

    result = backend.run(circuit, shots=1, state_classification=False).result()

    assert result.get_counts() == {"1": 1}
    execute_call = experiment.measurement_service.calls[0]
    assert execute_call["state_classification"] is False
    assert execute_call["time_integration"] is True


def test_provider_build_classifier_delegates_to_executor() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = DurationPulse()
            self.calls = []

        def build_classifier(self, **kwargs):
            self.calls.append(kwargs)
            return "classifier-result"

    experiment = FakeExperiment()
    provider = QubexProvider.from_experiment(experiment)

    result = provider.build_classifier(targets=["Q0"], shots=12)

    assert result == "classifier-result"
    assert experiment.calls == [
        {"targets": ["Q0"], "n_shots": 12, "plot": False}
    ]


def test_backend_build_classifier_delegates_to_executor() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.calls = []

        def build_classifier(self, **kwargs):
            self.calls.append(kwargs)
            return "classifier-result"

    experiment = FakeExperiment()
    backend = QubexProvider.from_experiment(experiment).get_backend()

    result = backend.build_classifier(shots="7")

    assert result == "classifier-result"
    assert experiment.calls == [
        {"targets": ["Q0"], "n_shots": 7, "plot": False}
    ]


def test_qubex_executor_applies_readout_mitigation(monkeypatch) -> None:
    class FakeMeasurementService:
        def __init__(self):
            self.execute_kwargs = None

        def execute(self, **kwargs):
            self.execute_kwargs = kwargs
            return {"counts": {"0": 70, "1": 30}}

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()
            self.measurement_service = FakeMeasurementService()

        def get_inverse_confusion_matrix(self, targets):
            assert targets == ["Q0"]
            return [[1.25, -0.25], [-0.25, 1.25]]

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

    result = backend.run(circuit, shots=100, readout_mitigation=True).result()

    assert result.get_counts() == {"0": 80, "1": 20}
    assert "readout_mitigation" not in experiment.measurement_service.execute_kwargs


def test_qubex_executor_reports_missing_classifier_with_provider_guidance(monkeypatch) -> None:
    class FakeMeasureResult:
        def get_counts(self, targets):
            raise ValueError("Classifier is not set")

    class FakeMeasurementService:
        def execute(self, **kwargs):
            return FakeMeasureResult()

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

    with pytest.raises(ValueError, match="provider.build_classifier"):
        backend.run(circuit, shots=1).result()


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

    with pytest.raises(ValueError, match="state_classification"):
        backend.run(circuit, shots=1, state_classification="False").result()


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
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(24e-9)
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(24e-9)
    assert backend.target["measure"][(0,)].duration == pytest.approx(20e-9)
    assert backend.target["rz"][(0,)].duration == 0.0


def test_provider_from_experiment_forwards_calibration_valid_days_to_duration_probe() -> None:
    class ValidDaysPulse(DurationPulse):
        def __init__(self):
            self.calls = []

        def x90(self, target, *, valid_days=None):
            self.calls.append(("x90", target, valid_days))
            return super().x90(target)

        def x180(self, target, *, valid_days=None):
            self.calls.append(("x180", target, valid_days))
            return super().x180(target)

    class FakeMeasurementService:
        def execute(self, **kwargs):
            return None

    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = ValidDaysPulse()
            self.measurement_service = FakeMeasurementService()

    experiment = FakeExperiment()

    QubexProvider.from_experiment(
        experiment,
        calibration_valid_days=3,
    ).get_backend()

    assert ("x90", "Q0", 3) in experiment.pulse.calls
    assert ("x180", "Q0", 3) in experiment.pulse.calls


def test_qubex_executor_warns_when_pulse_duration_cannot_be_inferred() -> None:
    class PartialPulse:
        def x90(self, target):
            return DurationObject(f"x90-{target}", 4)

        def x180(self, target):
            return DurationObject(f"x180-{target}", 8)

        def y90(self, target):
            return DurationObject(f"y90-{target}", 4)

        def y180(self, target):
            return DurationObject(f"y180-{target}", 8)

        def z90(self):
            return DurationVirtualZ(pi / 2)

        def z180(self):
            return DurationVirtualZ(pi)

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")

        def __init__(self):
            self.pulse = PartialPulse()

    executor = QubexPulseExecutor(FakeExperiment(), warn_duration_failures=True)

    with pytest.warns(RuntimeWarning, match="duration"):
        durations = executor.instruction_durations_seconds()

    assert durations["x"][(0,)] == pytest.approx(8e-9)
    assert "cx" not in durations
    assert any("cx" in failure for failure in executor.duration_failures)


def test_native_basis_target_exposes_cx_without_ecr() -> None:
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

    assert set(backend.target.operation_names) == {"rz", "sx", "cx", "measure", "delay"}
    assert backend.target["cx"][(0, 1)].duration == pytest.approx(24e-9)


def test_native_flag_target_exposes_cx_without_ecr() -> None:
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

    assert set(backend.target.operation_names) == {"rz", "sx", "cx", "measure", "delay"}


def test_native_basis_transpiles_to_minimal_native_gates() -> None:
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
    circuit.h(0)
    circuit.x(1)
    circuit.cx(0, 1)

    transpiled = transpile(circuit, backend, optimization_level=1)

    assert set(transpiled.count_ops()) <= {"rz", "sx", "cx", "delay"}
    assert "cx" in transpiled.count_ops()


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
    assert backend.target["sx"][(0,)].duration == pytest.approx(16e-9)
    assert backend._executor.qubit_labels == ("Q05", "Q07")

    refreshed = QubexProvider.from_experiment(
        FakeExperiment(),
        device_topology=topology,
        refresh_instruction_durations=True,
    ).get_backend()

    assert refreshed.target["sx"][(0,)].duration == pytest.approx(4e-9)


def test_provider_from_device_topology_native_flag_exposes_cx_without_ecr() -> None:
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

    assert set(backend.target.operation_names) == {"rz", "cx", "delay"}


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

    assert set(backend.target.operation_names) == {"rz", "sx", "cx", "measure", "delay"}


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

    assert set(backend.target.operation_names) == {"rz", "sx", "cx", "measure", "delay"}


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


def test_scheduled_two_qubit_gate_start_time_aligns_cr_channel(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0", "Q1", "Q2", "Q3")
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(4, 4)
    circuit.x(0)
    circuit.x(2)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.measure(range(4), range(4))
    circuit._op_start_times = [0, 0, 20, 20, 44, 44, 44, 44]
    circuit._duration = 64
    circuit._unit = "dt"

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    cr_blanks = [
        (op[1], op[2].duration)
        for op in schedule.ops
        if op[0] == "add" and op[1] in {"Q0-Q1", "Q2-Q3"} and op[2].name == "blank"
    ]
    assert cr_blanks == [("Q0-Q1", pytest.approx(20)), ("Q2-Q3", pytest.approx(20))]


def test_qubex_executor_annotates_schedule_for_qxsimulator(monkeypatch) -> None:
    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 1e-9

        def __init__(self):
            self.pulse = DurationPulse()

        def resolve_read_label(self, target, allow_legacy=False):
            return f"R{target}"

        def get_target(self, label):
            if label == "Q0":
                return SimpleNamespace(
                    label="Q0",
                    frequency=5.0,
                    object=SimpleNamespace(label="Q0"),
                )
            if label == "Q1":
                return SimpleNamespace(
                    label="Q1",
                    frequency=5.2,
                    object=SimpleNamespace(label="Q1"),
                )
            if label == "Q0-Q1":
                return SimpleNamespace(
                    label="Q0-Q1",
                    frequency=5.2,
                    object=SimpleNamespace(label="Q0"),
                )
            if label == "RQ0":
                return SimpleNamespace(
                    label="RQ0",
                    frequency=7.0,
                    object=SimpleNamespace(label="RQ0"),
                )
            raise ValueError(label)

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(2, 1)
    circuit.sx(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    assert schedule.get_frequency("Q0") == pytest.approx(5.0)
    assert schedule.get_target("Q0") == "Q0"
    assert schedule.get_frequency("Q1") == pytest.approx(5.2)
    assert schedule.get_target("Q1") == "Q1"
    assert schedule.get_frequency("Q0-Q1") == pytest.approx(5.2)
    assert schedule.get_target("Q0-Q1") == "Q0"
    assert schedule.get_frequency("RQ0") == pytest.approx(7.0)
    assert schedule.get_target("RQ0") == "RQ0"


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


def test_provider_from_experiment_forwards_cr_frame_sync_option() -> None:
    class FakeExperiment:
        qubit_labels = ("Q0",)

        def __init__(self):
            self.pulse = DurationPulse()

    backend = QubexProvider.from_experiment(
        FakeExperiment(),
        sync_cr_channel_frames=False,
    ).get_backend()

    assert backend._executor._sync_cr_channel_frames_enabled is False


def test_qubex_executor_syncs_frame_left_by_previous_cx(monkeypatch) -> None:
    class CxFramePulse(DurationPulse):
        def cx(self, control, target):
            schedule = DurationSchedule(
                [control, target, f"{control}-{target}"],
                duration=24,
                ops=[("cx", control, target)],
            )
            schedule.add(control, DurationVirtualZ(-pi / 2))
            return schedule

    class FakeExperiment:
        qubit_labels = ("Q0", "Q1")
        dt = 1e-9

        def __init__(self):
            self.pulse = CxFramePulse()

    monkeypatch.setattr(executor_module, "_import_pulse_schedule", lambda: DurationSchedule)
    monkeypatch.setattr(executor_module, "_import_blank", lambda: DurationBlank)
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.cx(1, 0)

    schedule = QubexPulseExecutor(FakeExperiment()).build_schedule(circuit)

    cr_frame_updates = [
        op[2].theta
        for op in schedule.ops
        if op[0] == "add" and op[1] == "Q1-Q0" and op[2].name == "virtual_z"
    ]
    assert cr_frame_updates == [pytest.approx(-pi / 2)]
    assert schedule.get_final_frame_shift("Q1-Q0") == pytest.approx(pi / 2)


def test_materialized_simulation_schedule_preserves_target_final_frames() -> None:
    qx = pytest.importorskip("qubex")
    experiment = FakeQubexExperiment()
    with qx.PulseSchedule(["Q00", "Q00-Q01"]) as schedule:
        schedule.add("Q00", qx.pulse.Blank(4))
        schedule.add("Q00", qx.pulse.VirtualZ(-pi / 2))
        schedule.add("Q00-Q01", qx.pulse.Blank(4))
    experiment._annotate_schedule_metadata(schedule)

    materialized = materialize_pulse_schedule_for_simulation(schedule)

    assert materialized.get_final_frame_shift("Q00") == pytest.approx(pi / 2)
    assert materialized.get_final_frame_shift("Q00-Q01") == pytest.approx(pi / 2)


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
