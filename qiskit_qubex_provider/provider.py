"""Provider service for Qubex-backed Qiskit backends."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qiskit.providers import BackendV2
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .backend import QubexBackend
from .device_topology import qid_to_label
from .estimator import QubexEstimatorV2
from .executor import QubexPulseExecutor
from .sampler import QubexSamplerV2
from .target import QUBEX_NATIVE_BASIS_GATES, QubexTargetSource


class QubexProvider:
    """Create Qiskit backends and primitives from Qubex system metadata."""

    def __init__(
        self,
        qubex: QubexTargetSource | None = None,
        *,
        name: str = "qubex_simulator",
        num_qubits: int | None = None,
        coupling_map: Iterable[tuple[int, int]] | None = None,
        basis_gates: Iterable[str] | None = None,
        native: bool = False,
        instruction_durations: Mapping[str, Mapping[tuple[int, ...], float]] | None = None,
        executor: Any | None = None,
        use_qubex_executor: bool = False,
        readout_stagger_ns: float = 0.0,
        readout_stagger_mode: str = "start",
        readout_multiplex_groups: Mapping[str, Any] | Sequence[Sequence[str]] | None = None,
        calibration_valid_days: int | None = None,
        warn_duration_failures: bool = False,
        backend_cls: type[QubexBackend] = QubexBackend,
        **backend_options: Any,
    ) -> None:
        if executor is None and use_qubex_executor:
            executor = QubexPulseExecutor(
                qubex,
                readout_stagger_ns=readout_stagger_ns,
                readout_stagger_mode=readout_stagger_mode,
                readout_multiplex_groups=readout_multiplex_groups,
                calibration_valid_days=calibration_valid_days,
                warn_duration_failures=warn_duration_failures,
            )
        if native and basis_gates is None:
            basis_gates = QUBEX_NATIVE_BASIS_GATES
        self._backend = backend_cls(
            qubex,
            name=name,
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            instruction_durations=instruction_durations,
            executor=executor,
            provider=self,
            **backend_options,
        )

    def backends(self, name: str | None = None, **filters: Any) -> list[BackendV2]:
        """Return provider backends matching the optional name and attributes."""
        backend = self._backend
        if name is not None and backend.name != name:
            return []
        for attr, expected in filters.items():
            if not hasattr(backend, attr) or getattr(backend, attr) != expected:
                return []
        return [backend]

    def get_backend(self, name: str | None = None, **filters: Any) -> BackendV2:
        """Return the single matching Qubex backend."""
        matches = self.backends(name=name, **filters)
        if not matches:
            requested = name or self._backend.name
            raise QiskitBackendNotFoundError(f"No backend matches {requested!r}.")
        return matches[0]

    def get_sampler(
        self,
        *,
        backend: QubexBackend | None = None,
        **options: Any,
    ) -> QubexSamplerV2:
        """Return a Qiskit V2 Sampler for a Qubex backend."""
        return QubexSamplerV2(backend or self._backend, **options)

    def get_estimator(
        self,
        *,
        backend: QubexBackend | None = None,
        **options: Any,
    ) -> QubexEstimatorV2:
        """Return a Qiskit V2 Estimator for a Qubex backend."""
        return QubexEstimatorV2(backend or self._backend, **options)

    def validate(
        self,
        run_input: Any,
        *,
        backend: QubexBackend | None = None,
    ) -> list[Any]:
        """Build and preflight Qubex pulse schedules without executing them."""
        return (backend or self._backend).validate(run_input)

    def build_classifier(
        self,
        targets: Sequence[str] | str | None = None,
        *,
        backend: QubexBackend | None = None,
        **options: Any,
    ) -> Any:
        """Build Qubex state classifiers used to convert raw IQ data to counts."""
        return (backend or self._backend).build_classifier(targets=targets, **options)

    @classmethod
    def from_device_topology(
        cls,
        device_topology: str | Path | Mapping[str, Any],
        *,
        name: str | None = None,
        native: bool = False,
        **backend_options: Any,
    ) -> "QubexProvider":
        """Create a provider from a device-gateway ``device_topology.json``."""
        topology = _load_device_topology(device_topology)
        return cls(
            topology,
            name=name
            or str(topology.get("name") or topology.get("device_id") or "qubex"),
            native=native,
            **backend_options,
        )

    @classmethod
    def from_experiment(
        cls,
        experiment: Any,
        *,
        name: str = "qubex",
        device_topology: str | Path | Mapping[str, Any] | None = None,
        qubit_labels: Sequence[str] | None = None,
        execute_options: dict[str, Any] | None = None,
        timing_policy: str = "qiskit",
        readout_stagger_ns: float = 0.0,
        readout_stagger_mode: str = "start",
        readout_multiplex_groups: Mapping[str, Any] | Sequence[Sequence[str]] | None = None,
        calibration_valid_days: int | None = None,
        warn_duration_failures: bool = False,
        refresh_instruction_durations: bool = False,
        native: bool = False,
        **backend_options: Any,
    ) -> "QubexProvider":
        """Create a provider from an already configured Qubex Experiment."""
        topology = (
            _load_device_topology(device_topology)
            if device_topology is not None
            else None
        )
        executor_qubit_labels = (
            tuple(str(label) for label in qubit_labels)
            if qubit_labels is not None
            else _device_topology_qubit_labels(topology)
        )
        executor = QubexPulseExecutor(
            experiment,
            qubit_labels=executor_qubit_labels,
            execute_options=execute_options,
            timing_policy=timing_policy,
            readout_stagger_ns=readout_stagger_ns,
            readout_stagger_mode=readout_stagger_mode,
            readout_multiplex_groups=readout_multiplex_groups,
            calibration_valid_days=calibration_valid_days,
            warn_duration_failures=warn_duration_failures,
        )
        backend_options.setdefault("dt", executor.dt_seconds())
        instruction_durations = (
            executor.instruction_durations_seconds()
            if topology is None or refresh_instruction_durations
            else None
        )
        return cls(
            topology or experiment,
            name=name,
            executor=executor,
            instruction_durations=instruction_durations,
            native=native,
            **backend_options,
        )

    @classmethod
    def from_experiment_config(
        cls,
        *,
        name: str = "qubex",
        system_id: str | None = None,
        chip_id: str | None = None,
        qubits: Iterable[str | int] | None = None,
        device_topology: str | Path | Mapping[str, Any] | None = None,
        qubit_labels: Sequence[str] | None = None,
        coupling_map: Iterable[tuple[int, int]] | None = None,
        basis_gates: Iterable[str] | None = None,
        native: bool = False,
        connect_devices: bool = False,
        execute_options: dict[str, Any] | None = None,
        timing_policy: str = "qiskit",
        readout_stagger_ns: float = 0.0,
        readout_stagger_mode: str = "start",
        readout_multiplex_groups: Mapping[str, Any] | Sequence[Sequence[str]] | None = None,
        calibration_valid_days: int | None = None,
        warn_duration_failures: bool = False,
        refresh_instruction_durations: bool = False,
        **experiment_options: Any,
    ) -> "QubexProvider":
        """Create a Qubex Experiment and wrap it in a provider.

        Hardware connection is opt-in. By default this loads the Qubex session
        configuration but does not connect devices.
        """
        try:
            from qubex import Experiment
        except ImportError as exc:
            raise ImportError(
                "QubexProvider.from_experiment_config requires qubex to be installed."
            ) from exc
        topology = (
            _load_device_topology(device_topology)
            if device_topology is not None
            else None
        )
        topology_qubit_labels = _device_topology_qubit_labels(topology)
        resolved_qubits = (
            list(qubits)
            if qubits is not None
            else list(qubit_labels or topology_qubit_labels or ())
        )
        if not resolved_qubits:
            raise ValueError(
                "qubits must be supplied unless device_topology or qubit_labels "
                "can provide the Qubex qubit order."
            )
        experiment = Experiment(
            system_id=system_id,
            chip_id=chip_id,
            qubits=resolved_qubits,
            **experiment_options,
        )
        if connect_devices:
            experiment.connect()
        return cls.from_experiment(
            experiment,
            name=name,
            device_topology=topology,
            qubit_labels=qubit_labels or topology_qubit_labels,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            native=native,
            execute_options=execute_options,
            timing_policy=timing_policy,
            readout_stagger_ns=readout_stagger_ns,
            readout_stagger_mode=readout_stagger_mode,
            readout_multiplex_groups=readout_multiplex_groups,
            calibration_valid_days=calibration_valid_days,
            warn_duration_failures=warn_duration_failures,
            refresh_instruction_durations=refresh_instruction_durations,
        )


def _load_device_topology(
    device_topology: str | Path | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(device_topology, Mapping):
        return device_topology
    path = Path(device_topology)
    return json.loads(path.read_text(encoding="utf-8"))


def _device_topology_qubit_labels(
    device_topology: Mapping[str, Any] | None,
) -> tuple[str, ...] | None:
    if device_topology is None:
        return None
    qubits = device_topology.get("qubits")
    if not isinstance(qubits, list):
        return None
    physical_ids = [
        int(qubit.get("physical_id", qubit.get("id", index)))
        for index, qubit in enumerate(qubits)
        if isinstance(qubit, Mapping)
    ]
    label_width_qubits = max(physical_ids, default=len(qubits) - 1) + 1
    labels = []
    for index, qubit in enumerate(qubits):
        if not isinstance(qubit, Mapping):
            labels.append(qid_to_label(index, len(qubits)))
            continue
        label = qubit.get("label")
        if label is not None:
            labels.append(str(label))
            continue
        physical_id = int(qubit.get("physical_id", qubit.get("id", index)))
        labels.append(qid_to_label(physical_id, label_width_qubits))
    return tuple(labels)
