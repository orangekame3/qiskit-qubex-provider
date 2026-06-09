"""Provider service for Qubex-backed Qiskit backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qiskit.providers import BackendV2
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .backend import QubexBackend
from .estimator import QubexEstimatorV2
from .executor import QubexPulseExecutor
from .sampler import QubexSamplerV2
from .target import QubexTargetSource


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
        executor: Any | None = None,
        use_qubex_executor: bool = False,
        backend_cls: type[QubexBackend] = QubexBackend,
        **backend_options: Any,
    ) -> None:
        if executor is None and use_qubex_executor:
            executor = QubexPulseExecutor(qubex)
        self._backend = backend_cls(
            qubex,
            name=name,
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
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

    @classmethod
    def from_experiment(
        cls,
        experiment: Any,
        *,
        name: str = "qubex",
        execute_options: dict[str, Any] | None = None,
        **backend_options: Any,
    ) -> "QubexProvider":
        """Create a provider from an already configured Qubex Experiment."""
        executor = QubexPulseExecutor(
            experiment,
            execute_options=execute_options,
        )
        qubit_labels = executor.qubit_labels
        return cls(
            experiment,
            name=name,
            executor=executor,
            **backend_options,
        )

    @classmethod
    def from_experiment_config(
        cls,
        *,
        name: str = "qubex",
        system_id: str | None = None,
        chip_id: str | None = None,
        qubits: Iterable[str | int],
        connect_devices: bool = False,
        execute_options: dict[str, Any] | None = None,
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
        experiment = Experiment(
            system_id=system_id,
            chip_id=chip_id,
            qubits=list(qubits),
            **experiment_options,
        )
        if connect_devices:
            experiment.connect()
        return cls.from_experiment(
            experiment,
            name=name,
            execute_options=execute_options,
        )
