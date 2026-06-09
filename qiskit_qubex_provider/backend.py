"""Qiskit BackendV2 implementation for Qubex targets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qiskit.providers import BackendV2, Options
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler import Target

from .target import QubexTargetSource, build_qubex_target


class QubexBackend(BackendV2):
    """BackendV2 exposing Qubex topology and metadata to Qiskit.

    The backend builds a Qiskit :class:`~qiskit.transpiler.Target` from a Qubex
    system, target registry, qubit count, or explicit coupling map. Execution is
    delegated to Qiskit's local :class:`~qiskit.providers.basic_provider.BasicSimulator`.
    This keeps the provider usable for transpilation and local workflow tests
    while preserving the Qubex object as backend metadata for future hardware
    execution adapters.
    """

    def __init__(
        self,
        qubex: QubexTargetSource | None = None,
        *,
        name: str = "qubex_simulator",
        num_qubits: int | None = None,
        coupling_map: Iterable[tuple[int, int]] | None = None,
        basis_gates: Iterable[str] | None = None,
        dt: float | None = 1e-9,
        executor: Any | None = None,
        simulator: BasicSimulator | None = None,
        provider: Any | None = None,
        **fields: Any,
    ) -> None:
        super().__init__(provider=provider, name=name, **fields)
        self._qubex = qubex
        self._target = build_qubex_target(
            qubex,
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            dt=dt,
            description=f"Qiskit target for {name}",
        )
        self._executor = executor
        self._simulator = simulator or BasicSimulator()

    @classmethod
    def _default_options(cls) -> Options:
        """Return run options accepted by the local simulator."""
        return Options(shots=1024, memory=False, seed_simulator=None)

    @property
    def target(self) -> Target:
        """Return the Qubex-derived Qiskit target."""
        return self._target

    @property
    def max_circuits(self) -> int | None:
        """Return the maximum number of circuits per job."""
        return None

    @property
    def qubex(self) -> QubexTargetSource | None:
        """Return the source Qubex object used to build this backend."""
        return self._qubex

    def run(self, run_input: Any, **options: Any):
        """Run circuits on Qiskit's local basic simulator.

        Args:
            run_input: A circuit or iterable of circuits.
            **options: Runtime options. Backend defaults are merged first, then
                overridden by these values.
        """
        run_options = dict(self.options.__dict__)
        run_options.update(options)
        run_options = {
            key: value for key, value in run_options.items() if value is not None
        }
        if self._executor is not None:
            return self._executor.run(run_input, **run_options)
        return self._simulator.run(run_input, **run_options)
