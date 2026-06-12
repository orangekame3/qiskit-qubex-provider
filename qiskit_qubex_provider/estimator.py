"""Estimator primitive wrapper for Qubex backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qiskit.primitives import BaseEstimatorV2, EstimatorPubLike

from .backend import QubexBackend


class QubexEstimatorV2(BaseEstimatorV2):
    """EstimatorV2 implementation for Qubex workflows.

    When the backend has a Qubex executor configured, expectation values are
    estimated from sampled backend executions through Qiskit's backend
    estimator, so they reflect real hardware runs. Without an executor the
    estimator delegates to Qiskit's exact statevector estimator, which works
    without live Qubex hardware while sharing the same backend target used for
    transpilation.
    """

    def __init__(self, backend: QubexBackend, **options: Any) -> None:
        self._backend = backend
        self._options = dict(options)
        self._delegate = self._make_delegate()

    @property
    def backend(self) -> QubexBackend:
        """Return the backend associated with this estimator."""
        return self._backend

    @property
    def options(self) -> dict[str, Any]:
        """Return estimator options."""
        return dict(self._options)

    def run(self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None):
        """Run estimator pubs and return a Qiskit primitive job."""
        if precision is not None:
            return self._delegate.run(pubs, precision=precision)
        return self._delegate.run(pubs)

    def _make_delegate(self):
        if getattr(self._backend, "executor", None) is not None:
            try:
                from qiskit.primitives import BackendEstimatorV2

                # Constructor options (default_precision, abelian_grouping,
                # seed_simulator) belong to the delegate; run() only accepts
                # precision.
                return BackendEstimatorV2(
                    backend=self._backend,
                    options=self._options or None,
                )
            except ImportError:
                pass
        from qiskit.primitives import StatevectorEstimator

        return StatevectorEstimator()
