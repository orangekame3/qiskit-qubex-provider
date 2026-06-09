"""Estimator primitive wrapper for Qubex backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qiskit.primitives import BaseEstimatorV2, EstimatorPubLike

from .backend import QubexBackend


class QubexEstimatorV2(BaseEstimatorV2):
    """EstimatorV2 implementation for Qubex workflows.

    The estimator delegates to Qiskit's statevector estimator. This provides a
    standards-compliant primitive that works without live Qubex hardware while
    sharing the same backend target used for transpilation.
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
        run_kwargs = dict(self._options)
        if precision is not None:
            run_kwargs["precision"] = precision
        return self._delegate.run(pubs, **run_kwargs)

    def _make_delegate(self):
        from qiskit.primitives import StatevectorEstimator

        return StatevectorEstimator()
