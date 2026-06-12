"""Sampler primitive wrapper for Qubex backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qiskit.primitives import BaseSamplerV2, SamplerPubLike

from .backend import QubexBackend


class QubexSamplerV2(BaseSamplerV2):
    """SamplerV2 implementation backed by Qiskit's backend sampler."""

    def __init__(self, backend: QubexBackend, **options: Any) -> None:
        self._backend = backend
        self._options = dict(options)
        self._delegate = self._make_delegate()

    @property
    def backend(self) -> QubexBackend:
        """Return the backend used by this sampler."""
        return self._backend

    @property
    def options(self) -> dict[str, Any]:
        """Return sampler options."""
        return dict(self._options)

    def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None):
        """Run sampler pubs and return a Qiskit primitive job."""
        if shots is not None:
            return self._delegate.run(pubs, shots=shots)
        return self._delegate.run(pubs)

    def _make_delegate(self):
        try:
            from qiskit.primitives import BackendSamplerV2

            # Constructor options (default_shots, seed_simulator, run_options)
            # belong to the delegate; run() only accepts shots.
            return BackendSamplerV2(
                backend=self._backend,
                options=self._options or None,
            )
        except ImportError:
            from qiskit.primitives import StatevectorSampler

            return StatevectorSampler()
