"""Job wrapper for synchronous Qubex executions."""

from __future__ import annotations

from typing import Any

from qiskit.providers import BackendV2, JobStatus, JobV1


class QubexJob(JobV1):
    """Completed Qiskit job backed by a synchronous Qubex execution."""

    def __init__(
        self,
        backend: BackendV2 | None,
        job_id: str,
        result: Any,
    ) -> None:
        super().__init__(backend, job_id)
        self._result = result

    def submit(self) -> None:
        """Submit is a no-op because Qubex execution has already completed."""

    def result(self, timeout: float | None = None) -> Any:
        """Return the completed result."""
        return self._result

    def status(self) -> JobStatus:
        """Return completed job status."""
        return JobStatus.DONE

    def cancel(self) -> bool:
        """Cancellation is not possible for an already completed job."""
        return False

    def cancelled(self) -> bool:
        """Return whether the job was cancelled."""
        return False

    def done(self) -> bool:
        """Return whether the job is done."""
        return True

    def running(self) -> bool:
        """Return whether the job is running."""
        return False
