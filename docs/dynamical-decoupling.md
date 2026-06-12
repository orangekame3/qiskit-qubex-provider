# Dynamical decoupling

Helpers for inserting dynamical decoupling (DD) sequences into idle windows
of scheduled circuits, using the backend target's calibrated durations.

Build the DD pass manager from the same backend target *after* layout and
routing have mapped the circuit to physical qubits:

```python
from qiskit import transpile
from qiskit_qubex_provider import build_dynamical_decoupling_pass_manager

physical = transpile(circuit, backend, optimization_level=1)
dd_passes = build_dynamical_decoupling_pass_manager(
    backend,
    sequence="xy4",
    scheduling_method="alap",
)
scheduled = dd_passes.run(physical)
```

Built-in sequences are `"xx"`, `"xy4"`, and `"x"`/`"hahn"`; a concrete list
of Qiskit gates can also be passed. Additional knobs (`qubits`, `spacing`,
`skip_reset_qubits`, `pulse_alignment`, `extra_slack_distribution`) are
forwarded to Qiskit's `PadDynamicalDecoupling`.

## Topology-aware DD

For coupling-context-aware X-sequence DD, prefer the explicit helper:

```python
from qiskit_qubex_provider import (
    build_topology_aware_dynamical_decoupling_pass_manager,
)

physical = transpile(circuit, backend, optimization_level=1)
dd_passes = build_topology_aware_dynamical_decoupling_pass_manager(
    backend,
    scheduling_method="alap",
)
scheduled = dd_passes.run(physical)
```

This wraps Qiskit's `ContextAwareDynamicalDecoupling`: it uses the backend
`Target` coupling map to choose mutually orthogonal X-sequences on adjacent
qubits and around CX/ECR-like interactions. It is topology-aware, but not a
global optimizer over every possible DD sequence.

The resulting scheduled circuit can be validated and executed like any other
(see [hardware-execution.md](hardware-execution.md)); inserted DD pulses and
delays become calibrated Qubex pulses and `Blank` padding. A runnable
example with pulse-level before/after timelines lives at
[examples/dynamical-decoupling.ipynb](../examples/dynamical-decoupling.ipynb).
