# qiskit-qubex-provider

Qiskit provider for [Qubex](https://github.com/amachino/qubex): build Qiskit
`Target`s from Qubex system metadata, transpile and schedule against real
calibration data, and execute circuits on Qubex hardware as calibrated pulse
schedules.

## Installation

```bash
uv pip install -e .
```

Install directly from Git:

```bash
pip install "qiskit-qubex-provider @ git+https://github.com/orangekame3/qiskit-qubex-provider.git"
```

`qubex` is intentionally optional because it is commonly installed from a
local checkout. To install this provider and Qubex from Git in one command:

```bash
pip install "qiskit-qubex-provider[qubex] @ git+https://github.com/orangekame3/qiskit-qubex-provider.git"
```

The `qubex` extra also pulls in `qxdriver-quel1` (from the Qubex
repository), which `qubex.Experiment` imports even in mock mode — a plain
`pip install qubex@git+...` does not include it and fails with
`ModuleNotFoundError: No module named 'qxdriver_quel1'`.

For local Qubex development, install the checkout explicitly. The
`qxdriver-quel1` backend package is required to construct an `Experiment`:

```bash
uv pip install -e ../qubex
uv pip install -e ../qubex/packages/qxdriver-quel1
```

## Quick start (no hardware)

Transpile against Qubex topology and run on a local simulator:

```python
from qiskit import QuantumCircuit, transpile
from qiskit_qubex_provider import QubexProvider

provider = QubexProvider(num_qubits=2, coupling_map=[(0, 1)])
backend = provider.get_backend()

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

transpiled = transpile(circuit, backend)
job = backend.run(transpiled, shots=1024)
print(job.result().get_counts())
```

A Device Gateway `device-topology.json` (or a Qubex system object) gives the
target real connectivity, qubit properties, and gate durations:

```python
provider = QubexProvider.from_device_topology("device-topology.json")
```

## Run on Qubex hardware

Configure a Qubex `Experiment` and inject it. The executor converts Qiskit
circuits into Qubex `PulseSchedule`s and executes them through
`measurement_service.execute(...)`:

```python
from qubex import Experiment
from qiskit import QuantumCircuit, transpile
from qiskit_qubex_provider import QubexProvider

exp = Experiment(system_id="64Q-HF-Q1", qubits=["Q00", "Q01"], ...)
exp.connect()

provider = QubexProvider.from_experiment(exp, device_topology="device-topology.json")
backend = provider.get_backend()

transpiled = transpile(circuit, backend, scheduling_method="alap")
backend.validate(transpiled)   # preflight without executing
job = backend.run(transpiled, shots=1024)
counts = job.result().get_counts()
```

For reproducible and low-overhead scheduling, generate `device_topology` with
Qubex pulse-derived durations ahead of execution:

```python
from qiskit_qubex_provider import build_device_topology

topology = build_device_topology(
    calib_note_path="qubex-config/calibration/calib_note.json",
    params_dir="qubex-config/params",
    pulse_source=exp,
    calibration_valid_days=7,
)
provider = QubexProvider.from_experiment(exp, device_topology=topology)
```

When `device_topology` is supplied, `from_experiment(...)` uses its durations
without probing Qubex pulse methods again. Pass
`refresh_instruction_durations=True` only when you intentionally want to
re-probe pulse durations while constructing the provider.

Pass `native=True` to transpile to the Qubex execution gate set (`rz`,
`sx`, and `cx`).

For readout-crosstalk benchmarks on shared readout hardware, pass
`readout_stagger_ns=...` to `from_experiment(...)` to offset measurements that
Qiskit scheduled at the same start time and on the same readout multiplex group
while leaving the default behavior unchanged. The default
`readout_stagger_mode="start"` offsets readout start times by that step; use
`readout_stagger_mode="sequential"` to start each readout after the previous
readout in the same group has ended, plus the configured gap. Groups can be
supplied with `readout_multiplex_groups=...`; otherwise the executor uses Qubex
readout channel metadata when available.

## Primitives

```python
sampler = provider.get_sampler()      # samples via backend.run (hardware when configured)
estimator = provider.get_estimator()  # hardware-sampled with an executor, exact statevector otherwise
```

## Documentation

| Topic | Where |
| --- | --- |
| Hardware execution: setup, run options, supported gates, validation | [docs/hardware-execution.md](docs/hardware-execution.md) |
| Hardware examples: 144Qv2 script and scheduling notebook | [examples/hardware/](examples/hardware/) |
| Device topology files and target generation (incl. CLI) | [docs/device-topology.md](docs/device-topology.md) |
| Dynamical decoupling pass managers | [docs/dynamical-decoupling.md](docs/dynamical-decoupling.md) |
| Comparing pulse schedules across scheduling methods | [docs/pulse-schedule-visualization.md](docs/pulse-schedule-visualization.md) |
| Internals: frame tracking, timing model, sampling grid | [docs/hardware-execution-notes.md](docs/hardware-execution-notes.md) |
| Notebook: end-to-end tour (topology → transpile → pulses → visualization) | [examples/simulation/tutorial.ipynb](examples/simulation/tutorial.ipynb) |
| Notebook: mid-circuit measurement | [examples/simulation/mid-circuit-measurement.ipynb](examples/simulation/mid-circuit-measurement.ipynb) |
| Notebook: dynamical decoupling | [examples/simulation/dynamical-decoupling.ipynb](examples/simulation/dynamical-decoupling.ipynb) |
| Notebook: DD fidelity demonstration (Pokharel et al. protocol) | [examples/simulation/dd-demonstration.ipynb](examples/simulation/dd-demonstration.ipynb) |
| Notebook: Heisenberg dynamics, new vs deprecated scheduling | [examples/simulation/heisenberg.ipynb](examples/simulation/heisenberg.ipynb) |

## Development

Set up the dev environment with Qubex (from Git, including its
`qxdriver-quel1` backend package) and the plotting extras in one command,
then run the tests:

```bash
uv sync --extra qubex --extra plot --extra dev
uv run pytest
```

Qubex is optional for development too — without the `qubex` extra the test
suite still passes; the tutorial notebook in `examples/` is the only part
that requires it. To use a local Qubex checkout instead of Git, see the
installation section above.

Repository tests use synthetic topology and calibration fixtures only;
private Qubex configuration or measured device data should stay out of CI.
