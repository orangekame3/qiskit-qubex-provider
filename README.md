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

Pass `native=True` to transpile to the native gate set (`ecr` instead of
`cx`/`cz`).

## Primitives

```python
sampler = provider.get_sampler()      # samples via backend.run (hardware when configured)
estimator = provider.get_estimator()  # hardware-sampled with an executor, exact statevector otherwise
```

## Documentation

| Topic | Where |
| --- | --- |
| Hardware execution: setup, run options, supported gates, validation | [docs/hardware-execution.md](docs/hardware-execution.md) |
| Device topology files and target generation (incl. CLI) | [docs/device-topology.md](docs/device-topology.md) |
| Dynamical decoupling pass managers | [docs/dynamical-decoupling.md](docs/dynamical-decoupling.md) |
| Comparing pulse schedules across scheduling methods | [docs/pulse-schedule-visualization.md](docs/pulse-schedule-visualization.md) |
| Internals: frame tracking, timing model, sampling grid | [docs/hardware-execution-notes.md](docs/hardware-execution-notes.md) |
| Notebook: end-to-end tour (topology → transpile → pulses → visualization) | [examples/tutorial.ipynb](examples/tutorial.ipynb) |
| Notebook: mid-circuit measurement | [examples/mid-circuit-measurement.ipynb](examples/mid-circuit-measurement.ipynb) |
| Notebook: dynamical decoupling | [examples/dynamical-decoupling.ipynb](examples/dynamical-decoupling.ipynb) |
| Notebook: DD fidelity demonstration (Pokharel et al. protocol) | [examples/dd-demonstration.ipynb](examples/dd-demonstration.ipynb) |

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
