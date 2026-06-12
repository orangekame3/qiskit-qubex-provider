# qiskit-qubex-provider

Qiskit provider for [Qubex](https://github.com/amachino/qubex): build Qiskit
`Target`s from Qubex system metadata, transpile and schedule against real
calibration data, and execute circuits on Qubex hardware as calibrated pulse
schedules.

## Installation

```bash
uv pip install -e .
```

`qubex` is intentionally not a hard dependency because it is commonly
installed from a local checkout. Some Qubex imports also need optional local
backend packages:

```bash
uv pip install -e ../qubex
uv pip install -e ../qubex/packages/qxdriver-quel1  # if required
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
| Internals: frame tracking, timing model, sampling grid | [docs/hardware-execution-notes.md](docs/hardware-execution-notes.md) |

## Development

```bash
uv run pytest
```

Repository tests use synthetic topology and calibration fixtures only;
private Qubex configuration or measured device data should stay out of CI.
