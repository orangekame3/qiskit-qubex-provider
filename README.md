# qiskit-qubex-provider

Qiskit provider for Qubex targets and local Qiskit execution.

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

You can also pass a Qubex-like system object with `qubits` and `cr_targets`
attributes. The provider converts that metadata into a Qiskit `Target` so
circuits can be transpiled against Qubex topology and qubit properties.

```python
provider = QubexProvider(qubex_experiment_system)
backend = provider.get_backend()
sampler = provider.get_sampler()
estimator = provider.get_estimator()
```

Device Gateway `device-topology.json` files can also be used directly for
transpilation and local simulation metadata:

```python
provider = QubexProvider.from_device_topology("device-topology.json")
backend = provider.get_backend()
```

The provider reads `qubits`, `couplings`, qubit lifetimes, and gate durations
from the topology file. Coupling `gate_duration.rzx90` is exposed as the
scheduled two-qubit duration for `cx`/`cz`.

If you have Qubex calibration files but not a generated Device Gateway topology
file yet, the provider can generate one from `calib_note.json` and the Qubex
`params` directory:

```bash
qiskit-qubex-device-topology \
  --calib-note qubex-config/64Qv3/calibration/calib_note.json \
  --params-dir qubex-config/64Qv3/params \
  --output-json device-topology.json
```

The same generator is available as Python API:

```python
from qiskit_qubex_provider import build_device_topology

topology = build_device_topology(
    calib_note_path="qubex-config/64Qv3/calibration/calib_note.json",
    params_dir="qubex-config/64Qv3/params",
)
provider = QubexProvider.from_device_topology(topology)
```

`qubex` is intentionally not a hard package dependency because the Qubex
repository is commonly installed from a local checkout:

```bash
uv pip install -e ../qubex
```

Some Qubex imports currently need optional local backend packages as well, for
example:

```bash
uv pip install -e ../qubex/packages/qxdriver-quel1
```

For hardware execution, pass an executor object with a `run(circuits,
**options)` method through `QubexProvider(..., executor=...)`. Without an
executor, `backend.run(...)` uses Qiskit's local `BasicSimulator`.

The package also includes `QubexPulseExecutor`, which converts supported Qiskit
circuits into Qubex `PulseSchedule` objects and calls
`measurement_service.execute(...)`. In production you should create and
configure a Qubex `Experiment` first, then inject it into the provider:

```python
from qubex import Experiment
from qiskit import QuantumCircuit, transpile
from qiskit_qubex_provider import QubexProvider

exp = Experiment(
    system_id="64Q-HF-Q1",
    qubits=["Q00", "Q01"],
    config_dir="...",
    params_dir="...",
)
exp.connect()

provider = QubexProvider.from_experiment(
    exp,
    device_topology="device-topology.json",
)
backend = provider.get_backend()

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

transpiled = transpile(circuit, backend)
job = backend.run(transpiled, shots=1024)
counts = job.result().get_counts()
```

`from_experiment(...)` is the recommended production path. Pass
`device_topology=...` when available: the topology file supplies the Qiskit
`Target` constraints for transpilation and scheduling, while the configured
Qubex `Experiment` supplies pulse generation, frame tracking, measurement, and
hardware execution. The executor infers calibrated pulse durations from
`Experiment.pulse` and exposes those durations plus the Qubex sampling period
to Qiskit. That lets Qiskit scheduling passes use the same timing grid as
Qubex:

```python
scheduled = transpile(circuit, backend, scheduling_method="asap")
scheduled = transpile(circuit, backend, scheduling_method="alap")
```

Scheduled Qiskit operation start times are preserved when building the Qubex
`PulseSchedule`: idle time and Qiskit `delay` instructions become Qubex
`Blank` pulses. Qiskit `measure` instructions become Qubex readout pulses at
their circuit positions, and classified pulse-aligned captures are mapped back
to the requested Qiskit clbits. `rz`, `s`, `sdg`, and `z` are emitted as
zero-duration `VirtualZ` frame shifts, so Qubex/qxpulse frame tracking remains
responsible for applying them to later physical pulses.

For simple setup code, the provider can create the `Experiment` for you. Device
connection is opt-in:

```python
provider = QubexProvider.from_experiment_config(
    system_id="64Q-HF-Q1",
    device_topology="device-topology.json",
    config_dir="...",
    params_dir="...",
    connect_devices=True,
)
```

When `device_topology` is supplied to `from_experiment_config(...)`, `qubits`
can be omitted and will be inferred from the topology's physical qubit order.
For unusual label widths, for example a subset of a 100+ qubit system, pass
`qubit_labels=[...]` explicitly so Qiskit physical qubit indices map to the
intended Qubex labels.

A bare Qubex `Measurement` object is not enough for gate-level Qiskit circuits,
because the executor needs the calibrated pulse methods provided by
`Experiment.pulse` (`x90`, `x180`, `cx`, and related operations). For custom
hardware paths, pass an object implementing `run(circuits, **options)` as
`executor=...`.

`QubexPulseExecutor` currently supports the calibrated gate-level subset
`id`, `x`, `sx`, `sxdg`, `y`, `h`, `s`, `sdg`, `z`, `rx(0|+/-pi/2|pi)`,
`ry(0|+/-pi/2|pi)`, `rz(theta)`, `cx`, `cz`, `barrier`, `delay`, and
measurements without same-shot classical feedback. Unsupported circuits should
be transpiled to this target or run through a custom executor.

Same-shot feedback is not supported by the built-in Qubex pulse executor yet.
That includes measurement-conditioned gates and control-flow operations such as
`if_else`. Initial reset operations are accepted as no-ops under the usual
shot-initialization assumption, but any reset after an active operation is
rejected.

Before execution, schedules are preflighted against Qubex target metadata when
physical channel information is available. Overlapping non-blank pulses on the
same hardware channel are rejected before calling `measurement_service.execute`.
Qubex/qxpulse still performs the backend-specific schedule and capture
validation for constraints that are not representable from target channel
metadata alone.
