# Running circuits on Qubex hardware

How to set up the provider for hardware execution, what options are
available, and what the built-in executor supports. For the underlying
mechanics (frame tracking, timing model, sampling grid), see
[hardware-execution-notes.md](hardware-execution-notes.md).

## Recommended setup: `from_experiment`

Create and configure a Qubex `Experiment` first, then inject it:

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

transpiled = transpile(circuit, backend, scheduling_method="alap")
backend.validate(transpiled)
job = backend.run(transpiled, shots=1024)
counts = job.result().get_counts()
```

Pass `device_topology=...` when available: the topology file supplies the
Qiskit `Target` constraints for transpilation and scheduling, while the
configured `Experiment` supplies pulse generation, frame tracking,
measurement, and hardware execution. For production workflows, write actual
Qubex pulse durations into the topology ahead of execution:

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

`build_device_topology(..., pulse_source=exp)` builds each calibrated pulse
or sub-schedule (`x90`, `x180`, `readout`, `zx90`, `cx`) and writes its
actual duration into `gate_duration`. When `device_topology` is supplied,
`from_experiment(...)` uses those durations directly and does not probe pulse
methods again. This keeps job setup fast and makes scheduling inputs
reproducible. Pass `refresh_instruction_durations=True` only when you
intentionally want provider construction to re-probe pulse durations.

### Creating the Experiment through the provider

For simple setup code, `from_experiment_config(...)` creates the
`Experiment` for you. Device connection is opt-in:

```python
provider = QubexProvider.from_experiment_config(
    system_id="64Q-HF-Q1",
    device_topology="device-topology.json",
    config_dir="...",
    params_dir="...",
    connect_devices=True,
)
```

When `device_topology` is supplied, `qubits` can be omitted and is inferred
from the topology's physical qubit order. For unusual label widths (for
example a subset of a 100+ qubit system), pass `qubit_labels=[...]`
explicitly so Qiskit physical qubit indices map to the intended Qubex labels.

## Native basis gates

Qubex's execution gate set exposed to Qiskit is `rz`, `sx`, and `cx`;
`measure` and `delay` are also available as circuit timing and readout
operations. To transpile to this target, pass `native=True`; Qiskit then
decomposes compatibility gates such as `x` and `h` while preserving `cx`:

```python
provider = QubexProvider.from_experiment(
    exp,
    device_topology="device-topology.json",
    native=True,
)
native = transpile(circuit, provider.get_backend(), optimization_level=1)
```

The default target still exposes additional compatibility gates when their
durations are known. For explicit control, pass
`basis_gates=QUBEX_NATIVE_BASIS_GATES` instead of `native=True`. During
execution, Qiskit `cx` instructions are emitted as Qubex `cx` pulse schedules.

## Preflight validation

`backend.validate(circuits)` builds the exact Qubex `PulseSchedule` without
executing it: it runs qxpulse schedule validation and the provider's
hardware-resource overlap check. Run it on the final transpiled/scheduled
circuit before submitting to hardware:

```python
backend.validate(scheduled)
job = backend.run(scheduled, shots=1024)
```

## Run options

```python
backend.run(
    scheduled,
    shots=1024,      # positive integer (or integer string)
    memory=True,     # optional Qiskit shot memory
    plot=False,      # Qubex option, boolean
)
```

- `shots` must be a positive integer. `memory`, `state_classification`,
  `final_measurement`, and `plot` must be booleans; string values such as
  `"False"` are rejected rather than interpreted through Python truthiness.
- `state_classification=True` is required (the executor converts classified
  results into Qiskit counts) and is the default.
- `final_measurement` defaults to `False` when the circuit has explicit
  `measure` instructions, and to `True` otherwise so the executor can still
  produce counts from the Qubex final measurement.
- Other Qubex execution options are passed through to
  `measurement_service.execute(...)`. Qiskit `seed_simulator` is ignored on
  the hardware path.

## Supported circuit subset

`QubexPulseExecutor` supports the calibrated gate-level subset:

`id`, `x`, `sx`, `sxdg`, `y`, `h`, `s`, `sdg`, `z`, `rx(0|±π/2|π)`,
`ry(0|±π/2|π)`, `rz(θ)`, `ecr`, `cx`, `barrier`, `delay`, and
measurements without same-shot classical feedback.

Not supported:

- Same-shot feedback: measurement-conditioned gates and control flow
  (`if_else`, loops) are rejected.
- Mid-circuit `reset`: resets are accepted only at the start of the circuit
  (as no-ops under the usual shot-initialization assumption).
- Arbitrary-angle `rx`/`ry`: only calibrated angles are available.

Unsupported circuits should be transpiled to the backend target or run
through a custom executor.

## Legacy Device Gateway timing policy

For migration comparisons with the old Device Gateway Qubex plugin:

```python
provider = QubexProvider.from_experiment(
    exp,
    device_topology="device-topology.json",
    timing_policy="legacy_device_gateway",
)
```

This deprecated policy ignores Qiskit operation start times and emits pulses
in instruction order, closely matching the legacy plugin for A/B validation.
New code should keep the default `timing_policy="qiskit"`, which respects
Qiskit scheduling, `delay` padding, and explicit measurement timing.

## Custom executors

A bare Qubex `Measurement` object is not enough for gate-level circuits; the
executor needs the calibrated pulse methods provided by `Experiment.pulse`
(`x90`, `x180`, `cx`, ...). For custom hardware paths, pass any object
implementing `run(circuits, **options)` as `QubexProvider(..., executor=...)`.
