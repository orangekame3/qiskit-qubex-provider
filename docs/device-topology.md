# Device topology targets

How to build Qiskit `Target` metadata from a Device Gateway
`device-topology.json` file or from Qubex calibration files.

## Using an existing topology file

```python
from qiskit_qubex_provider import QubexProvider

provider = QubexProvider.from_device_topology("device-topology.json")
backend = provider.get_backend()
```

The provider reads `qubits`, `couplings`, qubit lifetimes (T1/T2), and gate
durations from the file. Coupling `gate_duration.rzx90` is exposed as the
scheduled two-qubit duration for native `ecr` and compatibility `cx`/`cz`.

Backends built this way carry transpilation and scheduling metadata only —
`backend.run(...)` falls back to Qiskit's local `BasicSimulator` unless an
executor is configured. Combine the topology with a configured `Experiment`
through `from_experiment(device_topology=...)` for hardware execution (see
[hardware-execution.md](hardware-execution.md)).

## Generating a topology file from Qubex calibration

If you have Qubex calibration files but no generated topology file yet:

```bash
qiskit-qubex-device-topology \
  --calib-note qubex-config/64Qv3/calibration/calib_note.json \
  --params-dir qubex-config/64Qv3/params \
  --output-json device-topology.json
```

The same generator is available as a Python API:

```python
from qiskit_qubex_provider import build_device_topology

topology = build_device_topology(
    calib_note_path="qubex-config/64Qv3/calibration/calib_note.json",
    params_dir="qubex-config/64Qv3/params",
)
provider = QubexProvider.from_device_topology(topology)
```

## Qubit labels and physical IDs

Qiskit physical qubit index `i` maps to the i-th entry of the topology's
`qubits` list. Labels come from each entry's `label` field when present, and
otherwise from the `physical_id` formatted with the device label width
(`qid_to_label`/`label_to_qid` helpers). When targeting a subset of a large
system whose label width differs from the subset size, pass
`qubit_labels=[...]` explicitly.

## Other target sources

`QubexProvider`/`build_qubex_target` also accept:

- An integer qubit count or `num_qubits=...` with an optional
  `coupling_map=[...]` for synthetic targets.
- A Qubex `ExperimentSystem`/`QuantumSystem`-like object with `qubits` /
  `cr_targets` attributes; qubit frequencies and CR connectivity are
  converted into the Qiskit target.
