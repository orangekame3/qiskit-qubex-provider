# Device topology targets

How to build Qiskit `Target` metadata from a Device Gateway
`device-topology.json` file or from Qubex calibration files.

A small generated example is available at
[examples/device-topology.json](../examples/device-topology.json), with a
matching topology image at
[examples/device-topology.svg](../examples/device-topology.svg).

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

The CLI also writes a topology image next to the JSON by default:
`device-topology.svg`. Use `--output-image topology.svg` to choose a path, or
`--no-output-image` to skip image generation.

You can also pass a QDash-style request JSON to select target qubits, exclude
couplings, and apply fidelity ranges:

```json
{
  "name": "anemone",
  "device_id": "anemone",
  "qubits": ["0", "1", "2", "3"],
  "exclude_couplings": ["1-2"],
  "condition": {
    "qubit_fidelity": {
      "metric": "x90_gate_fidelity",
      "min": 0.9,
      "max": 1.0
    },
    "coupling_fidelity": {
      "metric": "zx90_gate_fidelity",
      "min": 0.3,
      "max": 1.0
    },
    "readout_fidelity": {
      "metric": "average_readout_fidelity",
      "min": 0.8,
      "max": 1.0,
      "is_within_24h": true
    },
    "only_maximum_connected": true
  }
}
```

```bash
qiskit-qubex-device-topology \
  --calib-note qubex-config/64Qv3/calibration/calib_note.json \
  --params-dir qubex-config/64Qv3/params \
  --request-json request.json \
  --output-json device-topology.json
```

The `metric` fields select which YAML metric file under `--params-dir` is used
for each filter. `is_within_24h` is accepted for QDash request compatibility,
but local generation uses the values present in the YAML files. Couplings are
emitted only when the selected coupling fidelity metric exists for that
coupling.

Static PNG/SVG export uses Plotly when the optional plot dependencies are
installed:

```bash
pip install "qiskit-qubex-provider[plot] @ git+https://github.com/orangekame3/qiskit-qubex-provider.git"
```

Then choose an image path:

```bash
qiskit-qubex-device-topology ... --output-image device-topology.png
qiskit-qubex-device-topology ... --output-image device-topology.html
```

The Plotly figure includes directed couplings, coupling fidelity hover text,
gate duration hover text, qubit fidelity color scale, readout fidelity, and
T1/T2 details. Without the optional Plotly dependencies, the default `.svg`
path still writes a dependency-free static fallback.

The same generator is available as a Python API:

```python
from qiskit_qubex_provider import build_device_topology

topology = build_device_topology(
    calib_note_path="qubex-config/64Qv3/calibration/calib_note.json",
    params_dir="qubex-config/64Qv3/params",
)
provider = QubexProvider.from_device_topology(topology)
```

To write both files from Python:

```python
from qiskit_qubex_provider import write_device_topology

write_device_topology(
    "device-topology.json",
    calib_note_path="qubex-config/64Qv3/calibration/calib_note.json",
    params_dir="qubex-config/64Qv3/params",
)
```

This writes `device-topology.json` and `device-topology.svg`. Pass
`output_image=False` to skip the SVG, or pass an explicit SVG path.

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
