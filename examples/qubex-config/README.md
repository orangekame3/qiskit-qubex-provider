# qubex-config — offline Qubex configuration for the example notebook

A minimal, self-contained Qubex system configuration that lets the example
notebooks ([tutorial.ipynb](../tutorial.ipynb),
[mid-circuit-measurement.ipynb](../mid-circuit-measurement.ipynb),
[dynamical-decoupling.ipynb](../dynamical-decoupling.ipynb))
construct a **real `qubex.Experiment`** without hardware:

```python
from qubex import Experiment

exp = Experiment(
    system_id="4Q-DEMO-SYS",
    muxes=[0],
    config_dir="qubex-config/config",
    params_dir="qubex-config/params",
    calib_note_path="qubex-config/calibration/calib_note.json",
    calibration_valid_days=10000,  # committed example calibration; ignore staleness
    mock_mode=True,  # no devices are contacted
)
```

The same files also feed the topology generator, which is how
[../device-topology.json](../device-topology.json) and
[../device-topology.svg](../device-topology.svg) were produced:

```bash
qiskit-qubex-device-topology \
  --calib-note qubex-config/calibration/calib_note.json \
  --params-dir qubex-config/params \
  --name 4Q-DEMO --device-id 4Q-DEMO
```

## Files

| File | Role |
| --- | --- |
| `config/chip.yaml` | Chip catalog: `16Q-DEMO`, a 16-qubit square lattice (16 qubits so Qubex uses two-digit labels `Q00`–`Q15`) |
| `config/system.yaml` | System catalog: `4Q-DEMO-SYS` runs the `16Q-DEMO` chip on the `quel1` backend |
| `config/box.yaml` | Control-box catalog: two fictitious boxes (`DEMO1` control, `DEMO2` readout) with placeholder addresses |
| `config/wiring.yaml` | Wires mux 0 (`Q00`–`Q03`) to those boxes; the other muxes stay unwired |
| `params/control_frequency.yaml` | Qubit drive frequencies (GHz) |
| `params/control_amplitude.yaml` | Calibrated drive amplitudes |
| `params/readout_frequency.yaml` | Resonator frequencies (GHz) |
| `params/readout_amplitude.yaml` | Readout amplitudes |
| `params/capture_delay.yaml` | Capture delays (`ndelay` units) |
| `params/x90_gate_fidelity.yaml` | X90 gate fidelities — used as qubit fidelity in the generated topology |
| `params/zx90_gate_fidelity.yaml` | ZX90 gate fidelities — used as coupling fidelity in the generated topology |
| `params/average_readout_fidelity.yaml` | Readout fidelities — used for `meas_error` in the generated topology |
| `params/t1.yaml`, `params/t2_echo.yaml` | Coherence times (µs) — used for `qubit_lifetime` in the generated topology |
| `calibration/calib_note.json` | Calibration note: `drag_hpi_params` / `drag_pi_params` for `Q00`–`Q03` and `cr_params` for the three couplings (`Q00-Q01`, `Q00-Q02`, `Q03-Q02` — a connected graph) |

## Where the numbers come from

Fidelities, coherence times, and the calibration-note entries are
**representative demo values**: they are modeled on a real device
calibration so magnitudes and qubit-to-qubit variation look right (down to
the bad qubit with a short T1), but every value has been perturbed and
rounded — nothing here is an actual device readout. The `Q03-Q02` coupling
in particular has no real counterpart at all; its CR entry and ZX90
fidelity were invented for the demo so the topology stays connected.

The catalog files (`chip`/`system`/`box`/`wiring`) and the frequency /
amplitude / capture-delay parameters are example numbers (borrowed from the
Qubex repository's own `docs/examples/system` catalog). Everything is fine
for building and visualizing pulse schedules, meaningless for real
hardware.

Calibration-note entries carry timestamps and expire after
`calibration_valid_days` (default 14), which is why the notebook passes a
large value.
