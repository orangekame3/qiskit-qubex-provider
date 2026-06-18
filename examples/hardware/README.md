# Hardware examples

These scripts connect to real Qubex hardware. The directory includes `qubex-config` device trees such as `64Qv3` and `144Qv2`.

## 64Qv3 Bell pair

Dry-run validation:

```bash
python examples/hardware/bell_state.py
```

Execute on hardware:

```bash
python examples/hardware/bell_state.py \
  --execute \
  --shots 1024
```

By default the script uses `examples/hardware/qubex-config` and targets `64Qv3` with `Q24,Q25,Q26,Q27,Q28,Q29,Q30,Q31,Q40,Q41,Q42`; the Bell pair is `Q28 -> Q25`. To use a different checkout, pass `--config-root /path/to/qubex-config` or set `QUBEX_CONFIG_ROOT`. The config root must contain a device subdirectory such as `64Qv3/config`, `64Qv3/params`, and `64Qv3/calibration`.

The script validates the transpiled schedule first, builds QUBEX classifiers through `provider.build_classifier(...)`, then runs `backend.run(...)` with the provider's software-classification path.

## Scheduling comparison notebooks

Open [`scheduling_comparison.ipynb`](scheduling_comparison.ipynb) to run a Bell-pair 2-qubit Heisenberg comparison on the qubits configured by `bell_state.py`. Open [`scheduling_comparison_simulation.ipynb`](scheduling_comparison_simulation.ipynb) to run the same 2-qubit scheduling comparison with `mock_mode=True`, without connecting to hardware. Open [`scheduling_comparison_4q.ipynb`](scheduling_comparison_4q.ipynb) to connect to all labels in `bell_state.DEFAULT_QUBIT_LABELS` while running the 4-qubit workload on `bell_state.DEFAULT_4Q_WORKLOAD_LABELS`; the physical spin chain is `bell_state.DEFAULT_4Q_CHAIN_LABELS`.

Both hardware notebooks plot an ideal simulation reference, compare `timing_policy="qiskit"` with ALAP scheduling against `timing_policy="legacy_device_gateway"`, and can run the same sweep on hardware. Because this is under `examples/hardware`, `RUN_ON_HARDWARE` defaults to `True`. Set it to `False` in the setup cell when you only want compilation, schedule validation, and the ideal simulation.

The simulation notebook additionally builds a `qxsimulator.QuantumSystem` from the generated `device-topology`, filters the validated QUBEX `PulseSchedule` down to active channels, and runs `QuantumSimulator(...).mesolve(...)` for a sample pulse-level simulation. That population-dynamics step is more than a scheduling check: the real-device `DEVICE_ID/config`, `DEVICE_ID/params`, and `DEVICE_ID/calibration/calib_note.json` may be enough to generate pulses, but meaningful qxsimulator dynamics require calibration that is consistent with the simulator Hamiltonian, channel targets, amplitude scaling, frequencies, and units.
