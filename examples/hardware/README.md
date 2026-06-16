# Hardware examples

These scripts connect to real Qubex hardware. The directory includes a `qubex-config/144Qv2` tree for the 144Qv2 Bell example.

## 144Qv2 Bell pair

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

By default the script uses `examples/hardware/qubex-config`. To use a different checkout, pass `--config-root /path/to/qubex-config` or set `QUBEX_CONFIG_ROOT`. The config root must contain a device subdirectory such as `144Qv2/config`, `144Qv2/params`, and `144Qv2/calibration`.

The script validates the transpiled schedule first, builds QUBEX classifiers through `provider.build_classifier(...)`, then runs `backend.run(...)` with the provider's software-classification path.

## Scheduling comparison notebooks

Open [`scheduling_comparison.ipynb`](scheduling_comparison.ipynb) to run a Bell-pair 2-qubit Heisenberg comparison on the qubits configured by `bell_state.py`. Open [`scheduling_comparison_4q.ipynb`](scheduling_comparison_4q.ipynb) to run the 4-qubit version over all labels in `bell_state.DEFAULT_QUBIT_LABELS`.

Both notebooks plot an ideal simulation reference, compare `timing_policy="qiskit"` with ALAP scheduling against `timing_policy="legacy_device_gateway"`, and can run the same sweep on hardware. Because this is under `examples/hardware`, `RUN_ON_HARDWARE` defaults to `True`. Set it to `False` in the setup cell when you only want compilation, schedule validation, and the ideal simulation.
