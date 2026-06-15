# Pulse schedule visualization

Helpers for judging at a glance how a change in pulse scheduling — a
different `timing_policy`, a different transpiler `scheduling_method`, or a
provider code change — affects the Qubex `PulseSchedule` that actually runs
on hardware.

Schedules are built without execution via `backend.validate(...)` (or
`executor.build_schedule(circuit)`), so no hardware time is consumed:

```python
provider = QubexProvider.from_experiment(exp, timing_policy="qiskit")
schedule = provider.get_backend().validate(circuit)[0]
```

The result is a plain Qubex `PulseSchedule`, so everything below — and
Qubex's own plotting — applies to it directly.

A runnable tutorial notebook is available at
[examples/simulation/tutorial.ipynb](../examples/simulation/tutorial.ipynb).

## Gantt-style timeline (Qiskit `timeline_drawer`-like)

`build_pulse_schedule_timeline_figure` renders one horizontal lane per
channel, with a labeled box for every pulse (colored by waveform name) and a
tick marker for every virtual-Z phase shift — the same reading experience as
Qiskit's `timeline_drawer` or IQM's playlist views, but for the Qubex
schedule that actually runs:

```python
from qiskit_qubex_provider import build_pulse_schedule_timeline_figure

build_pulse_schedule_timeline_figure(schedule, title="timing_policy=qiskit").show()
```

To judge a scheduling change, render one figure per variant and compare them
side by side. Plotting requires the optional `plot` extra
(`pip install 'qiskit-qubex-provider[plot]'`).

Hover shows exact start and duration in ns. The underlying data is also
available programmatically via `extract_pulse_timeline(schedule)`, which
returns per-channel `{"kind", "name", "start_ns", "duration_ns"}` entries.

To save the figure instead of showing it (e.g. for a PR description):

```python
from qiskit_qubex_provider import write_pulse_schedule_timeline_image

write_pulse_schedule_timeline_image(schedule, "timeline.html")
# .png / .svg also work with Kaleido installed
```

## Waveform detail via Qubex

For the actual waveform envelopes (I/Q components and frame phase per
channel), call Qubex's own plot on the schedule — no provider API needed:

```python
schedule.plot()                          # logical X/Y/phase view
schedule.plot(show_physical_pulse=True)  # physical I/Q view
```

## Judge numerically

`diff_pulse_schedules` compares two schedules sample by sample — useful in
notebooks and as a regression check when refactoring the scheduler:

```python
from qiskit_qubex_provider import diff_pulse_schedules

diff = diff_pulse_schedules(schedule_a, schedule_b)
diff["equal"]     # True if every channel matches sample-for-sample
diff["channels"]  # per-channel status: equal / changed / length_mismatch /
                  # only_in_a / only_in_b, with max_abs_diff where comparable
```

`summarize_pulse_schedule` reports per-channel timing facts (total duration
and the first/last non-blank sample boundaries), which is often enough to
spot where a scheduling change moved a pulse:

```python
from qiskit_qubex_provider import summarize_pulse_schedule

summarize_pulse_schedule(schedule)
# {"Q00": {"duration_ns": 120.0, "active_start_ns": 0.0,
#          "active_end_ns": 30.0, "n_samples": 60}, ...}
```

## Gate-level view via Qiskit

With `timing_policy="qiskit"`, the schedule follows the scheduled circuit's
timing, so Qiskit's own gate-level timeline applies directly to the
transpiled circuit:

```python
from qiskit import transpile
from qiskit.visualization import timeline_drawer

scheduled = transpile(circuit, backend, scheduling_method="alap")
timeline_drawer(scheduled)
```

Use that for the gate-level picture and the helpers above for the
pulse-level picture of the same change.
