# Hardware execution notes

Special considerations when executing Qiskit circuits on Qubex hardware
through `QubexPulseExecutor`. These notes document behavior that is easy to
get wrong and the invariants the executor maintains; read them before
changing the schedule builder or debugging unexpected hardware results.

## Frame tracking

Qubex/qxpulse implements `rz`-family gates as *virtual Z* rotations: no pulse
is played, the rotating frame of the drive channel is shifted instead, and the
shift is applied to the phase of every later physical pulse on that channel.
The executor maps Qiskit `rz(θ)`, `s`, `sdg`, and `z` to zero-duration
`VirtualZ` objects, so qxpulse frame tracking — not the executor — is
responsible for rotating subsequent pulses.

Three details matter:

### Sign convention

`qxpulse.VirtualZ(theta)` is a `PhaseShift` that stores `-theta`. A Qiskit
`rz(θ)` therefore becomes `VirtualZ(θ)` directly. When the executor needs to
apply a *raw* frame shift of `δ` (see CR mirroring below), it must construct
`VirtualZ(-δ)` to undo the negation. `PulseSchedule.get_final_frame_shift`
returns the accumulated raw shift wrapped into `[-π, π)`; because frame shifts
are modular in 2π, comparing wrapped values is safe.

### Cross-resonance channels share the target qubit's frame

A cross-resonance channel labeled `Qc-Qt` drives qubit `Qc` *at the frequency
and in the frame of* `Qt`. Any virtual-Z accumulated on `Qt`'s drive channel
must therefore also rotate the frame of later CR pulses on `Qc-Qt`, or the
effective CR phase is wrong and two-qubit gates silently degrade.

Production Qubex handles this inside its `cnot`/`cz` constructions by adding
the same `z180` to the CR channel whenever it adds one to the qubit channel
(see `PulseService.cnot`/`cz` in the qubex repository). The executor extends
that bookkeeping across gate boundaries: a Qiskit circuit can place `rz` gates
*between* two-qubit gates, and those frame shifts also have to reach the CR
channel.

Before every `schedule.call(...)` of a two-qubit sub-schedule, the executor
runs `_sync_cr_channel_frames`:

1. For each CR-style label `Qc-Qt` in the sub-schedule, read the frequency
   target `Qt` from the label.
2. Compare the outer schedule's accumulated frame shift on `Qt` with the one
   on `Qc-Qt`.
3. If they differ by `δ`, add `VirtualZ(-δ)` to the CR channel so both frames
   agree when the sub-schedule starts.

Because the comparison uses the outer schedule's *final* frame shifts, it
automatically covers every source of frame drift: explicit `rz`/`s`/`z`
gates, the `z180` inside the `h` decomposition (`Z180·Y90`), and the
virtual-Z corrections embedded in previous `cx`/`cz`/`ecr` sub-schedules.

Readout channels do not need mirroring: Z-basis state classification is
insensitive to the qubit drive frame.

### Hadamard carries a hidden frame shift

`pulse.hadamard` returns a `PulseArray` of `VirtualZ(π)` followed by a
physical `y90` pulse. Its waveform duration equals the `y90` duration, but it
*also* shifts the channel frame by π. Code that reasons about frames must use
`get_final_frame_shift` on the schedule rather than assuming only `rz`-family
gates move frames.

## Timing model

The executor builds one `PulseSchedule` per circuit. qxpulse channels are
independent timelines that only synchronize at barriers and `call(...)`
boundaries, so the executor maintains a per-channel offset table
(`channel_offsets`, in ns) that must always equal the actual end position of
each channel's content. Every branch that adds a waveform must advance both
the schedule and the table by the same amount — if they diverge, later
alignment inserts wrong-sized `Blank`s and pulses land at the wrong time.

### Scheduled circuits (recommended)

Transpile with `scheduling_method="asap"` or `"alap"` so the circuit carries
`op_start_times`. The executor converts each start time to ns using the
circuit time unit and the backend `dt`, and pads each involved channel with
`Blank` up to the operation's start. Idle time and explicit Qiskit `delay`
instructions become `Blank` pulses. An operation that would start *before*
the channel's current end raises an error instead of silently overlapping.

`QubexProvider.from_experiment(...)` feeds calibrated pulse durations and the
Qubex sampling period into the Qiskit `Target`, so Qiskit scheduling uses the
same timing grid as Qubex.

### Unscheduled circuits

Without `op_start_times` the executor falls back to per-channel sequential
semantics: each instruction is appended at its channel's current end, and
channels synchronize only at Qiskit `barrier`s, at two-qubit `call(...)`
boundaries, and at measurements. Cross-channel ordering between single-qubit
gates on different qubits is *not* preserved — if your circuit relies on it,
either insert barriers or transpile with a scheduling method.

### Measurement timing

`measure` instructions add the calibrated readout waveform on the resolved
readout channel (`RQxx`). Two invariants keep readout aligned:

- In the unscheduled path, the executor barriers the qubit channel and its
  readout channel immediately before adding the readout pulse. Without this,
  the readout channel would start at t=0 and the readout would overlap
  earlier gates.
- In both paths, the executor adds a `Blank` of the readout duration on the
  *drive* channel of the measured qubit. This keeps the tracked offset equal
  to the real channel end and guarantees that later gates on that qubit
  (mid-circuit measurement) start after the readout window instead of
  overlapping it.

Circuits without explicit `measure` instructions run with
`final_measurement=True`, in which case Qubex appends the final readout
itself and none of the above applies.

## Sampling grid

qxpulse waveforms live on a fixed sampling grid (`DEFAULT_SAMPLING_PERIOD` is
2 ns); durations that are not integer multiples of the sampling period within
a 1e-9 tolerance are rejected by qxpulse. `from_experiment(...)` sets the
backend `dt` to the Qubex sampling period, so `delay`s and start times
expressed in `dt` units always stay on the grid. Be careful with `delay`
durations given in `s`/`us`/`ns` units — they are converted verbatim and will
fail schedule construction if they fall off the grid. Backends built without
an experiment (for example `from_device_topology`) default to `dt=1e-9` and
are intended for transpilation metadata, not pulse execution.

## Measurement results and counts

- `state_classification=True` is required; the executor converts classified
  bitstrings into Qiskit counts and rejects anything else.
- Mid-circuit measurement is supported without feedback. Each measurement of
  a qubit becomes a `(label, capture_index)` pair, and Qubex's
  `MultipleMeasureResult.get_counts` is queried with exactly those pairs, so
  repeated measurements of one qubit map to distinct clbits. A runnable
  example with the pulse-level timeline lives at
  [examples/mid-circuit-measurement.ipynb](../examples/mid-circuit-measurement.ipynb).
- Bit ordering follows Qiskit conventions: clbit 0 is the least significant
  bit of the hex count keys.
- Count totals (and memory length when `memory=True`) are validated against
  the requested shots; mismatches raise instead of returning skewed
  distributions.
- Same-shot feedback (`if_else`, measurement-conditioned gates) and
  mid-circuit `reset` are rejected up front. Initial resets are accepted as
  no-ops under the usual shot-initialization assumption.

## Primitives

- `QubexSamplerV2` delegates to Qiskit's `BackendSamplerV2` over
  `backend.run(...)`, so it samples from hardware whenever an executor is
  configured. Constructor options (`default_shots`, `seed_simulator`, ...)
  are forwarded to the delegate's constructor.
- `QubexEstimatorV2` delegates to `BackendEstimatorV2` (sampled, hardware)
  when the backend has an executor, and to the exact `StatevectorEstimator`
  otherwise. Expectation values from the hardware path carry shot noise by
  design; do not compare them against exact simulator values without error
  bars.

## Preflight validation

`backend.validate(circuits)` builds the exact Qubex `PulseSchedule` without
executing it. It runs qxpulse's own schedule validation (`is_valid`) and a
provider-side resource check: non-blank pulse windows of two logical channels
that resolve to the same physical hardware channel (for example a CR channel
and the drive channel sharing an AWG port) must not overlap. Run it on the
final transpiled/scheduled circuit before submitting to hardware.
