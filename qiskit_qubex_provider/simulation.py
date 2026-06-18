"""Helpers for qxsimulator pulse-level simulation."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .device_topology import label_to_qid, qid_to_label


@dataclass
class FakeQubexExperiment:
    """Small simulation fixture with QUBEX-like calibration metadata."""

    name: str = "fake-qubex-cr-demo"
    device_id: str = "fake-qubex-cr-demo"
    qubit_labels: tuple[str, str] = ("Q00", "Q01")
    qubit_frequencies: tuple[float, float] = (7.157231, 8.032295)
    qubit_anharmonicities: tuple[float, float] = (-0.393715, -0.487412)
    coupling_strength: float = 0.005
    qubit_lifetime: tuple[float, float] = (20.0, 20.0)
    qubit_lifetimes: tuple[tuple[float, float], ...] | None = None
    hpi_duration: float = 24.0
    pi_duration: float = 24.0
    readout_duration: float = 1000.0
    rzx90_duration: float | None = None
    cx_duration: float | None = None
    single_qubit_fidelity: float | None = None
    two_qubit_fidelity: float | None = None
    readout_assignment_error: float | None = None
    positions: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (1.0, 0.0),
    )
    calibrated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    drag_hpi_pulses: dict[str, Any] = field(default_factory=dict, init=False)
    drag_pi_pulses: dict[str, Any] = field(default_factory=dict, init=False)
    cr_params: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    cx_frame_params: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    classifiers: dict[str, Any] = field(default_factory=dict, init=False)
    readout_assignment_errors: dict[str, float] = field(default_factory=dict, init=False)
    _clifford_generator: Any | None = field(default=None, init=False, repr=False)

    @classmethod
    def two_qubit_cr_demo(cls, **overrides: Any) -> "FakeQubexExperiment":
        """Return a two-qubit CR calibration model matching the QUBEX examples."""
        return cls(**overrides)

    def device_topology(self) -> dict[str, Any]:
        """Build a device-gateway compatible ``device-topology.json`` mapping."""
        qubits = [
            self._qubit_topology(index, label)
            for index, label in enumerate(self.qubit_labels)
        ]
        topology: dict[str, Any] = {
            "name": self.name,
            "device_id": self.device_id,
            "qubits": qubits,
            "couplings": [self._coupling_topology()],
        }
        if self.calibrated_at is not None:
            topology["calibrated_at"] = self.calibrated_at
        if self.metadata:
            topology["metadata"] = dict(self.metadata)
        return topology

    def write_device_topology(
        self,
        path: str | Path,
        *,
        image_path: str | Path | None = None,
    ) -> Path:
        """Write ``device-topology.json`` and optionally the matching SVG image."""
        output_path = Path(path)
        topology = self.device_topology()
        output_path.write_text(
            json.dumps(topology, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if image_path is not None:
            from .device_topology import write_device_topology_image

            write_device_topology_image(topology, image_path)
        return output_path

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        repetitions: int = 20,
        plot: bool | None = True,
        **_: Any,
    ) -> Any:
        """Calibrate DRAG half-pi pulses with an experiment-like API."""
        np, _pd, qx, _qt, Result, _StateClassifierGMM, Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        simulator = QuantumSimulator(self._qx_system())
        target_labels = self._target_labels(targets)
        data: dict[str, Any] = {}
        figures: dict[str, Any] = {}
        for target in target_labels:
            index = self.qubit_labels.index(target)
            beta = -0.5 / (2 * np.pi * self.qubit_anharmonicities[index])
            hpi_pulse = self._area_normalized_drag(self.hpi_duration, np.pi / 2, beta)
            pi_pulse = self._area_normalized_drag(self.pi_duration, np.pi, beta)
            self.drag_hpi_pulses[target] = hpi_pulse
            self.drag_pi_pulses[target] = pi_pulse
            repeat_result = self.repeat_sequence(
                {target: hpi_pulse},
                repetitions=repetitions,
                plot=plot,
                _simulator=simulator,
            )
            repeat_df = repeat_result["dataframe"]
            data[target] = {
                "target": target,
                "duration": self.hpi_duration,
                "amplitude": float(np.max(hpi_pulse.real)),
                "beta": float(beta),
                "repeat": repeat_df,
            }
            if repeat_result.figure is not None:
                figures[target] = repeat_result.figure
        return Result(data=data, figures=figures)

    def calibrate_drag_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        plot: bool | None = True,
        **_: Any,
    ) -> Any:
        """Calibrate DRAG pi pulses from the same DRAG beta estimate."""
        np, _pd, _qx, _qt, Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        data: dict[str, Any] = {}
        for target in self._target_labels(targets):
            index = self.qubit_labels.index(target)
            beta = -0.5 / (2 * np.pi * self.qubit_anharmonicities[index])
            pi_pulse = self._area_normalized_drag(self.pi_duration, np.pi, beta)
            self.drag_pi_pulses[target] = pi_pulse
            data[target] = {
                "target": target,
                "duration": self.pi_duration,
                "amplitude": float(np.max(pi_pulse.real)),
                "beta": float(beta),
            }
        return Result(data=data)

    def obtain_cr_params(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        tomography_duration: float = 400.0,
        tomography_samples: int = 80,
        plot: bool | None = True,
        **_: Any,
    ) -> Any:
        """Obtain echoed ZX90 CR parameters with a QUBEX-like API."""
        np, _pd, qx, _qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        simulator = QuantumSimulator(self._qx_system())
        control_index = self.qubit_labels.index(control_qubit)
        target_index = self.qubit_labels.index(target_qubit)
        frequency_diff = (
            self.qubit_frequencies[control_index] - self.qubit_frequencies[target_index]
        )
        cr_amplitude_max = 0.75 * abs(frequency_diff)
        cr_phase_offset = 0.25 * np.pi
        cr_ramptime = 16
        xt_ratio = 0.01
        xt_phase_offset = 0.5 * np.pi
        rotary_multiple = 17
        duration_unit = 16
        pi_pulse = self.drag_pi_pulses.get(control_qubit)
        if pi_pulse is None:
            self.calibrate_drag_pi_pulse([control_qubit], plot=False)
            pi_pulse = self.drag_pi_pulses[control_qubit]

        def cr_sequence(
            *,
            duration: float,
            cr_amplitude: float,
            cr_phase: float,
            cancel_amplitude: float,
            cancel_phase: float,
            echo: bool,
        ) -> Any:
            cr_beta = -1 / (2 * np.pi * frequency_diff)
            cr_waveform = qx.pulse.FlatTop(
                duration=duration,
                amplitude=2 * np.pi * cr_amplitude,
                tau=cr_ramptime,
                phase=cr_phase + cr_phase_offset,
                beta=cr_beta,
            )
            cancel_waveform = qx.pulse.FlatTop(
                duration=duration,
                amplitude=2 * np.pi * cancel_amplitude,
                tau=cr_ramptime,
                phase=cancel_phase,
                beta=-1 / (2 * np.pi * self.qubit_anharmonicities[target_index]),
            )
            crosstalk_waveform = qx.pulse.FlatTop(
                duration=duration,
                amplitude=2 * np.pi * cr_amplitude * xt_ratio,
                tau=cr_ramptime,
                phase=xt_phase_offset,
                beta=cr_beta,
            )
            channels = [
                qx.PulseChannel(
                    label="Control",
                    frequency=self.qubit_frequencies[control_index],
                    target=control_qubit,
                ),
                qx.PulseChannel(
                    label="CR",
                    frequency=self.qubit_frequencies[target_index],
                    target=control_qubit,
                ),
                qx.PulseChannel(
                    label="Crosstalk",
                    frequency=self.qubit_frequencies[target_index],
                    target=target_qubit,
                ),
                qx.PulseChannel(
                    label="Target",
                    frequency=self.qubit_frequencies[target_index],
                    target=target_qubit,
                ),
            ]
            with qx.PulseSchedule(channels) as cr:
                cr.add("CR", cr_waveform)
                cr.add("Crosstalk", crosstalk_waveform)
                cr.add("Target", cancel_waveform)
            if not echo:
                return cr
            with qx.PulseSchedule(channels) as ecr:
                ecr.call(cr)
                ecr.barrier()
                ecr.add("Control", pi_pulse)
                ecr.barrier()
                ecr.call(cr.scaled(-1))
                ecr.barrier()
                ecr.add("Control", pi_pulse)
            return ecr

        def simulate_cr(
            *,
            cr_amplitude: float,
            cr_phase: float,
            cancel_amplitude: float,
            cancel_phase: float,
            duration: float,
            echo: bool,
            control_state: str,
            n_samples: int = 120,
        ) -> Any:
            return simulator.mesolve(
                controls=cr_sequence(
                    duration=duration,
                    cr_amplitude=cr_amplitude,
                    cr_phase=cr_phase,
                    cancel_amplitude=cancel_amplitude,
                    cancel_phase=cancel_phase,
                    echo=echo,
                ),
                initial_state={control_qubit: control_state, target_qubit: "0"},
                n_samples=n_samples,
            )

        def hamiltonian_tomography(
            *,
            cr_amplitude: float,
            cr_phase: float,
            cancel_amplitude: float,
            cancel_phase: float,
        ) -> dict[str, Any]:
            result_0 = simulate_cr(
                cr_amplitude=cr_amplitude,
                cr_phase=cr_phase,
                cancel_amplitude=cancel_amplitude,
                cancel_phase=cancel_phase,
                duration=tomography_duration,
                echo=False,
                control_state="0",
                n_samples=tomography_samples,
            )
            result_1 = simulate_cr(
                cr_amplitude=cr_amplitude,
                cr_phase=cr_phase,
                cancel_amplitude=cancel_amplitude,
                cancel_phase=cancel_phase,
                duration=tomography_duration,
                echo=False,
                control_state="1",
                n_samples=tomography_samples,
            )
            times = result_0.times
            vectors_0 = result_0.get_bloch_vectors(target_qubit)
            vectors_1 = result_1.get_bloch_vectors(target_qubit)
            indices = (times >= cr_ramptime) & (times < times[-1] - cr_ramptime)
            times_fit = times[indices] - cr_ramptime * 0.5
            fit_0 = qx.fit.fit_rotation(times_fit, vectors_0[indices], plot=False)
            fit_1 = qx.fit.fit_rotation(times_fit, vectors_1[indices], plot=False)
            omega = np.concatenate(
                [
                    0.5 * (fit_0["Omega"] + fit_1["Omega"]),
                    0.5 * (fit_0["Omega"] - fit_1["Omega"]),
                ]
            )
            coeffs = dict(
                zip(["IX", "IY", "IZ", "ZX", "ZY", "ZZ"], omega / (2 * np.pi), strict=True)
            )
            xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
            cr_rotation = coeffs["ZX"] + 1j * coeffs["ZY"]
            return {
                "coeffs": coeffs,
                "xt_rotation": xt_rotation,
                "cr_rotation": cr_rotation,
                "zx90_duration": float(1 / (4 * np.abs(cr_rotation))),
            }

        tomography_1 = hamiltonian_tomography(
            cr_amplitude=cr_amplitude_max,
            cr_phase=0.0,
            cancel_amplitude=0.0,
            cancel_phase=0.0,
        )
        cr_phase = -float(np.angle(tomography_1["cr_rotation"]))
        tomography_2 = hamiltonian_tomography(
            cr_amplitude=cr_amplitude_max,
            cr_phase=cr_phase,
            cancel_amplitude=0.0,
            cancel_phase=0.0,
        )
        cancel_pulse_max = -tomography_2["xt_rotation"]
        tomography_3 = hamiltonian_tomography(
            cr_amplitude=cr_amplitude_max,
            cr_phase=cr_phase,
            cancel_amplitude=float(np.abs(cancel_pulse_max)),
            cancel_phase=float(np.angle(cancel_pulse_max)),
        )
        zx_rate = float(tomography_3["coeffs"]["ZX"])
        cr_duration = float(
            ((tomography_3["zx90_duration"] / 2 + cr_ramptime) // duration_unit + 1)
            * duration_unit
        )

        def measure_target_z(cr_amplitude: float) -> float:
            ratio = cr_amplitude / cr_amplitude_max
            cancel_pulse = (cancel_pulse_max + zx_rate * rotary_multiple) * ratio
            result = simulate_cr(
                cr_amplitude=cr_amplitude,
                cr_phase=cr_phase,
                cancel_amplitude=float(np.abs(cancel_pulse)),
                cancel_phase=float(np.angle(cancel_pulse)),
                duration=cr_duration,
                echo=True,
                control_state="0",
                n_samples=2,
            )
            return float(result.get_bloch_vectors(target_qubit)[-1][2])

        amplitude_range = np.linspace(cr_amplitude_max * 0.8, cr_amplitude_max * 1.2, 20)
        z_values = np.array([measure_target_z(float(amplitude)) for amplitude in amplitude_range])
        fit_result = qx.fit.fit_polynomial(
            x=amplitude_range,
            y=z_values,
            degree=3,
            plot=False,
        )
        cr_amplitude = float(fit_result["root"])
        ratio = cr_amplitude / cr_amplitude_max
        cancel_pulse = (cancel_pulse_max + zx_rate * rotary_multiple) * ratio
        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = {
            "control": control_qubit,
            "target": target_qubit,
            "duration": cr_duration,
            "ramptime": cr_ramptime,
            "cr_amplitude": cr_amplitude,
            "cr_phase": cr_phase,
            "cancel_amplitude": float(np.abs(cancel_pulse)),
            "cancel_phase": float(np.angle(cancel_pulse)),
            "cr_amplitude_max": cr_amplitude_max,
        }
        self.cr_params[cr_label] = cr_param
        self.rzx90_duration = 2.0 * cr_duration + 2.0 * self.pi_duration
        self.cx_duration = self.rzx90_duration + self.hpi_duration
        figures = {}
        if plot:
            fig = qx.viz.make_plot_figure(
                x=amplitude_range,
                y=z_values,
                title="Echoed ZX90 amplitude calibration",
                xlabel="CR amplitude [GHz]",
                ylabel="target <Z>",
                ylim=[-1.1, 1.1],
            )
            fig.add_hline(y=0.0, line_dash="dot", line_color="#64748b")
            figures["amplitude"] = fig
        return Result(
            data={
                "cr_param": cr_param,
                "tomography": [tomography_1, tomography_2, tomography_3],
                "amplitude_scan": {
                    "amplitudes": amplitude_range,
                    "z_values": z_values,
                    "fit": fit_result,
                },
            },
            figure=figures.get("amplitude"),
            figures=figures,
        )

    def calibrate_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        amplitude_range: Any | None = None,
        n_repetitions: int = 4,
        n_points: int = 15,
        plot: bool | None = True,
        **_: Any,
    ) -> Any:
        """Fine tune the calibrated echoed ZX90 amplitude with qxsimulator."""
        np, _pd, qx, _qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        cr_label = f"{control_qubit}-{target_qubit}"
        if cr_label not in self.cr_params:
            self.obtain_cr_params(control_qubit, target_qubit, plot=False)
        param = self.cr_params[cr_label]
        base_amplitude = float(param["cr_amplitude"])
        if amplitude_range is None:
            amplitude_range = np.linspace(base_amplitude * 0.9, base_amplitude * 1.1, n_points)
        else:
            amplitude_range = np.asarray(amplitude_range, dtype=float)
        simulator = QuantumSimulator(self._qx_system(include_decoherence=False))
        original = dict(param)

        ideal_p1 = np.array(
            [np.sin(repeats * np.pi / 4) ** 2 for repeats in range(1, n_repetitions + 1)]
        )

        def repeated_p1(amplitude: float) -> list[float]:
            ratio = float(amplitude) / base_amplitude
            param["cr_amplitude"] = float(amplitude)
            param["cancel_amplitude"] = float(original["cancel_amplitude"]) * ratio
            values = []
            for repeats in range(1, n_repetitions + 1):
                sequence = self.zx90(control_qubit, target_qubit).repeated(repeats)
                result = simulator.mesolve(
                    sequence,
                    initial_state={control_qubit: "0", target_qubit: "0"},
                    n_samples=2,
                )
                populations = result._get_population(result.get_substates(target_qubit)[-1])
                values.append(float(np.real(populations[1])))
            return values

        p1_values = np.array(
            [repeated_p1(float(amplitude)) for amplitude in amplitude_range],
            dtype=float,
        )
        errors = np.mean((p1_values - ideal_p1) ** 2, axis=1)
        fit_degree = min(2, len(amplitude_range) - 1)
        coefficients = np.polyfit(amplitude_range, errors, deg=fit_degree)
        tuned_amplitude = float(amplitude_range[np.argmin(errors)])
        if fit_degree == 2 and coefficients[0] > 0:
            tuned_amplitude = float(-coefficients[1] / (2 * coefficients[0]))
        fit_result = {
            "degree": fit_degree,
            "coefficients": coefficients,
            "argmin": tuned_amplitude,
        }
        if not (float(np.min(amplitude_range)) <= tuned_amplitude <= float(np.max(amplitude_range))):
            tuned_amplitude = float(amplitude_range[np.argmin(errors)])
        ratio = tuned_amplitude / base_amplitude
        param["cr_amplitude"] = tuned_amplitude
        param["cancel_amplitude"] = float(original["cancel_amplitude"]) * ratio
        self.rzx90_duration = 2.0 * float(param["duration"]) + 2.0 * self.pi_duration
        self.cx_duration = self.rzx90_duration + self.hpi_duration

        figures = {}
        if plot:
            fig = qx.viz.make_plot_figure(
                x=amplitude_range,
                y=errors,
                title="ZX90 amplitude fine tune",
                xlabel="CR amplitude [GHz]",
                ylabel=f"Repeat-sequence MSE ({n_repetitions} repeats)",
            )
            fig.add_vline(x=tuned_amplitude, line_dash="dot", line_color="#dc2626")
            figures["amplitude"] = fig

        return Result(
            data={
                "cr_param": dict(param),
                "amplitudes": amplitude_range,
                "p1_values": p1_values,
                "ideal_p1_values": ideal_p1,
                "errors": errors,
                "fit": fit_result,
            },
            figure=figures.get("amplitude"),
            figures=figures,
        )

    def zx90(self, control_qubit: str, target_qubit: str) -> Any:
        """Build the calibrated echoed ZX90 pulse schedule."""
        np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        cr_label = f"{control_qubit}-{target_qubit}"
        if cr_label not in self.cr_params:
            self.obtain_cr_params(control_qubit, target_qubit, plot=False)
        param = self.cr_params[cr_label]
        control_index = self.qubit_labels.index(control_qubit)
        target_index = self.qubit_labels.index(target_qubit)
        frequency_diff = (
            self.qubit_frequencies[control_index] - self.qubit_frequencies[target_index]
        )
        pi_pulse = self.drag_pi_pulses.get(control_qubit)
        if pi_pulse is None:
            self.calibrate_drag_pi_pulse([control_qubit], plot=False)
            pi_pulse = self.drag_pi_pulses[control_qubit]
        cr_beta = -1 / (2 * np.pi * frequency_diff)
        cr_waveform = qx.pulse.FlatTop(
            duration=param["duration"],
            amplitude=2 * np.pi * param["cr_amplitude"],
            tau=param["ramptime"],
            phase=param["cr_phase"] + 0.25 * np.pi,
            beta=cr_beta,
        )
        cancel_waveform = qx.pulse.FlatTop(
            duration=param["duration"],
            amplitude=2 * np.pi * param["cancel_amplitude"],
            tau=param["ramptime"],
            phase=param["cancel_phase"],
            beta=-1 / (2 * np.pi * self.qubit_anharmonicities[target_index]),
        )
        channels = [
            qx.PulseChannel(
                label=control_qubit,
                frequency=self.qubit_frequencies[control_index],
                target=control_qubit,
            ),
            qx.PulseChannel(
                label=f"{control_qubit}-{target_qubit}",
                frequency=self.qubit_frequencies[target_index],
                target=control_qubit,
            ),
            qx.PulseChannel(
                label=target_qubit,
                frequency=self.qubit_frequencies[target_index],
                target=target_qubit,
            ),
        ]
        with qx.PulseSchedule(channels) as cr:
            cr.add(f"{control_qubit}-{target_qubit}", cr_waveform)
            cr.add(target_qubit, cancel_waveform)
        with qx.PulseSchedule(channels) as ecr:
            ecr.call(cr)
            ecr.barrier()
            ecr.add(control_qubit, pi_pulse)
            ecr.barrier()
            ecr.call(cr.scaled(-1))
            ecr.barrier()
            ecr.add(control_qubit, pi_pulse)
        return ecr

    @property
    def pulse(self) -> "FakeQubexExperiment":
        """Expose an experiment-like pulse service."""
        return self

    @property
    def dt(self) -> float:
        """Sampling period used by the fake experiment, in seconds."""
        return 1e-9

    def x90(self, target: str) -> Any:
        if target not in self.drag_hpi_pulses:
            self.calibrate_drag_hpi_pulse([target], plot=False)
        return self.drag_hpi_pulses[target]

    def x90m(self, target: str) -> Any:
        return self.x90(target).scaled(-1)

    def x180(self, target: str) -> Any:
        if target not in self.drag_pi_pulses:
            self.calibrate_drag_pi_pulse([target], plot=False)
        return self.drag_pi_pulses[target]

    def y90(self, target: str) -> Any:
        return self.x90(target).shifted(0.5 * _simulation_dependencies()[0].pi)

    def y90m(self, target: str) -> Any:
        return self.x90(target).shifted(-0.5 * _simulation_dependencies()[0].pi)

    def y180(self, target: str) -> Any:
        return self.x180(target).shifted(0.5 * _simulation_dependencies()[0].pi)

    def z90(self) -> Any:
        _np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        return qx.pulse.VirtualZ(0.5 * _np.pi)

    def z180(self) -> Any:
        _np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        return qx.pulse.VirtualZ(_np.pi)

    def hadamard(self, target: str) -> Any:
        _np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        return qx.PulseArray([self.z180(), self.y90(target)])

    def readout(self, target: str) -> Any:
        _np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        return qx.pulse.Blank(self.readout_duration)

    def cx(self, control_qubit: str, target_qubit: str) -> Any:
        _np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        zx90 = self.zx90(control_qubit, target_qubit)
        with qx.PulseSchedule(list(zx90.labels)) as schedule:
            schedule.call(zx90)
            schedule.add(control_qubit, qx.pulse.VirtualZ(-0.5 * _np.pi))
            schedule.add(target_qubit, self.x90(target_qubit).scaled(-1))
        return schedule

    def cnot(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: Any | None = None,
        **_: Any,
    ) -> Any:
        """QUBEX pulse-service alias for ``cx`` with optional ZX90 injection."""
        if zx90 is None:
            return self.cx(control_qubit, target_qubit)
        _np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        with qx.PulseSchedule(list(zx90.labels)) as schedule:
            schedule.call(zx90)
            schedule.add(control_qubit, qx.pulse.VirtualZ(-0.5 * _np.pi))
            schedule.add(target_qubit, self.x90(target_qubit).scaled(-1))
        return schedule

    def resolve_read_label(self, target: str, allow_legacy: bool = False) -> str:
        return f"R{target}"

    def get_target(self, label: str) -> Any:
        from types import SimpleNamespace

        if label in self.qubit_labels:
            index = self.qubit_labels.index(label)
            return SimpleNamespace(
                label=label,
                frequency=self.qubit_frequencies[index],
                object=SimpleNamespace(label=label),
            )
        if "-" in label:
            control, target = label.split("-", 1)
            target_index = self.qubit_labels.index(target)
            return SimpleNamespace(
                label=label,
                frequency=self.qubit_frequencies[target_index],
                object=SimpleNamespace(label=control),
            )
        if label.startswith("R") and label[1:] in self.qubit_labels:
            return SimpleNamespace(
                label=label,
                frequency=0.0,
                object=SimpleNamespace(label=label),
            )
        raise ValueError(label)

    def repeat_sequence(
        self,
        sequence: Any,
        *,
        repetitions: int = 20,
        initial_state: Mapping[str, str] | None = None,
        plot: bool | None = True,
        _simulator: Any | None = None,
    ) -> Any:
        """Repeat a pulse or pulse schedule and return a QUBEX-style result."""
        np, pd, qx, _qt, Result, _StateClassifierGMM, Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        simulator = _simulator or QuantumSimulator(self._qx_system())
        initial = {label: "0" for label in self.qubit_labels}
        if initial_state:
            initial.update(initial_state)
        rows = []
        figure = None
        if isinstance(sequence, Mapping):
            if len(sequence) != 1:
                raise ValueError("Fake repeat_sequence currently supports one pulse target.")
            target, pulse = next(iter(sequence.items()))
            index = self.qubit_labels.index(target)
            for repeats in range(repetitions + 1):
                if repeats == 0:
                    rows.append({"repeats": 0, "0": 1.0, "1": 0.0, "2": 0.0})
                    continue
                result = simulator.mesolve(
                    controls=[
                        Control(
                            target=target,
                            frequency=self.qubit_frequencies[index],
                            waveform=qx.PulseArray([pulse] * repeats),
                        )
                    ],
                    initial_state=initial,
                    n_samples=2,
                )
                populations = result._get_population(result.get_substates(target)[-1])
                rows.append(
                    {"repeats": repeats}
                    | {str(i): float(value) for i, value in enumerate(np.real(populations))}
                )
            df = pd.DataFrame(rows)
            if plot:
                figure = qx.viz.make_plot_figure(
                    x=df["repeats"],
                    y=df["1"],
                    title=f"Repeat sequence : {target}",
                    xlabel="Number of repetitions",
                    ylabel="Normalized signal",
                    ylim=[0, 1],
                )
            return Result(data={"dataframe": df}, figure=figure)

        target = self.qubit_labels[1]
        for repeats in range(repetitions + 1):
            if repeats == 0:
                rows.append({"repeats": 0, "P0": 1.0, "P1": 0.0, "P2": 0.0})
                continue
            result = simulator.mesolve(
                controls=sequence.repeated(repeats),
                initial_state=initial,
                n_samples=2,
            )
            populations = result._get_population(result.get_substates(target)[-1])
            rows.append(
                {"repeats": repeats}
                | {f"P{i}": float(value) for i, value in enumerate(np.real(populations))}
            )
        df = pd.DataFrame(rows)
        if plot:
            figure = qx.viz.make_plot_figure(
                x=df["repeats"],
                y=df["P1"],
                title=f"Repeat sequence : {target}",
                xlabel="Number of repetitions",
                ylabel="Normalized signal",
                ylim=[0, 1],
            )
        return Result(data={"dataframe": df}, figure=figure)

    def pulse_tomography(
        self,
        sequence: Any,
        *,
        initial_state: Mapping[str, str] | None = None,
        n_samples: int | None = None,
        plot: bool | None = True,
        **_: Any,
    ) -> Any:
        """Run simulated pulse tomography with the QUBEX experiment API shape."""
        np, _pd, qx, _qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        initial = {label: "0" for label in self.qubit_labels}
        if initial_state:
            initial.update(initial_state)
        result = QuantumSimulator(self._qx_system()).mesolve(
            self._annotate_schedule_metadata(sequence),
            initial_state=initial,
            n_samples=n_samples or 100,
        )
        data = {
            target: result.get_bloch_vectors(target)
            for target in self._tomography_targets(sequence)
        }
        figures = {}
        if plot:
            for target, vectors in data.items():
                fig = qx.viz.make_figure()
                for axis, values in zip(("X", "Y", "Z"), np.asarray(vectors).T, strict=True):
                    fig.add_scatter(x=result.times, y=values, mode="lines", name=axis)
                fig.update_layout(
                    title=f"State evolution : {target}",
                    xaxis_title="Time [ns]",
                    yaxis_title="Bloch vector",
                    yaxis_range=[-1.1, 1.1],
                )
                figures[target] = fig
        return Result(data=data, figures=figures, figure=next(iter(figures.values()), None))

    def measure_bell_state(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        control_basis: str | None = None,
        target_basis: str | None = None,
        zx90: Any | None = None,
        plot: bool | None = True,
        **_: Any,
    ) -> Any:
        """Measure a simulated Bell state in the requested Pauli basis."""
        np, _pd, qx, _qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        control_basis = control_basis or "Z"
        target_basis = target_basis or "Z"
        schedule = self._bell_measurement_schedule(
            control_qubit,
            target_qubit,
            control_basis=control_basis,
            target_basis=target_basis,
            zx90=zx90,
        )
        result = QuantumSimulator(self._qx_system()).mesolve(
            self._annotate_schedule_metadata(schedule),
            initial_state={control_qubit: "0", target_qubit: "0"},
            n_samples=2,
        )
        probabilities = self._computational_probabilities(result)
        raw = np.array([probabilities[label] for label in ("00", "01", "10", "11")])
        figure = None
        if plot:
            figure = qx.viz.make_figure()
            labels = ["|00>", "|01>", "|10>", "|11>"]
            figure.add_bar(x=labels, y=raw, name="Raw")
            figure.add_bar(x=labels, y=raw, name="Mitigated")
            figure.update_layout(
                title=f"Bell state measurement: {control_qubit}-{target_qubit}",
                xaxis_title=f"State ({control_basis}{target_basis} basis)",
                yaxis_title="Probability",
                barmode="group",
                yaxis_range=[0, 1],
            )
        return Result(
            data={
                "raw": raw,
                "mitigated": raw.copy(),
                "probabilities": probabilities,
                "basis": f"{control_basis}{target_basis}",
            },
            figure=figure,
        )

    def bell_state_tomography(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        readout_mitigation: bool | None = True,
        zx90: Any | None = None,
        plot: bool | None = True,
        mle_fit: bool | None = False,
        **_: Any,
    ) -> Any:
        """Perform lightweight Bell-state tomography from simulated basis probabilities."""
        np, _pd, qx, _qt, Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        probabilities = {}
        for control_basis in ("X", "Y", "Z"):
            for target_basis in ("X", "Y", "Z"):
                result = self.measure_bell_state(
                    control_qubit,
                    target_qubit,
                    control_basis=control_basis,
                    target_basis=target_basis,
                    zx90=zx90,
                    plot=False,
                )
                probabilities[f"{control_basis}{target_basis}"] = (
                    result["mitigated"] if readout_mitigation else result["raw"]
                )

        paulis = {
            "I": np.array([[1, 0], [0, 1]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        expected_values = {}
        rho = np.zeros((4, 4), dtype=complex)
        for control_basis, control_pauli in paulis.items():
            for target_basis, target_pauli in paulis.items():
                basis = f"{control_basis}{target_basis}"
                if basis == "II":
                    p = probabilities["ZZ"]
                    expectation = p[0] + p[1] + p[2] + p[3]
                elif control_basis == "I":
                    p = probabilities[f"Z{target_basis}"]
                    expectation = p[0] - p[1] + p[2] - p[3]
                elif target_basis == "I":
                    p = probabilities[f"{control_basis}Z"]
                    expectation = p[0] + p[1] - p[2] - p[3]
                else:
                    p = probabilities[basis]
                    expectation = p[0] - p[1] - p[2] + p[3]
                expected_values[basis] = float(np.real(expectation))
                rho += expectation * np.kron(control_pauli, target_pauli)
        rho = rho / 4
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        fidelity = float(np.real(np.conj(bell) @ rho @ bell))
        figure = None
        if plot:
            figure = qx.viz.make_figure()
            figure.add_heatmap(
                z=np.real(rho),
                x=["00", "01", "10", "11"],
                y=["00", "01", "10", "11"],
                colorscale="RdBu",
                zmid=0,
            )
            figure.update_layout(
                title=f"Bell state tomography: {control_qubit}-{target_qubit} (F={fidelity:.4f})",
                width=600,
                height=420,
            )
        return Result(
            data={
                "probabilities": probabilities,
                "expected_values": expected_values,
                "density_matrix": rho,
                "fidelity": fidelity,
                "mle_fit": bool(mle_fit),
            },
            figure=figure,
        )

    def zx90_gate_fidelity(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: Any | None = None,
        include_decoherence: bool = False,
        **_: Any,
    ) -> Any:
        """Compute qxsimulator average gate fidelity for the calibrated ZX90 pulse."""
        np, _pd, _qx, qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        target_unitary = qt.Qobj(
            [
                [1, -1j, 0, 0],
                [-1j, 1, 0, 0],
                [0, 0, 1, 1j],
                [0, 0, 1j, 1],
            ]
        ) / np.sqrt(2)
        target_unitary.dims = [[2, 2], [2, 2]]
        fidelity = QuantumSimulator(
            self._qx_system(include_decoherence=include_decoherence)
        ).gate_fidelity(
            controls=self._annotate_schedule_metadata(
                zx90 or self.zx90(control_qubit, target_qubit)
            ),
            target_unitary=target_unitary,
        )
        self.two_qubit_fidelity = float(fidelity)
        return Result(
            data={
                "gate": "zx90",
                "control": control_qubit,
                "target": target_qubit,
                "fidelity": float(fidelity),
                "include_decoherence": include_decoherence,
            }
        )

    def cx_gate_fidelity(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        include_decoherence: bool = False,
        **_: Any,
    ) -> Any:
        """Compute qxsimulator average gate fidelity for the synthesized Qiskit CX."""
        _np, _pd, _qx, qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        fidelity = QuantumSimulator(
            self._qx_system(include_decoherence=include_decoherence)
        ).gate_fidelity(
            controls=materialize_pulse_schedule_for_simulation(
                self._annotate_schedule_metadata(self.cx(control_qubit, target_qubit))
            ),
            target_unitary=self._cx_target_unitary(qt),
        )
        return Result(
            data={
                "gate": "cx",
                "control": control_qubit,
                "target": target_qubit,
                "fidelity": float(fidelity),
                "include_decoherence": include_decoherence,
            }
        )

    def calibrate_cx_frame(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        phase_grid: Any | None = None,
        include_decoherence: bool = False,
        **_: Any,
    ) -> Any:
        """Estimate local-Z frame dressing required by the synthesized CX pulse."""
        np, _pd, _qx, qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        if phase_grid is None:
            phase_grid = np.linspace(-np.pi, np.pi, 25)
        else:
            phase_grid = np.asarray(phase_grid, dtype=float)

        simulator = QuantumSimulator(
            self._qx_system(include_decoherence=include_decoherence)
        )
        superop = simulator.propagator(
            materialize_pulse_schedule_for_simulation(
                self._annotate_schedule_metadata(self.cx(control_qubit, target_qubit))
            )
        )
        superop = simulator.system.truncate_superoperator(superop)
        cx_unitary = self._cx_target_unitary(qt)
        base_fidelity = float(qt.average_gate_fidelity(superop, cx_unitary))

        best_fidelity = base_fidelity
        best_control_phase = 0.0
        best_target_phase = 0.0
        for control_phase in phase_grid:
            for target_phase in phase_grid:
                dressed = self._dressed_cx_target_unitary(
                    qt,
                    np,
                    float(control_phase),
                    float(target_phase),
                )
                fidelity = float(qt.average_gate_fidelity(superop, dressed))
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_control_phase = float(control_phase)
                    best_target_phase = float(target_phase)

        frame_param = {
            "control": control_qubit,
            "target": target_qubit,
            "base_fidelity": base_fidelity,
            "dressed_fidelity": best_fidelity,
            "post_control_z": best_control_phase,
            "post_target_z": best_target_phase,
            "include_decoherence": include_decoherence,
        }
        self.cx_frame_params[f"{control_qubit}-{target_qubit}"] = frame_param
        return Result(data=frame_param)

    def measure_state_distribution(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: int = 2,
        n_shots: int = 10_000,
        **_: Any,
    ) -> Any:
        """Generate synthetic IQ distributions for classifier calibration."""
        np, _pd, _qx, _qt, Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        rng = np.random.default_rng(7)
        centers = {0: 0.10 + 0.02j, 1: 0.86 + 0.34j, 2: 1.14 - 0.20j}
        sigma = 0.16
        target_labels = self._target_labels(targets)
        distributions = []
        for state in range(n_states):
            shot_map = {}
            for target in target_labels:
                sampled_states = np.full(n_shots, state)
                noise = sigma * (
                    rng.normal(size=n_shots) + 1j * rng.normal(size=n_shots)
                )
                shot_map[target] = np.array([centers[int(s)] for s in sampled_states]) + noise
            distributions.append(shot_map)
        return Result(data={"distributions": distributions})

    def build_classifier(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: int = 2,
        n_shots: int = 10_000,
        plot: bool | None = True,
        **_: Any,
    ) -> Any:
        """Build synthetic state classifiers with the QUBEX experiment API shape."""
        np, _pd, qx, _qt, Result, StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        target_labels = self._target_labels(targets)
        distributions = self.measure_state_distribution(
            target_labels,
            n_states=n_states,
            n_shots=n_shots,
        )["distributions"]
        data = {
            target: {state: distributions[state][target] for state in range(n_states)}
            for target in target_labels
        }
        classifiers = {
            target: StateClassifierGMM.fit(data[target], phase=0.0)
            for target in target_labels
        }
        classified = {}
        fidelities = {}
        figures = {}
        for target in target_labels:
            classified[target] = []
            for state in range(n_states):
                predicted = classifiers[target].predict(data[target][state])
                counts = np.bincount(predicted, minlength=classifiers[target].n_states)
                classified[target].append(
                    {label: int(counts[label]) for label in range(len(counts))}
                )
                if plot:
                    fig = qx.viz.make_classification_figure(
                        target=target,
                        data=data[target][state],
                        labels=predicted,
                        centers=classifiers[target].centers,
                        stddevs=classifiers[target].stddevs,
                    )
                    fig.update_layout(title=f"{target} prepared as |{state}>")
                    figures[f"{target}:{state}"] = fig
            fidelities[target] = [
                classified[target][state][state] / sum(classified[target][state].values())
                for state in range(n_states)
            ]
            self.classifiers[target] = classifiers[target]
            self.readout_assignment_errors[target] = 1.0 - float(np.mean(fidelities[target]))
        return Result(
            data={
                "data": data,
                "classifiers": classifiers,
                "classified": classified,
                "readout_fidelities": fidelities,
                "average_readout_fidelity": {
                    target: float(np.mean(fidelities[target])) for target in target_labels
                },
            },
            figures=figures,
        )

    @property
    def clifford_generator(self) -> Any:
        """Return QUBEX's Clifford generator used by RB sequence builders."""
        from qubex.clifford import CliffordGenerator

        if self._clifford_generator is None:
            self._clifford_generator = CliffordGenerator()
        return self._clifford_generator

    @property
    def clifford(self) -> dict[str, Any]:
        """Return the QUBEX Clifford dictionary."""
        return self.clifford_generator.cliffords

    def rb_sequence(
        self,
        target: str,
        *,
        n: int,
        x90: Any | Mapping[str, Any] | None = None,
        zx90: Any | None = None,
        interleaved_waveform: Any | None = None,
        interleaved_clifford: Any | None = None,
        seed: int | None = None,
    ) -> Any:
        """Build a QUBEX-style randomized benchmarking pulse schedule."""
        if "-" in target:
            if x90 is not None and not isinstance(x90, Mapping):
                raise ValueError("x90 must be a mapping for 2Q RB.")
            return self._rb_sequence_2q(
                target,
                n=n,
                x90=x90,
                zx90=zx90,
                interleaved_waveform=interleaved_waveform,
                interleaved_clifford=interleaved_clifford,
                seed=seed,
            )
        if isinstance(x90, Mapping):
            x90 = x90.get(target)
        return self._rb_sequence_1q(
            target,
            n=n,
            x90=x90,
            interleaved_waveform=interleaved_waveform,
            interleaved_clifford=interleaved_clifford,
            seed=seed,
        )

    def randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: Any | None = None,
        n_trials: int | None = None,
        seeds: Any | None = None,
        max_n_cliffords: int | None = None,
        x90: Mapping[str, Any] | None = None,
        zx90: Mapping[str, Any] | None = None,
        plot: bool | None = True,
        include_decoherence: bool = False,
        **_: Any,
    ) -> Any:
        """Run pulse-schedule RB through qxsimulator with the QUBEX API shape."""
        return self._rb_experiment(
            targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            interleaved_clifford=None,
            interleaved_waveform=None,
            plot=plot,
            include_decoherence=include_decoherence,
        )

    def interleaved_randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        interleaved_clifford: str | Any,
        interleaved_waveform: Mapping[str, Any] | None = None,
        n_cliffords_range: Any | None = None,
        n_trials: int | None = None,
        seeds: Any | None = None,
        max_n_cliffords: int | None = None,
        x90: Mapping[str, Any] | None = None,
        zx90: Mapping[str, Any] | None = None,
        plot: bool | None = True,
        include_decoherence: bool = False,
        **_: Any,
    ) -> Any:
        """Run interleaved pulse-schedule RB through qxsimulator."""
        if isinstance(interleaved_clifford, str):
            interleaved_clifford = self.clifford[interleaved_clifford]
        return self._rb_experiment(
            targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            interleaved_clifford=interleaved_clifford,
            interleaved_waveform=interleaved_waveform,
            plot=plot,
            include_decoherence=include_decoherence,
        )

    def _rb_sequence_1q(
        self,
        target: str,
        *,
        n: int,
        x90: Any | None = None,
        interleaved_waveform: Any | None = None,
        interleaved_clifford: Any | None = None,
        seed: int | None = None,
    ) -> Any:
        np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        x90 = x90 or self.x90(target)
        z90 = qx.pulse.VirtualZ(np.pi / 2)
        if interleaved_clifford is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="1Q",
                seed=seed,
            )
        else:
            if interleaved_waveform is None:
                if interleaved_clifford.name == "X90":
                    interleaved_waveform = self.x90(target)
                elif interleaved_clifford.name == "X180":
                    interleaved_waveform = self.x180(target)
                else:
                    raise ValueError("interleaved_waveform must be provided.")
            cliffords, inverse = self.clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="1Q",
                seed=seed,
            )
        sequence = []

        def add_gate(gate: str) -> None:
            if gate == "X90":
                sequence.append(x90)
            elif gate == "Z90":
                sequence.append(z90)
            else:
                raise ValueError(f"Invalid 1Q Clifford gate {gate!r}.")

        for clifford in cliffords:
            for gate in clifford:
                add_gate(gate)
            if interleaved_waveform is not None:
                sequence.append(interleaved_waveform)
        for gate in inverse:
            add_gate(gate)
        with qx.PulseSchedule([target]) as schedule:
            schedule.add(target, qx.PulseArray(sequence))
        return schedule

    def _rb_sequence_2q(
        self,
        target: str,
        *,
        n: int,
        x90: Mapping[str, Any] | None = None,
        zx90: Any | None = None,
        interleaved_waveform: Any | None = None,
        interleaved_clifford: Any | None = None,
        seed: int | None = None,
    ) -> Any:
        np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        control_qubit, target_qubit = target.split("-", 1)
        xi90 = x90.get(control_qubit) if x90 is not None else None
        ix90 = x90.get(target_qubit) if x90 is not None else None
        xi90 = xi90 or self.x90(control_qubit)
        ix90 = ix90 or self.x90(target_qubit)
        z90 = qx.pulse.VirtualZ(np.pi / 2)
        zx90 = zx90 or self.zx90(control_qubit, target_qubit)
        if interleaved_clifford is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="2Q",
                seed=seed,
            )
        else:
            if interleaved_waveform is None:
                if interleaved_clifford.name == "ZX90":
                    interleaved_waveform = self.zx90(control_qubit, target_qubit)
                else:
                    raise ValueError("interleaved_waveform must be provided.")
            cliffords, inverse = self.clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="2Q",
                seed=seed,
            )
        with qx.PulseSchedule([control_qubit, target, target_qubit]) as schedule:

            def add_gate(gate: str) -> None:
                if gate == "XI90":
                    schedule.add(control_qubit, xi90)
                elif gate == "IX90":
                    schedule.add(target_qubit, ix90)
                elif gate == "ZI90":
                    schedule.add(control_qubit, z90)
                elif gate == "IZ90":
                    schedule.add(target_qubit, z90)
                    schedule.add(target, z90)
                elif gate == "ZX90":
                    schedule.barrier()
                    schedule.call(zx90)
                    schedule.barrier()
                else:
                    raise ValueError(f"Invalid 2Q Clifford gate {gate!r}.")

            for clifford in cliffords:
                for gate in clifford:
                    add_gate(gate)
                if interleaved_waveform is not None:
                    schedule.barrier()
                    schedule.call(interleaved_waveform)
                    schedule.barrier()
            for gate in inverse:
                add_gate(gate)
        return schedule

    def _rb_experiment(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: Any | None,
        n_trials: int | None,
        seeds: Any | None,
        max_n_cliffords: int | None,
        x90: Mapping[str, Any] | None,
        zx90: Mapping[str, Any] | None,
        interleaved_clifford: Any | None,
        interleaved_waveform: Mapping[str, Any] | None,
        plot: bool | None,
        include_decoherence: bool,
    ) -> Any:
        np, pd, qx, _qt, Result, _StateClassifierGMM, _Control, QuantumSimulator = (
            _simulation_dependencies()
        )
        from qubex.analysis import fitting

        target_labels = [targets] if isinstance(targets, str) else list(targets)
        if n_trials is None:
            n_trials = 8
        if n_cliffords_range is None:
            max_n = 16 if max_n_cliffords is None else int(max_n_cliffords)
            n_cliffords_range = [0]
            n = 1
            while n <= max_n:
                n_cliffords_range.append(n)
                n *= 2
        n_cliffords_range = np.asarray(n_cliffords_range, dtype=int)
        if seeds is None:
            seeds = np.arange(n_trials, dtype=int)
        else:
            seeds = np.asarray(seeds, dtype=int)
            n_trials = len(seeds)

        simulator = QuantumSimulator(self._qx_system(include_decoherence=include_decoherence))
        data: dict[str, Any] = {}
        figures: dict[str, Any] = {}
        for target in target_labels:
            rows = []
            trials = []
            for n_cliffords in n_cliffords_range:
                values = []
                for seed in seeds:
                    sequence = self.rb_sequence(
                        target,
                        n=int(n_cliffords),
                        x90=x90,
                        zx90=zx90.get(target) if zx90 else None,
                        interleaved_waveform=interleaved_waveform.get(target)
                        if interleaved_waveform
                        else None,
                        interleaved_clifford=interleaved_clifford,
                        seed=int(seed),
                    )
                    values.append(
                        self._rb_survival_probability(
                            target,
                            sequence,
                            simulator=simulator,
                        )
                    )
                values_array = np.asarray(values, dtype=float)
                trials.append(values_array)
                rows.append(
                    {
                        "n_cliffords": int(n_cliffords),
                        "mean": float(np.mean(values_array)),
                        "std": float(np.std(values_array)),
                    }
                )
            df = pd.DataFrame(rows)
            dimension = 4 if "-" in target else 2
            fit_result = fitting.fit_rb(
                target=target,
                x=n_cliffords_range,
                y=df["mean"].to_numpy(),
                error_y=df["std"].to_numpy() if n_trials > 1 else None,
                dimension=dimension,
                bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                title="Interleaved randomized benchmarking"
                if interleaved_clifford
                else "Randomized benchmarking",
                xlabel="Number of Cliffords",
                ylabel="Survival probability",
                plot=False,
            )
            fit_payload = dict(fit_result) if getattr(fit_result, "data", None) else {}
            figure = fit_result.get_figure() if hasattr(fit_result, "get_figure") else None
            if plot:
                if figure is None:
                    figure = qx.viz.make_figure()
                    figure.add_scatter(
                        x=df["n_cliffords"],
                        y=df["mean"],
                        error_y={"type": "data", "array": df["std"]},
                        mode="lines+markers",
                        name=target,
                    )
                figure.update_layout(yaxis_range=[0, 1])
            if figure is not None:
                figures[target] = figure
            data[target] = {
                "n_cliffords": n_cliffords_range,
                "mean": df["mean"].to_numpy(),
                "std": df["std"].to_numpy(),
                "trials": np.vstack(trials),
                "seeds": np.asarray(seeds, dtype=int),
                "dataframe": df,
                "include_decoherence": include_decoherence,
                **fit_payload,
            }
        return Result(data=data, figures=figures, figure=next(iter(figures.values()), None))

    def _rb_survival_probability(
        self,
        target: str,
        sequence: Any,
        *,
        simulator: Any,
    ) -> float:
        np, _pd, _qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        schedule = materialize_pulse_schedule_for_simulation(
            self._annotate_schedule_metadata(sequence)
        )
        pulse_ranges = schedule.get_pulse_ranges() if hasattr(schedule, "get_pulse_ranges") else {}
        has_active_pulse = any(
            pulse_range.start != pulse_range.stop
            for ranges in pulse_ranges.values()
            for pulse_range in ranges
        )
        if not has_active_pulse:
            return 1.0
        if "-" in target:
            control_qubit, target_qubit = target.split("-", 1)
            result = simulator.mesolve(
                schedule,
                initial_state={control_qubit: "0", target_qubit: "0"},
                n_samples=2,
            )
            probabilities = self._computational_probabilities(result)
            return float(probabilities["00"])
        result = simulator.mesolve(
            schedule,
            initial_state={target: "0"},
            n_samples=2,
        )
        populations = result._get_population(result.get_substates(target)[-1])
        return float(np.real(populations[0]))

    def _qubit_topology(self, index: int, label: str) -> dict[str, Any]:
        t1, t2 = self._qubit_lifetime(index)
        x, y = self.positions[index]
        qubit = {
            "id": index,
            "physical_id": label_to_qid(label),
            "label": label,
            "position": {"x": x, "y": y},
            "frequency": self.qubit_frequencies[index],
            "anharmonicity": self.qubit_anharmonicities[index],
            "qubit_lifetime": {"t1": t1, "t2": t2},
            "gate_duration": {
                "rz": 0,
                "sx": self.hpi_duration,
                "sxdg": self.hpi_duration,
                "x": self.pi_duration,
                "y": self.pi_duration,
                "measure": self.readout_duration,
            },
        }
        if self.single_qubit_fidelity is not None:
            qubit["fidelity"] = self.single_qubit_fidelity
        readout_assignment_error = self.readout_assignment_errors.get(
            label,
            self.readout_assignment_error,
        )
        if readout_assignment_error is not None:
            qubit["meas_error"] = {
                "prob_meas1_prep0": readout_assignment_error / 2,
                "prob_meas0_prep1": readout_assignment_error / 2,
                "readout_assignment_error": readout_assignment_error,
            }
        return qubit

    def _coupling_topology(self) -> dict[str, Any]:
        gate_duration = {}
        if self.rzx90_duration is not None:
            gate_duration["rzx90"] = self.rzx90_duration
        if self.cx_duration is not None:
            gate_duration["cx"] = self.cx_duration
        coupling = {
            "control": 0,
            "target": 1,
            "coupling_strength_mhz": self.coupling_strength * 1000.0,
        }
        if self.two_qubit_fidelity is not None:
            coupling["fidelity"] = self.two_qubit_fidelity
        if gate_duration:
            coupling["gate_duration"] = gate_duration
        return coupling

    def _target_labels(self, targets: Collection[str] | str | None) -> list[str]:
        if targets is None:
            return list(self.qubit_labels)
        if isinstance(targets, str):
            return [targets]
        return list(targets)

    def _qubit_lifetime(self, index: int) -> tuple[float, float]:
        if self.qubit_lifetimes is not None:
            return self.qubit_lifetimes[index]
        return self.qubit_lifetime

    def _qx_system(self, *, include_decoherence: bool = True) -> Any:
        return build_qxsimulator_system(
            self.device_topology(),
            qubit_dimension=3,
            include_decoherence=include_decoherence,
        )

    def _area_normalized_drag(self, duration: float, area: float, beta: float) -> Any:
        np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        pulse = qx.pulse.Drag(duration=duration, amplitude=1.0, beta=beta, type="Gaussian")
        norm_factor = area / float(np.sum(np.abs(pulse.values) * pulse.SAMPLING_PERIOD))
        return pulse.scaled(norm_factor)

    def _tomography_targets(self, sequence: Any) -> list[str]:
        labels = getattr(sequence, "labels", self.qubit_labels)
        targets = []
        for label in labels:
            target = label.split("-", 1)[-1] if "-" in label else label
            if target in self.qubit_labels and target not in targets:
                targets.append(target)
        return targets or list(self.qubit_labels)

    def _bell_measurement_schedule(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        control_basis: str,
        target_basis: str,
        zx90: Any | None = None,
    ) -> Any:
        _np, _pd, qx, _qt, _Result, _StateClassifierGMM, _Control, _QuantumSimulator = (
            _simulation_dependencies()
        )
        cnot = self.cnot(control_qubit, target_qubit, zx90=zx90)
        with qx.PulseSchedule(list(cnot.labels)) as schedule:
            schedule.add(control_qubit, self.y90(control_qubit))
            schedule.call(cnot)
            if control_basis == "X":
                schedule.add(control_qubit, self.y90m(control_qubit))
            elif control_basis == "Y":
                schedule.add(control_qubit, self.x90(control_qubit))
            elif control_basis != "Z":
                raise ValueError(f"Unsupported control_basis: {control_basis!r}")
            if target_basis == "X":
                schedule.add(target_qubit, self.y90m(target_qubit))
            elif target_basis == "Y":
                schedule.add(target_qubit, self.x90(target_qubit))
            elif target_basis != "Z":
                raise ValueError(f"Unsupported target_basis: {target_basis!r}")
        return schedule

    def _annotate_schedule_metadata(self, schedule: Any) -> Any:
        if not hasattr(schedule, "labels"):
            return schedule
        channels = getattr(schedule, "_channels", {})
        for label in getattr(schedule, "labels", ()):
            target = None
            frequency = None
            if label in self.qubit_labels:
                index = self.qubit_labels.index(label)
                target = label
                frequency = self.qubit_frequencies[index]
            elif "-" in str(label):
                control, target_label = str(label).split("-", 1)
                if target_label in self.qubit_labels:
                    target = control
                    frequency = self.qubit_frequencies[self.qubit_labels.index(target_label)]
            if frequency is not None and hasattr(schedule, "set_frequency"):
                schedule.set_frequency(label, frequency)
            channel = channels.get(label) if isinstance(channels, dict) else None
            if channel is not None and target is not None:
                channel.target = target
        return schedule

    @staticmethod
    def _cx_target_unitary(qt: Any) -> Any:
        target_unitary = qt.Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )
        target_unitary.dims = [[2, 2], [2, 2]]
        return target_unitary

    @classmethod
    def _dressed_cx_target_unitary(
        cls,
        qt: Any,
        np: Any,
        control_phase: float,
        target_phase: float,
    ) -> Any:
        def rz(theta: float) -> Any:
            return np.diag(
                [
                    np.exp(-0.5j * theta),
                    np.exp(0.5j * theta),
                ]
            )

        dressed = np.kron(rz(control_phase), rz(target_phase)) @ np.asarray(
            cls._cx_target_unitary(qt).full()
        )
        target_unitary = qt.Qobj(dressed)
        target_unitary.dims = [[2, 2], [2, 2]]
        return target_unitary

    @staticmethod
    def _computational_probabilities(result: Any) -> dict[str, float]:
        import numpy as np

        populations = np.real(result.final_state.diag())
        return {
            label: float(populations[result.system.basis_labels.index(label)])
            for label in ("00", "01", "10", "11")
        }


def filter_pulse_schedule_for_simulation(
    schedule: Any,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    active_only: bool = True,
) -> Any:
    """Return a copy of a Qubex ``PulseSchedule`` with irrelevant channels removed."""
    filtered = schedule.copy() if hasattr(schedule, "copy") else schedule
    keep = set(str(label) for label in labels) if labels is not None else set(getattr(filtered, "labels", ()))
    if active_only:
        pulse_ranges = filtered.get_pulse_ranges() if hasattr(filtered, "get_pulse_ranges") else {}
        active_labels = {
            label
            for label, ranges in pulse_ranges.items()
            if any(pulse_range.start != pulse_range.stop for pulse_range in ranges)
        }
        keep &= active_labels
    channels = getattr(filtered, "_channels", None)
    if isinstance(channels, dict):
        filtered._channels = {
            label: channel
            for label, channel in channels.items()
            if label in keep
        }
    offsets = getattr(filtered, "_offsets", None)
    if isinstance(offsets, dict):
        for label in list(offsets):
            if label not in keep:
                offsets.pop(label, None)
    return filtered


def materialize_pulse_schedule_for_simulation(
    schedule: Any,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Convert a QUBEX pulse schedule into numeric waveforms for qxsimulator."""
    try:
        import numpy as np
        import qubex as qx
    except ImportError as exc:
        raise ImportError(
            "materialize_pulse_schedule_for_simulation requires qubex and numpy."
        ) from exc

    source_labels = list(labels or getattr(schedule, "labels", ()))
    schedule_labels = set(getattr(schedule, "labels", ()))
    channels = []
    final_frame_shifts: dict[str, float] = {}
    for label in source_labels:
        frequency = (
            schedule.get_frequency(label)
            if hasattr(schedule, "get_frequency") and label in schedule_labels
            else None
        )
        target = (
            schedule.get_target(label)
            if hasattr(schedule, "get_target") and label in schedule_labels
            else None
        )
        channels.append(qx.PulseChannel(label=label, frequency=frequency, target=target))
        if label in schedule_labels and hasattr(schedule, "get_final_frame_shift"):
            target_label = _simulation_target_label(target)
            frame_source = target_label if target_label in schedule_labels else label
            final_frame_shifts[label] = float(schedule.get_final_frame_shift(frame_source))

    with qx.PulseSchedule(channels) as materialized:
        for label in source_labels:
            if label not in schedule_labels:
                continue
            sequence = schedule._channels[label].sequence
            values = np.asarray(sequence.values, dtype=complex)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            materialized.add(
                label,
                qx.pulse.Arbitrary(
                    values=values,
                    sampling_period=getattr(sequence, "SAMPLING_PERIOD", None),
                ),
            )
            final_frame_shift = final_frame_shifts.get(label, 0.0)
            if abs(final_frame_shift) > 1e-12:
                materialized.add(label, qx.pulse.VirtualZ(-final_frame_shift))
    return materialized


def _simulation_target_label(target: Any) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        return target
    label = getattr(target, "label", None)
    if label is not None:
        return str(label)
    obj = getattr(target, "object", None)
    obj_label = getattr(obj, "label", None)
    if obj_label is not None:
        return str(obj_label)
    return str(target)


def build_qxsimulator_system(
    device_topology: str | Path | Mapping[str, Any],
    *,
    qubit_labels: list[str] | tuple[str, ...] | None = None,
    qubit_dimension: int = 3,
    include_decoherence: bool = True,
    include_resonators: bool = False,
    resonator_dimension: int = 2,
    default_coupling_strength: float | None = None,
):
    """Build a ``qxsimulator.QuantumSystem`` from Device Gateway topology data.

    Frequencies and anharmonicities are interpreted in GHz. Qubit lifetimes in
    ``qubit_lifetime`` are interpreted in microseconds. Coupling strengths from
    ``coupling_strength_mhz`` are converted to GHz for qxsimulator.
    """
    try:
        from qxsimulator import Coupling, QuantumSystem, Resonator, Transmon
    except ImportError as exc:
        raise ImportError(
            "build_qxsimulator_system requires qxsimulator to be installed."
        ) from exc

    topology = _load_topology(device_topology)
    all_qubits = list(topology.get("qubits", ()))
    qubits = list(all_qubits)
    if qubit_labels is not None:
        selected = set(str(label) for label in qubit_labels)
        qubits = [
            qubit
            for qubit in qubits
            if _qubit_label(qubit, all_qubits) in selected
        ]
    if not qubits:
        raise ValueError("device_topology must contain at least one qubit.")

    label_by_logical_id = {
        int(qubit["id"]): _qubit_label(qubit, all_qubits)
        for qubit in qubits
    }

    objects = []
    for qubit in qubits:
        label = _qubit_label(qubit, all_qubits)
        frequency = _required_float(qubit.get("frequency"), f"frequency for {label}")
        lifetime = qubit.get("qubit_lifetime") or {}
        relaxation_rate, dephasing_rate = _decoherence_rates(
            lifetime.get("t1"),
            lifetime.get("t2"),
            include_decoherence=include_decoherence,
        )
        objects.append(
            Transmon(
                label=label,
                dimension=qubit_dimension,
                frequency=frequency,
                anharmonicity=_optional_float(qubit.get("anharmonicity")),
                relaxation_rate=relaxation_rate,
                dephasing_rate=dephasing_rate,
            )
        )
        if include_resonators:
            resonator_frequency = _optional_float(
                qubit.get("resonator_frequency", qubit.get("readout_frequency"))
            )
            if resonator_frequency is not None:
                objects.append(
                    Resonator(
                        label=_readout_label(label),
                        dimension=resonator_dimension,
                        frequency=resonator_frequency,
                    )
                )

    couplings = []
    seen_coupling_pairs = set()
    for coupling in topology.get("couplings", ()):
        control = int(coupling["control"])
        target = int(coupling["target"])
        if control not in label_by_logical_id or target not in label_by_logical_id:
            continue
        pair = (label_by_logical_id[control], label_by_logical_id[target])
        pair_key = frozenset(pair)
        if pair_key in seen_coupling_pairs:
            continue
        strength = _coupling_strength_ghz(
            coupling,
            default_coupling_strength=default_coupling_strength,
        )
        if strength is None:
            continue
        seen_coupling_pairs.add(pair_key)
        couplings.append(Coupling(pair=pair, strength=strength))

    return QuantumSystem(objects=objects, couplings=couplings)


def _load_topology(device_topology: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(device_topology, Mapping):
        return device_topology
    return json.loads(Path(device_topology).read_text(encoding="utf-8"))


def _qubit_label(qubit: Mapping[str, Any], qubits: list[Mapping[str, Any]]) -> str:
    label = qubit.get("label")
    if label is not None:
        return str(label)
    physical_id = int(qubit.get("physical_id", qubit.get("id")))
    return qid_to_label(physical_id, _label_width_base(qubits))


def _label_width_base(qubits: list[Mapping[str, Any]]) -> int:
    physical_ids = [
        int(qubit.get("physical_id", qubit.get("id", index)))
        for index, qubit in enumerate(qubits)
    ]
    return max(max(physical_ids, default=0) + 1, len(qubits))


def _readout_label(qubit_label: str) -> str:
    return f"R{qubit_label}"


def _required_float(value: Any, name: str) -> float:
    converted = _optional_float(value)
    if converted is None:
        raise ValueError(f"device_topology is missing {name}.")
    return converted


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _decoherence_rates(
    t1_us: Any,
    t2_us: Any,
    *,
    include_decoherence: bool,
) -> tuple[float, float]:
    if not include_decoherence:
        return 0.0, 0.0
    relaxation_rate = _lifetime_rate_per_ns(t1_us)
    t2_rate = _lifetime_rate_per_ns(t2_us)
    dephasing_rate = max(0.0, t2_rate - 0.5 * relaxation_rate)
    return relaxation_rate, dephasing_rate


def _lifetime_rate_per_ns(value_us: Any) -> float:
    value = _optional_float(value_us)
    if value is None or value <= 0.0:
        return 0.0
    return 1.0 / (value * 1000.0)


def _coupling_strength_ghz(
    coupling: Mapping[str, Any],
    *,
    default_coupling_strength: float | None,
) -> float | None:
    if coupling.get("coupling_strength_mhz") is not None:
        return float(coupling["coupling_strength_mhz"]) / 1000.0
    if coupling.get("coupling_strength") is not None:
        return float(coupling["coupling_strength"])
    return default_coupling_strength


def _simulation_dependencies() -> tuple[Any, ...]:
    try:
        import numpy as np
        import pandas as pd
        import qutip as qt
        import qubex as qx
        from qubex.experiment.models.result import Result
        from qubex.measurement.classifiers.state_classifier_gmm import StateClassifierGMM
        from qxsimulator import Control, QuantumSimulator
    except ImportError as exc:
        raise ImportError(
            "FakeQubexExperiment calibration methods require qubex, qutip, "
            "qxsimulator, numpy, and pandas to be installed."
        ) from exc
    return np, pd, qx, qt, Result, StateClassifierGMM, Control, QuantumSimulator
