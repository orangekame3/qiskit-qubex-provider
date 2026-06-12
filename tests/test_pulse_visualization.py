"""Tests for pulse schedule inspection and visualization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qiskit_qubex_provider import (
    build_pulse_schedule_timeline_figure,
    diff_pulse_schedules,
    extract_pulse_timeline,
    summarize_pulse_schedule,
    write_pulse_schedule_timeline_image,
)


class FakeSampledSchedule:
    """Minimal stand-in for the qxpulse PulseSchedule sampling API."""

    def __init__(self, channels, sampling_period=2.0):
        self._channels = {
            label: np.asarray(values, dtype=complex)
            for label, values in channels.items()
        }
        self._sampling_period = sampling_period

    @property
    def labels(self):
        return list(self._channels)

    @property
    def length(self):
        return max((len(v) for v in self._channels.values()), default=0)

    @property
    def duration(self):
        return self.length * self._sampling_period

    def get_sampled_sequences(self):
        return dict(self._channels)


class FakeElement:
    """Stand-in for a qxpulse Waveform element."""

    def __init__(self, name, duration, amplitude=1.0):
        self.name = name
        self.duration = duration
        samples = max(1, int(duration // 2))
        self.values = np.full(samples, complex(amplitude))


class FakePhaseShift:
    """Stand-in for a qxpulse PhaseShift (no duration attribute)."""

    def __init__(self, theta):
        self.theta = theta


class FakeSequence:
    def __init__(self, elements):
        self.flattened_elements = elements


class FakeElementSchedule(FakeSampledSchedule):
    """Schedule exposing per-element structure like qxpulse PulseSchedule."""

    def __init__(self, channel_elements, sampling_period=2.0):
        self._sequences = {
            label: FakeSequence(elements)
            for label, elements in channel_elements.items()
        }
        channels = {}
        for label, elements in channel_elements.items():
            values = []
            for element in elements:
                if hasattr(element, "duration"):
                    values.extend(element.values)
            channels[label] = values
        super().__init__(channels, sampling_period=sampling_period)

    def get_sequences(self):
        return dict(self._sequences)


def test_summarize_pulse_schedule_reports_active_window() -> None:
    schedule = FakeSampledSchedule(
        {
            "Q00": [0, 0, 1 + 1j, 1, 0],
            "Q01": [0, 0, 0, 0, 0],
        },
        sampling_period=2.0,
    )

    summary = summarize_pulse_schedule(schedule)

    assert summary["Q00"] == {
        "duration_ns": 10.0,
        "active_start_ns": 4.0,
        "active_end_ns": 8.0,
        "n_samples": 5,
    }
    assert summary["Q01"]["active_start_ns"] is None
    assert summary["Q01"]["active_end_ns"] is None


def test_diff_pulse_schedules_detects_equal_and_changed_channels() -> None:
    schedule_a = FakeSampledSchedule({"Q00": [1, 0], "Q01": [0, 1]})
    schedule_b = FakeSampledSchedule({"Q00": [1, 0], "Q01": [0, 0.5]})

    diff = diff_pulse_schedules(schedule_a, schedule_b)

    assert diff["equal"] is False
    assert diff["channels"]["Q00"]["status"] == "equal"
    assert diff["channels"]["Q01"]["status"] == "changed"
    assert diff["channels"]["Q01"]["max_abs_diff"] == pytest.approx(0.5)
    assert diff["duration_ns_a"] == pytest.approx(4.0)

    same = diff_pulse_schedules(schedule_a, schedule_a)
    assert same["equal"] is True


def test_diff_pulse_schedules_reports_structural_mismatches() -> None:
    schedule_a = FakeSampledSchedule({"Q00": [1, 0], "Q01": [0, 1]})
    schedule_b = FakeSampledSchedule({"Q00": [1, 0, 0], "Q02": [1]})

    diff = diff_pulse_schedules(schedule_a, schedule_b)

    assert diff["equal"] is False
    assert diff["channels"]["Q00"]["status"] == "length_mismatch"
    assert diff["channels"]["Q00"]["n_samples_a"] == 2
    assert diff["channels"]["Q00"]["n_samples_b"] == 3
    assert diff["channels"]["Q01"]["status"] == "only_in_a"
    assert diff["channels"]["Q02"]["status"] == "only_in_b"


def test_extract_pulse_timeline_uses_element_structure() -> None:
    schedule = FakeElementSchedule(
        {
            "Q00": [
                FakeElement("Blank", 10.0, amplitude=0.0),
                FakeElement("FlatTop", 30.0),
                FakePhaseShift(1.5708),
                FakeElement("Drag", 20.0),
            ],
        }
    )

    timeline = extract_pulse_timeline(schedule)

    assert timeline["Q00"] == [
        {"kind": "pulse", "name": "FlatTop", "start_ns": 10.0, "duration_ns": 30.0},
        {
            "kind": "phase",
            "name": "FakePhaseShift",
            "start_ns": 40.0,
            "duration_ns": 0.0,
            "theta": 1.5708,
        },
        {"kind": "pulse", "name": "Drag", "start_ns": 40.0, "duration_ns": 20.0},
    ]


def test_extract_pulse_timeline_falls_back_to_sample_segments() -> None:
    schedule = FakeSampledSchedule(
        {"Q00": [0, 1, 1, 0, 0, 1, 0]},
        sampling_period=2.0,
    )

    timeline = extract_pulse_timeline(schedule)

    assert timeline["Q00"] == [
        {"kind": "pulse", "name": "pulse", "start_ns": 2.0, "duration_ns": 4.0},
        {"kind": "pulse", "name": "pulse", "start_ns": 10.0, "duration_ns": 2.0},
    ]


def test_build_pulse_schedule_timeline_figure_groups_by_pulse_name() -> None:
    pytest.importorskip("plotly")
    schedule = FakeElementSchedule(
        {
            "Q00": [
                FakeElement("FlatTop", 30.0),
                FakePhaseShift(0.5),
                FakeElement("Drag", 20.0),
            ],
            "Q01": [FakeElement("FlatTop", 100.0)],
        }
    )

    figure = build_pulse_schedule_timeline_figure(schedule)

    bar_traces = [trace for trace in figure.data if trace.type == "bar"]
    scatter_traces = [trace for trace in figure.data if trace.type == "scatter"]
    assert {trace.name for trace in bar_traces} == {"FlatTop", "Drag"}
    assert len(scatter_traces) == 1  # virtual-Z markers
    flat_top = next(trace for trace in bar_traces if trace.name == "FlatTop")
    assert list(flat_top.y) == ["Q00", "Q01"]
    assert list(flat_top.base) == [0.0, 0.0]
    assert list(flat_top.x) == [30.0, 100.0]


def test_build_pulse_schedule_timeline_figure_rejects_empty_schedule() -> None:
    pytest.importorskip("plotly")
    with pytest.raises(ValueError, match="no channels"):
        build_pulse_schedule_timeline_figure(FakeSampledSchedule({}))


def test_write_pulse_schedule_timeline_image_writes_html(tmp_path) -> None:
    pytest.importorskip("plotly")
    schedule = FakeSampledSchedule({"Q00": [1, 0]})
    output = tmp_path / "timeline.html"

    write_pulse_schedule_timeline_image(schedule, output)

    assert output.exists()
    assert "plotly" in output.read_text(encoding="utf-8")
