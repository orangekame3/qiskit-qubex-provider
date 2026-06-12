"""Visualize and inspect Qubex pulse schedules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_PULSE_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#ca8a04",
    "#db2777",
)
_PHASE_COLOR = "#475569"


def summarize_pulse_schedule(schedule: Any) -> dict[str, dict[str, Any]]:
    """Return per-channel timing facts for a Qubex ``PulseSchedule``.

    Each channel maps to ``duration_ns``, ``active_start_ns``,
    ``active_end_ns`` (first/last non-zero sample boundaries, ``None`` for an
    all-blank channel), and ``n_samples``.
    """
    sampling_period = _sampling_period(schedule)
    summary: dict[str, dict[str, Any]] = {}
    for label, values in _sampled_sequences(schedule).items():
        nonzero = np.flatnonzero(np.abs(values) > 0)
        summary[label] = {
            "duration_ns": len(values) * sampling_period,
            "active_start_ns": (
                float(nonzero[0]) * sampling_period if nonzero.size else None
            ),
            "active_end_ns": (
                float(nonzero[-1] + 1) * sampling_period if nonzero.size else None
            ),
            "n_samples": int(len(values)),
        }
    return summary


def diff_pulse_schedules(
    schedule_a: Any,
    schedule_b: Any,
    *,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Compare two Qubex ``PulseSchedule`` objects sample by sample.

    Returns a mapping with ``equal`` (overall verdict), per-schedule
    ``duration_ns``, and a ``channels`` mapping whose values report one of the
    statuses ``equal``, ``changed``, ``length_mismatch``, ``only_in_a`` or
    ``only_in_b`` together with ``max_abs_diff`` where comparable.
    """
    sampled_a = _sampled_sequences(schedule_a)
    sampled_b = _sampled_sequences(schedule_b)
    channels: dict[str, dict[str, Any]] = {}
    for label in _ordered_union([list(sampled_a), list(sampled_b)]):
        values_a = sampled_a.get(label)
        values_b = sampled_b.get(label)
        if values_a is None:
            channels[label] = {"status": "only_in_b", "max_abs_diff": None}
        elif values_b is None:
            channels[label] = {"status": "only_in_a", "max_abs_diff": None}
        elif len(values_a) != len(values_b):
            channels[label] = {
                "status": "length_mismatch",
                "max_abs_diff": None,
                "n_samples_a": int(len(values_a)),
                "n_samples_b": int(len(values_b)),
            }
        else:
            max_abs_diff = (
                float(np.max(np.abs(values_a - values_b))) if len(values_a) else 0.0
            )
            channels[label] = {
                "status": "changed" if max_abs_diff > atol else "equal",
                "max_abs_diff": max_abs_diff,
            }
    return {
        "equal": all(entry["status"] == "equal" for entry in channels.values()),
        "duration_ns_a": float(schedule_a.duration),
        "duration_ns_b": float(schedule_b.duration),
        "channels": channels,
    }


def extract_pulse_timeline(schedule: Any) -> dict[str, list[dict[str, Any]]]:
    """Return per-channel timeline entries for a Qubex ``PulseSchedule``.

    Each channel maps to a time-ordered list of entries. Pulse entries carry
    ``{"kind": "pulse", "name", "start_ns", "duration_ns"}`` where ``name`` is
    the Qubex waveform class name (e.g. ``FlatTop``); blank (idle) waveforms
    are skipped. Virtual-Z operations appear as zero-duration
    ``{"kind": "phase", ..., "theta"}`` entries.

    Falls back to contiguous non-blank sample segments (named ``"pulse"``)
    when the schedule does not expose per-element structure.
    """
    timeline = _timeline_from_elements(schedule)
    if timeline is not None:
        return timeline
    return _timeline_from_samples(schedule)


def build_pulse_schedule_timeline_figure(
    schedule: Any,
    *,
    title: str = "Pulse Schedule Timeline",
    width: int = 900,
):
    """Return a Plotly Gantt-style timeline figure for one pulse schedule.

    Renders one horizontal lane per channel with a labeled box for every
    pulse (colored by waveform name) and a tick marker for every virtual-Z
    phase shift — similar in spirit to Qiskit's ``timeline_drawer``. This
    view emphasizes *when* operations are scheduled; for full I/Q and phase
    detail use Qubex's ``PulseSchedule.plot()``.
    """
    try:
        import plotly.graph_objects as go
    except (ImportError, ValueError) as exc:
        raise ImportError(
            "Pulse schedule plotting requires the optional 'plot' extra: "
            "pip install 'qiskit-qubex-provider[plot]'."
        ) from exc

    timeline = extract_pulse_timeline(schedule)
    if not timeline:
        raise ValueError("schedule contains no channels to plot.")
    labels = list(timeline)
    lane_order = list(reversed(labels))  # first channel on top

    pulses_by_name: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    phases: list[tuple[str, dict[str, Any]]] = []
    for label in labels:
        for entry in timeline[label]:
            if entry["kind"] == "pulse":
                pulses_by_name.setdefault(entry["name"], []).append((label, entry))
            else:
                phases.append((label, entry))

    fig = go.Figure()
    for index, (name, pulses) in enumerate(pulses_by_name.items()):
        fig.add_trace(
            go.Bar(
                y=[label for label, _ in pulses],
                x=[entry["duration_ns"] for _, entry in pulses],
                base=[entry["start_ns"] for _, entry in pulses],
                orientation="h",
                name=name,
                marker=dict(
                    color=_PULSE_COLORS[index % len(_PULSE_COLORS)],
                    line=dict(color="#ffffff", width=1),
                ),
                text=name,
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    name + "<br>start=%{base:.1f} ns"
                    "<br>duration=%{x:.1f} ns<extra></extra>"
                ),
            )
        )
    if phases:
        # Virtual-Z operations take no time; render them as tall labeled
        # ticks drawn on top of the bars so they stay visible.
        fig.add_trace(
            go.Scatter(
                y=[label for label, _ in phases],
                x=[entry["start_ns"] for _, entry in phases],
                mode="markers+text",
                name="virtual Z",
                text=[
                    f"VZ({entry.get('theta', 0.0):+.2f})" for _, entry in phases
                ],
                textposition="top center",
                textfont=dict(size=11, color=_PHASE_COLOR),
                cliponaxis=False,
                marker=dict(
                    symbol="line-ns-open",
                    size=22,
                    color=_PHASE_COLOR,
                    line=dict(color=_PHASE_COLOR, width=3),
                ),
                customdata=[
                    [entry["name"], entry.get("theta")] for _, entry in phases
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>t=%{x:.1f} ns"
                    "<br>theta=%{customdata[1]:.3f} rad<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        width=width,
        height=44 * len(labels) + 170,
        barmode="overlay",
        bargap=0.3,
        xaxis=dict(title="Time (ns)", rangemode="tozero"),
        yaxis=dict(categoryorder="array", categoryarray=lane_order),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    return fig


def write_pulse_schedule_timeline_image(
    schedule: Any,
    output_image: str | Path,
    *,
    title: str = "Pulse Schedule Timeline",
    width: int = 900,
) -> None:
    """Write the timeline figure to HTML or a static image file.

    HTML output requires Plotly; static image formats such as PNG and SVG
    additionally require Plotly's Kaleido backend.
    """
    output_path = Path(output_image)
    suffix = output_path.suffix.lower()
    try:
        figure = build_pulse_schedule_timeline_figure(
            schedule, title=title, width=width
        )
        if suffix == ".html":
            figure.write_html(str(output_path), include_plotlyjs="cdn")
        else:
            figure.write_image(str(output_path), scale=2)
    except ImportError as exc:
        if suffix == ".html":
            raise RuntimeError(
                "Writing pulse schedule plots to HTML requires Plotly. Install "
                "the 'plot' extra: pip install 'qiskit-qubex-provider[plot]'."
            ) from exc
        raise RuntimeError(
            "Writing pulse schedule plots to static image formats requires "
            "Plotly and a working Kaleido backend. Install the 'plot' extra "
            "and use a .html output path if Kaleido is unavailable."
        ) from exc


def _timeline_from_elements(schedule: Any) -> dict[str, list[dict[str, Any]]] | None:
    try:
        sequences = schedule.get_sequences()
    except AttributeError:
        return None
    timeline: dict[str, list[dict[str, Any]]] = {}
    for label, sequence in sequences.items():
        try:
            elements = sequence.flattened_elements
        except AttributeError:
            return None
        entries: list[dict[str, Any]] = []
        cursor = 0.0
        for element in elements:
            duration = getattr(element, "duration", None)
            if duration is None:
                # Zero-duration phase shift (virtual Z).
                entries.append(
                    {
                        "kind": "phase",
                        "name": type(element).__name__,
                        "start_ns": cursor,
                        "duration_ns": 0.0,
                        "theta": float(getattr(element, "theta", 0.0)),
                    }
                )
                continue
            name = getattr(element, "name", type(element).__name__)
            values = np.asarray(getattr(element, "values", ()), dtype=complex)
            is_idle = name == "Blank" or not np.any(np.abs(values) > 0)
            if not is_idle:
                entries.append(
                    {
                        "kind": "pulse",
                        "name": name,
                        "start_ns": cursor,
                        "duration_ns": float(duration),
                    }
                )
            cursor += float(duration)
        timeline[label] = entries
    return timeline


def _timeline_from_samples(schedule: Any) -> dict[str, list[dict[str, Any]]]:
    sampling_period = _sampling_period(schedule)
    timeline: dict[str, list[dict[str, Any]]] = {}
    for label, values in _sampled_sequences(schedule).items():
        active = np.concatenate([[False], np.abs(values) > 0, [False]])
        edges = np.flatnonzero(np.diff(active.astype(int)))
        timeline[label] = [
            {
                "kind": "pulse",
                "name": "pulse",
                "start_ns": float(start) * sampling_period,
                "duration_ns": float(stop - start) * sampling_period,
            }
            for start, stop in zip(edges[0::2], edges[1::2])
        ]
    return timeline


def _sampled_sequences(schedule: Any) -> dict[str, np.ndarray]:
    values = schedule.get_sampled_sequences()
    return {
        label: np.asarray(channel_values, dtype=complex)
        for label, channel_values in values.items()
    }


def _sampling_period(schedule: Any) -> float:
    length = int(schedule.length)
    if length <= 0:
        return 0.0
    return float(schedule.duration) / length


def _ordered_union(label_lists: list[list[str]]) -> list[str]:
    ordered: list[str] = []
    for labels in label_lists:
        for label in labels:
            if label not in ordered:
                ordered.append(label)
    return ordered
