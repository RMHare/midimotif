from __future__ import annotations

from dataclasses import replace

from motiflab.core.constants import DEFAULT_BAR_BEATS
from motiflab.core.event_model import NoteEvent


def assign_rough_beats(events: list[NoteEvent], ticks_per_beat: int) -> list[NoteEvent]:
    """Assign beat and bar positions without changing raw timing."""
    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be > 0")
    beat_scale = 1.0 / ticks_per_beat

    output: list[NoteEvent] = []
    for event in events:
        onset_beat = event.onset_tick * beat_scale
        duration_beats = event.duration_tick * beat_scale
        beat_index = onset_beat % DEFAULT_BAR_BEATS
        bar_index = int(onset_beat // DEFAULT_BAR_BEATS)
        output.append(
            replace(
                event,
                onset_beat=onset_beat,
                duration_beats=duration_beats,
                beat_index=beat_index,
                bar_index=bar_index,
            )
        )
    return output
