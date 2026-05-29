from __future__ import annotations

from dataclasses import replace

from motiflab.core.event_model import NoteEvent


DEFAULT_BAR_BEATS = 4.0


def assign_rough_beats(events: list[NoteEvent], ticks_per_beat: int) -> list[NoteEvent]:
    """Assign beat and bar positions without changing raw timing."""
    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be > 0")

    output: list[NoteEvent] = []
    for event in events:
        onset_beat = event.onset_tick / ticks_per_beat
        duration_beats = event.duration_tick / ticks_per_beat
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
