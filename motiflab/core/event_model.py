from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(slots=True)
class NoteEvent:
    file_id: str
    track_id: int
    channel: int | None
    instrument_program: int | None
    is_drum: bool
    pitch: int
    pitch_class: int
    onset_seconds: float
    offset_seconds: float
    duration_seconds: float
    onset_tick: int
    offset_tick: int
    duration_tick: int
    velocity: int
    estimated_tempo: float | None
    bar_index: int | None
    beat_index: float | None
    onset_beat: float | None
    duration_beats: float | None
    microtiming_ms: float | None
    metrical_strength: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TrackSummary:
    track_id: int
    instrument_program: int | None
    is_drum: bool
    note_count: int
    pitch_min: int
    pitch_max: int
    median_pitch: float
    median_duration_beats: float
    density_notes_per_bar: float
    polyphony_rate: float
    syncopation_score: float
    likely_role: Literal[
        "melody",
        "bass",
        "drums",
        "chords",
        "pad",
        "arp",
        "countermelody",
        "lead",
        "unknown",
    ]
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)
