from motiflab.core.event_model import NoteEvent
from motiflab.core.harmonic_analysis import estimate_bar_chords, estimate_global_key


def _event(
    index: int, pitch: int, bar_index: int, track_id: int = 0, is_drum: bool = False
) -> NoteEvent:
    onset_tick = index * 480
    return NoteEvent(
        file_id="demo",
        track_id=track_id,
        channel=0,
        instrument_program=0,
        is_drum=is_drum,
        pitch=pitch,
        pitch_class=pitch % 12,
        onset_seconds=index * 0.5,
        offset_seconds=index * 0.5 + 0.5,
        duration_seconds=0.5,
        onset_tick=onset_tick,
        offset_tick=onset_tick + 480,
        duration_tick=480,
        velocity=90,
        estimated_tempo=120.0,
        bar_index=bar_index,
        beat_index=float(index % 4),
        onset_beat=float(index),
        duration_beats=1.0,
        microtiming_ms=0.0,
        metrical_strength=1.0,
    )


def test_estimate_bar_chords_detects_simple_major_triad() -> None:
    events = [
        _event(0, 60, 0),  # C
        _event(1, 64, 0),  # E
        _event(2, 67, 0),  # G
    ]

    chords = estimate_bar_chords(events)

    assert len(chords) == 1
    assert chords[0].chord == "C:major"
    assert chords[0].confidence == 1.0


def test_estimate_global_key_picks_most_common_root_and_mode() -> None:
    events = [
        _event(0, 60, 0),
        _event(1, 64, 0),
        _event(2, 67, 0),
        _event(4, 60, 1),
        _event(5, 64, 1),
        _event(6, 67, 1),
    ]
    chords = estimate_bar_chords(events)

    assert estimate_global_key(chords) == "C major"


def test_estimate_bar_chords_marks_single_pitch_bar_as_unknown() -> None:
    events = [_event(0, 60, 0)]

    chords = estimate_bar_chords(events)

    assert len(chords) == 1
    assert chords[0].chord == "C:unknown"
    assert chords[0].quality == "unknown"
    assert chords[0].confidence == 0.0


def test_estimate_global_key_returns_none_for_unknown_chords_only() -> None:
    events = [_event(0, 60, 0), _event(4, 72, 1)]
    chords = estimate_bar_chords(events)

    assert estimate_global_key(chords) is None


def test_estimate_bar_chords_keeps_single_pitch_root_label() -> None:
    events = [_event(0, 62, 0)]

    chords = estimate_bar_chords(events)

    assert len(chords) == 1
    assert chords[0].chord == "D:unknown"
    assert chords[0].root_pc == 2


def test_estimate_bar_chords_aggregates_notes_across_tracks() -> None:
    events = [
        _event(0, 60, 0, track_id=0),
        _event(1, 64, 0, track_id=1),
        _event(2, 67, 0, track_id=2),
    ]

    chords = estimate_bar_chords(events)

    assert len(chords) == 1
    assert chords[0].chord == "C:major"
    assert chords[0].confidence == 1.0
