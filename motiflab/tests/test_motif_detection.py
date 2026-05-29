from motiflab.analysis.motif import find_motif_candidates
from motiflab.core.event_model import NoteEvent


def _event(index: int, pitch: int) -> NoteEvent:
    onset_tick = index * 480
    return NoteEvent(
        file_id="demo",
        track_id=0,
        channel=0,
        instrument_program=0,
        is_drum=False,
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
        bar_index=index // 4,
        beat_index=float(index % 4),
        onset_beat=float(index),
        duration_beats=1.0,
        microtiming_ms=0.0,
        metrical_strength=1.0,
    )


def test_motif_detection_finds_exact_and_transposed_repetition() -> None:
    pitches = [60, 62, 64, 60, 62, 64, 65, 67, 69]
    events = [_event(i, pitch) for i, pitch in enumerate(pitches)]

    motifs = find_motif_candidates(events, min_notes=3, max_notes=3, min_occurrences=2)

    assert motifs
    first = motifs[0]
    transformations = {occ.transformation for occ in first.occurrences}
    assert "exact" in transformations
    assert "transposition" in transformations
