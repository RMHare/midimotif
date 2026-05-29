from motiflab.core.event_model import NoteEvent
from motiflab.core.timing import assign_rough_beats


def test_assign_rough_beats_preserves_raw_values_and_sets_metrical_fields() -> None:
    event = NoteEvent(
        file_id="demo",
        track_id=0,
        channel=0,
        instrument_program=0,
        is_drum=False,
        pitch=60,
        pitch_class=0,
        onset_seconds=0.0,
        offset_seconds=0.5,
        duration_seconds=0.5,
        onset_tick=960,
        offset_tick=1440,
        duration_tick=480,
        velocity=90,
        estimated_tempo=120.0,
        bar_index=None,
        beat_index=None,
        onset_beat=None,
        duration_beats=None,
        microtiming_ms=None,
        metrical_strength=None,
    )

    [updated] = assign_rough_beats([event], ticks_per_beat=480)

    assert updated.onset_tick == 960
    assert updated.duration_tick == 480
    assert updated.onset_beat == 2.0
    assert updated.duration_beats == 1.0
    assert updated.bar_index == 0
    assert updated.beat_index == 2.0
