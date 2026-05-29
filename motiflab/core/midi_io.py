from __future__ import annotations

from collections import defaultdict
from statistics import median

from motiflab.core.constants import DEFAULT_BAR_BEATS
from motiflab.core.event_model import NoteEvent, TrackSummary


def _likely_role(summary: dict) -> tuple[str, float]:
    if summary["is_drum"]:
        return "drums", 0.95
    if summary["median_pitch"] <= 50 and summary["polyphony_rate"] < 0.2:
        return "bass", 0.75
    if summary["polyphony_rate"] > 0.45 and summary["median_duration_beats"] >= 0.75:
        return "chords", 0.7
    if summary["polyphony_rate"] < 0.25 and summary["median_pitch"] >= 60:
        return "melody", 0.6
    return "unknown", 0.4


def load_midi_file(path: str, file_id: str | None = None) -> tuple[list[NoteEvent], list[TrackSummary], int]:
    try:
        import mido
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mido is required to parse MIDI files") from exc

    midi = mido.MidiFile(path)
    file_id = file_id or path
    ticks_per_beat = midi.ticks_per_beat
    if ticks_per_beat <= 0:
        raise ValueError("Invalid MIDI file: ticks_per_beat must be > 0")

    events: list[NoteEvent] = []
    tempo_us_per_beat = 500000

    for track_id, track in enumerate(midi.tracks):
        track_ticks = 0
        active_notes: dict[tuple[int | None, int], list[tuple[int, int]]] = defaultdict(list)
        program = None

        for msg in track:
            track_ticks += msg.time
            if msg.type == "set_tempo":
                tempo_us_per_beat = msg.tempo
            if msg.type == "program_change":
                program = msg.program
            if msg.type == "note_on" and msg.velocity > 0:
                active_notes[(getattr(msg, "channel", None), msg.note)].append((track_ticks, msg.velocity))
                continue

            is_note_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
            if not is_note_off:
                continue

            key = (getattr(msg, "channel", None), msg.note)
            if not active_notes[key]:
                continue
            onset_tick, velocity = active_notes[key].pop(0)
            offset_tick = track_ticks
            if offset_tick <= onset_tick:
                continue

            onset_seconds = mido.tick2second(onset_tick, ticks_per_beat, tempo_us_per_beat)
            offset_seconds = mido.tick2second(offset_tick, ticks_per_beat, tempo_us_per_beat)
            duration_tick = offset_tick - onset_tick
            duration_seconds = max(0.0, offset_seconds - onset_seconds)
            onset_beat = onset_tick / ticks_per_beat

            events.append(
                NoteEvent(
                    file_id=file_id,
                    track_id=track_id,
                    channel=getattr(msg, "channel", None),
                    instrument_program=program,
                    is_drum=getattr(msg, "channel", None) == 9,
                    pitch=msg.note,
                    pitch_class=msg.note % 12,
                    onset_seconds=onset_seconds,
                    offset_seconds=offset_seconds,
                    duration_seconds=duration_seconds,
                    onset_tick=onset_tick,
                    offset_tick=offset_tick,
                    duration_tick=duration_tick,
                    velocity=velocity,
                    estimated_tempo=(60_000_000 / tempo_us_per_beat),
                    bar_index=int(onset_beat // DEFAULT_BAR_BEATS),
                    beat_index=onset_beat % DEFAULT_BAR_BEATS,
                    onset_beat=onset_beat,
                    duration_beats=duration_tick / ticks_per_beat,
                    microtiming_ms=None,
                    metrical_strength=None,
                )
            )

    track_events: dict[int, list[NoteEvent]] = defaultdict(list)
    for event in events:
        track_events[event.track_id].append(event)

    summaries: list[TrackSummary] = []
    for track_id, tr_events in sorted(track_events.items()):
        pitches = [event.pitch for event in tr_events]
        durations = [event.duration_beats or 0.0 for event in tr_events]
        bars = {event.bar_index for event in tr_events if event.bar_index is not None}
        bars_count = max(1, len(bars))

        intervals = sorted((event.onset_tick, event.offset_tick) for event in tr_events)
        active_offsets: list[int] = []
        overlap_count = 0
        for onset_tick, offset_tick in intervals:
            active_offsets = [active for active in active_offsets if active > onset_tick]
            if active_offsets:
                overlap_count += 1
            active_offsets.append(offset_tick)
        polyphony_rate = overlap_count / max(1, len(intervals))

        summary_data = {
            "is_drum": any(event.is_drum for event in tr_events),
            "median_pitch": float(median(pitches)),
            "polyphony_rate": polyphony_rate,
            "median_duration_beats": float(median(durations)) if durations else 0.0,
        }
        role, confidence = _likely_role(summary_data)

        summaries.append(
            TrackSummary(
                track_id=track_id,
                instrument_program=tr_events[0].instrument_program,
                is_drum=summary_data["is_drum"],
                note_count=len(tr_events),
                pitch_min=min(pitches),
                pitch_max=max(pitches),
                median_pitch=summary_data["median_pitch"],
                median_duration_beats=summary_data["median_duration_beats"],
                density_notes_per_bar=len(tr_events) / bars_count,
                polyphony_rate=polyphony_rate,
                syncopation_score=0.0,
                likely_role=role,
                confidence=confidence,
            )
        )

    return events, summaries, ticks_per_beat
