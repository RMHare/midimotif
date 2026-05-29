from __future__ import annotations

from dataclasses import asdict, dataclass

from motiflab.analysis.transformations import detect_exact_repetition, detect_transposition
from motiflab.core.event_model import NoteEvent


@dataclass(slots=True)
class MotifOccurrence:
    start_index: int
    end_index: int
    transformation: str
    transposition: int | None


@dataclass(slots=True)
class MotifCandidate:
    id: str
    track_id: int
    pitch_pattern: list[int]
    interval_vector: list[int]
    occurrences: list[MotifOccurrence]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "track_id": self.track_id,
            "pitch_pattern": self.pitch_pattern,
            "interval_vector": self.interval_vector,
            "occurrences": [asdict(o) for o in self.occurrences],
        }


def _intervals(pitches: list[int]) -> list[int]:
    return [b - a for a, b in zip(pitches, pitches[1:])]


def find_motif_candidates(
    events: list[NoteEvent],
    min_notes: int = 3,
    max_notes: int = 8,
    min_occurrences: int = 2,
) -> list[MotifCandidate]:
    if len(events) < min_notes:
        return []

    by_track: dict[int, list[NoteEvent]] = {}
    for event in events:
        by_track.setdefault(event.track_id, []).append(event)

    results: list[MotifCandidate] = []
    motif_id = 0

    for track_id, track_events in by_track.items():
        track_events = sorted(track_events, key=lambda event: (event.onset_tick, event.pitch))
        used_patterns: set[tuple[int, ...]] = set()
        max_window_size = min(max_notes, len(track_events))

        for width in range(min_notes, max_window_size + 1):
            for start in range(0, len(track_events) - width + 1):
                base = track_events[start : start + width]
                base_pitches = [event.pitch for event in base]
                base_key = tuple(_intervals(base_pitches))

                if base_key in used_patterns:
                    continue

                occurrences: list[MotifOccurrence] = [
                    MotifOccurrence(
                        start_index=start,
                        end_index=start + width,
                        transformation="source",
                        transposition=0,
                    )
                ]

                for other_start in range(start + 1, len(track_events) - width + 1):
                    other = track_events[other_start : other_start + width]
                    other_pitches = [event.pitch for event in other]

                    transformation = None
                    transposition: int | None = None
                    if detect_exact_repetition(base_pitches, other_pitches):
                        transformation = "exact"
                        transposition = 0
                    else:
                        shift = detect_transposition(base_pitches, other_pitches)
                        if shift is not None:
                            transformation = "transposition"
                            transposition = shift

                    if transformation is not None:
                        occurrences.append(
                            MotifOccurrence(
                                start_index=other_start,
                                end_index=other_start + width,
                                transformation=transformation,
                                transposition=transposition,
                            )
                        )

                if len(occurrences) >= min_occurrences:
                    used_patterns.add(base_key)
                    motif_id += 1
                    results.append(
                        MotifCandidate(
                            id=f"motif_{motif_id}",
                            track_id=track_id,
                            pitch_pattern=base_pitches,
                            interval_vector=list(base_key),
                            occurrences=occurrences,
                        )
                    )

    return results
