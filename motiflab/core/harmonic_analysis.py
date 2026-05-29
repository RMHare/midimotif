from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from motiflab.core.constants import DEFAULT_BAR_BEATS
from motiflab.core.event_model import NoteEvent

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TRIADS = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
}


@dataclass(slots=True)
class ChordEstimate:
    bar_index: int
    chord: str
    root_pc: int
    quality: str
    confidence: float
    pitch_classes: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


def _bar_index_for_event(event: NoteEvent) -> int:
    if event.bar_index is not None:
        return event.bar_index
    if event.onset_beat is not None:
        return int(event.onset_beat // DEFAULT_BAR_BEATS)
    return 0


def _best_triad(pitch_classes: list[int]) -> tuple[int, str, float]:
    unique = set(pitch_classes)
    if not unique:
        return 0, "unknown", 0.0
    if len(unique) == 1:
        return next(iter(unique)), "unknown", 0.0

    best_root = 0
    best_quality = "unknown"
    best_score = 0.0

    for root in range(12):
        for quality, intervals in TRIADS.items():
            triad = {(root + interval) % 12 for interval in intervals}
            overlap = len(unique & triad)
            score = overlap / len(triad)
            if score > best_score:
                best_root = root
                best_quality = quality
                best_score = score

    return best_root, best_quality, best_score


def estimate_bar_chords(events: list[NoteEvent]) -> list[ChordEstimate]:
    by_bar: dict[int, list[int]] = defaultdict(list)
    for event in events:
        if event.is_drum:
            continue
        by_bar[_bar_index_for_event(event)].append(event.pitch_class)

    estimates: list[ChordEstimate] = []
    for bar_index in sorted(by_bar):
        pcs = by_bar[bar_index]
        root_pc, quality, confidence = _best_triad(pcs)
        label_quality = quality if confidence > 0.0 else "unknown"
        chord = f"{PITCH_CLASS_NAMES[root_pc]}:{label_quality}"
        estimates.append(
            ChordEstimate(
                bar_index=bar_index,
                chord=chord,
                root_pc=root_pc,
                quality=label_quality,
                confidence=round(confidence, 3),
                pitch_classes=sorted(set(pcs)),
            )
        )
    return estimates


def estimate_global_key(chords: list[ChordEstimate]) -> str | None:
    if not chords:
        return None
    tonal_chords = [chord for chord in chords if chord.quality in {"major", "minor"}]
    if not tonal_chords:
        return None
    roots = Counter(chord.root_pc for chord in tonal_chords)
    top_root, _ = roots.most_common(1)[0]
    major_count = sum(1 for chord in tonal_chords if chord.quality == "major")
    minor_count = sum(1 for chord in tonal_chords if chord.quality == "minor")
    mode = "major" if major_count >= minor_count else "minor"
    return f"{PITCH_CLASS_NAMES[top_root]} {mode}"
