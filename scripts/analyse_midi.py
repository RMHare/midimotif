from __future__ import annotations

import argparse
import json
from pathlib import Path

from motiflab.analysis.motif import find_motif_candidates
from motiflab.core.midi_io import load_midi_file
from motiflab.core.timing import assign_rough_beats


def run(input_midi: Path, output_json: Path) -> None:
    events, track_summaries, ticks_per_beat = load_midi_file(str(input_midi))
    beat_events = assign_rough_beats(events, ticks_per_beat)
    motifs = find_motif_candidates(beat_events)

    output = {
        "input": str(input_midi),
        "ticks_per_beat": ticks_per_beat,
        "event_count": len(beat_events),
        "track_summaries": [summary.to_dict() for summary in track_summaries],
        "motifs": [motif.to_dict() for motif in motifs],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a MIDI file and export symbolic analysis JSON")
    parser.add_argument("input_midi", type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis.json"))
    args = parser.parse_args()
    run(args.input_midi, args.output)


if __name__ == "__main__":
    main()
